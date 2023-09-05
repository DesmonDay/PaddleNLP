# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning jointly dataset. """

import random
from collections import namedtuple
from dataclasses import dataclass
from typing import List

import numpy as np
import paddle
from dataset.base import MultiSourceDatset
from paddle.io import IterableDataset, get_worker_info

from paddlenlp.utils.log import logger


@dataclass
class Example:
    src: List[str]
    tgt: List[str]
    response: List[str]
    sort: List
    score: List
    label: List


@dataclass
class Sequences:
    token_ids: List[List[int]]
    position_ids: List[List[int]]
    labels: List[List[int]]
    loss_mask: List[List[int]]
    rm_loss_mask: List
    rm_labels: List[int]


def pad_batch_data(
    insts,
    pad_idx=0,
    return_pos=False,
    max_seq_len=None,
    return_input_mask=False,
    return_max_len=False,
    return_num_token=False,
    return_seq_lens=False,
):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max_seq_len if max_seq_len is not None else max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array([inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len])]

    # position data
    if return_pos:
        inst_pos = np.array([list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst)) for inst in insts])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


def create_dataset(**dataset_config):
    """Create RM dataset."""
    example_dataset = MultiSourceDatset(
        task_group_filename=dataset_config["task_group_filename"],
        sub_dataset_type="file",
        process_fn=process_example,
    )
    sequence_dataset = SequenceJointDataset(
        dataset=example_dataset,
        tokenizer=dataset_config["tokenizer"],
        max_seq_len=dataset_config["max_seq_len"],
        is_valid=dataset_config.get("is_valid", False),
        random_seed=dataset_config["random_seed"],
        num_comparisons=dataset_config["num_comparisons"],
        use_cls=dataset_config["use_cls"],
    )
    return sequence_dataset


def collate_fn(batch: List[Sequences], tokenizer, max_seq_len: int):
    """Convert batch data into tensor."""
    input_keys = [
        "input_ids",
        "position_ids",
        "attention_mask",
        "labels",
        "loss_mask",
        "rm_loss_indices",
        "rm_loss_mask",
        "rm_labels",
    ]
    batch_token_ids = sum([sequences.token_ids for sequences in batch], [])
    batch_position_ids = sum([sequences.position_ids for sequences in batch], [])
    batch_loss_mask = sum([sequences.loss_mask for sequences in batch], [])
    batch_labels = sum([sequences.labels for sequences in batch], [])
    batch_rm_loss_mask = [sequences.rm_loss_mask for sequences in batch]
    batch_rm_labels = sum([sequences.rm_labels for sequences in batch], [])

    rm_loss_indices = []
    for i, token_ids in enumerate(batch_token_ids):
        # 默认 使用最后一个 token 预测 RM score
        rm_loss_indices.append([i, len(token_ids) - 1])
    batch_loss_indices = np.array(rm_loss_indices, dtype="int64")

    # padding
    padded_token_ids = pad_batch_data(batch_token_ids, pad_idx=tokenizer.pad_token_id, max_seq_len=max_seq_len)
    padded_position_ids = pad_batch_data(batch_position_ids, pad_idx=0, max_seq_len=max_seq_len)
    padded_labels = pad_batch_data(batch_labels, pad_idx=tokenizer.pad_token_id, max_seq_len=max_seq_len)
    padded_loss_mask = pad_batch_data(batch_loss_mask, pad_idx=0, max_seq_len=max_seq_len)
    padded_labels = np.where(padded_loss_mask == 1, padded_labels, tokenizer.pad_token_id)

    # add in-batch mask
    input_mask = np.tril(np.ones([len(batch_token_ids), 1, max_seq_len, max_seq_len]), 0)

    return_list = [
        padded_token_ids,
        padded_position_ids,
        input_mask,
        padded_labels,
        padded_loss_mask,
        batch_loss_indices,
        batch_rm_loss_mask,
        batch_rm_labels,
    ]

    return_list = [paddle.to_tensor(tensor_list) for tensor_list in return_list]
    return_list[-1] = return_list[-1].reshape([-1, 1])

    input_dict = dict(zip(input_keys, return_list))
    return input_dict


def process_example(data, input_file):
    """Convert raw json format example to Example."""
    if isinstance(data["src"], str):
        data["src"] = [data["src"]]
    if isinstance(data["tgt"], str):
        data["tgt"] = [data["tgt"]]

    for item in data["src"] + data["tgt"]:
        if len(item.strip()) == 0:
            return None

    if not (len(data["src"]) == len(data["tgt"]) + 1):
        return None

    if not (len(data["response"]) == len(data["sort"])):
        return None

    if data["score"] is None or data["score"] == []:
        data["score"] = [-100] * len(data["sort"])

    if "label" not in data:
        label = []
        for s in data["score"]:
            label.append([0] * len(data["tgt"]) + [1 if int(s) > 2 else 0])
        data["label"] = label

    return Example(
        src=data["src"],
        tgt=data["tgt"],
        response=data["response"],
        score=data["score"],
        sort=data["sort"],
        label=data["label"],
    )


class SequenceJointDataset(IterableDataset):
    """Create sequences from Example Dataset.

    This is a stateful dataset.
    """

    def __init__(
        self,
        dataset: MultiSourceDatset,
        tokenizer,
        max_seq_len: int = 4096,
        is_valid: bool = False,
        random_seed: int = 11,
        num_comparisons: int = 6,
        use_cls: bool = True,
    ):
        self.example_dataset = dataset
        self.tokenizer = tokenizer
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.sep_token = self.break_token = "<sep>"
        self.cls_token = self.break_turn_token = "<cls>"
        self.max_seq_len = max_seq_len
        self.use_anti_k_sampling = False
        self.add_break_token_multi_turn_for_nontrigger_data = True
        self.is_valid = is_valid
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self.num_comparisons = num_comparisons
        self.use_cls = use_cls
        self.epoch_index = 0
        self.tokenizer.markup_tokens = []

    def __iter__(self):
        # epoch_rng only use in this epoch.
        headers = ["src", "tgt", "label"]
        Record = namedtuple("Record", headers)
        epoch_rng = np.random.RandomState(self.epoch_index)
        worker_info = get_worker_info()

        # prepare epoch data
        logger.debug("prepare SequenceJointDataset ...")
        examples_per_task = []
        examples_all = []
        for task in self.example_dataset._task_group:
            examples = [ex for ex in task["dataset"]]
            epoch_rng.shuffle(examples)
            if worker_info is not None:
                examples = examples[worker_info.id :: worker_info.num_workers]
            examples_per_task.append(examples)
            examples_all.extend(examples)
        epoch_rng.shuffle(examples_all)
        epoch_rng.shuffle(examples_per_task)
        logger.debug(f"prepare SequenceJointDataset done: total number of examples is {len(examples_all)}")

        # [NOTE] random_seed is strange (use the same random seed in shuffle epoch data), need to double check
        for index, example in enumerate(examples_all):
            responses = example.response
            sort = list(map(int, example.sort))
            score = list(map(float, example.score))
            label = example.label

            # sort responses by rank
            all_sort, all_response, all_score, all_label = map(
                list, zip(*sorted(zip(sort, responses, score, label), key=lambda x: x[0]))
            )
            all_response = all_response[: self.num_comparisons]
            all_sort = all_sort[: self.num_comparisons]
            all_score = all_score[: self.num_comparisons]
            if len(set(all_sort)) <= 1:
                continue

            all_sort_np = np.asarray(all_sort)
            x = (all_sort_np[None, :] != all_sort_np[:, None]).astype(np.int32)
            score_mask = np.zeros([self.num_comparisons, self.num_comparisons])
            score_mask[: len(all_response), : len(all_response)] = x

            while len(all_score) < self.num_comparisons and len(all_score) > 0:
                all_response.append(all_response[-1])
                all_score.append(-100)
                all_label.append([0] * len(all_label[-1]))

            records = []
            for label, response in zip(all_label, all_response):
                cur_data = [example.src, example.tgt + [response], label]
                cur_example = Record(*cur_data)
                record = self._postprocess_sequence(cur_example)
                records.append(record)

            if None in records:
                continue

            token_ids, pos_ids, labels, loss_mask = map(list, zip(*records))
            sequence = Sequences(
                token_ids=token_ids,
                position_ids=pos_ids,
                labels=labels,
                loss_mask=loss_mask,
                rm_loss_mask=score_mask,
                rm_labels=all_score,
            )
            yield sequence

        self.epoch_index += 1

    def _postprocess_sequence(self, example):
        """Post process sequence: tokenization & truncation."""
        tokens = []
        loss_mask = []
        previous_cur_len = 2 if self.use_cls else 3  # <s>, <cls>, </s>
        reserved_multi_turn_break_length = 2  # break_token, break_token_multi_turn

        turn_index = len(example.src) - 1

        while turn_index >= 0:
            src, tgt = example.src[turn_index].strip(), example.tgt[turn_index].strip()
            tokens_src, tokens_target = self.tokenizer.tokenize(src), self.tokenizer.tokenize(tgt)
            is_parts_a_truncated, is_parts_b_truncated = self._truncate_seq_pair(
                tokens_src, tokens_target, self.max_seq_len - previous_cur_len - reserved_multi_turn_break_length
            )

            if is_parts_b_truncated or is_parts_a_truncated:
                break

            cur_tokens = tokens_src + [self.break_token] + tokens_target + [self.break_turn_token]

            loss_mask = (
                [0] * len(tokens_src) + [example.label[turn_index]] * (len(tokens_target) + 1) + [0] + loss_mask
            )

            tokens = cur_tokens + tokens
            previous_cur_len += len(cur_tokens)
            turn_index -= 1

        if len(tokens) <= 2:
            return None

        if self.add_break_token_multi_turn_for_nontrigger_data:
            tokens = [self.start_token, self.break_turn_token] + tokens
            loss_mask = [0] * 2 + loss_mask  # 2 for start_token & break_turn_token
        else:
            tokens = [self.start_token] + tokens
            loss_mask = [0] * 1 + loss_mask  # 1 for start_token

        if len(tokens) > self.max_seq_len:
            raise RuntimeError(f"token_ids is too long: {len(tokens)}")

        tokens = tokens + [self.end_token]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        labels = token_ids[1:]
        if self.use_cls:
            token_ids = token_ids[:-1]
        else:
            labels += [self.tokenizer.convert_token_to_ids("<pad>")]
            loss_mask += [0]
        pos_ids = list(range(len(token_ids)))

        return token_ids, pos_ids, labels, loss_mask

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.

        markups_poped_a = []
        markups_poped_b = []

        def pop(tokens_list, poped_list):
            while len(tokens_list) > 0:
                poped_token = tokens_list.pop()
                if poped_token in self.tokenizer.markup_tokens:
                    poped_list.append(poped_token)
                else:
                    break

        is_parts_a_truncated, is_parts_b_truncated = False, False
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(markups_poped_a) + len(markups_poped_b)
            if total_length <= max_length or (len(tokens_a) + len(tokens_b) == 0):
                break
            if len(tokens_a) > len(tokens_b):
                is_parts_a_truncated = True
                pop(tokens_a, markups_poped_a)
            else:
                is_parts_b_truncated = True
                pop(tokens_b, markups_poped_b)

        if len(tokens_a) + len(tokens_b) != 0:
            tokens_a.extend(markups_poped_a)
            tokens_b.extend(markups_poped_b)

        return is_parts_a_truncated, is_parts_b_truncated
