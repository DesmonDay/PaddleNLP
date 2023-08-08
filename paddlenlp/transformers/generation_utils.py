# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
from __future__ import annotations

import inspect
from abc import ABC
from collections import OrderedDict
from typing import Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.common_ops_import import convert_dtype
from paddle.fluid.dygraph.base import in_declarative_mode
from paddle.utils import map_structure

try:
    from paddle import top_p_sampling

    is_top_p_sampling_avaliable = True
except:
    is_top_p_sampling_avaliable = False

from paddlenlp.utils.log import logger

from .model_outputs import ModelOutput
from .utils import get_scale_by_dtype

__all__ = ["GenerationMixin"]


def get_unfinished_flag(
    input_ids: Tensor, unfinished_flag: Tensor, eos_token_id: Union[int, list[int], list[list[int]]]
) -> Tensor:
    """get unfinished flag for generation step

    Args:
        input_ids (Tensor): the input_ids
        eos_token_id (Union[int, list[int], list[list[int]]]): the end os sentence flag, which can be:
            * single token id, eg: 10
            * multiple token ids to stop generation, eg: [10, 10]
            * some more tokens to stop generations, eg: [[10], [20, 20], [30, 30, 30]]

    Returns:
        Tensor: the unfinished flag tensor
    """
    if isinstance(eos_token_id, int):
        unfinished_flag = paddle.logical_and(unfinished_flag, input_ids[:, -1:] != eos_token_id)
    elif isinstance(eos_token_id[0], int):
        eos_token_id_tensor = paddle.to_tensor([eos_token_id])
        is_last_tokens_equal = paddle.all(
            paddle.equal(input_ids[:, -len(eos_token_id) :], eos_token_id_tensor), axis=-1
        ).unsqueeze(-1)
        unfinished_flag = paddle.logical_and(unfinished_flag, ~is_last_tokens_equal)
    else:
        batch_unfinish_flag = None
        for batch_eos_token_id in eos_token_id:
            if batch_unfinish_flag is None:
                batch_unfinish_flag = ~get_unfinished_flag(input_ids, unfinished_flag, batch_eos_token_id)
            else:
                batch_unfinish_flag = paddle.logical_or(
                    batch_unfinish_flag, ~get_unfinished_flag(input_ids, unfinished_flag, batch_eos_token_id)
                )

        unfinished_flag = ~batch_unfinish_flag
    return unfinished_flag


class BeamHypotheses:
    def __init__(self, num_beams, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = get_scale_by_dtype()

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs, origin_len=0):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (((hyp.shape[-1] - origin_len + 5) / 6) ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len, origin_len=0):
        """
        If there are enough hypotheses and that none of the hypotheses being
        generated can become better than the worst one in the heap, then we
        are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / ((cur_len - origin_len + 5) / 6) ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class BeamSearchScorer(object):
    """
    implementing standard beam search decoding.
    """

    def __init__(
        self,
        batch_size,
        max_length,
        num_beams,
        length_penalty=1.0,
        do_early_stopping=False,
        num_beam_hyps_to_keep=1,
        num_beam_groups=1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams, length_penalty=self.length_penalty, early_stopping=self.do_early_stopping
            )
            for _ in range(batch_size)
        ]
        self._done = paddle.to_tensor([0 for _ in range(batch_size)], dtype="int64")

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                "`num_beams` has to be an integer strictly greater than 1, but "
                "received {}. For `num_beams` == 1, one should make use of "
                "`greedy_search` instead.".format(num_beams)
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than "
                "`num_beams` and `num_beams` has to be divisible by "
                "`num_beam_groups`, but received num_beam_groups={}, num_beams="
                "{}.".format(num_beam_groups, num_beams)
            )

    @property
    def is_done(self):
        return paddle.min(self._done) == 1

    def process(
        self, input_ids, next_scores, next_tokens, next_indices, origin_len=0, pad_token_id=None, eos_token_id=None
    ):
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        next_beam_scores = paddle.zeros([batch_size, self.group_size], dtype=next_scores.dtype)
        next_beam_tokens = paddle.zeros([batch_size, self.group_size], dtype=next_tokens.dtype)
        next_beam_indices = paddle.zeros([batch_size, self.group_size], dtype=next_indices.dtype)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx] == 1:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # If beam_token does not belong to top num_beams tokens,
                    # it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(input_ids[batch_beam_idx.item()].clone(), next_score.item(), origin_len)

                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token.item()
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx.item()
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    "At most {} tokens in `next_tokens[batch_idx]` can be equal "
                    "to `eos_token_id: {}`. Make sure `next_tokens[batch_idx]` "
                    "are corrected.".format(self.group_size, eos_token_id)
                )

            # Check if we are done so that we can save a pad step if all(done)
            if beam_hyp.is_done(next_scores[batch_idx].max().item(), cur_len, origin_len):
                self._done[batch_idx] = 1

        return {
            "next_beam_scores": next_beam_scores.reshape([-1]),
            "next_beam_tokens": next_beam_tokens.reshape([-1]),
            "next_beam_indices": next_beam_indices.reshape([-1]),
        }

    def finalize(
        self,
        input_ids,
        final_beam_scores,
        final_beam_tokens,
        final_beam_indices,
        origin_len=0,
        pad_token_id=None,
        eos_token_id=None,
    ):
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx] == 1:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score, origin_len=origin_len)

        # select the best hypotheses
        sent_lengths = paddle.zeros([batch_size * self.num_beam_hyps_to_keep], dtype=input_ids.dtype)
        best = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_score, best_hyp = sorted_hyps.pop()
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append([best_hyp, best_score])

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded = paddle.zeros([batch_size * self.num_beam_hyps_to_keep, sent_max_len], dtype=input_ids.dtype)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded[:, :] = pad_token_id
        decoded_score = paddle.zeros([batch_size * self.num_beam_hyps_to_keep, 1])

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, score) in enumerate(best):
            decoded[i, : sent_lengths[i].item()] = hypo.numpy()
            decoded_score[i] = score
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i].item()] = eos_token_id
        return decoded, decoded_score


class GenerationMixin(object):
    r"""
    This class implements the interface for generation task.

    It's used as the base class of `paddlenlp.transformers.PretrainedModel
    <https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.model_utils.html>`__.
    """
    # enable `to_static` method for CausalLM Model
    enable_to_static_method = False

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
        return paddle.ones([batch_size, 1], dtype="int64") * bos_token_id

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids == pad_token_id).astype(paddle.get_default_dtype()) * get_scale_by_dtype(
                return_positive=False
            )
        else:
            attention_mask = paddle.zeros_like(input_ids, dtype=paddle.get_default_dtype())
        return paddle.unsqueeze(attention_mask, axis=[1, 2])

    @staticmethod
    def prepare_seq_len_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            seq_len = paddle.sum(input_ids != pad_token_id, axis=1).unsqueeze(-1)
        else:
            seq_len = paddle.full((input_ids.shape[0], 1), input_ids.shape[1], dtype="int64")
        return seq_len

    def get_logits_processor(
        self,
        min_length=None,
        max_length=None,
        eos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        num_beams=1,
        num_beam_groups=1,
        diversity_rate=0.0,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        logits_processors=None,
    ):
        processors = LogitsProcessorList()

        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if num_beam_groups > 1 and diversity_rate > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_rate=diversity_rate, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        # TODO
        # Add more pre_processing for distribution

        if logits_processors is not None:
            custom_processors = LogitsProcessorList()
            custom_processors_type = [type(lp) for lp in logits_processors]

            for processor in processors:
                if type(processor) not in custom_processors_type:
                    custom_processors.append(processor)
            custom_processors.extend(logits_processors)

            return custom_processors
        else:
            return processors

    @staticmethod
    def expand_inputs_for_generation(input_ids, expand_size, attention_mask=None, **model_kwargs):

        index = paddle.tile(
            paddle.arange(paddle.shape(input_ids)[0], dtype="int64").unsqueeze(-1), [1, expand_size]
        ).reshape([-1])

        input_ids = paddle.gather(input_ids, index)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = paddle.gather(attention_mask, index)

        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.gather(token_type_ids, index)

        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.gather(position_ids, index)

        if "seq_len" in model_kwargs and model_kwargs["seq_len"] is not None:
            seq_len = model_kwargs["seq_len"]
            model_kwargs["seq_len"] = paddle.gather(seq_len, index)

        if "encoder_output" in model_kwargs and model_kwargs["encoder_output"] is not None:
            encoder_output = model_kwargs["encoder_output"]
            model_kwargs["encoder_output"] = paddle.gather(encoder_output, index)

        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.gather(role_ids, index)

        return input_ids, model_kwargs

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["cache"] = outputs[1]
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, ModelOutput) and "past_key_values" in outputs:
            model_kwargs["cache"] = outputs.past_key_values
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat([token_type_ids, token_type_ids[:, -1:]], axis=-1)

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        # update attention_mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # nn.Pad2D don't support the data type `bool`
            if convert_dtype(attention_mask.dtype) == "bool":
                attention_mask = paddle.cast(attention_mask, "int64")
            if len(attention_mask.shape) == 4:
                cur_device = paddle.get_device()
                if cur_device.split(":")[0] == "npu":
                    attention_mask = nn.Pad2D([0, 0, 0, 1], mode="constant")(attention_mask)
                    attention_mask = nn.Pad2D([0, 1, 0, 0], value=0)(attention_mask)
                else:
                    attention_mask = nn.Pad2D([0, 0, 0, 1], mode="replicate")(attention_mask)
                    attention_mask = nn.Pad2D([0, 1, 0, 0], value=get_scale_by_dtype(return_positive=False))(
                        attention_mask
                    )

                dtype = convert_dtype(attention_mask.dtype)
                if "int" in dtype:
                    attention_mask[:, :, -1, -1] = 1
                elif "float" in dtype:
                    attention_mask[:, :, -1, -1] = 0.0
                else:
                    raise ValueError("The data type of input `attention_mask` must " "be bool, int or float")
            else:
                attention_mask = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype="int64")], axis=-1
                )
            model_kwargs["attention_mask"] = attention_mask

        # update role_ids
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat([role_ids, role_ids[:, -1:]], axis=-1)

        return model_kwargs

    @staticmethod
    def update_scores_for_generation(scores, next_scores, length, unfinished_flag):
        # update scores

        unfinished_scores = (scores * length + next_scores) / (length + 1)
        scores = paddle.where(unfinished_flag, unfinished_scores, scores)
        return scores

    def prepare_encoder_decoder_kwargs_for_generation(self, input_ids, model_kwargs):
        if "encoder_output" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (
                    argument.startswith("decoder_") or argument.startswith("cross_attn") or argument == "use_cache"
                )
            }
            # Use inputs_embeds as the priority if inputs_embeds exists
            if "inputs_embeds" in encoder_kwargs:
                model_kwargs["encoder_output"] = encoder(**encoder_kwargs)
            else:
                model_kwargs["encoder_output"] = encoder(input_ids=input_ids, **encoder_kwargs)
        return model_kwargs

    def prepare_decoder_input_ids_for_generation(self, input_ids, decoder_start_token_id=None, bos_token_id=None):
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else bos_token_id

        decoder_input_ids = paddle.ones([input_ids.shape[0], 1], dtype="int64") * decoder_start_token_id

        return decoder_input_ids

    def get_decoder_start_token_id(self, decoder_start_token_id=None, bos_token_id=None):
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif self.config.decoder_start_token_id is not None:
            return self.config.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif self.config.bos_token_id is not None:
            return self.config.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Implement in subclasses for custom behavior to prepare inputs in the
        # generate method.

        return {"input_ids": input_ids}

    def adjust_logits_during_generation(self, logits):
        # Implement in subclasses for custom behavior to adjust the logits in
        # the generate method.

        return logits

    def prepare_fast_entry(self, kwargs):
        return False

    def _convert_to_fast(self, kwargs):
        # try general convert
        pass

    def _build_fast(self, kwargs):
        self._fast_entry = False
        if kwargs["num_beam_groups"] != 1:
            # not support for group_beam_search yet in the fast version
            raise AttributeError("'num_beam_groups != 1' is not supported yet in the fast version")
        if paddle.get_default_dtype() == "float16" and kwargs["use_fp16_decoding"] is False:
            logger.info(
                "Since the default dtype is float16, float16 would be used " "though 'use_fp16_decoding=False'."
            )
            kwargs["use_fp16_decoding"] = True
        self.prepare_fast_entry(kwargs)

    @paddle.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        max_length=20,
        min_length=0,
        decode_strategy="greedy_search",
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        num_beams=1,
        num_beam_groups=1,
        length_penalty=0.0,
        early_stopping=False,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        decoder_start_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        no_repeat_ngram_size=None,
        num_return_sequences=1,
        diversity_rate=0.0,
        use_cache=True,
        use_fast=False,
        use_fp16_decoding=False,
        **model_kwargs
    ):
        r"""
        The interface for generation task. This method can generate sequences
        by using decoding strategy. Currently, there are three decoding
        strategies supported: "greedy_search", "sampling" and "beam_search".

        Args:
            input_ids (Tensor, optional): The input sequence ids for the
                generation. It is a Tensor with shape [batch_size, sequence_length].
                The data type should be int32 or int64. Default to None, which
                we will initialize it as a Tensor with shape [1, 1], filled
                with the value `bos_token_id`.
            max_length (int, optional): The maximum length of the sequence to
                be generated. Default to 20.
            min_length (int, optional): The minimum length of the sequence to
                be generated. Default to 0.
            decode_strategy (str, optional): The decoding strategy in generation.
                Currently, there are three decoding strategies supported:
                "greedy_search", "sampling" and "beam_search". Default to
                "greedy_search".
            temperature (float, optional): The value used to module the next
                token probabilities in the "sampling" strategy. Default to 1.0,
                which means no effect.
            top_k (int, optional): The number of highest probability tokens to
                keep for top-k-filtering in the "sampling" strategy. Default to
                0, which means no effect.
            top_p (float, optional): The cumulative probability for
                top-p-filtering in the "sampling" strategy. The value should
                satisfy :math:`0 <= top\_p < 1`. Default to 1.0, which means no
                effect.
            repetition_penalty (float, optional):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details. Defaults to 1.0.
            num_beams (int, optional): The number of beams in the "beam_search"
                strategy. Default to 1.
            num_beam_groups (int, optional):
                Number of groups to divide `num_beams` into in order to use DIVERSE
                BEAM SEARCH. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__
                for more details. Default to 1.
            length_penalty (float, optional): The exponential penalty to the
                sequence length in the "beam_search" strategy. The larger this
                param is, the more that the model would generate shorter
                sequences. Default to 0.0, which means no penalty.
            early_stopping (bool, optional): Whether to stop searching in the
                "beam_search" strategy when at least `num_beams` sentences are
                finished per batch or not. Default to False.
            bos_token_id (int, optional): The id of the `bos_token`. Default to
                None.
            eos_token_id (int, optional): The id of the `eos_token`. Default to
                None.
            pad_token_id (int, optional): The id of the `pad_token`. Default to
                None.
            decoder_start_token_id (int, optional): The start token id for
                encoder-decoder models. Default to None.
            forced_bos_token_id (int, optional): The id of the token to force as
                the first generated token. Usually use for multilingual models.
                Default to None.
            forced_eos_token_id (int, optional): The id of the token to force as
                the last generated token. Default to None.
            num_return_sequences (int, optional): The number of returned
                sequences for each sequence in the batch. Default to 1.
            diversity_rate (float, optional): If num_beam_groups is 1, this is the
                diversity_rate for Diverse Siblings Search. See
                `this paper https://arxiv.org/abs/1611.08562`__ for more details.
                If not, this is the diversity_rate for DIVERSE BEAM SEARCH.
            use_cache: (bool, optional): Whether to use the model cache to
                speed up decoding. Default to True.
            use_fast: (bool, optional): Whether to use fast entry of model
                for FastGeneration. Default to False.
            use_fp16_decoding: (bool, optional): Whether to use fp16 for decoding.
                Only works when fast entry is avalible. Default to False.
            model_kwargs (dict): It can be used to specify additional kwargs
                passed to the model.

        Returns:
            tuple[Tensor]: It is a tuple contains two elements: ids and scores.
            Each element is a Tensor.

            With the fields:

            - ids (Tensor):
                The ids of the generated sequences. It is a Tensor with shape
                [batch_size * num_return_sequences, sequence_length]. The data
                type is same as the input `input_ids`.
            - scores (Tensor):
                The scores of the generated sequences. It is a Tensor with shape
                [batch_size * num_return_sequences, 1]. The data type is float32
                or float64, which is the same as the parameters in the model.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import (
                    UnifiedTransformerLMHeadModel,
                    UnifiedTransformerTokenizer
                )

                paddle.seed(2)

                # Initialize the model and tokenizer
                model_name_or_path = 'unified_transformer-12L-cn-luge'
                model = UnifiedTransformerLMHeadModel.from_pretrained(model_name_or_path)
                tokenizer = UnifiedTransformerTokenizer.from_pretrained(model_name_or_path)

                # Prepare the model inputs.
                history = "早上好，今天空气质量不错。"
                inputs = tokenizer.dialogue_encode(history, task_type='chitchat',
                    add_start_token_as_response=True, return_tensors=True)

            .. code-block::

                # Generate the sequence by using "greedy_search" strategy
                ids, scores = model.generate(
                    input_ids=inputs['input_ids'],
                    token_type_ids=inputs['token_type_ids'],
                    position_ids=inputs['position_ids'],
                    attention_mask=inputs['attention_mask'],
                    decode_strategy="greedy_search")
                print(ids.shape, scores.shape)
                # [1, 3] [1, 1]
                sequence_ids = ids.numpy().tolist()[0]
                sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                response = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                print(response)
                # 是的

            .. code-block::

                # Generate 2 sequences by using "sampling" strategy (top_k=5)
                ids, scores = model.generate(
                    input_ids=inputs['input_ids'],
                    token_type_ids=inputs['token_type_ids'],
                    position_ids=inputs['position_ids'],
                    attention_mask=inputs['attention_mask'],
                    decode_strategy="sampling",
                    top_k=5,
                    num_return_sequences=2)
                print(ids.shape, scores.shape)
                # [2, 7] [2, 1]
                response = []
                for sequence_ids in ids.numpy().tolist():
                    sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                    text = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                    response.append(text)
                print(response)
                # ['天气好,心情也好', '你也是']

            .. code-block::

                # Generate 2 sequences by using "beam_search" strategy (num_beams=5)
                ids, scores = model.generate(
                    input_ids=inputs['input_ids'],
                    token_type_ids=inputs['token_type_ids'],
                    position_ids=inputs['position_ids'],
                    attention_mask=inputs['attention_mask'],
                    decode_strategy="beam_search",
                    num_beams=5,
                    num_return_sequences=2)
                print(ids.shape, scores.shape)
                # [2, 3] [2, 1]
                response = []
                for sequence_ids in ids.numpy().tolist():
                    sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                    text = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                    response.append(text)
                print(response)
                # ['是的', '嗯嗯']
        """
        assert decode_strategy in [
            "greedy_search",
            "sampling",
            "beam_search",
        ], "`decode_strategy` must be one of 'greedy_search', 'sampling' or 'beam_search' but received {}.".format(
            decode_strategy
        )

        # Whether to dynamic to static
        is_tracing = False
        if in_declarative_mode():
            is_tracing = True

        if is_tracing:
            assert decode_strategy in [
                "sampling",
            ], "`generate()` only supports 'sampling' temporarily but received {}.".format(decode_strategy)

        if getattr(self, "deprecated_warnings", None) is None:
            self.deprecated_warnings = {}

        if "use_faster" in model_kwargs:
            use_fast = model_kwargs.pop("use_faster")
            if not self.deprecated_warnings.get("use_faster", False):
                logger.warning("`use_faster` will be deprecated in near future. Please use `use_fast` instead. ")
                self.deprecated_warnings["use_faster"] = True

        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )

        if is_tracing:
            self._fast_entry = None

        if getattr(self, "_fast_entry", None) is not False and use_fast:
            args = locals()
            args.pop("self")
            args.pop("__class__", None)
            model_kwargs = args.pop("model_kwargs")
            args.update(model_kwargs)
            try:
                if getattr(self, "_fast_entry", None) is None:
                    self._build_fast(args)
                if self._fast_entry:
                    output = self._fast_entry(**args)
                    if isinstance(output, tuple):
                        output_ids, dummy_srore = output
                    else:
                        output_ids = output
                        # make result and fast result oneconsistent
                        dummy_srore = None
                    if decode_strategy == "beam_search":
                        output_ids = output_ids.transpose([1, 2, 0])
                        output_ids = output_ids[:, :num_return_sequences, :].reshape([-1, output_ids.shape[-1]])
                        if dummy_srore is not None:
                            dummy_srore = dummy_srore[:, :num_return_sequences].flatten()
                    else:
                        output_ids = output_ids.transpose([1, 0])
                    return output_ids, dummy_srore

            except Exception as e:
                args["model_kwargs"] = model_kwargs
                # TODO
                # Prevent self._convert_to_fast to throw Exception
                self._convert_to_fast(args)
                logger.warning(e)
                logger.warning("FastGeneration is not available, " "and the original version would be used instead.")

        # params check
        if input_ids is None and "inputs_embeds" not in model_kwargs:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)
        elif "inputs_embeds" in model_kwargs:
            # Add input embeds support
            input_ids = self.prepare_input_ids_for_generation(
                bos_token_id, encoder_output=model_kwargs["inputs_embeds"]
            )

        # Add to model_kwargs
        model_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            model_kwargs["position_ids"] = position_ids

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self.prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )
        self.is_encoder_decoder = (
            getattr(self, "encoder", None) is not None and getattr(self, "decoder", None) is not None
        )
        if self.is_encoder_decoder:
            model_kwargs = self.prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)
            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self.prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id, bos_token_id
                )
        if pad_token_id is None and eos_token_id is not None:
            print("Setting `pad_token_id` to `eos_token_id`:{} for " "open-end generation.".format(eos_token_id))
            pad_token_id = eos_token_id

        model_kwargs["use_cache"] = use_cache

        if is_tracing and not paddle.is_tensor(max_length):
            if hasattr(paddle.framework, "_no_check_dy2st_diff"):
                with paddle.framework._no_check_dy2st_diff():
                    min_len = input_ids.shape[-1]
                    max_len = input_ids.shape[-1]
                    paddle.increment(min_len, min_length)
                    paddle.increment(max_len, max_length)
            else:
                min_len = input_ids.shape[-1]
                max_len = input_ids.shape[-1]
                paddle.increment(min_len, min_length)
                paddle.increment(max_len, max_length)
        else:
            input_len = input_ids.shape[-1]
            min_len = input_len + min_length
            max_len = input_len + max_length

        logits_processors = self.get_logits_processor(
            min_length=min_len if min_length > 0 else None,
            max_length=max_len,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_rate=diversity_rate,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            logits_processors=model_kwargs["logits_processors"]
            if "logits_processors" in model_kwargs
            and isinstance(model_kwargs["logits_processors"], LogitsProcessorList)
            else None,
        )
        if "logits_processors" in model_kwargs:
            model_kwargs.pop("logits_processors")

        if decode_strategy == "greedy_search":
            if num_return_sequences > 1:
                raise ValueError(
                    "`num_return_sequences` has to be 1, but is {} "
                    "when doing greedy search.".format(num_return_sequences)
                )
            return self.greedy_search(
                input_ids, logits_processors, max_len, pad_token_id, eos_token_id, **model_kwargs
            )

        elif decode_strategy == "sampling":
            if num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_return_sequences, **model_kwargs
                )

            if is_tracing:
                return self.sample_d2s(
                    input_ids,
                    logits_processors,
                    max_len,
                    pad_token_id,
                    eos_token_id,
                    top_k,
                    top_p,
                    temperature,
                    **model_kwargs,
                )
            else:
                return self.sample(
                    input_ids,
                    logits_processors,
                    max_len,
                    pad_token_id,
                    eos_token_id,
                    top_k,
                    top_p,
                    temperature,
                    **model_kwargs,
                )

        elif decode_strategy == "beam_search":
            batch_size = input_ids.shape[0]
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to "
                    "`num_beams`. But received `num_return_sequences` is {}, "
                    "`num_beams` is {}".format(num_return_sequences, num_beams)
                )
            if num_beams <= 1:
                raise ValueError(
                    "`num_beams` has to be bigger than 1. But received "
                    "`num_beams` is {}. If `num_beams` is 1, `decode_strategy` "
                    "should be 'greedy_search'".format(num_beams)
                )
            if num_beam_groups > 1:
                diverse_beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_len,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_early_stopping=early_stopping,
                    num_beam_hyps_to_keep=num_return_sequences,
                    num_beam_groups=num_beam_groups,
                )

                # interleave with `num_beams`
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_beams, **model_kwargs
                )

                return self.group_beam_search(
                    input_ids,
                    diverse_beam_scorer,
                    logits_processors,
                    max_len,
                    pad_token_id,
                    eos_token_id,
                    **model_kwargs,
                )
            else:
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_len,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_early_stopping=early_stopping,
                    num_beam_hyps_to_keep=num_return_sequences,
                )

                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_beams, **model_kwargs
                )

                return self.beam_search(
                    input_ids,
                    beam_scorer,
                    logits_processors,
                    max_len,
                    diversity_rate,
                    pad_token_id,
                    eos_token_id,
                    **model_kwargs,
                )

    def greedy_search(self, input_ids, logits_processors, max_length, pad_token_id, eos_token_id, **model_kwargs):
        model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()
        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())
        while cur_len < max_length:

            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :]

            # pre-process distribution
            next_token_logits = self.adjust_logits_during_generation(next_token_logits)
            next_tokens_scores = logits_processors(input_ids, next_token_logits)
            # greedy
            probs = F.softmax(next_tokens_scores)
            probs = paddle.log(probs)
            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(probs.astype("float32"), next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1

            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag):
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )

        return input_ids[:, origin_len:], scores

    def sample(
        self,
        input_ids,
        logits_processors,
        max_length,
        pad_token_id,
        eos_token_id,
        top_k=None,
        top_p=None,
        temperature=None,
        min_tokens_to_keep=1,
        **model_kwargs
    ):
        model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)

        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # sample
            origin_probs = F.softmax(logits)
            origin_probs = paddle.log(origin_probs)
            if temperature is not None and temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits)
            if top_k is not None and top_k != 0:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
            if top_p is not None and top_p < 1.0:
                probs = TopPProcess(probs, top_p, min_tokens_to_keep)

            # multinomial not support fp16 and bf16 currently, issue: https://github.com/PaddlePaddle/Paddle/issues/51852
            if probs.dtype == paddle.bfloat16 and top_k == 1:
                probs = probs.astype("float32")
                next_tokens = paddle.unsqueeze(paddle.argmax(probs, axis=-1), -1)
            else:
                next_tokens = paddle.multinomial(probs)

            if self.config.tensor_parallel_degree > 1:
                paddle.distributed.broadcast(next_tokens, 0)

            next_scores = paddle.index_sample(origin_probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag):
                break
            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
        return input_ids[:, origin_len:], scores

    def to_static(self, path: str, config: dict):
        """export generation model to static

        Args:
            path (str): path of saved inference model
            config (dict): configuration for generation
                bos_token_id (int): token id of begin-of-sentence
                eos_token_id (int): token id of end-of-sentence
                pad_token_id (int): token id of pad token
                use_top_p (bool): whether use top_p decoding strategy
        """

        use_top_p = config.get("use_top_p", True)

        top_k_spec = paddle.static.InputSpec(shape=[1], dtype="int64") if not use_top_p else 0

        top_p_spec = paddle.static.InputSpec(shape=[1], dtype="float32") if use_top_p else 1.0
        temperature = paddle.static.InputSpec(shape=[1], dtype="float32") if use_top_p else 1.0

        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # attention_mask
            None,  # position_ids
            paddle.static.InputSpec(shape=[1], dtype="int64"),  # max_length
            0,  # min_length
            "sampling",  # decode_strategy
            temperature,  # temperature
            top_k_spec,  # top_k
            top_p_spec,  # top_p
            1,  # repetition_penalty
            # num_beams
            1,
            # num_beam_groups
            1,
            # length_penalty
            0.0,
            # early_stopping
            False,
            # bos_token_id
            config.get("bos_token_id", 0),
            # eos_token_id
            config.get("eos_token_id", 0),
            # pad_token_id
            config.get("pad_token_id", 0),
            # decoder_start_token_id
            None,
            # forced_bos_token_id
            None,
            # forced_eos_token_id
            None,
            # no_repeat_ngram_size
            None,
            # num_return_sequences
            1,
            # diversity_rate
            0.0,
            # use_cache
            True,
            # use_fast=False,
            False,
            # use_fp16_decoding=False,
            False,
        ]

        model = paddle.jit.to_static(self.generate, input_spec=input_spec)

        paddle.jit.save(model, path)

    def sample_d2s(
        self,
        input_ids,
        logits_processors,
        max_length,
        pad_token_id,
        eos_token_id,
        top_k=None,
        top_p=None,
        temperature=None,
        min_tokens_to_keep=1,
        **model_kwargs
    ):

        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        if paddle.is_tensor(top_k) and not paddle.is_tensor(top_p):
            use_top_p = False
        elif not paddle.is_tensor(top_k) and paddle.is_tensor(top_p):
            use_top_p = True

        # top_k and top_p are the const value
        elif isinstance(top_p, float) or isinstance(top_k, int):
            use_top_p = True
        else:
            if top_p is None and top_k is None:
                raise ValueError("top_k and top_p should not be None")
            raise ValueError(
                "you should not specify InputSpec for top_k and top_p parameters, one of InputSpec is expected"
            )

        use_topp_sampling_op = is_top_p_sampling_avaliable or model_kwargs.get("use_fuse_topp_sampling", False)
        return_scores = model_kwargs.get("return_scores", True)

        batch_size, cur_len = paddle.shape(input_ids)
        # used for compute on gpu, avoid memcpy D2H
        cur_len_gpu = paddle.full([1], cur_len, dtype="int64")

        origin_len = paddle.shape(input_ids)[1]
        # used for compute on gpu, avoid memcpy D2H
        origin_len_gpu = paddle.full([1], origin_len, dtype="int64")

        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        if return_scores:
            scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())
        else:
            scores = None

        # use_cache is immutable, we split it off other mutable kwargs.
        assert "use_cache" in model_kwargs
        immutable = {"use_cache": model_kwargs["use_cache"]}
        del model_kwargs["use_cache"]

        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **args, **immutable)
            assert "use_cache" in model_inputs
            del model_inputs["use_cache"]
            return self(**model_inputs, **immutable)

        def _post_process_(outputs, input_ids, cur_len, origin_len, scores, unfinished_flag, model_kwargs):
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)

            logits = logits_processors(input_ids, logits)
            probs = F.softmax(logits)

            # sample
            if return_scores:
                origin_probs = F.softmax(logits)
                origin_probs = paddle.log(origin_probs)

            # compute next_tokens
            if use_top_p:
                logits = logits / temperature
                if use_topp_sampling_op:
                    top_ps_tensor = paddle.full(shape=[paddle.shape(probs)[0], 1], fill_value=top_p, dtype=probs.dtype)
                    _, next_tokens = top_p_sampling(probs, top_ps_tensor)
                else:
                    probs = TopPProcess(probs, top_p, min_tokens_to_keep)
                    next_tokens = paddle.multinomial(probs)
            else:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
                if top_k == 1:
                    next_tokens = paddle.unsqueeze_(paddle.argmax(probs, axis=-1), -1)
                else:
                    next_tokens = paddle.multinomial(probs)

            if return_scores:
                next_scores = paddle.index_sample(origin_probs, next_tokens)
                scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )

            return input_ids, scores, unfinished_flag, model_kwargs

        outputs = _forward_(**model_kwargs)
        input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
            outputs, input_ids, cur_len_gpu, origin_len_gpu, scores, unfinished_flag, model_kwargs
        )

        if hasattr(paddle.framework, "_no_check_dy2st_diff"):
            with paddle.framework._no_check_dy2st_diff():
                paddle.increment(cur_len)
                paddle.increment(cur_len_gpu)
        else:
            paddle.increment(cur_len)
            paddle.increment(cur_len_gpu)

        attn_mask = model_kwargs["attention_mask"]
        # make the shape of attention_mask = (-1, -1, -1, -1) in dy2static.
        model_kwargs["attention_mask"] = paddle.reshape(attn_mask, paddle.shape(attn_mask))
        model_kwargs["cache"] = outputs[1] if isinstance(outputs, tuple) else None
        max_length = paddle.full([1], max_length, dtype="int64")

        if hasattr(paddle.framework, "_no_check_dy2st_diff"):
            with paddle.framework._no_check_dy2st_diff():
                while cur_len < max_length and paddle.any(unfinished_flag):
                    input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
                        _forward_(**model_kwargs),
                        input_ids,
                        cur_len_gpu,
                        origin_len_gpu,
                        scores,
                        unfinished_flag,
                        model_kwargs,
                    )
                paddle.increment(cur_len)
                paddle.increment(cur_len_gpu)
        else:
            while cur_len < max_length and paddle.any(unfinished_flag):
                input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
                    _forward_(**model_kwargs),
                    input_ids,
                    cur_len_gpu,
                    origin_len_gpu,
                    scores,
                    unfinished_flag,
                    model_kwargs,
                )
            paddle.increment(cur_len)
            paddle.increment(cur_len_gpu)

        return input_ids[:, origin_len:], scores

    def beam_search(
        self,
        input_ids,
        beam_scorer,
        logits_processors,
        max_length,
        diversity_rate,
        pad_token_id,
        eos_token_id,
        **model_kwargs
    ):
        model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)

        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        batch_beam_size, cur_len = input_ids.shape
        origin_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size
        )

        beam_scores = paddle.zeros((batch_size, num_beams), dtype=paddle.get_default_dtype())

        beam_scores[:, 1:] = get_scale_by_dtype(return_positive=False)
        beam_scores = paddle.reshape(beam_scores, [-1])

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            # beam search
            # [batch_size * num_beams, vocab_size]
            next_scores = F.softmax(logits)
            next_scores = paddle.log(next_scores)
            next_scores = logits_processors(input_ids, next_scores)
            next_scores = next_scores + beam_scores.unsqueeze(-1)

            vocab_size = next_scores.shape[-1]
            if diversity_rate == 0.0:
                # reshape for beam search
                next_scores = next_scores.reshape([batch_size, num_beams * vocab_size])

                next_scores, next_tokens = paddle.topk(next_scores, 2 * num_beams, axis=1)

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

            else:
                next_scores, next_tokens = paddle.topk(next_scores, 2 * num_beams, axis=1)

                sibling_score = paddle.arange(1, 2 * num_beams + 1, dtype="int64").unsqueeze(0) * diversity_rate

                diversed_score = next_scores - sibling_score

                next_scores = next_scores.reshape([batch_size, 2 * num_beams * num_beams])
                next_tokens = next_tokens.reshape([batch_size, 2 * num_beams * num_beams])

                diversed_score = diversed_score.reshape([batch_size, 2 * num_beams * num_beams])
                diversed_score, diversed_tokens = paddle.topk(diversed_score, 2 * num_beams, axis=1)

                # TODO
                # Use gather_nd() to select origan token and score
                next_scores = paddle.stack(
                    [paddle.index_select(next_scores[i], diversed_tokens[i]) for i in range(next_scores.shape[0])]
                )
                next_tokens = paddle.stack(
                    [paddle.index_select(next_tokens[i], diversed_tokens[i]) for i in range(next_tokens.shape[0])]
                )

                next_indices = diversed_tokens // (2 * num_beams)

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                origin_len=origin_len,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            cur_len += 1
            input_ids = paddle.concat(
                [paddle.index_select(input_ids, beam_idx), beam_next_tokens.unsqueeze(-1)], axis=-1
            )

            if beam_scorer.is_done:
                break
            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            if "cache" in model_kwargs and model_kwargs["cache"] is not None:
                # reorder the cache
                model_kwargs["cache"] = map_structure(
                    lambda x: paddle.index_select(x, beam_idx), model_kwargs["cache"]
                )

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            origin_len=origin_len,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        return pred_ids[:, origin_len:], scores

    def group_beam_search(
        self, input_ids, beam_scorer, logits_processors, max_length, pad_token_id, eos_token_id, **model_kwargs
    ):
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups

        batch_beam_size, cur_len = input_ids.shape
        origin_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size
        )

        beam_scores = paddle.full((batch_size, num_beams), get_scale_by_dtype(return_positive=False), dtype="float32")
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = paddle.reshape(beam_scores, [-1])

        while cur_len < max_length:
            # predicted tokens in cur_len step
            current_tokens = paddle.zeros(shape=[batch_size * num_beams], dtype=input_ids.dtype)

            # indices which will form the beams in the next time step
            reordering_indices = paddle.zeros(shape=[batch_size * num_beams], dtype="int64")
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs)

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )

                group_input_ids = input_ids[batch_group_indices]

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif isinstance(outputs, ModelOutput):
                    logits = outputs.logits
                else:
                    logits = outputs

                logits = logits[:, -1, :]
                logits = paddle.index_select(logits, paddle.to_tensor(batch_group_indices))
                logits = self.adjust_logits_during_generation(logits)

                next_scores = F.softmax(logits)
                next_scores = paddle.log(next_scores)
                vocab_size = next_scores.shape[-1]

                next_scores = logits_processors(
                    group_input_ids, next_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )

                next_scores = next_scores + beam_scores[batch_group_indices].unsqueeze(-1)

                # reshape for beam search
                next_scores = next_scores.reshape([batch_size, group_size * vocab_size])

                next_scores, next_tokens = paddle.topk(next_scores, 2 * group_size, axis=1)

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_scores,
                    next_tokens,
                    next_indices,
                    origin_len=origin_len,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )

                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = paddle.concat(
                    [paddle.index_select(group_input_ids, index=beam_idx), beam_next_tokens.unsqueeze(-1)], axis=-1
                )
                current_tokens[batch_group_indices] = beam_next_tokens

                reordering_indices[batch_group_indices] = (
                    num_beams * (beam_idx // group_size) + group_start_idx + (beam_idx % group_size)
                )

            input_ids = paddle.concat([input_ids, current_tokens.unsqueeze(-1)], axis=-1)

            cur_len += 1
            if beam_scorer.is_done:
                break
            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            if "cache" in model_kwargs and model_kwargs["cache"] is not None:
                # reorder the cache
                model_kwargs["cache"] = map_structure(
                    lambda x: paddle.index_select(x, reordering_indices), model_kwargs["cache"]
                )

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            origin_len=origin_len,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        return pred_ids[:, origin_len:], scores


class LogitsProcessorList:
    """use ordered dict to store processors"""

    def __init__(self, processors: list[LogitsProcessor] = None) -> None:
        self._processors = OrderedDict()
        processors = processors or []
        for processor in processors:
            self.append(processor)

    def __call__(self, input_ids, logits, **kwargs):
        for processor in self._processors.values():
            processor_args = inspect.signature(processor.__call__).parameters
            if len(processor_args) > 2:
                assert all(
                    arg in kwargs for arg in list(processor_args.keys())[2:]
                ), f"The parameters don't match for {processor.__class__}"
                logits = processor(input_ids, logits, **kwargs)
            else:
                logits = processor(input_ids, logits)
        return logits

    def append(self, processor):
        self._processors[len(self._processors)] = processor


class LogitsProcessor(ABC):
    """
    Abstract base class for all logit processors that can be applied during
    generation.
    """

    def __call__(self, input_ids, logits):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. " "Only classes inheriting this class can be called."
        )


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    Enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (int): The minimum length of generation sequence.
        eos_token_id (int): The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length, eos_token_id):
        if min_length < 0 and not in_declarative_mode():
            raise ValueError("`min_length` should be a positive integer, but get {}".format(min_length))

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError("`eos_token_id` should be a positive integer, but get {}".format(eos_token_id))

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, logits):
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            logits[:, self.eos_token_id] = -float("inf")
        return logits


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    Enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (float):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: float):
        if not (penalty > 0) and not in_declarative_mode():
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids, logits):
        score = paddle.index_sample(logits, input_ids)
        score = paddle.where(score < 0, score * self.penalty, score / self.penalty)
        input_ids = input_ids + paddle.arange(logits.shape[0], dtype="int64").unsqueeze(-1) * logits.shape[-1]
        outputs = paddle.scatter(logits.flatten(), input_ids.flatten(), score.flatten()).reshape(logits.shape)
        return outputs


def _get_ngrams(ngram_size, prev_input_ids, num_hypos):
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(ngram_size, prev_input_ids, num_hypos, cur_len):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)

    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).
    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(self, input_ids, scores):
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores


class HammingDiversityLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces diverse beam search. Note that this logits
    processor is only effective for `group_beam_search`. See
    `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.

    Args:
        diversity_rate (float): This value is subtracted from a beam's score if
            it generates a token same as any beam from other group at a particular
            time.
        num_beams (int): Number of beams used for group beam search.
        num_beam_groups (int): Number of groups to divide `num_beams` into in order
            to ensure diversity among different groups of beams.
    """

    def __init__(self, diversity_rate, num_beams, num_beam_groups):
        if not isinstance(diversity_rate, float) or (not diversity_rate > 0.0):
            raise ValueError("`diversity_rate` should be a float strictly larger than 0.")
        self._diversity_rate = diversity_rate
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(self, input_ids, scores, current_tokens, beam_group_idx):
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        for batch_idx in range(batch_size):
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ]
            token_frequency = paddle.bincount(previous_group_tokens, minlength=vocab_size)
            scores[batch_idx * group_size : (batch_idx + 1) * group_size] -= self._diversity_rate * token_frequency

        return scores


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the first generated token to be the selected `forced_bos_token`.

    Args:
        forced_bos_token_id (:obj:`int`):
            The id of the token to be generated as the first token.
    """

    def __init__(self, forced_bos_token_id):
        self.forced_bos_token_id = forced_bos_token_id

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            scores[:] = -float("inf")
            scores[:, self.forced_bos_token_id] = 0
        return scores


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the last generated token to be the selected `forced_eos_token`.

    Args:
        max_length (int): The maximum length of the sequence to be generated.
        forced_eos_token_id (int): The id of the token to be generated as the last token.
    """

    def __init__(self, max_length, forced_eos_token_id):
        self.max_length = max_length
        self.forced_eos_token_id = forced_eos_token_id

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            scores[:] = -1e9  # TODO change back to -inf after paddle.topk is fixed
            scores[:, self.forced_eos_token_id] = 0
        return scores


def TopKProcess(probs, top_k, min_tokens_to_keep):
    top_k = min(max(top_k, min_tokens_to_keep), probs.shape[-1])
    # Remove all tokens with a probability less than the last token of the top-k
    # cast to float16 to support generation & d2s
    if probs.dtype == paddle.bfloat16:
        probs = paddle.cast(probs, paddle.float32)
        topk_probs, _ = paddle.topk(probs, k=top_k)
        topk_probs = paddle.cast(topk_probs, paddle.bfloat16)
    else:
        topk_probs, _ = paddle.topk(probs, k=top_k)

    probs = paddle.where(probs >= topk_probs[:, -1:], probs, paddle.full_like(probs, 0.0))
    return probs


def TopPProcess(probs, top_p, min_tokens_to_keep):
    sorted_indices = paddle.argsort(probs, descending=True)
    if isinstance(sorted_indices, tuple):
        sorted_probs, sorted_indices = sorted_indices
    else:
        sorted_probs = paddle.sort(probs, descending=True)

    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

    # Remove tokens with cumulative probs above the top_p, But keep at
    # least min_tokens_to_keep tokens
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        # Set 'min_tokens_to_keep - 1' because the first token is kept
        sorted_indices_to_remove[:, : min_tokens_to_keep - 1] = 0
    # Keep the first token
    sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    # Scatter sorted tensors to original indexing
    sorted_indices = sorted_indices + paddle.arange(probs.shape[0], dtype="int64").unsqueeze(-1) * probs.shape[-1]
    condition = paddle.scatter(
        sorted_indices_to_remove.flatten(), sorted_indices.flatten(), sorted_indices_to_remove.flatten()
    )
    condition = paddle.cast(condition, "bool").reshape(probs.shape)
    probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
    return probs
