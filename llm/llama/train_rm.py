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
""" Training Llama Reward Model  """

import json
import os
from dataclasses import dataclass, field
from functools import partial

import paddle
from utils import LlamaTrainer

from paddlenlp.trainer import (
    PdArgumentParser,
    TrainingArguments,
    get_last_checkpoint,
    set_seed,
)
from paddlenlp.transformers import LlamaRewardModel, LlamaTokenizer
from paddlenlp.utils.log import logger


@dataclass
class DataArgument:
    train_task_config: str = field(default="./task_rm.json", metadata={"help": "Path to the training task config."})
    eval_task_config: str = field(default=None, metadata={"help": "Path to the evaluation task config."})
    max_seq_len: int = field(default=4096, metadata={"help": "Maximum sequence length."})
    num_comparisons: int = field(default=6, metadata={"help": "Number of candidate responses."})
    use_cls: bool = field(default=True, metadata={"help": "Whether to use cls to predict RM score."})


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default="facebook/llama-7b", metadata={"help": "Pretrained model name or path to local directory."}
    )
    model_type: str = field(
        default="ErnieBotRewardModel",
        metadata={"help": "Pretrained model type to select model module for initialization."},
    )
    tensor_parallel_output: bool = field(
        default=True,
        metadata={
            "help": "True for cross-entropy loss calculation in parallel models with "
            "fleet.meta_parallel.ParallelCrossEntropy."
        },
    )
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use Flash Attention."})
    recompute_granularity: str = field(
        default="full",
        metadata={"help": "Choose among ['full', 'core_attn', 'full_attn']"},
    )


def main():
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    set_seed(training_args.seed)
    paddle.set_device(training_args.device)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: "
        f"{training_args.world_size}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 1:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set the dtype for loading model
    dtype = paddle.get_default_dtype()
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    logger.info("Start to load model ...")
    model_config = json.load(open(f"{model_args.model_name_or_path}/config.json"))
    assert model_args.model_type == "LlamaRewardModel", "Only support reward model training in this script!"
    model_class = LlamaRewardModel
    logger.info("Use model type: LlamaRewardModel")
    if model_config["architectures"][0] != model_args.model_type:
        logger.warning(
            f"""WARNING: define model type as {model_args.model_type} from model_args.model_type. But load pretrain architectures from {model_config["architectures"][0]}, check whether is what you need."""
        )

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        load_state_as_np=True,
        dtype=dtype,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        use_recompute=training_args.recompute,
        recompute_granularity=model_args.recompute_granularity,
        use_flash_attention=model_args.use_flash_attention,
        tensor_parallel_output=model_args.tensor_parallel_output,
    )

    logger.info("Loading model successfully!")

    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)
    if isinstance(tokenizer, LlamaTokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Start to create dataset ...")
    config_dataset = {
        "tokenizer": tokenizer,
        "num_comparisons": data_args.num_comparisons,
        "use_cls": data_args.use_cls,
        "max_seq_len": data_args.max_seq_len,
        "random_seed": training_args.seed,
    }
    logger.info("Using dataset type from dataset.joint for ErnieBotRewardModel")
    from dataset.joint import collate_fn, create_dataset

    train_dataset = create_dataset(task_group_filename=data_args.train_task_config, **config_dataset)
    if training_args.do_eval:
        eval_dataset = create_dataset(task_group_filename=data_args.eval_task_config, is_valid=True, **config_dataset)
    logger.info("Creating dataset successfully ...")

    trainer = LlamaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        do_generation=False,
        data_collator=partial(collate_fn, tokenizer=tokenizer, max_seq_len=data_args.max_seq_len),
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
