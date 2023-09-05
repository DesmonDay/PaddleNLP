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

set -x
unset CUDA_VISIBLE_DEVICES
task_name="llama_hybid_rm"
#rm -rf output/$task_name/
#rm -rf "output/$task_name""_log"

PYTHONPATH=../../:$PYTHONPATH  \
python -m paddle.distributed.launch \
    --log_dir "output/$task_name""_log" \
    train_rm.py \
    --model_name_or_path "/root/.paddlenlp/models/facebook/llama-7b/" \
    --model_type LlamaRewardModel \
    --output_dir "output/$task_name" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --train_task_config task_rm.json \
    --eval_task_config task_rm.json \
    --max_steps 8500 \
    --num_train_epochs 20 \
    --save_steps 500 \
    --logging_steps 1 \
    --eval_steps 1000000000000 \
    --weight_decay 0.01 \
    --evaluation_strategy steps \
    --tensor_parallel_degree 8 \
    --max_seq_len 2048 \
    --seed 23 \
    --warmup_steps 600 \
    --lr_scheduler_type "linear" \
    --learning_rate 1e-5 \
    --bf16 \
    --fp16_opt_level O2 \
    --disable_tqdm True \
    --num_comparisons 6 \
    --use_flash_attention true \
    --recompute true \
    --recompute_granularity full \
    --do_train \
    --dataloader_num_workers 1
