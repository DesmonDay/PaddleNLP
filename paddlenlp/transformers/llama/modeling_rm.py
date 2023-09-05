# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Llama reward model"""

from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers.model_outputs import ModelOutput

from .configuration import LlamaConfig
from .modeling import LlamaForCausalLM, LlamaLMHead

__all__ = [
    "LlamaRewardModel",
]


class LlamaRewardModelOutput(ModelOutput):
    """
    Output class for outputs of reward models.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`paddle.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None
    rm_logits: Optional[paddle.Tensor] = None
    rm_loss: Optional[paddle.Tensor] = None
    nll_loss: Optional[paddle.Tensor] = None


class LlamaRewardCriterion(nn.Layer):
    """
    Criterion for Llama reward model. It calculates the final loss.
    """

    def __init__(self, use_point_loss=False):
        super(LlamaRewardCriterion, self).__init__()
        self.use_point_loss = use_point_loss
        if use_point_loss:
            self.mse_loss_func = paddle.nn.loss.MSELoss(reduction="mean")
        self.ranking_loss = 0
        self.point_loss = 0

    def forward(self, logits, loss_mask, rm_labels, compute_loss=True):
        if self.use_point_loss:
            point_loss_idx = paddle.nonzero(loss_mask.sum(axis=-1).reshape(logits.shape))
            point_logits = logits.gather_nd(point_loss_idx)
            rm_labels = rm_labels.gather_nd(point_loss_idx)
            point_loss_idx = paddle.nonzero(rm_labels != -100)
            point_logits = point_logits.gather_nd(point_loss_idx)
            rm_labels = rm_labels.gather_nd(point_loss_idx)
            if point_loss_idx.shape[0] != 0:
                point_loss = self.mse_loss_func(point_logits, rm_labels * 2 - 4)

        micro_bsz = loss_mask.shape[-1]
        logits = logits.reshape([-1, 1, micro_bsz])

        # broadcast x to shape of [bsz, micro_bsz, micro_bsz]
        logits = paddle.expand(logits, [-1, micro_bsz, -1])
        logits_b = logits

        # compute the score gap
        score_sub = paddle.subtract(logits.transpose([0, 2, 1]), logits_b)
        loss_mask = paddle.tensor.tril(loss_mask, diagonal=-1)

        loss = 0
        if compute_loss:
            score_log = paddle.nn.functional.log_sigmoid(score_sub)
            loss = -paddle.mean(
                paddle.multiply(loss_mask, score_log).sum(axis=[1, 2]) / (loss_mask.sum(axis=[1, 2]) + 1e-5)
            )
            self.ranking_loss = loss.item()
            if self.use_point_loss:
                if point_loss_idx.shape[0] != 0:
                    self.point_loss = point_loss.item()
                    loss = 0.9 * loss + 0.1 * point_loss
        else:
            loss = paddle.mean(
                (paddle.multiply(loss_mask, score_sub) > 0).sum(axis=[1, 2]) / loss_mask.sum(axis=[1, 2])
            )
        return loss


class LlamaValueHead(nn.Layer):
    def __init__(self, config):
        super(LlamaValueHead, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.pooled = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(mean=0.0, std=self.config.initializer_range)
            ),
            bias_attr=False,
        )
        self.score = nn.Linear(
            self.hidden_size,
            1,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(mean=0.0, std=self.config.initializer_range)
            ),
            bias_attr=False,
        )

    def forward(self, hidden_states):
        hidden_states = F.tanh(self.pooled(hidden_states))
        logits = self.score(hidden_states)
        return logits


class LlamaRewardModel(LlamaForCausalLM):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super(LlamaRewardModel, self).__init__(config)
        self.use_point_loss = False
        self.use_rank_loss = True
        self.use_lm_loss = config.get("use_lm_loss", False)

        if self.use_lm_loss:
            self.lm_head = LlamaLMHead(config)
            self.tie_weights()

        self.value_head = LlamaValueHead(config)
        self.criterion = LlamaRewardCriterion(use_point_loss=self.use_point_loss)
        self.nll_loss = 0
        self.rm_loss = 0
        self.point_loss = 0

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        loss_mask=None,
        rm_loss_indices=None,
        rm_loss_mask=None,
        rm_labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.llama(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        loss = 0.0
        nll_loss, lm_logits, rm_loss, rm_logits = None, None, None, None
        rm_logits = self.value_head(hidden_states)
        if rm_loss_indices is not None:
            rm_logits = rm_logits.gather_nd(rm_loss_indices)
            rm_loss = self.criterion(
                rm_logits, rm_loss_mask, rm_labels, compute_loss=True
            )  # no lm_logits input currently
            loss += rm_loss
            self.rm_loss = self.criterion.ranking_loss

        if self.use_lm_loss:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            outputs = (lm_logits,) + outputs[1:] + (rm_logits,)
            return ((loss,) + outputs) if loss != 0 else outputs

        return LlamaRewardModelOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            rm_logits=rm_logits,
            rm_loss=rm_loss,
            nll_loss=nll_loss,
        )
