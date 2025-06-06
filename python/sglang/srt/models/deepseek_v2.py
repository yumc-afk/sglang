# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference-only DeepseekV2 model."""

import logging
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import PretrainedConfig

from sglang.srt import two_batch_overlap
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
    decode_attention_fwd_grouped_rope,
)
from sglang.srt.layers.dp_attention import (
    dp_gather_partial,
    dp_scatter,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
    tp_all_gather,
    tp_reduce_scatter,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, EPMoE
from sglang.srt.layers.moe.ep_moe.token_dispatcher import (
    DEEPEP_NUM_SMS,
    DeepEPDispatcher,
)
from sglang.srt.layers.moe.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.deep_gemm import _ENABLE_JIT_DEEPGEMM
from sglang.srt.layers.quantization.fp8_kernel import (
    per_tensor_quant_mla_deep_gemm_masked_fp8,
    per_tensor_quant_mla_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    block_quant_to_tensor_quant,
    channel_quant_to_tensor_quant,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope, get_rope_wrapper
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.managers.schedule_batch import (
    get_global_expert_location_metadata,
    global_server_args_dict,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.weight_utils import (
    ModelParamNameInfo,
    ModelParamNameInfoMoe,
    ModelParamNameInfoOthers,
    default_weight_loader,
)
from sglang.srt.two_batch_overlap import model_forward_split_inputs
from sglang.srt.utils import (
    BumpAllocator,
    DeepEPMode,
    add_prefix,
    configure_deep_gemm_num_sms,
    get_bool_env_var,
    get_int_env_var,
    is_cuda,
    is_hip,
)

_is_hip = is_hip()
_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import awq_dequantize, bmm_fp8, merge_state_v2

    from sglang.srt.layers.quantization.deep_gemm import (
        grouped_gemm_nt_f8f8bf16_masked as deep_gemm_grouped_gemm_nt_f8f8bf16_masked,
    )
else:
    from vllm._custom_ops import awq_dequantize

if _is_hip:
    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
        decode_attention_fwd_grouped_rope,
    )

logger = logging.getLogger(__name__)


def _enable_moe_dense_fully_dp():
    return global_server_args_dict["moe_dense_tp_size"] == 1


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x, forward_mode: Optional[ForwardMode] = None):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MoEGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts))
            )
        else:
            self.e_score_correction_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)
        return logits


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: int = -999,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.n_share_experts_fusion = global_server_args_dict["n_share_experts_fusion"]
        self.layer_id = layer_id
        self.tp_rank = get_tensor_model_parallel_rank()

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(config=config, prefix=add_prefix("gate", prefix))

        MoEImpl = (
            DeepEPMoE
            if global_server_args_dict["enable_deepep_moe"]
            else (EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE)
        )

        self.experts = MoEImpl(
            num_experts=config.n_routed_experts
            + self.n_share_experts_fusion
            + global_server_args_dict["ep_num_redundant_experts"],
            top_k=config.num_experts_per_tok + min(self.n_share_experts_fusion, 1),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
            **(
                dict(deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]])
                if global_server_args_dict["enable_deepep_moe"]
                else {}
            ),
        )

        if config.n_shared_experts is not None and self.n_share_experts_fusion == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # disable tp for shared experts when enable deepep moe
            if not global_server_args_dict["enable_deepep_moe"]:
                self.shared_experts = DeepseekV2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=add_prefix("shared_experts", prefix),
                )
            else:
                self.shared_experts = DeepseekV2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=add_prefix("shared_experts", prefix),
                    tp_rank=0,
                    tp_size=1,
                )

        if global_server_args_dict["enable_deepep_moe"]:
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + global_server_args_dict["ep_num_redundant_experts"]
            )
            self.top_k = config.num_experts_per_tok
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

            self.deepep_dispatcher = self._create_deepep_dispatcher(config)

        if global_server_args_dict["enable_two_batch_overlap"]:
            # TODO maybe we do not need to create 2+1 dispatchers, but can reuse the one above
            self.tbo_deepep_dispatchers = [
                self._create_deepep_dispatcher(config) for i in range(2)
            ]

    def _create_deepep_dispatcher(self, config):
        return DeepEPDispatcher(
            group=parallel_state.get_tp_group().device_group,
            router_topk=self.top_k,
            permute_fusion=True,
            num_experts=config.n_routed_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            num_local_experts=config.n_routed_experts // self.tp_size,
            hidden_size=config.hidden_size,
            params_dtype=config.torch_dtype,
            deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
            async_finish=True,  # TODO
            return_recv_hook=True,
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_mode: Optional[ForwardMode] = None
    ) -> torch.Tensor:
        if not global_server_args_dict["enable_deepep_moe"]:
            return self.forward_normal(hidden_states)
        else:
            return self.forward_deepep(hidden_states, forward_mode)

    def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_output = self._forward_shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states *= self.routed_scaling_factor
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_mode: ForwardMode
    ) -> torch.Tensor:
        shared_output = None
        if (
            forward_mode is not None
            and not forward_mode.is_idle()
            and hidden_states.shape[0] > 0
        ):
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            shared_output = self._forward_shared_experts(hidden_states)
        else:
            router_logits = None

        self._forward_deepep_dispatch_a(
            self.deepep_dispatcher, forward_mode, hidden_states, router_logits
        )
        (
            hidden_states,
            topk_idx,
            topk_weights,
            reorder_topk_ids,
            num_recv_tokens_per_expert,
            seg_indptr,
            masked_m,
            expected_m,
        ) = self.deepep_dispatcher.dispatch_b()

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            reorder_topk_ids=reorder_topk_ids,
            seg_indptr=seg_indptr,
            masked_m=masked_m,
            expected_m=expected_m,
            num_recv_tokens_per_expert=num_recv_tokens_per_expert,
            forward_mode=forward_mode,
        )

        if self.ep_size > 1:
            final_hidden_states = self.deepep_dispatcher.combine(
                hidden_states=final_hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                forward_mode=forward_mode,
            )
        final_hidden_states *= self.routed_scaling_factor

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states

    def _forward_deepep_shared_output(self, forward_mode, hidden_states):
        if (
            forward_mode is not None
            and not forward_mode.is_idle()
            and hidden_states.shape[0] > 0
            and self.n_shared_experts is not None
        ):
            return self.shared_experts(hidden_states)
        return None

    def _forward_deepep_dispatch_a(
        self, chosen_deepep_dispatcher, forward_mode, hidden_states, router_logits
    ):
        if (
            forward_mode is not None
            and not forward_mode.is_idle()
            and hidden_states.shape[0] > 0
        ):
            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=True,
                renormalize=self.renormalize,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                correction_bias=self.correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    ep_rank=self.tp_rank,
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_idx = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            topk_weights = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )
        chosen_deepep_dispatcher.dispatch_a(
            hidden_states,
            topk_idx,
            topk_weights,
            forward_mode=forward_mode,
        )

    # TODO hacky, refactor
    def _forward_deepep_dispatch_a_part_one(
        self, forward_mode, hidden_states, router_logits
    ):
        if (
            forward_mode is not None
            and not forward_mode.is_idle()
            and hidden_states.shape[0] > 0
        ):
            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=True,
                renormalize=self.renormalize,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                correction_bias=self.correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    ep_rank=self.tp_rank,
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_idx = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            topk_weights = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )

        # NOTE HACK this is originally in DeepEPDispatcherImplLowLatency.dispatch_a, but we copy it here to reduce a kernel there
        topk_idx = topk_idx.to(torch.int64)

        return topk_weights, topk_idx

    def _forward_deepep_dispatch_a_part_two(
        self,
        chosen_deepep_dispatcher,
        forward_mode,
        hidden_states,
        topk_idx,
        topk_weights,
    ):
        chosen_deepep_dispatcher.dispatch_a(
            hidden_states,
            topk_idx,
            topk_weights,
            forward_mode=forward_mode,
        )

    # ----------------------------------------- TBO-related --------------------------------------------

    def _forward_tbo_op_gate(self, state):
        state.router_logits = self.gate(state.hidden_states_after_post_attn_ln)

    def _forward_tbo_op_mlp(self, state):
        state.expert_output_hidden_states = self.experts(
            hidden_states=state.pop("hidden_states_from_dispatch"),
            topk_idx=state.topk_idx_from_dispatch,
            topk_weights=state.topk_weights_from_dispatch,
            reorder_topk_ids=state.pop("reorder_topk_ids_from_dispatch"),
            seg_indptr=state.pop("seg_indptr_from_dispatch"),
            masked_m=state.pop("masked_m_from_dispatch"),
            expected_m=state.pop("expected_m_from_dispatch"),
            num_recv_tokens_per_expert=state.pop(
                "num_recv_tokens_per_expert_from_dispatch"
            ),
            forward_mode=state.forward_batch.forward_mode,
        )

    def _forward_tbo_op_dispatch_a_part_one(self, state):
        state.topk_weights, state.topk_idx = self._forward_deepep_dispatch_a_part_one(
            forward_mode=state.forward_batch.forward_mode,
            hidden_states=state.hidden_states_after_post_attn_ln,
            router_logits=state.pop("router_logits"),
        )

    def _forward_tbo_op_dispatch_a_part_two(self, state):
        self._forward_deepep_dispatch_a_part_two(
            chosen_deepep_dispatcher=self.tbo_deepep_dispatchers[
                state.tbo_subbatch_index
            ],
            forward_mode=state.forward_batch.forward_mode,
            hidden_states=state.hidden_states_after_post_attn_ln,
            topk_idx=state.pop("topk_idx"),
            topk_weights=state.pop("topk_weights"),
        )

    def _forward_tbo_op_dispatch_b(self, state, tbo_child_index: int):
        dispatcher = self.tbo_deepep_dispatchers[state.tbo_subbatch_index]
        with get_global_expert_distribution_recorder().with_current_layer(
            self.layer_id
        ), get_global_expert_distribution_recorder().with_debug_name(
            ["child_a", "child_b"][tbo_child_index]
        ):
            (
                state.hidden_states_from_dispatch,
                state.topk_idx_from_dispatch,
                state.topk_weights_from_dispatch,
                state.reorder_topk_ids_from_dispatch,
                state.num_recv_tokens_per_expert_from_dispatch,
                state.seg_indptr_from_dispatch,
                state.masked_m_from_dispatch,
                state.expected_m_from_dispatch,
            ) = dispatcher.dispatch_b()

    def _forward_tbo_op_combine_a(self, state):
        self.tbo_deepep_dispatchers[state.tbo_subbatch_index].combine_a(
            hidden_states=state.pop("expert_output_hidden_states"),
            topk_idx=state.pop("topk_idx_from_dispatch"),
            topk_weights=state.pop("topk_weights_from_dispatch"),
            forward_mode=state.forward_batch.forward_mode,
        )

    def _forward_tbo_op_combine_b(self, state):
        dispatcher = self.tbo_deepep_dispatchers[state.tbo_subbatch_index]
        hidden_states = dispatcher.combine_b()
        # hidden_states *= self.routed_scaling_factor
        # state.hidden_states_from_combine = hidden_states
        state.hidden_states_from_combine_without_scaling = hidden_states

    def _forward_tbo_op_shared(self, state):
        if get_bool_env_var("SGLANG_HACK_SLOW_BETWEEN_COMMUNICATION", "false"):
            for i in range(3):
                self.shared_experts(state.hidden_states_after_post_attn_ln)

        state.shared_output = self._forward_deepep_shared_output(
            state.forward_batch.forward_mode,
            state.pop("hidden_states_after_post_attn_ln"),
        )

    def _forward_shared_experts(self, hidden_states):
        if self.n_share_experts_fusion == 0:
            return self.shared_experts(hidden_states)
        else:
            return None


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2AttentionMLA(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # For tensor parallel attention
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb.forward = self.rotary_emb.forward_native

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.w_kc = None
        self.w_vc = None
        self.w_scale = None

        self.w_scale_k = None
        self.w_scale_v = None
        self.use_deep_gemm_bmm = False

        self.flashinfer_mla_disable_ragged = global_server_args_dict[
            "flashinfer_mla_disable_ragged"
        ]
        self.disable_chunked_prefix_cache = global_server_args_dict[
            "disable_chunked_prefix_cache"
        ]
        self.attention_backend = global_server_args_dict["attention_backend"]
        self.rocm_fused_decode_mla = get_bool_env_var(
            "SGLANG_ROCM_FUSED_DECODE_MLA", "false"
        )

        # TODO: Design a finer way to determine the threshold
        self.chunked_prefix_cache_threshold = get_int_env_var(
            "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD", 8192
        )

    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> AttnForwardMethod:
        if self.attention_backend == "flashinfer":
            # Flashinfer MLA: Do not absorb when enabling ragged prefill
            if (
                not self.flashinfer_mla_disable_ragged
                and forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu) == 0
            ):
                return AttnForwardMethod.MHA
            else:
                return AttnForwardMethod.MLA
        elif self.attention_backend == "fa3":
            # Flash Attention: Use MHA with chunked KV cache when prefilling on long sequences.
            if forward_batch.extend_prefix_lens_cpu is not None:
                sum_extend_prefix_lens = sum(forward_batch.extend_prefix_lens_cpu)
            if (
                forward_batch.forward_mode.is_extend()
                and not self.disable_chunked_prefix_cache
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and (
                    sum_extend_prefix_lens >= self.chunked_prefix_cache_threshold
                    or sum_extend_prefix_lens == 0
                )
            ):
                return AttnForwardMethod.MHA_CHUNKED_KV
            else:
                return AttnForwardMethod.MLA
        else:
            # Triton: Use normal computation for prefill and use weight absorption for extend/decode
            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and sum(forward_batch.extend_prefix_lens_cpu) == 0
            ):
                return AttnForwardMethod.MHA
            else:
                return AttnForwardMethod.MLA

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states

        attn_forward_method = self.dispatch_attn_forward_method(forward_batch)

        if attn_forward_method == AttnForwardMethod.MHA:
            return self.forward_normal(positions, hidden_states, forward_batch)
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            return self.forward_normal_chunked_kv(
                positions, hidden_states, forward_batch
            )
        else:
            if _is_hip:
                if (
                    self.rocm_fused_decode_mla
                    and forward_batch.forward_mode.is_decode()
                ):
                    return self.forward_absorb_fused_mla_rope(
                        positions, hidden_states, forward_batch
                    )
                else:
                    return self.forward_absorb(
                        positions, hidden_states, forward_batch, zero_allocator
                    )
            else:
                return self.forward_absorb(
                    positions, hidden_states, forward_batch, zero_allocator
                )

    def forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_absorb(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        state = self.forward_absorb_stage_prepare(
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
        )
        return self.forward_absorb_stage_core(state, zero_allocator)

    def forward_absorb_stage_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        # TODO optimize this part
        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return (hidden_states,)

        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_tensor_quant_mla_deep_gemm_masked_fp8(
                    q_nope.transpose(0, 1), dtype=torch.float8_e4m3fn
                )
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif self.w_kc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                zero_allocator.allocate(1),
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        k_nope = latent_cache[..., : self.kv_lora_rank]
        k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        return q_nope_out, q_pe, k_nope, k_pe, forward_batch

    def forward_absorb_stage_core(
        self,
        state,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        # TODO optimize this part
        if len(state) == 1:
            (hidden_states,) = state
            return hidden_states

        q_nope_out, q_pe, k_nope, k_pe, forward_batch = state

        # NOTE this line is deleted in PR5638, be careful when git merge!
        # q = torch.cat([q_nope_out, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        if self.attention_backend == "fa3":
            attn_output = self.attn_mqa(
                q_nope_out, k, k_nope, forward_batch, q_rope=q_pe
            )
        else:
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            attn_output = self.attn_mqa(q, k, k_nope, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_tensor_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1), dtype=torch.float8_e4m3fn
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = attn_bmm_output[:, :expected_m, :]
        elif self.w_vc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            if (num_repeat := get_int_env_var("SGLANG_HACK_SLOW_ATTN_NUM_REPEAT")) > 0:
                for i in range(num_repeat):
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)

            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        if get_bool_env_var("SGLANG_HACK_SLOW_BETWEEN_COMMUNICATION", "false"):
            for i in range(3):
                self.o_proj(attn_output)

        return output

    def forward_absorb_fused_mla_rope(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        enable_rope_fusion = (
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
        )
        q_len = hidden_states.shape[0]
        q_input = hidden_states.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if self.w_kc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input

        if not enable_rope_fusion:
            k_pe = k_input[..., self.kv_lora_rank :]
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q_input[..., self.kv_lora_rank :] = q_pe
            k_input[..., self.kv_lora_rank :] = k_pe
            k_pe_output = None
        else:
            k_pe_output = torch.empty_like(k_input[..., self.kv_lora_rank :])

        q_input[..., self.kv_lora_rank :] = q_pe

        # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # Use Fused ROPE with use_rope=OFF.
        attn_output = torch.empty(
            (q_len, self.num_local_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
            forward_batch.attn_backend.forward_metadata
        )
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        num_kv_split = forward_batch.attn_backend.num_kv_splits
        sm_scale = self.attn_mqa.scaling
        if attn_logits is None:
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_local_heads,
                    num_kv_split,
                    self.kv_lora_rank + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )

        # save current latent cache.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mqa, forward_batch.out_cache_loc, k_input, None
        )
        key_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mqa.layer_id
        )
        val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]

        decode_attention_fwd_grouped_rope(
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            self.kv_lora_rank,
            self.rotary_emb.rotary_dim,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            logit_cap=self.attn_mqa.logit_cap,
            use_rope=enable_rope_fusion,
            is_neox_style=self.rotary_emb.is_neox_style,
        )

        if enable_rope_fusion:
            k_input[..., self.kv_lora_rank :] = k_pe_output
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.w_vc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        return output

    def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            # Fetch latent cache from memory pool with precomputed chunked kv indices
            latent_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[
                forward_batch.prefix_chunk_kv_indices[i]
            ].contiguous()

            kv_a_normed, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a_normed = kv_a_normed.squeeze(1).contiguous()
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            k = torch.empty(
                (
                    k_nope.shape[0],
                    self.num_local_heads,
                    self.qk_nope_head_dim + self.qk_rope_head_dim,
                ),
                dtype=v.dtype,
                device=v.device,
            )
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
            lse = torch.transpose(lse, 0, 1).contiguous()
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse

        return accum_output

    def forward_normal_chunked_kv(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # In normal mha, the k and v tensors will become overly large when the prefix length is long.
        # To avoid this, we split the kv cache into chunks and process them one after another.
        # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.
        # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
        # will be helpful for understanding the purpose of this function.

        # First do normal mha forward to get output for extended part
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k_pe = latent_cache[:, :, self.kv_lora_rank :]

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )

        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        lse = torch.transpose(lse, 0, 1).contiguous()

        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if any(forward_batch.extend_prefix_lens_cpu):
            # Only initialize the info once
            if forward_batch.num_prefix_chunks is None:
                forward_batch.prepare_chunked_prefix_cache_info(q.device)

            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class _FFNInputMode(Enum):
    # The MLP sublayer requires 1/tp_size tokens as input
    SCATTERED = auto()
    # The MLP sublayer requires all tokens as input
    FULL = auto()


@dataclass
class _DecoderLayerInfo:
    is_sparse: bool
    ffn_input_mode: _FFNInputMode


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]
        self.layer_id = layer_id
        self.local_dp_size = get_local_attention_dp_size()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
        )

        self.info = self._compute_info(config, layer_id=layer_id, is_nextn=is_nextn)
        previous_layer_info = self._compute_info(
            config, layer_id=layer_id - 1, is_nextn=False
        )

        if self.info.is_sparse:
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
            )
        else:
            if _enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_is_scattered = (
            layer_id > 0
            and previous_layer_info.ffn_input_mode == _FFNInputMode.SCATTERED
        )
        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @staticmethod
    def _compute_info(config: PretrainedConfig, layer_id: int, is_nextn: bool):
        is_sparse = is_nextn or (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        )
        ffn_input_mode = (
            _FFNInputMode.SCATTERED
            if (global_server_args_dict["enable_deepep_moe"] and is_sparse)
            or (_enable_moe_dense_fully_dp() and not is_sparse)
            else _FFNInputMode.FULL
        )
        return _DecoderLayerInfo(is_sparse=is_sparse, ffn_input_mode=ffn_input_mode)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        if self.info.ffn_input_mode == _FFNInputMode.SCATTERED:
            return self.forward_ffn_with_scattered_input(
                positions, hidden_states, forward_batch, residual, zero_allocator
            )
        elif self.info.ffn_input_mode == _FFNInputMode.FULL:
            return self.forward_ffn_with_full_input(
                positions, hidden_states, forward_batch, residual, zero_allocator
            )
        else:
            raise NotImplementedError

    def forward_ffn_with_full_input(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}, {self.layer_id}, {self.__class__.__name__}] forward_ffn_with_full_input start {hidden_states.shape=}")

        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            hidden_states, residual = self._forward_input_layernorm(
                hidden_states, residual
            )

            assert not (
                self.attn_tp_size != 1 and self.input_is_scattered
            ), "moe_layer_freq > 1 is not supported when attn_tp_size > 1"

            # Self Attention
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )

        # Gather
        if get_tensor_model_parallel_world_size() > 1:
            # all gather and all reduce
            if self.local_dp_size != 1:
                if self.attn_tp_rank == 0:
                    hidden_states += residual
                hidden_states, local_hidden_states = (
                    forward_batch.gathered_buffer,
                    hidden_states,
                )
                dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
                dp_scatter(residual, hidden_states, forward_batch)
                # TODO extract this bugfix
                if hidden_states.shape[0] != 0:
                    hidden_states = self.post_attention_layernorm(hidden_states)
            else:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                # TODO extract this bugfix
                if hidden_states.shape[0] != 0:
                    hidden_states, residual = self.post_attention_layernorm(
                        hidden_states, residual
                    )
        else:
            # TODO extract this bugfix
            if hidden_states.shape[0] != 0:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )

        # Fully Connected
        hidden_states = self.mlp(hidden_states)
        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}, {self.layer_id}, {self.__class__.__name__}] forward_ffn_with_full_input after-mlp {hidden_states.shape=}")

        # TODO(ch-wan): ues reduce-scatter in MLP to avoid this scatter
        # Scatter
        if self.local_dp_size != 1:
            # important: forward batch.gathered_buffer is used both after scatter and after gather.
            # be careful about this!
            hidden_states, global_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            dp_scatter(hidden_states, global_hidden_states, forward_batch)

        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}, {self.layer_id}, {self.__class__.__name__}] forward_ffn_with_full_input end {self.local_dp_size=} {hidden_states.shape=}")
        return hidden_states, residual

    def forward_ffn_with_scattered_input(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}, {self.layer_id}, {self.__class__.__name__}] forward_ffn_with_scattered_input start {hidden_states.shape=}")
        # print(f"hi [{get_tensor_model_parallel_rank()}, {self.__class__.__name__}] forward_deepep start {self.layer_id=} {self.mlp.__class__.__name__=} "
        #       f"{hidden_states.shape=} {hidden_states[:1, :5]=} {residual[:1, :5] if residual is not None else None=}")

        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            hidden_states, residual = self._forward_input_layernorm(
                hidden_states, residual
            )

        if self.attn_tp_size != 1 and self.input_is_scattered:
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            tp_all_gather(
                list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
            )

        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        if self.attn_tp_size != 1:
            if self.input_is_scattered:
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                tp_reduce_scatter(hidden_states, tensor_list)
                if hidden_states.shape[0] != 0:
                    hidden_states, residual = self.post_attention_layernorm(
                        hidden_states, residual
                    )
            else:
                if self.attn_tp_rank == 0:
                    hidden_states += residual
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                tp_reduce_scatter(hidden_states, tensor_list)
                residual = hidden_states
                if hidden_states.shape[0] != 0:
                    hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            if hidden_states.shape[0] != 0:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )

        if not (
            _enable_moe_dense_fully_dp()
            and (not self.info.is_sparse)
            and hidden_states.shape[0] == 0
        ):
            hidden_states = self.mlp(hidden_states, forward_batch.forward_mode)

        if self.is_last_layer and self.attn_tp_size != 1:
            hidden_states += residual
            residual = None
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            tp_all_gather(
                list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
            )

        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}, {self.layer_id}, {self.__class__.__name__}] forward_ffn_with_scattered_input end {hidden_states.shape=}")
        # print(f"hi [{get_tensor_model_parallel_rank()}, {self.__class__.__name__}] forward_deepep end {self.layer_id=} {self.mlp.__class__.__name__=} "
        #       f"{hidden_states.shape=} {hidden_states[:1, :5]=} {residual[:1, :5] if residual is not None else None=}")
        return hidden_states, residual

    def _forward_input_layernorm(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states, residual

    # ----------------------------------------- TBO-related --------------------------------------------

    def get_forward_tbo_operations(
        self, forward_mode: ForwardMode, tbo_child_index: int
    ):
        if forward_mode == ForwardMode.EXTEND:
            operations = [
                self._forward_tbo_op_input_layernorm,
                self._forward_tbo_op_prefill_attn,
                self._forward_tbo_op_post_attn_layernorm,
                self.mlp._forward_tbo_op_gate,
                self.mlp._forward_tbo_op_dispatch_a_part_one,
                self.mlp._forward_tbo_op_dispatch_a_part_two,
                two_batch_overlap.YieldOperation(),
                partial(
                    self.mlp._forward_tbo_op_dispatch_b, tbo_child_index=tbo_child_index
                ),
                self.mlp._forward_tbo_op_mlp,
                self.mlp._forward_tbo_op_combine_a,
                two_batch_overlap.YieldOperation(),
                self.mlp._forward_tbo_op_shared,
                self.mlp._forward_tbo_op_combine_b,
                self._forward_tbo_op_compute_layer_output,
            ]
        elif forward_mode == ForwardMode.DECODE:
            operations = [
                self._forward_tbo_op_input_layernorm,
                self._forward_tbo_op_decode_attn_0,
                two_batch_overlap.YieldOperation(),
                self._forward_tbo_op_decode_attn_1,
                self._forward_tbo_op_post_attn_layernorm,
                self.mlp._forward_tbo_op_gate,
                self.mlp._forward_tbo_op_dispatch_a_part_one,
                two_batch_overlap.YieldOperation(),
                self.mlp._forward_tbo_op_dispatch_a_part_two,
                self.mlp._forward_tbo_op_shared,
                two_batch_overlap.YieldOperation(),
                partial(
                    self.mlp._forward_tbo_op_dispatch_b, tbo_child_index=tbo_child_index
                ),
                self.mlp._forward_tbo_op_mlp,
                self.mlp._forward_tbo_op_combine_a,
                two_batch_overlap.YieldOperation(),
                self.mlp._forward_tbo_op_combine_b,
                self._forward_tbo_op_compute_layer_output,
                two_batch_overlap.YieldOperation(),
            ]
        else:
            raise NotImplementedError(f"Unsupported {forward_mode=}")
        return two_batch_overlap.decorate_operations(
            operations, debug_name_prefix=f"L{self.layer_id}-"
        )

    def _forward_tbo_op_input_layernorm(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tbo_subbatch_index: int,
    ):
        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}, {self.layer_id}] _forward_tbo_op_input_layernorm start {forward_batch.input_ids.shape=} {hidden_states.shape=}")

        # TODO adhoc code, avoid copy-pasting these
        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            hidden_states, residual = self._forward_input_layernorm(
                hidden_states, residual
            )

        if self.attn_tp_size != 1 and self.input_is_scattered:
            assert (
                forward_batch.gathered_buffer is not None
            ), "please use moe_dense_tp_size=1"
            # print(
            #     f"hi [{get_tensor_model_parallel_rank()}, {self.layer_id}] _forward_tbo_op_input_layernorm {forward_batch.input_ids.shape=} {hidden_states.shape=} {forward_batch.gathered_buffer.shape=}")
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            tp_all_gather(
                list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
            )

        state.update(
            dict(
                hidden_states_after_input_ln=hidden_states,
                residual_after_input_ln=residual,
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def _forward_tbo_op_prefill_attn(self, state):
        state.hidden_states_after_attn = self.self_attn(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_input_ln"),
            forward_batch=state.forward_batch,
            # TODO hack
            zero_allocator=BumpAllocator(
                buffer_size=2, dtype=torch.float32, device="cuda"
            ),
        )

    def _forward_tbo_op_decode_attn_0(self, state):
        state.self_attn_state = self.self_attn.forward_absorb_stage_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_input_ln"),
            forward_batch=state.forward_batch,
            # TODO hack
            zero_allocator=BumpAllocator(
                buffer_size=2, dtype=torch.float32, device="cuda"
            ),
        )

    def _forward_tbo_op_decode_attn_1(self, state):
        assert (
            (get_tensor_model_parallel_world_size() > 1)
            and global_server_args_dict["enable_dp_attention"]
            and global_server_args_dict["enable_deepep_moe"]
            and isinstance(self.mlp, DeepseekV2MoE)
        )
        state.hidden_states_after_attn = self.self_attn.forward_absorb_stage_core(
            state.pop("self_attn_state"),
            # TODO hack
            zero_allocator=BumpAllocator(
                buffer_size=2, dtype=torch.float32, device="cuda"
            ),
        )

    def _forward_tbo_op_post_attn_layernorm(self, state):
        hidden_states, residual = (
            state.pop("hidden_states_after_attn"),
            state.pop("residual_after_input_ln"),
        )

        # TODO adhoc code, do not copy-paste
        if self.attn_tp_size != 1:
            if self.input_is_scattered:
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                tp_reduce_scatter(hidden_states, tensor_list)
                if hidden_states.shape[0] != 0:
                    hidden_states, residual = self.post_attention_layernorm(
                        hidden_states, residual
                    )
            else:
                if self.attn_tp_rank == 0:
                    hidden_states += residual
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                tp_reduce_scatter(hidden_states, tensor_list)
                residual = hidden_states
                if hidden_states.shape[0] != 0:
                    hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            if hidden_states.shape[0] != 0:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )

        state.hidden_states_after_post_attn_ln, state.residual_after_post_attn_ln = (
            hidden_states,
            residual,
        )

    # TODO some logic should be in MLP, refactor this
    def _forward_tbo_op_compute_layer_output(self, state):
        hidden_states = state.pop("hidden_states_from_combine_without_scaling")
        residual = state.pop("residual_after_post_attn_ln")

        if (shared_output := state.pop("shared_output")) is not None:
            # TODO beautify
            x = shared_output
            x.add_(hidden_states, alpha=self.mlp.routed_scaling_factor)
            hidden_states = x
        else:
            hidden_states *= self.mlp.routed_scaling_factor

        # TODO do not copy paste
        if self.is_last_layer and self.attn_tp_size != 1:
            hidden_states += residual
            residual = None
            hidden_states, local_hidden_states = (
                state.forward_batch.gathered_buffer[
                    : state.forward_batch.input_ids.shape[0]
                ],
                hidden_states,
            )
            tp_all_gather(
                list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
            )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )
        state.clear(expect_keys={"positions", "forward_batch", "tbo_subbatch_index"})
        return output


class DeepseekV2Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        zero_allocator = BumpAllocator(
            buffer_size=len(self.layers) * 2,
            dtype=torch.float32,
            device=(
                input_embeds.device if input_embeds is not None else input_ids.device
            ),
        )

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None

        normal_num_layers = (
            self.first_k_dense_replace
            if forward_batch.can_run_tbo
            else len(self.layers)
        )
        for i in range(normal_num_layers):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual, zero_allocator
                )

            # if i == 2:
            #     print(f"hi [{get_tensor_model_parallel_rank()}, {self.__class__.__name__}] forward after-layer-{i} "
            #           f"{forward_batch.tbo_split_seq_index=} "
            #           f"{hidden_states[:, :3] if hidden_states is not None else None=} "
            #           f"{residual[:, :3] if residual is not None else None=}"
            #           )

        hidden_states, residual = self._forward_tbo_layers(
            positions=positions,
            forward_batch=forward_batch,
            hidden_states=hidden_states,
            residual=residual,
            start_layer=normal_num_layers,
        )

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def _forward_tbo_layers(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        start_layer: int,
    ):
        end_layer = len(self.layers)
        if start_layer == end_layer:
            return hidden_states, residual

        def compute_operations(tbo_child_index: int):
            return [
                op
                for i in range(start_layer, end_layer)
                for op in self.layers[i].get_forward_tbo_operations(
                    forward_batch.global_forward_mode, tbo_child_index
                )
            ]

        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}] forward_tbo_layers start {forward_batch.tbo_split_seq_index=} {hidden_states.shape=}")
        if self.attn_tp_size != 1:
            hidden_states += residual
            residual = None

            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            tp_all_gather(
                list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
            )

        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}] forward_tbo_layers gathered {hidden_states.shape=}")
        inputs_a, inputs_b = model_forward_split_inputs(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=residual,
        )
        del hidden_states, residual

        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}] forward_tbo_layers TBO-split {inputs_a['hidden_states'].shape=} {inputs_b['hidden_states'].shape=}")

        def _postprocess_splitted_inputs(hidden_states, residual, **kwargs):
            if self.attn_tp_size != 1:
                assert residual is None
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]

            return dict(hidden_states=hidden_states, residual=residual, **kwargs)

        inputs_a = _postprocess_splitted_inputs(**inputs_a)
        inputs_b = _postprocess_splitted_inputs(**inputs_b)
        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}] forward_tbo_layers postprocessed {inputs_a['hidden_states'].shape=} {inputs_b['hidden_states'].shape=}")

        # TODO do not hardcode
        total_num_sm = torch.cuda.get_device_properties(
            device="cuda"
        ).multi_processor_count
        extend_mode_communication_num_sm = DEEPEP_NUM_SMS
        num_sm_context = (
            configure_deep_gemm_num_sms(
                num_sms=total_num_sm - extend_mode_communication_num_sm
            )
            if forward_batch.forward_mode.is_extend()
            else nullcontext()
        )
        with num_sm_context:
            return two_batch_overlap.model_forward_execute_two_batch(
                inputs_a=inputs_a,
                inputs_b=inputs_b,
                operations_a=compute_operations(0),
                operations_b=compute_operations(1),
                delta_stages={
                    ForwardMode.EXTEND: 0,
                    ForwardMode.DECODE: 2,
                }[forward_batch.global_forward_mode],
            )


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_n_share_experts_fusion()
        self.model = DeepseekV2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            enable_tp=not _enable_moe_dense_fully_dp(),  # TODO: replace it with DP attention
        )
        self.logits_processor = LogitsProcessor(config)

    def determine_n_share_experts_fusion(
        self, architecture: str = "DeepseekV3ForCausalLM"
    ):
        self.n_share_experts_fusion = global_server_args_dict["n_share_experts_fusion"]
        if self.n_share_experts_fusion > 0:
            # Only Deepseek V3/R1 can use shared experts fusion optimization now.
            if (
                self.config.architectures[0] != architecture
                or self.config.n_routed_experts != 256
            ):
                self.n_share_experts_fusion = 0
                global_server_args_dict["n_share_experts_fusion"] = 0
                logger.info(
                    "Only Deepseek V3/R1 can use shared experts fusion optimization. Shared experts fusion optimization is disabled."
                )
            else:
                assert (
                    self.n_share_experts_fusion == self.tp_size
                ), f"Shared experts fusion optimization is enabled in DeepSeek V3/R1, set it to {self.tp_size} can get best optimized performace."
        elif self.n_share_experts_fusion == 0:
            if (
                torch.cuda.get_device_capability("cuda") >= (9, 0)
                and self.config.architectures[0] == architecture
                and self.config.n_routed_experts == 256
                and (not global_server_args_dict["enable_deepep_moe"])
            ):
                self.n_share_experts_fusion = self.tp_size
                global_server_args_dict["n_share_experts_fusion"] = self.tp_size
                logger.info(
                    "Deepseek V3/R1 with fp8 can use shared experts fusion optimization when SM version >=90. Shared experts fusion optimization is enabled."
                )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}, {self.__class__.__name__}] forward start {forward_batch.tbo_split_seq_index=} {input_ids.shape=} {input_ids=} {positions=}")

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        # print(
        #     f"hi [{get_tensor_model_parallel_rank()}, {self.__class__.__name__}] forward end {forward_batch.tbo_split_seq_index=} {hidden_states[:, :3] if hidden_states is not None else None=}")

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def post_load_weights(self, enable_mla_postprocess: bool = True):

        # Perform post-processing after loading weights
        if enable_mla_postprocess:
            for layer_id in range(self.config.num_hidden_layers):
                self_attn = self.model.layers[layer_id].self_attn
                if hasattr(self_attn.kv_b_proj, "qweight"):
                    # AWQ compatible
                    if _is_cuda:
                        w = awq_dequantize(
                            self_attn.kv_b_proj.qweight,
                            self_attn.kv_b_proj.scales,
                            self_attn.kv_b_proj.qzeros,
                        ).T
                    else:
                        w = awq_dequantize(
                            self_attn.kv_b_proj.qweight,
                            self_attn.kv_b_proj.scales,
                            self_attn.kv_b_proj.qzeros,
                            0,
                            0,
                            0,
                        ).T
                else:
                    w = self_attn.kv_b_proj.weight
                # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`.
                # This may affect the accuracy of fp8 model.
                # Fix deepseek v3 blockwise bmm by using deep_gemm
                use_deep_gemm_bmm = False
                model_dtype = torch.get_default_dtype()

                if w.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    if hasattr(self.quant_config, "weight_block_size"):
                        weight_block_size = self.quant_config.weight_block_size
                        if weight_block_size is not None:
                            assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                            if _is_hip:
                                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                    weight=w,
                                    weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                                    input_scale=None,
                                )
                            else:
                                weight = w
                                weight_scale = self_attn.kv_b_proj.weight_scale_inv

                            if (
                                _is_cuda
                                and weight_block_size[0] == 128
                                and weight_block_size[1] == 128
                                and model_dtype == torch.bfloat16
                            ):
                                if _ENABLE_JIT_DEEPGEMM and get_bool_env_var(
                                    "SGL_USE_DEEPGEMM_BMM", "false"
                                ):
                                    block_scale = weight_scale
                                    use_deep_gemm_bmm = True
                                else:
                                    w = block_quant_dequant(
                                        weight,
                                        weight_scale,
                                        weight_block_size,
                                        model_dtype,
                                    )
                            else:
                                w, scale = block_quant_to_tensor_quant(
                                    weight, weight_scale, weight_block_size
                                )
                                self_attn.w_scale = scale
                    else:
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale
                        w, scale = channel_quant_to_tensor_quant(weight, weight_scale)
                        self_attn.w_scale = scale

                if w.dtype == torch.int8:
                    if hasattr(self.quant_config, "weight_block_size"):
                        # block-wise int8 need it
                        weight_block_size = self.quant_config.weight_block_size
                        if weight_block_size is not None:
                            assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale_inv
                            w = int8_block_dequant(
                                weight, weight_scale, weight_block_size
                            ).to(torch.bfloat16)
                    else:
                        # channel-wise int8 need it
                        w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                            torch.bfloat16
                        )

                w_kc, w_vc = w.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                if not use_deep_gemm_bmm:
                    self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                    self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                    if (
                        hasattr(self_attn.kv_b_proj, "weight_scale")
                        and self_attn.w_scale is None
                    ):
                        self_attn.w_scale = self_attn.kv_b_proj.weight_scale
                        if _is_hip:
                            self_attn.w_scale *= 2.0
                else:
                    num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1]
                    num_tiles_n = self_attn.v_head_dim // weight_block_size[0]
                    ws_kc, ws_vc = block_scale.unflatten(
                        0, (-1, (num_tiles_k + num_tiles_n))
                    ).split([num_tiles_k, num_tiles_n], dim=1)
                    self_attn.w_scale_k = ws_kc.transpose(1, 2).contiguous()
                    self_attn.w_scale_v = ws_vc.contiguous()
                    self_attn.w_kc = w_kc.transpose(1, 2).contiguous()
                    self_attn.w_vc = w_vc.contiguous()
                    self_attn.use_deep_gemm_bmm = True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        if self.n_share_experts_fusion > 0:
            weights_list = list(weights)
            weights_dict = dict(weights_list)
            if self.quant_config is None or self.quant_config.get_name() == "w8a8_int8":
                suffix_list = [
                    "down_proj.weight",
                    "down_proj.weight_scale",
                    "gate_proj.weight",
                    "gate_proj.weight_scale",
                    "up_proj.weight",
                    "up_proj.weight_scale",
                ]
            else:
                suffix_list = [
                    "down_proj.weight",
                    "down_proj.weight_scale_inv",
                    "gate_proj.weight",
                    "gate_proj.weight_scale_inv",
                    "up_proj.weight",
                    "up_proj.weight_scale_inv",
                ]
            names_to_remove = []
            for moe_layer in tqdm(
                range(
                    self.config.first_k_dense_replace,
                    self.config.num_hidden_layers,
                    self.config.moe_layer_freq,
                ),
                desc=f"Cloning {self.n_share_experts_fusion} "
                "replicas of the shared expert into MoE",
            ):
                for suffix in suffix_list:
                    shared_expert_weight_name = (
                        f"model.layers.{moe_layer}.mlp.shared_experts.{suffix}"
                    )
                    for num_repeat in range(self.n_share_experts_fusion):
                        weights_list.append(
                            (
                                f"model.layers.{moe_layer}."
                                f"mlp.experts."
                                f"{self.config.n_routed_experts + num_repeat}"
                                f".{suffix}",
                                weights_dict[shared_expert_weight_name],
                            )
                        )
                    names_to_remove += [shared_expert_weight_name]
            weights = [w for w in weights_list if w[0] not in names_to_remove]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = (
            DeepEPMoE
            if global_server_args_dict["enable_deepep_moe"]
            else (EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE)
        )
        expert_params_mapping = MoEImpl.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.n_share_experts_fusion,
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        params_dict = dict(self.named_parameters())
        exist_mla_weights = False
        for name, loaded_weight in weights:
            exist_mla_weights |= "self_attn" in name

            # TODO(HandH1998): Modify it when nextn is supported.
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                if num_nextn_layers > 0 and name.startswith("model.layers"):
                    name_list = name.split(".")
                    if (
                        len(name_list) >= 3
                        and int(name_list[2]) >= self.config.num_hidden_layers
                    ):
                        continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if fuse_qkv_a_proj and (
                        "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                    ):
                        cached_a_proj[name] = loaded_weight
                        q_a_proj_name = (
                            name
                            if "q_a_proj" in name
                            else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                        )
                        kv_a_proj_name = (
                            name
                            if "kv_a_proj_with_mqa" in name
                            else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                        )

                        # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                        if (
                            q_a_proj_name in cached_a_proj
                            and kv_a_proj_name in cached_a_proj
                        ):
                            q_a_proj_weight = cached_a_proj[q_a_proj_name]
                            kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                            fused_weight = torch.cat(
                                [q_a_proj_weight, kv_a_proj_weight], dim=0
                            )

                            param_name = name.replace(
                                "q_a_proj", "fused_qkv_a_proj_with_mqa"
                            )
                            param = params_dict[param_name]

                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, fused_weight)
                            cached_a_proj.pop(q_a_proj_name)
                            cached_a_proj.pop(kv_a_proj_name)
                    else:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

        self.post_load_weights(enable_mla_postprocess=exist_mla_weights)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_param_name_info(self, name: str) -> ModelParamNameInfo:
        if ".experts." in name:
            return ModelParamNameInfoMoe(
                layer_id=int(re.search(r"layers\.(\d+)", name).group(1)),
                expert_id=int(re.search(r"experts\.(\d+)", name).group(1)),
            )
        return ModelParamNameInfoOthers()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM]
