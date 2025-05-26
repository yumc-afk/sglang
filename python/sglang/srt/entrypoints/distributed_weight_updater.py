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
"""Distributed weight update utilities for VERL integration."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from sglang.srt.utils import MultiprocessingSerializer


@dataclass
class DistributedUpdateRequest:
    """Request for distributed weight update."""
    tp_rank: int
    tp_size: int
    source_tp_size: Optional[int] = None  # For TP resharding
    local_tensors: List[Tuple[str, bytes]] = None
    load_format: Optional[str] = None


class DistributedWeightUpdater:
    """Handles distributed weight updates without gathering to rank 0."""
    
    def __init__(self, tp_rank: int, tp_size: int, device_mesh=None):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.device_mesh = device_mesh
        
    def prepare_distributed_update(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        source_tp_size: Optional[int] = None,
    ) -> DistributedUpdateRequest:
        """Prepare tensors for distributed update."""
        local_tensors = []
        
        for name, tensor in named_tensors:
            # Handle DTensor
            if isinstance(tensor, DTensor):
                local_shard = tensor.to_local()
            else:
                local_shard = tensor
            
            # Serialize
            serialized = MultiprocessingSerializer.serialize(local_shard)
            local_tensors.append((name, serialized))
        
        return DistributedUpdateRequest(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            source_tp_size=source_tp_size or self.tp_size,
            local_tensors=local_tensors,
        )
    
    def send_distributed_weights(
        self,
        request: DistributedUpdateRequest,
        engine,
        use_broadcast: bool = False,
    ):
        """Send weights in a distributed manner."""
        if use_broadcast:
            # Each rank broadcasts its weights to all others
            self._broadcast_weights(request, engine)
        else:
            # Point-to-point communication
            self._send_p2p_weights(request, engine)
    
    def _broadcast_weights(self, request: DistributedUpdateRequest, engine):
        """Use broadcast within groups for efficient communication."""
        # This is more efficient for TP resharding scenarios
        # Implementation depends on the specific resharding pattern
        pass
    
    def _send_p2p_weights(self, request: DistributedUpdateRequest, engine):
        """Point-to-point weight sending."""
        # For now, still coordinate through rank 0
        # but without gathering all data
        for name, serialized in request.local_tensors:
            # Create sparse tensor list
            sparse_values = [None] * self.tp_size
            sparse_values[self.tp_rank] = serialized
            
            # Only rank 0 sends to engine
            if self.tp_rank == 0:
                from sglang.srt.model_executor.model_runner import LocalSerializedTensor
                engine.update_weights_from_tensor(
                    named_tensors=[(name, LocalSerializedTensor(values=sparse_values))],
                    load_format=request.load_format,
                    flush_cache=False,
                )
            
            # Synchronize
            if self.device_mesh is not None:
                dist.barrier(group=self.device_mesh.get_group())


def get_placement_for_param(param_name: str) -> List:
    """Determine the placement strategy for a parameter."""
    from torch.distributed.tensor import Replicate, Shard
    
    # Column parallel parameters (output dimension sharded)
    if any(x in param_name for x in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
        return [Shard(0)]
    
    # Row parallel parameters (input dimension sharded)
    if any(x in param_name for x in ["o_proj", "down_proj"]):
        return [Shard(1)]
    
    # MoE expert weights
    if "experts" in param_name:
        if "w1" in param_name or "w3" in param_name:
            return [Shard(2)]  # Shard on hidden dimension
        elif "w2" in param_name:
            return [Shard(2)]
    
    # Non-parallel parameters (replicated)
    return [Replicate()]