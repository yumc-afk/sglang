# VERL 分布式 TP Resharding 实现指南

## 问题描述

当前 VerlEngine 在处理权重更新时存在两个严重问题：

### 1. 破坏 DTensor 分布式特性
```python
# python/sglang/srt/entrypoints/verl_engine.py:126
def _preprocess_tensor_for_update_weights(self, tensor):
    if isinstance(tensor, torch.distributed.tensor.DTensor):
        tensor = tensor.full_tensor()  # ❌ 这破坏了分布式！
    return tensor
```

### 2. 使用 gather_object 收集到 rank 0
```python
# python/sglang/srt/entrypoints/verl_engine.py:141
torch.distributed.gather_object(
    serialized_tensor, 
    object_gather_list, 
    dst=0, 
    group=self._device_mesh.get_group()
)
```

这导致：
- 所有权重聚集到 rank 0，造成内存瓶颈
- 大模型（如 DeepSeek V3 671B）会 OOM
- 违反 Zero-Copy 原则

## 修改方案

### 步骤 1：保持 DTensor 特性

修改 `_preprocess_tensor_for_update_weights`：
```python
def _preprocess_tensor_for_update_weights(self, tensor):
    # 保持 DTensor，不要转换！
    return tensor
```

### 步骤 2：实现分布式权重更新

#### 方案 A：利用 DTensor（推荐）

```python
def update_weights_from_tensor(
    self, 
    named_tensors: Iterable[Tuple[str, torch.Tensor]],
    load_format: Optional[str] = None
):
    """分布式版本，不使用 gather"""
    if self._tp_size == 1:
        # 单卡保持原逻辑
        return self._update_weights_single_device(named_tensors, load_format)
    
    # 多卡分布式处理
    distributed_updates = []
    
    for name, tensor in named_tensors:
        # 保持或创建 DTensor
        if isinstance(tensor, dt.DTensor):
            # 已经是 DTensor，保持
            distributed_tensor = tensor
        else:
            # 转换为 DTensor
            distributed_tensor = dt.DTensor.from_local(
                tensor,
                self._device_mesh,
                self._get_placement_for_param(name)
            )
        
        # 序列化 DTensor 的本地分片
        local_shard = distributed_tensor.to_local()
        serialized = MultiprocessingSerializer.serialize((name, local_shard))
        distributed_updates.append(serialized)
    
    # 创建分布式更新请求
    update_req = DistributedUpdateWeightsReq(
        tp_rank=self._tp_rank,
        tp_size=self._tp_size,
        serialized_tensors=distributed_updates,
        load_format=load_format
    )
    
    # 发送到引擎（每个 rank 只发送自己的数据）
    self._engine.update_weights_distributed(update_req)
```

#### 方案 B：自定义分布式通信

如果不能使用 DTensor，实现自定义通信：
```python
def update_weights_from_tensor_distributed(self, named_tensors, load_format=None):
    """每个 rank 只处理自己的数据"""
    local_data = {
        'tp_rank': self._tp_rank,
        'tp_size': self._tp_size,
        'tensors': []
    }
    
    for name, tensor in named_tensors:
        serialized = MultiprocessingSerializer.serialize(tensor)
        local_data['tensors'].append((name, serialized))
    
    # 直接发送，不 gather
    self._send_to_engine_distributed(local_data, load_format)
```

### 步骤 3：扩展 Engine 接口

在 `python/sglang/srt/entrypoints/engine.py` 添加：
```python
def update_weights_distributed(self, req: DistributedUpdateWeightsReq):
    """处理分布式权重更新"""
    # 每个 rank 的数据独立处理
    obj = UpdateWeightsFromTensorReqInput()
    obj.tp_rank = req.tp_rank
    obj.tp_size = req.tp_size
    obj.serialized_named_tensors = req.serialized_tensors
    obj.load_format = req.load_format
    
    # 发送给对应的 worker
    self.tokenizer_manager.update_weights_distributed(obj)
```

### 步骤 4：ModelRunner 支持 TP Resharding

在 `python/sglang/srt/model_executor/model_runner.py` 添加：

```python
def update_weights_from_tensor(self, named_tensors, load_format=None):
    # 检查是否需要 resharding
    source_tp = getattr(self.server_args, 'source_tp_size', self.tp_size)
    
    if source_tp != self.tp_size:
        # 需要 TP resharding
        named_tensors = self._reshard_weights_for_tp(
            named_tensors, 
            source_tp, 
            self.tp_size
        )
    
    # 继续原有加载逻辑
    # ...

def _reshard_weights_for_tp(self, named_tensors, source_tp, target_tp):
    """分布式 TP resharding"""
    if source_tp < target_tp:
        # 扩展场景：TP=4 → TP=16
        return self._expand_tp_distributed(named_tensors, source_tp, target_tp)
    else:
        # 收缩场景：暂不支持
        raise NotImplementedError("source_tp > target_tp not supported yet")

def _expand_tp_distributed(self, named_tensors, source_tp, target_tp):
    """分布式扩展 TP"""
    expansion_factor = target_tp // source_tp
    source_rank = self.tp_rank // expansion_factor
    rank_in_expansion = self.tp_rank % expansion_factor
    
    # 创建通信组
    group_start = source_rank * expansion_factor
    subgroup = dist.new_group(
        ranks=list(range(group_start, group_start + expansion_factor))
    )
    
    resharded = []
    for name, tensor in named_tensors:
        # 广播权重
        if rank_in_expansion == 0:
            dist.broadcast(tensor, src=self.tp_rank, group=subgroup)
        else:
            tensor = torch.empty_like(tensor)
            dist.broadcast(tensor, src=group_start, group=subgroup)
        
        # 本地切分
        sliced_tensor = self._slice_for_rank(
            tensor, name, rank_in_expansion, expansion_factor
        )
        resharded.append((name, sliced_tensor))
    
    return resharded
```

## 文件修改列表

1. `python/sglang/srt/entrypoints/verl_engine.py`
   - 修改 `_preprocess_tensor_for_update_weights`
   - 替换 `gather_object` 逻辑

2. `python/sglang/srt/entrypoints/engine.py`
   - 添加 `update_weights_distributed` 方法

3. `python/sglang/srt/model_executor/model_runner.py`
   - 添加 TP resharding 支持

4. 可能需要修改：
   - `tokenizer_manager.py`
   - `scheduler.py`
   - `tp_worker.py`

## 测试验证

1. **单元测试**
   ```python
   def test_no_gather():
       # 确保没有 gather_object 调用
       # 验证每个 rank 只处理自己的数据
   
   def test_tp_resharding():
       # 测试 TP=4 → TP=16
       # 验证内存使用 < 1.5x
   ```

2. **集成测试**
   - 使用小模型端到端验证
   - 监控 CPU-GPU 传输（应该为 0）
   - 验证大模型场景（模拟 671B）

## 参考资源

- SGLang DTensor 支持：`model_parallel.py`
- VERL 主仓库文档：
  - `/workspaces/verl/CODEX_TP_RESHARDING_TASK.md`
  - `/workspaces/verl/GEMINI_ANALYSIS_DISTRIBUTED_SOLUTION.md`

---
*创建时间：2025-01-26*
*目标：支持大模型分布式训练和推理*