import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops

# from torch.distributed.fsdp import (
    # FullyShardedDataParallel as FSDP,
    # CPUOffload,
    MixedPrecision,
    # BackwardPrefetch,
    # ShardingStrategy,
)

# requires grad scaler in main loop
fpSixteen = MixedPrecision(
    param_dtype=ms.float16,
    # Gradient communication precision.
    reduce_dtype=ms.float16,
    # Buffer precision.
    buffer_dtype=ms.float16,
)  # 'torch.distributed.fsdp.MixedPrecision' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

bfSixteen = MixedPrecision(
    param_dtype=ms.bfloat16,
    # Gradient communication precision.
    reduce_dtype=ms.bfloat16,
    # Buffer precision.
    buffer_dtype=ms.bfloat16,
)  # 'torch.distributed.fsdp.MixedPrecision' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

bfSixteen_working = MixedPrecision(
    param_dtype=ms.float32,
    reduce_dtype=ms.bfloat16,
    buffer_dtype=ms.bfloat16,
)  # 'torch.distributed.fsdp.MixedPrecision' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

fp32_policy = MixedPrecision(
    param_dtype=ms.float32,
    reduce_dtype=ms.float32,
    buffer_dtype=ms.float32,
)  # 'torch.distributed.fsdp.MixedPrecision' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
