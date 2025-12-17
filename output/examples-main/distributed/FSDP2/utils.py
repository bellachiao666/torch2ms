import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from model import Transformer
# from torch.distributed.fsdp import FSDPModule
from ms.Tensor import Shard


# 'torch.distributed.fsdp.FSDPModule' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
def inspect_model(model: FSDPModule):
    assert isinstance(model, Transformer)
    assert isinstance(model, FSDPModule)

    if mint.distributed.get_rank() == 0:
        print(model)

    for param in model.parameters():
        assert param.placements == (Shard(0),)  # 'torch.distributed.tensor.Shard' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        assert param.dtype == ms.float32
        # print(param.get_local_tensor())


# 'torch.distributed.fsdp.FSDPModule' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
def inspect_mixed_precision(model: FSDPModule):
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == ms.bfloat16
    model.reshard()
