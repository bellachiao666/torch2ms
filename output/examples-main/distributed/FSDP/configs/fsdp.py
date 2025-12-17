import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from dataclasses import dataclass, field
from typing import ClassVar
# from torch.distributed.fsdp import ShardingStrategy
# from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class fsdp_config:
    mixed_precision: bool=True
    use_fp16: bool=False
    seed: int=42
    fsdp_activation_checkpointing: bool=False
    limit_all_gathers: bool=True
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD  # HYBRID_SHARD, SHARD_GRAD_OP; 'torch.distributed.fsdp.ShardingStrategy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT  # alternatively can use SHARDED_STATE_DICT to avoid OOMs; 'torch.distributed.fsdp.fully_sharded_data_parallel.StateDictType' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    save_optimizer: bool=False
    
    
    
    