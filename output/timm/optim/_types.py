import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from typing import Any, Dict, Iterable, Union, Protocol, Type
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias
try:
    from typing import TypeVar
except ImportError:
    from typing_extensions import TypeVar

# import torch

try:
    # from torch.optim.optimizer import ParamsT
except (ImportError, TypeError):
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


OptimType = Type[torch.optim.Optimizer]


class OptimizerCallable(Protocol):
    """Protocol for optimizer constructor signatures."""

    # 类型标注 'torch.optim.optimizer.ParamsT' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    # 类型标注 'torch.optim.Optimizer' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    def __call__(self, params: ParamsT, **kwargs) -> torch.optim.Optimizer: ...


__all__ = ['ParamsT', 'OptimType', 'OptimizerCallable']