import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import types
""" Normalization layers and wrappers

Norm layer definitions that support fast norm and consistent channel arg order (always first arg).

Hacked together by / Copyright 2022 Ross Wightman
"""
import numbers
from typing import Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from .fast_norm import (
    is_fast_norm,
    fast_group_norm,
    fast_layer_norm,
    fast_rms_norm,
    rms_norm2d,
    fast_rms_norm2d,
    fast_simple_norm,
    simple_norm,
)

try:
    # from torch.nn.functional import rms_norm
    pass
except ImportError:
    from .fast_norm import rms_norm

# 轻量 torch / fallback nn stub
class _TorchStub:
    def __init__(self):
        class _Final:
            def __class_getitem__(cls, item):
                return bool
        self.jit = types.SimpleNamespace(Final=_Final)


torch = _TorchStub()

_fallback_norms = {
    "GroupNorm": getattr(msnn, "GroupNorm", None) or msnn.BatchNorm2d,
    "LayerNorm": getattr(msnn, "LayerNorm", None),
    "BatchNorm2d": msnn.BatchNorm2d,
}
for _name, _cls in _fallback_norms.items():
    if _cls is not None and not hasattr(nn, _name):
        setattr(nn, _name, _cls)


class FrozenBatchNorm2d(msnn.BatchNorm2d):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.0):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=False, track_running_stats=False)

    def construct(self, x):
        return super().construct(x)


class GroupNorm(nn.GroupNorm):
    _fast_norm: torch.jit.Final[bool]  # 'torch.jit.Final' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def __init__(
            self,
            num_channels: int,
            num_groups: int = 32,
            eps: float = 1e-5,
            affine: bool = True,
            **kwargs,
    ):
        # NOTE num_channels is swapped to first arg for consistency in swapping norm layers with BN
        super().__init__(num_groups, num_channels, eps=eps, affine=affine, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x):
        if self._fast_norm:
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)  # 'torch.nn.functional.group_norm' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


class GroupNorm1(nn.GroupNorm):
    """ Group Normalization with 1 group.
    Input: tensor in shape [B, C, *]
    """
    _fast_norm: torch.jit.Final[bool]  # 'torch.jit.Final' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(1, num_channels, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: ms.Tensor) -> ms.Tensor:
        if self._fast_norm:
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)  # 'torch.nn.functional.group_norm' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


class LayerNorm(nn.LayerNorm):
    """ LayerNorm w/ fast norm option
    """
    _fast_norm: torch.jit.Final[bool]  # 'torch.jit.Final' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            **kwargs,
    ):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: ms.Tensor) -> ms.Tensor:
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)  # 'torch.nn.functional.layer_norm' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        return x


class LayerNormFp32(nn.LayerNorm):
    """ LayerNorm
    """

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            **kwargs,
    ):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    def forward(self, x: ms.Tensor) -> ms.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        x = F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)  # 'torch.nn.functional.layer_norm' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.nn.functional.layer_norm.to' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        return x


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    _fast_norm: torch.jit.Final[bool]  # 'torch.jit.Final' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            **kwargs,
    ):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: ms.Tensor) -> ms.Tensor:
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)  # 'torch.nn.functional.layer_norm' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNorm2dFp32(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            **kwargs,
    ):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    def forward(self, x: ms.Tensor) -> ms.Tensor:
        x = x.permute(0, 2, 3, 1)
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        x = F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)  # 'torch.nn.functional.layer_norm' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.nn.functional.layer_norm.to' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        x = x.permute(0, 3, 1, 2)
        return x


def _is_contiguous(tensor: ms.Tensor) -> bool:
    # jit is oh so lovely :/
    # 'torch.jit.is_scripting' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


def _layer_norm_cf(x: ms.Tensor, weight: ms.Tensor, bias: ms.Tensor, eps: float):
    s, u = mint.var_mean(x, dim=1, unbiased=False, keepdim=True)
    x = (x - u) * mint.rsqrt(s + eps)
    x = x * weight[:, None, None] + bias[:, None, None]
    return x


def _layer_norm_cf_sqm(x: ms.Tensor, weight: ms.Tensor, bias: ms.Tensor, eps: float):
    u = x.mean(dim=1, keepdim=True)
    s = ((x * x).mean(dim=1, keepdim=True) - (u * u)).clamp(0)
    x = (x - u) * mint.rsqrt(s + eps)
    x = x * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
    return x


class LayerNormExp2d(nn.LayerNorm):
    """ LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).

    Experimental implementation w/ manual norm for tensors non-contiguous tensors.

    This improves throughput in some scenarios (tested on Ampere GPU), esp w/ channels_last
    layout. However, benefits are not always clear and can perform worse on other GPUs.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__(num_channels, eps=eps)

    def forward(self, x) -> ms.Tensor:
        if _is_contiguous(x):
            x = mint.permute(0, 3, 1, 2)  # 'torch.nn.functional.layer_norm' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        else:
            x = _layer_norm_cf(x, self.weight, self.bias, self.eps)
        return x


class RmsNorm(msnn.Cell):
    """ RmsNorm w/ fast (apex) norm if available
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', '_fast_norm']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    _fast_norm: bool

    def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

        if self.elementwise_affine:
            self.weight = ms.Parameter(mint.empty(self.normalized_shape, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # 'torch.nn.init.ones_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # NOTE fast norm fallback needs our rms norm impl, so both paths through here.
        # Since there is no built-in PyTorch impl, always uses APEX RmsNorm if installed.
        if self._fast_norm:
            x = fast_rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:
            x = rms_norm(x, self.normalized_shape, self.weight, self.eps)  # 'torch.nn.functional.rms_norm' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        return x


class RmsNormFp32(msnn.Cell):
    """ RmsNorm w/ fast (apex) norm if available
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine

        if self.elementwise_affine:
            self.weight = ms.Parameter(mint.empty(self.normalized_shape, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # 'torch.nn.init.ones_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        x = rms_norm(x.float(), self.normalized_shape, weight, self.eps).to(x.dtype)  # 'torch.nn.functional.rms_norm' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.nn.functional.rms_norm.to' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        return x


class RmsNorm2d(msnn.Cell):
    """ RmsNorm2D for NCHW tensors, w/ fast apex or cast norm if available

    NOTE: It's currently (2025-05-10) faster to use an eager 2d kernel that does reduction
    on dim=1 than to permute and use internal PyTorch F.rms_norm, this may change if something
    like https://github.com/pytorch/pytorch/pull/150576 lands.
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', '_fast_norm']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    _fast_norm: bool

    def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

        if self.elementwise_affine:
            self.weight = ms.Parameter(mint.empty(self.normalized_shape, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # 'torch.nn.init.ones_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # NOTE fast norm fallback needs our rms norm impl, so both paths through here.
        # Since there is no built-in PyTorch impl, always use APEX RmsNorm if is installed.
        if self._fast_norm:
            x = fast_rms_norm2d(x, self.normalized_shape, self.weight, self.eps)
        else:
            x = rms_norm2d(x, self.normalized_shape, self.weight, self.eps)
        return x


class RmsNorm2dFp32(msnn.Cell):
    """ RmsNorm2D for NCHW tensors, w/ fast apex or cast norm if available

    NOTE: It's currently (2025-05-10) faster to use an eager 2d kernel that does reduction
    on dim=1 than to permute and use internal PyTorch F.rms_norm, this may change if something
    like https://github.com/pytorch/pytorch/pull/150576 lands.
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine

        if self.elementwise_affine:
            self.weight = ms.Parameter(mint.empty(self.normalized_shape, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # 'torch.nn.init.ones_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        x = rms_norm2d(x.float(), self.normalized_shape, weight, self.eps).to(x.dtype)
        return x


class SimpleNorm(msnn.Cell):
    """ SimpleNorm (x / std(x))
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', '_fast_norm']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    _fast_norm: bool

    def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

        if self.elementwise_affine:
            self.weight = ms.Parameter(mint.empty(self.normalized_shape, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # 'torch.nn.init.ones_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self._fast_norm:
            x = fast_simple_norm(x, self.normalized_shape, self.weight, self.eps)
        else:
            x = simple_norm(x, self.normalized_shape, self.weight, self.eps)
        return x


class SimpleNormFp32(msnn.Cell):
    """ SimpleNorm (x / std(x))
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine

        if self.elementwise_affine:
            self.weight = ms.Parameter(mint.empty(self.normalized_shape, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # 'torch.nn.init.ones_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        x = simple_norm(x.float(), self.normalized_shape, weight, self.eps).to(x.dtype)
        return x


class SimpleNorm2d(msnn.Cell):
    """ SimpleNorm for NCHW tensors
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', '_fast_norm']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    _fast_norm: bool

    def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

        if self.elementwise_affine:
            self.weight = ms.Parameter(mint.empty(self.normalized_shape, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # 'torch.nn.init.ones_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_simple_norm(x, self.normalized_shape, self.weight, self.eps)
        else:
            x = simple_norm(x, self.normalized_shape, self.weight, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class SimpleNorm2dFp32(msnn.Cell):
    """ SimpleNorm for NCHW tensors
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine

        if self.elementwise_affine:
            self.weight = ms.Parameter(mint.empty(self.normalized_shape, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # 'torch.nn.init.ones_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = x.permute(0, 2, 3, 1)
        weight = self.weight.float() if self.weight is not None else None
        x = simple_norm(x.float(), self.normalized_shape, weight, self.eps).to(x.dtype)
        x = x.permute(0, 3, 1, 2)
        return x
