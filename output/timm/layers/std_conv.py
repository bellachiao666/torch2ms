import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" Convolution with Weight Standardization (StdConv and ScaledStdConv)

StdConv:
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1903.10520},
  year      = {2019},
}
Code: https://github.com/joe-siyuan-qiao/WeightStandardization

ScaledStdConv:
Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
    - https://arxiv.org/abs/2101.08692
Official Deepmind JAX code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

Hacked together by / copyright Ross Wightman, 2021.
"""
from typing import Optional, Tuple, Union

# import torch
# import torch.nn as nn

from ._fx import register_notrace_module
from .padding import get_padding, get_padding_value, pad_same


class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self,
            in_channel: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Optional[Union[int, Tuple[int, int]]] = None,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = False,
            eps: float = 1e-6,
            device=None,
            dtype=None,
    ):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias, device=device, dtype=dtype)
        self.eps = eps

    def forward(self, x):
        weight = nn.functional.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None, training = True, momentum = 0., eps = self.eps).reshape_as(self.weight)  # 'torch.nn.functional.batch_norm.reshape_as' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        x = nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


@register_notrace_module
class StdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for ViT Hybrid model.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self,
            in_channel: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: str = 'SAME',
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = False,
            eps: float = 1e-6,
            device=None,
            dtype=None,
    ):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, device=device, dtype=dtype)
        self.same_pad = is_dynamic
        self.eps = eps

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = nn.functional.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None, training = True, momentum = 0., eps = self.eps).reshape_as(self.weight)  # 'torch.nn.functional.batch_norm.reshape_as' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        x = nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Optional[Union[int, Tuple[int, int], str]] = None,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            gamma: float = 1.0,
            eps: float = 1e-6,
            gain_init: float = 1.0,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, **dd)
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps
        self.gain_init = gain_init

        self.gain = ms.Parameter(mint.empty((self.out_channels, 1, 1, 1), **dd))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Only initialize gain if it exists (for the second call)
        if hasattr(self, 'gain'):
            torch.nn.init.constant_(self.gain, self.gain_init)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            # Also reset parent parameters if needed
            super().reset_parameters()
        # On first call (from super().__init__), do nothing

    def forward(self, x):
        weight = nn.functional.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None, weight = (self.gain * self.scale).view(-1), training = True, momentum = 0., eps = self.eps).reshape_as(self.weight)  # 'torch.nn.functional.batch_norm.reshape_as' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        return nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


@register_notrace_module
class ScaledStdConv2dSame(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME padding support

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: str = 'SAME',
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            gamma: float = 1.0,
            eps: float = 1e-6,
            gain_init: float = 1.0,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, **dd)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.same_pad = is_dynamic
        self.eps = eps
        self.gain_init = gain_init

        self.gain = ms.Parameter(mint.empty((self.out_channels, 1, 1, 1), **dd))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Only initialize gain if it exists (for the second call)
        if hasattr(self, 'gain'):
            torch.nn.init.constant_(self.gain, self.gain_init)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            # Also reset parent parameters if needed
            super().reset_parameters()
        # On first call (from super().__init__), do nothing

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = nn.functional.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None, weight = (self.gain * self.scale).view(-1), training = True, momentum = 0., eps = self.eps).reshape_as(self.weight)  # 'torch.nn.functional.batch_norm.reshape_as' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        return nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
