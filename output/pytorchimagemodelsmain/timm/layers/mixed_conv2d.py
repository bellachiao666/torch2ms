import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
if not hasattr(nn, "ModuleDict"):
    class ModuleDict(msnn.Cell):
        def __init__(self, modules=None):
            super().__init__()
            self._modules = {}
            modules = modules or {}
            for name, module in modules.items():
                self.add_module(name, module)

        def add_module(self, name, module):
            self._modules[name] = module
            self.insert_child_to_cell(name, module)

        def items(self):
            return self._modules.items()

    nn.ModuleDict = ModuleDict
""" PyTorch Mixed Convolution

Paper: MixConv: Mixed Depthwise Convolutional Kernels (https://arxiv.org/abs/1907.09595)

Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import List, Union
# from torch import nn as nn

from .conv2d_same import create_conv2d_pad


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


# 'torch.nn.ModuleDict' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution

    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, List[int]] = 3,
            stride: int = 1,
            padding: str = '',
            dilation: int = 1,
            depthwise: bool = False,
            **kwargs
    ):
        super().__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = in_ch if depthwise else 1
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch,
                    out_ch,
                    k,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=conv_groups,
                    **kwargs,
                )
            )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.splits = in_splits

    def forward(self, x):
        x_split = mint.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = mint.cat(x_out, 1)
        return x
