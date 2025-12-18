import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" Global Response Normalization Module

Based on the GRN layer presented in
`ConvNeXt-V2 - Co-designing and Scaling ConvNets with Masked Autoencoders` - https://arxiv.org/abs/2301.00808

This implementation
* works for both NCHW and NHWC tensor layouts
* uses affine param names matching existing torch norm layers
* slightly improves eager mode performance via fused addcmul

Hacked together by / Copyright 2023 Ross Wightman
"""

# import torch


class GlobalResponseNorm(msnn.Cell):
    """ Global Response Normalization layer
    """
    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            channels_last: bool = True,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        if channels_last:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            self.wb_shape = (1, 1, 1, -1)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1)

        self.weight = ms.Parameter(mint.zeros(dim, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.bias = ms.Parameter(mint.zeros(dim, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    def construct(self, x):
        x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)  # 'torch.addcmul' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
