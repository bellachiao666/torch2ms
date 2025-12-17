import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" Bilinear-Attention-Transform and Non-Local Attention

Paper: `Non-Local Neural Networks With Grouped Bilinear Attentional Transforms`
    - https://openaccess.thecvf.com/content_CVPR_2020/html/Chi_Non-Local_Neural_Networks_With_Grouped_Bilinear_Attentional_Transforms_CVPR_2020_paper.html
Adapted from original code: https://github.com/BA-Transform/BAT-Image-Classification
"""
from typing import Optional, Type
# from torch import nn
# from torch.nn import functional as F

from ._fx import register_notrace_module
from .conv_bn_act import ConvNormAct
from .helpers import make_divisible
from .trace_utils import _assert


class NonLocalAttn(msnn.Cell):
    """Spatial NL block for image classification.

    This was adapted from https://github.com/BA-Transform/BAT-Image-Classification
    Their NonLocal impl inspired by https://github.com/facebookresearch/video-nonlocal-net.
    """

    def __init__(
            self,
            in_channels,
            use_scale=True,
            rd_ratio=1/8,
            rd_channels=None,
            rd_divisor=8,
            device=None,
            dtype=None,
            **_,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.scale = in_channels ** -0.5 if use_scale else 1.0
        self.t = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.p = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.g = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.z = nn.Conv2d(rd_channels, in_channels, kernel_size=1, stride=1, bias=True, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.norm = nn.BatchNorm2d(in_channels, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.reset_parameters()

    def construct(self, x):
        shortcut = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        B, C, H, W = t.size()
        t = t.view(B, C, -1).permute(0, 2, 1)
        p = p.view(B, C, -1)
        g = g.view(B, C, -1).permute(0, 2, 1)

        att = mint.bmm(t, p) * self.scale
        att = nn.functional.softmax(att, dim = 2)
        x = mint.bmm(att, g)

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.z(x)
        x = self.norm(x) + shortcut

        return x

    def reset_parameters(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')  # 'torch.nn.init.kaiming_normal_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
                if len(list(m.parameters())) > 1:
                    nn.init.constant_(m.bias, 0.0)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
                nn.init.constant_(m.bias, 0)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
                nn.init.constant_(m.bias, 0)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


@register_notrace_module
class BilinearAttnTransform(msnn.Cell):

    def __init__(
            self,
            in_channels: int,
            block_size: int,
            groups: int,
            act_layer: Type[msnn.Cell] = nn.ReLU,
            norm_layer: Type[msnn.Cell] = nn.BatchNorm2d,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, groups, 1, act_layer=act_layer, norm_layer=norm_layer, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.conv_p = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(block_size, 1), **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.conv_q = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(1, block_size), **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.conv2 = ConvNormAct(in_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.block_size = block_size
        self.groups = groups
        self.in_channels = in_channels

    def resize_mat(self, x, t: int):
        B, C, block_size, block_size1 = x.shape
        _assert(block_size == block_size1, '')
        if t <= 1:
            return x
        x = x.view(B * C, -1, 1, 1)
        x = x * mint.eye(t, t, dtype=x.dtype, device=x.device)
        x = x.view(B * C, block_size, block_size, t, t)
        x = mint.cat(mint.split(x, 1, dim=1), dim=3)
        x = mint.cat(mint.split(x, 1, dim=2), dim=4)
        x = x.view(B, C, block_size * t, block_size * t)
        return x

    def construct(self, x):
        _assert(x.shape[-1] % self.block_size == 0, '')
        _assert(x.shape[-2] % self.block_size == 0, '')
        B, C, H, W = x.shape
        out = self.conv1(x)
        rp = F.adaptive_max_pool2d(out, (self.block_size, 1))  # 'torch.nn.functional.adaptive_max_pool2d' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        cp = F.adaptive_max_pool2d(out, (1, self.block_size))  # 'torch.nn.functional.adaptive_max_pool2d' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        p = self.conv_p(rp).view(B, self.groups, self.block_size, self.block_size).sigmoid()
        q = self.conv_q(cp).view(B, self.groups, self.block_size, self.block_size).sigmoid()
        p = p / p.sum(dim=3, keepdim=True)
        q = q / q.sum(dim=2, keepdim=True)
        p = p.view(B, self.groups, 1, self.block_size, self.block_size).expand(x.size(
            0), self.groups, C // self.groups, self.block_size, self.block_size).contiguous()
        p = p.view(B, C, self.block_size, self.block_size)
        q = q.view(B, self.groups, 1, self.block_size, self.block_size).expand(x.size(
            0), self.groups, C // self.groups, self.block_size, self.block_size).contiguous()
        q = q.view(B, C, self.block_size, self.block_size)
        p = self.resize_mat(p, H // self.block_size)
        q = self.resize_mat(q, W // self.block_size)
        y = p.matmul(x)
        y = y.matmul(q)

        y = self.conv2(y)
        return y


class BatNonLocalAttn(msnn.Cell):
    """ BAT
    Adapted from: https://github.com/BA-Transform/BAT-Image-Classification
    """

    def __init__(
            self,
            in_channels: int,
            block_size: int = 7,
            groups: int = 2,
            rd_ratio: float = 0.25,
            rd_channels: Optional[int] = None,
            rd_divisor: int = 8,
            drop_rate: float = 0.2,
            act_layer: Type[msnn.Cell] = nn.ReLU,
            norm_layer: Type[msnn.Cell] = nn.BatchNorm2d,
            device=None,
            dtype=None,
            **_,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.conv1 = ConvNormAct(in_channels, rd_channels, 1, act_layer=act_layer, norm_layer=norm_layer, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.ba = BilinearAttnTransform(
            rd_channels,
            block_size,
            groups,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **dd,
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.conv2 = ConvNormAct(rd_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.dropout = nn.Dropout2d(p = drop_rate)

    def construct(self, x):
        xl = self.conv1(x)
        y = self.ba(xl)
        y = self.conv2(y)
        y = self.dropout(y)
        return y + x
