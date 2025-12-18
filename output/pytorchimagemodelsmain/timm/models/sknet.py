import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" Selective Kernel Networks (ResNet base)

Paper: Selective Kernel Networks (https://arxiv.org/abs/1903.06586)

This was inspired by reading 'Compounding the Performance Improvements...' (https://arxiv.org/abs/2001.06268)
and a streamlined impl at https://github.com/clovaai/assembled-cnn but I ended up building something closer
to the original paper with some modifications of my own to better balance param count vs accuracy.

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from typing import Optional, Type

# from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectiveKernel, ConvNormAct, create_attn
from ._builder import build_model_with_cfg
from ._registry import register_model, generate_default_cfgs
from .resnet import ResNet


class SelectiveKernelBasic(msnn.Cell):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[msnn.Cell] = None,
            cardinality: int = 1,
            base_width: int = 64,
            sk_kwargs: Optional[dict] = None,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[msnn.Cell] = nn.ReLU,
            norm_layer: Type[msnn.Cell] = nn.BatchNorm2d,
            attn_layer: Optional[Type[msnn.Cell]] = None,
            aa_layer: Optional[Type[msnn.Cell]] = None,
            drop_block: Optional[msnn.Cell] = None,
            drop_path: Optional[msnn.Cell] = None,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()

        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(act_layer=act_layer, norm_layer=norm_layer, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = SelectiveKernel(
            inplanes,
            first_planes,
            stride=stride,
            dilation=first_dilation,
            aa_layer=aa_layer,
            drop_layer=drop_block,
            **conv_kwargs,
            **sk_kwargs,
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.conv2 = ConvNormAct(
            first_planes,
            outplanes,
            kernel_size=3,
            dilation=dilation,
            apply_act=False,
            **conv_kwargs,
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.se = create_attn(attn_layer, outplanes, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.conv2.bn, 'weight', None) is not None:
            nn.init.zeros_(self.conv2.bn.weight)  # 'torch.nn.init.zeros_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act(x)
        return x


class SelectiveKernelBottleneck(msnn.Cell):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[msnn.Cell] = None,
            cardinality: int = 1,
            base_width: int = 64,
            sk_kwargs: Optional[dict] = None,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[msnn.Cell] = nn.ReLU,
            norm_layer: Type[msnn.Cell] = nn.BatchNorm2d,
            attn_layer: Optional[Type[msnn.Cell]] = None,
            aa_layer: Optional[Type[msnn.Cell]] = None,
            drop_block: Optional[msnn.Cell] = None,
            drop_path: Optional[msnn.Cell] = None,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()

        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(act_layer=act_layer, norm_layer=norm_layer, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = ConvNormAct(inplanes, first_planes, kernel_size=1, **conv_kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.conv2 = SelectiveKernel(
            first_planes,
            width,
            stride=stride,
            dilation=first_dilation,
            groups=cardinality,
            aa_layer=aa_layer,
            drop_layer=drop_block,
            **conv_kwargs,
            **sk_kwargs,
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.conv3 = ConvNormAct(width, outplanes, kernel_size=1, apply_act=False, **conv_kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.se = create_attn(attn_layer, outplanes, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.conv3.bn, 'weight', None) is not None:
            nn.init.zeros_(self.conv3.bn.weight)  # 'torch.nn.init.zeros_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act(x)
        return x


def _create_skresnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet,
        variant,
        pretrained,
        **kwargs,
    )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        'license': 'apache-2.0',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'skresnet18.ra_in1k': _cfg(hf_hub_id='timm/'),
    'skresnet34.ra_in1k': _cfg(hf_hub_id='timm/'),
    'skresnet50.untrained': _cfg(),
    'skresnet50d.untrained': _cfg(
        first_conv='conv1.0'),
    'skresnext50_32x4d.ra_in1k': _cfg(hf_hub_id='timm/'),
})


@register_model
def skresnet18(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Selective Kernel ResNet-18 model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(rd_ratio=1 / 8, rd_divisor=16, split_input=True)
    model_args = dict(
        block=SelectiveKernelBasic, layers=[2, 2, 2, 2], block_args=dict(sk_kwargs=sk_kwargs),
        zero_init_last=False)
    return _create_skresnet('skresnet18', pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;


@register_model
def skresnet34(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Selective Kernel ResNet-34 model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(rd_ratio=1 / 8, rd_divisor=16, split_input=True)
    model_args = dict(
        block=SelectiveKernelBasic, layers=[3, 4, 6, 3], block_args=dict(sk_kwargs=sk_kwargs),
        zero_init_last=False)
    return _create_skresnet('skresnet34', pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;


@register_model
def skresnet50(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Select Kernel ResNet-50 model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(split_input=True)
    model_args = dict(
        block=SelectiveKernelBottleneck, layers=[3, 4, 6, 3], block_args=dict(sk_kwargs=sk_kwargs),
        zero_init_last=False)
    return _create_skresnet('skresnet50', pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;


@register_model
def skresnet50d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Select Kernel ResNet-50-D model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(split_input=True)
    model_args = dict(
        block=SelectiveKernelBottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(sk_kwargs=sk_kwargs), zero_init_last=False)
    return _create_skresnet('skresnet50d', pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;


@register_model
def skresnext50_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Select Kernel ResNeXt50-32x4d model. This should be equivalent to
    the SKNet-50 model in the Select Kernel Paper
    """
    sk_kwargs = dict(rd_ratio=1/16, rd_divisor=32, split_input=False)
    model_args = dict(
        block=SelectiveKernelBottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        block_args=dict(sk_kwargs=sk_kwargs), zero_init_last=False)
    return _create_skresnet('skresnext50_32x4d', pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

