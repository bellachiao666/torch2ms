import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
"""SHViT
SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design
Code: https://github.com/ysj9909/SHViT
Paper: https://arxiv.org/abs/2401.16456

@inproceedings{yun2024shvit,
  author={Yun, Seokju and Ro, Youngmin},
  title={SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={5756--5767},
  year={2024}
}
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

# import torch
# import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import GroupNorm1, SqueezeExcite, SelectAdaptivePool2d, LayerType, trunc_normal_
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['SHViT']


class Residual(msnn.Cell):
    def __init__(self, m: msnn.Cell):
        super().__init__()
        self.m = m

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return x + self.m(x)

    @torch.no_grad()
    def fuse(self) -> msnn.Cell:
        if isinstance(self.m, Conv2dNorm):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = mint.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class Conv2dNorm(msnn.SequentialCell):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            bn_weight_init: int = 1,
            device=None,
            dtype=None,
            **kwargs,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.add_module('c', nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False, **dd, **kwargs))  # 存在 *args/**kwargs，需手动确认参数映射;
        self.add_module('bn', nn.BatchNorm2d(out_channels, **dd))  # 存在 *args/**kwargs，需手动确认参数映射;
        nn.init.constant_(self.bn.weight, bn_weight_init)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        nn.init.constant_(self.bn.bias, 0)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            in_channels = w.size(1) * self.c.groups, out_channels = w.size(0), kernel_size = w.shape[2:], stride = self.c.stride, padding = self.c.padding, dilation = self.c.dilation, groups = self.c.groups, dtype = c.weight.dtype)  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device' (position 9);
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class NormLinear(msnn.SequentialCell):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            std: float = 0.02,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(in_features, **dd))  # 存在 *args/**kwargs，需手动确认参数映射;
        self.add_module('l', nn.Linear(in_features, out_features, bias=bias, **dd))  # 存在 *args/**kwargs，需手动确认参数映射;
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    @torch.no_grad()
    def fuse(self) -> nn.Linear:
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = nn.Linear(w.size(1), w.size(0), dtype = l.weight.dtype)  # 'torch.nn.Linear':没有对应的mindspore参数 'device' (position 3);
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(msnn.Cell):
    def __init__(
            self,
            dim: int,
            out_dim: int,
            act_layer: Type[msnn.Cell] = nn.ReLU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2dNorm(dim, hid_dim, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.act1 = act_layer()
        self.conv2 = Conv2dNorm(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.act2 = act_layer()
        self.se = SqueezeExcite(hid_dim, 0.25, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.conv3 = Conv2dNorm(hid_dim, out_dim, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.conv3(x)
        return x


class FFN(msnn.Cell):
    def __init__(
            self,
            dim: int,
            embed_dim: int,
            act_layer: Type[msnn.Cell] = nn.ReLU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.pw1 = Conv2dNorm(dim, embed_dim, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.act = act_layer()
        self.pw2 = Conv2dNorm(embed_dim, dim, bn_weight_init=0, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x


class SHSA(msnn.Cell):
    """Single-Head Self-Attention"""
    def __init__(
            self,
            dim: int,
            qk_dim: int,
            pdim: int,
            norm_layer: Type[msnn.Cell] = GroupNorm1,
            act_layer: Type[msnn.Cell] = nn.ReLU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = norm_layer(pdim, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        self.qkv = Conv2dNorm(pdim, qk_dim * 2 + pdim, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.proj = msnn.SequentialCell(act_layer(), Conv2dNorm(dim, dim, bn_weight_init=0, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        B, _, H, W = x.shape
        x1, x2 = mint.split(x, [self.pdim, self.dim - self.pdim], dim = 1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = mint.split(qkv, [self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(mint.cat([x1, x2], dim = 1))
        return x


class BasicBlock(msnn.Cell):
    def __init__(
            self,
            dim: int,
            qk_dim: int,
            pdim: int,
            type: str,
            norm_layer: Type[msnn.Cell] = GroupNorm1,
            act_layer: Type[msnn.Cell] = nn.ReLU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.conv = Residual(Conv2dNorm(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        if type == "s":
            self.mixer = Residual(SHSA(dim, qk_dim, pdim, norm_layer, act_layer, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.mixer = msnn.Identity()
        self.ffn = Residual(FFN(dim, int(dim * 2), **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.conv(x)
        x = self.mixer(x)
        x = self.ffn(x)
        return x


class StageBlock(msnn.Cell):
    def __init__(
            self,
            prev_dim: int,
            dim: int,
            qk_dim: int,
            pdim: int,
            type: str,
            depth: int,
            norm_layer: Type[msnn.Cell] = GroupNorm1,
            act_layer: Type[msnn.Cell] = nn.ReLU,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.grad_checkpointing = False
        self.downsample = msnn.SequentialCell(
            Residual(Conv2dNorm(prev_dim, prev_dim, 3, 1, 1, groups=prev_dim, **dd)),
            Residual(FFN(prev_dim, int(prev_dim * 2), act_layer, **dd)),
            PatchMerging(prev_dim, dim, act_layer, **dd),
            Residual(Conv2dNorm(dim, dim, 3, 1, 1, groups=dim, **dd)),
            Residual(FFN(dim, int(dim * 2), act_layer, **dd)),
        ) if prev_dim != dim else msnn.Identity()  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        self.blocks = msnn.SequentialCell(*[
            BasicBlock(dim, qk_dim, pdim, type, norm_layer, act_layer, **dd) for _ in range(depth)
        ])  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class SHViT(msnn.Cell):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: Tuple[int, int, int] = (128, 256, 384),
            partial_dim: Tuple[int, int, int] = (32, 64, 96),
            qk_dim: Tuple[int, int, int] = (16, 16, 16),
            depth: Tuple[int, int, int] = (1, 2, 3),
            types: Tuple[str, str, str] = ("s", "s", "s"),
            drop_rate: float = 0.,
            norm_layer: Type[msnn.Cell] = GroupNorm1,
            act_layer: Type[msnn.Cell] = nn.ReLU,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        # Patch embedding
        stem_chs = embed_dim[0]
        self.patch_embed = msnn.SequentialCell(
            Conv2dNorm(in_chans, stem_chs // 8, 3, 2, 1, **dd),
            act_layer(),
            Conv2dNorm(stem_chs // 8, stem_chs // 4, 3, 2, 1, **dd),
            act_layer(),
            Conv2dNorm(stem_chs // 4, stem_chs // 2, 3, 2, 1, **dd),
            act_layer(),
            Conv2dNorm(stem_chs // 2, stem_chs, 3, 2, 1, **dd)
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        # Build SHViT blocks
        stages = []
        prev_chs = stem_chs
        for i in range(len(embed_dim)):
            stages.append(StageBlock(
                prev_dim=prev_chs,
                dim=embed_dim[i],
                qk_dim=qk_dim[i],
                pdim=partial_dim[i],
                type=types[i],
                depth=depth[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                **dd,
            ))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
            prev_chs = embed_dim[i]
            self.feature_info.append(dict(num_chs=prev_chs, reduction=2**(i+4), module=f'stages.{i}'))
        self.stages = msnn.SequentialCell(*stages)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        # Classifier head
        self.num_features = self.head_hidden_size = embed_dim[-1]
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = mint.flatten(1) if global_pool else msnn.Identity()  # don't flatten if pooling disabled
        self.head = NormLinear(self.head_hidden_size, num_classes, **dd) if num_classes > 0 else msnn.Identity()  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    @ms.jit
    def no_weight_decay(self) -> Set:
        return set()

    @ms.jit
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        matcher = dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
            ]
        )
        return matcher

    @ms.jit
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @ms.jit
    def get_classifier(self) -> msnn.Cell:
        return self.head.l

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = mint.flatten(1) if global_pool else msnn.Identity()  # don't flatten if pooling disabled
        self.head = NormLinear(self.head_hidden_size, num_classes) if num_classes > 0 else msnn.Identity()

    def forward_intermediates(
            self,
            x: ms.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[ms.Tensor], Tuple[ms.Tensor, List[ms.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        # forward pass
        x = self.patch_embed(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: ms.Tensor) -> ms.Tensor:
        x = self.patch_embed(x)
        x = self.stages(x)
        return x

    def forward_head(self, x: ms.Tensor, pre_logits: bool = False) -> ms.Tensor:
        x = self.global_pool(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = nn.functional.dropout(x, p = self.drop_rate, training = self.training)
        return x if pre_logits else self.head(x)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @torch.no_grad()
    def fuse(self):
        def fuse_children(net):
            for child_name, child in net.named_children():
                if hasattr(child, 'fuse'):
                    fused = child.fuse()
                    setattr(net, child_name, fused)
                    fuse_children(fused)
                else:
                    fuse_children(child)

        fuse_children(self)


def checkpoint_filter_fn(state_dict: Dict[str, ms.Tensor], model: msnn.Cell) -> Dict[str, ms.Tensor]:
    state_dict = state_dict.get('model', state_dict)

    # out_dict = {}
    # import re
    # replace_rules = [
    #     (re.compile(r'^blocks1\.'), 'stages.0.blocks.'),
    #     (re.compile(r'^blocks2\.'), 'stages.1.blocks.'),
    #     (re.compile(r'^blocks3\.'), 'stages.2.blocks.'),
    # ]
    # downsample_mapping = {}
    # for i in range(1, 3):
    #     downsample_mapping[f'^stages\\.{i}\\.blocks\\.0\\.0\\.'] = f'stages.{i}.downsample.0.'
    #     downsample_mapping[f'^stages\\.{i}\\.blocks\\.0\\.1\\.'] = f'stages.{i}.downsample.1.'
    #     downsample_mapping[f'^stages\\.{i}\\.blocks\\.1\\.'] = f'stages.{i}.downsample.2.'
    #     downsample_mapping[f'^stages\\.{i}\\.blocks\\.2\\.0\\.'] = f'stages.{i}.downsample.3.'
    #     downsample_mapping[f'^stages\\.{i}\\.blocks\\.2\\.1\\.'] = f'stages.{i}.downsample.4.'
    #     for j in range(3, 10):
    #         downsample_mapping[f'^stages\\.{i}\\.blocks\\.{j}\\.'] = f'stages.{i}.blocks.{j - 3}.'
    #
    # downsample_patterns = [
    #     (re.compile(pattern), replacement) for pattern, replacement in downsample_mapping.items()]
    #
    # for k, v in state_dict.items():
    #     for pattern, replacement in replace_rules:
    #         k = pattern.sub(replacement, k)
    #     for pattern, replacement in downsample_patterns:
    #         k = pattern.sub(replacement, k)
    #     out_dict[k] = v

    return state_dict


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (4, 4),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.0.c', 'classifier': 'head.l',
        'license': 'mit',
        'paper_ids': 'arXiv:2401.16456',
        'paper_name': 'SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design',
        'origin_url': 'https://github.com/ysj9909/SHViT',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'shvit_s1.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s1.pth',
    ),
    'shvit_s2.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s2.pth',
    ),
    'shvit_s3.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s3.pth',
    ),
    'shvit_s4.in1k': _cfg(
        hf_hub_id='timm/',
        #url='https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s4.pth',
        input_size=(3, 256, 256),
    ),
})


def _create_shvit(variant: str, pretrained: bool = False, **kwargs: Any) -> SHViT:
    model = build_model_with_cfg(
        SHViT, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2), flatten_sequential=True),
        **kwargs,
    )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def shvit_s1(pretrained: bool = False, **kwargs: Any) -> SHViT:
    model_args = dict(
        embed_dim=(128, 224, 320), depth=(2, 4, 5), partial_dim=(32, 48, 68), types=("i", "s", "s"))
    return _create_shvit('shvit_s1', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;


@register_model
def shvit_s2(pretrained: bool = False, **kwargs: Any) -> SHViT:
    model_args = dict(
        embed_dim=(128, 308, 448), depth=(2, 4, 5), partial_dim=(32, 66, 96), types=("i", "s", "s"))
    return _create_shvit('shvit_s2', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;


@register_model
def shvit_s3(pretrained: bool = False, **kwargs: Any) -> SHViT:
    model_args = dict(
        embed_dim=(192, 352, 448), depth=(3, 5, 5), partial_dim=(48, 75, 96), types=("i", "s", "s"))
    return _create_shvit('shvit_s3', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;


@register_model
def shvit_s4(pretrained: bool = False, **kwargs: Any) -> SHViT:
    model_args = dict(
        embed_dim=(224, 336, 448), depth=(4, 7, 6), partial_dim=(48, 72, 96), types=("i", "s", "s"))
    return _create_shvit('shvit_s4', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
