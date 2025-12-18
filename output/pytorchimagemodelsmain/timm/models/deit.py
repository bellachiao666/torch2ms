import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" DeiT - Data-efficient Image Transformers

DeiT model defs and weights from https://github.com/facebookresearch/deit, original copyright below

paper: `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

paper: `DeiT III: Revenge of the ViT` - https://arxiv.org/abs/2204.07118

Modifications copyright 2021, Ross Wightman
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial
from typing import Optional, Type

# import torch
# from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer, trunc_normal_, checkpoint_filter_fn
from ._builder import build_model_with_cfg
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['VisionTransformerDistilled']  # model_registry will add each entrypoint fn to this


class VisionTransformerDistilled(VisionTransformer):
    """ Vision Transformer w/ Distillation Token and Head

    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(*args, **kwargs, weight_init='skip')  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        assert self.global_pool in ('token',)
        dd = {'device': kwargs.get('device', None), 'dtype': kwargs.get('dtype', None)}

        self.num_prefix_tokens = 2
        self.dist_token = ms.Parameter(mint.zeros(1, 1, self.embed_dim, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.pos_embed = ms.Parameter(
            mint.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes, **dd) if self.num_classes > 0 else msnn.Identity()  # 存在 *args/**kwargs，需手动确认参数映射;
        self.distilled_training = False  # must set this True to train w/ distillation token

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode)

    @ms.jit
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_token',
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )

    @ms.jit
    def get_classifier(self) -> msnn.Cell:
        return self.head, self.head_dist

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else msnn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else msnn.Identity()

    @ms.jit
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def _pos_embed(self, x):
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            x = mint.cat((
                self.cls_token.expand(x.shape[0], -1, -1),
                self.dist_token.expand(x.shape[0], -1, -1),
                x),
                dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            x = mint.cat((
                self.cls_token.expand(x.shape[0], -1, -1),
                self.dist_token.expand(x.shape[0], -1, -1),
                x),
                dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def forward_head(self, x, pre_logits: bool = False) -> ms.Tensor:
        x, x_dist = x[:, 0], x[:, 1]
        if pre_logits:
            return (x + x_dist) / 2
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        # 'torch.jit.is_scripting' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2


def _create_deit(variant, pretrained=False, distilled=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model_cls = VisionTransformerDistilled if distilled else VisionTransformer
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        'license': 'apache-2.0',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # deit models (FB weights)
    'deit_tiny_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'deit_small_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'deit_base_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'),
    'deit_base_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'deit_tiny_distilled_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    'deit3_small_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth'),
    'deit3_small_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_medium_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_1k.pth'),
    'deit3_base_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth'),
    'deit3_base_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_large_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_1k.pth'),
    'deit3_large_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_huge_patch14_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_1k.pth'),

    'deit3_small_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_21k.pth',
        crop_pct=1.0),
    'deit3_small_patch16_384.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_medium_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_21k.pth',
        crop_pct=1.0),
    'deit3_base_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth',
        crop_pct=1.0),
    'deit3_base_patch16_384.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_large_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_21k.pth',
        crop_pct=1.0),
    'deit3_large_patch16_384.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_huge_patch14_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_21k_v1.pth',
        crop_pct=1.0),
})


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_deit('deit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_deit('deit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit('deit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit('deit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs) -> VisionTransformerDistilled:
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_deit(
        'deit_tiny_distilled_patch16_224', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs) -> VisionTransformerDistilled:
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_deit(
        'deit_small_distilled_patch16_224', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs) -> VisionTransformerDistilled:
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit(
        'deit_base_distilled_patch16_224', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs) -> VisionTransformerDistilled:
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit(
        'deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit3_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit3_small_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 small model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_small_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit3_medium_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 medium model @ 224x224 (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=512, depth=12, num_heads=8, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_medium_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit3_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit3_base_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit3_large_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit3_large_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 large model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_large_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def deit3_huge_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, no_embed_class=True, init_values=1e-6)
    model = _create_deit('deit3_huge_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


register_model_deprecations(__name__, {
    'deit3_small_patch16_224_in21ft1k': 'deit3_small_patch16_224.fb_in22k_ft_in1k',
    'deit3_small_patch16_384_in21ft1k': 'deit3_small_patch16_384.fb_in22k_ft_in1k',
    'deit3_medium_patch16_224_in21ft1k': 'deit3_medium_patch16_224.fb_in22k_ft_in1k',
    'deit3_base_patch16_224_in21ft1k': 'deit3_base_patch16_224.fb_in22k_ft_in1k',
    'deit3_base_patch16_384_in21ft1k': 'deit3_base_patch16_384.fb_in22k_ft_in1k',
    'deit3_large_patch16_224_in21ft1k': 'deit3_large_patch16_224.fb_in22k_ft_in1k',
    'deit3_large_patch16_384_in21ft1k': 'deit3_large_patch16_384.fb_in22k_ft_in1k',
    'deit3_huge_patch14_224_in21ft1k': 'deit3_huge_patch14_224.fb_in22k_ft_in1k'
})
