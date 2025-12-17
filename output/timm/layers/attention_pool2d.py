import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" Attention Pool 2D

Implementations of 2D spatial feature pooling using multi-head attention instead of average pool.

Based on idea in CLIP by OpenAI, licensed Apache 2.0
https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional, Union, Tuple

# import torch
# import torch.nn as nn

from .config import use_fused_attn
from .helpers import to_2tuple
from .pos_embed import resample_abs_pos_embed
from .pos_embed_sincos import apply_rot_embed_cat, create_rope_embed
from .weight_init import trunc_normal_


class RotAttentionPool2d(msnn.Cell):
    """ Attention based 2D feature pooling w/ rotary (relative) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    Adapted from the AttentionPool2d in CLIP w/ rotary embedding instead of learned embed.
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: While this impl does not require a fixed feature size, performance at differeing resolutions from
    train varies widely and falls off dramatically. I'm not sure if there is a way around this... -RW
    """
    fused_attn: torch.jit.Final[bool]  # 'torch.jit.Final' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def __init__(
            self,
            in_features: int,
            out_features: Optional[int] = None,
            ref_feat_size: Union[int, Tuple[int, int]] = 7,
            embed_dim: Optional[int] = None,
            head_dim: Optional[int] = 64,
            num_heads: Optional[int] = None,
            qkv_bias: bool = True,
            qkv_separate: bool = False,
            pool_type: str = 'token',
            class_token: bool = False,
            drop_rate: float = 0.,
            rope_type: str = 'cat',
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        assert pool_type in ('', 'token')
        self.embed_dim = embed_dim = embed_dim or in_features
        self.in_features = in_features
        self.out_features = out_features or in_features
        ref_feat_size = to_2tuple(ref_feat_size)
        if num_heads is not None:
            assert embed_dim % num_heads == 0
            head_dim = embed_dim // num_heads
        else:
            assert embed_dim % head_dim == 0
            num_heads = embed_dim // head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pool_type = pool_type.lower()
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.rope_type = rope_type

        if class_token:
            self.cls_token = ms.Parameter(mint.zeros(1, embed_dim, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.cls_token = None

        if qkv_separate:
            self.q = nn.Linear(in_features, embed_dim, bias=qkv_bias, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
            self.k = nn.Linear(in_features, embed_dim, bias=qkv_bias, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
            self.v = nn.Linear(in_features, embed_dim, bias=qkv_bias, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
            self.qkv = None
        else:
            self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(embed_dim, self.out_features, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;

        self.pos_embed = create_rope_embed(
            rope_type=rope_type,
            dim=embed_dim,
            num_heads=num_heads,
            in_pixels=False,
            ref_feat_shape=ref_feat_size,
            rotate_half=False,
            **dd,
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    def init_weights(self, zero_init_last: bool = False):
        if self.qkv is None:
            in_features = self.q.in_features
            trunc_normal_(self.q.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.q.bias)  # 'torch.nn.init.zeros_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            trunc_normal_(self.k.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.k.bias)  # 'torch.nn.init.zeros_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            trunc_normal_(self.v.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.v.bias)  # 'torch.nn.init.zeros_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        else:
            in_features = self.qkv.in_features
            trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.qkv.bias)  # 'torch.nn.init.zeros_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def reset(self, num_classes: Optional[int] = None, pool_type: Optional[str] = None):
        # NOTE: this module is being used as a head, so need compatible reset()
        if pool_type is not None:
            assert pool_type in ('', 'token')
            self.pool_type = pool_type
        if num_classes is not None:
            self.proj = nn.Linear(self.in_features, num_classes) if num_classes > 0 else msnn.Identity()
            self.out_features = num_classes if num_classes > 0 else self.embed_dim

    def _pool(self, x: ms.Tensor, H: int, W: int) -> ms.Tensor:
        if self.pool_type == 'token':
            x = x[:, 0]
        else:
            # if not pooled, return spatial output without token
            x = x[:, 1:].reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
        return x

    def construct(self, x, pre_logits: bool = False):
        B, _, H, W = x.shape
        N = H * W
        x = x.flatten(2).transpose(1, 2)
        if self.cls_token is None:
            x = mint.cat([x.mean(1, keepdim=True), x], dim=1)
        else:
            x = mint.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        if self.qkv is None:
            q = self.q(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            x = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = x.unbind(0)

        rope = self.pos_embed.get_embed((H, W))
        if isinstance(rope, tuple):
            # RotaryEmbedding returns (sin, cos) tuple - concatenate for apply_rot_embed_cat
            rope = mint.cat(rope, dim=-1)
        q = mint.cat([q[:, :, :1, :], apply_rot_embed_cat(q[:, :, 1:, :], rope)], dim=2).type_as(v)
        k = mint.cat([k[:, :, :1, :], apply_rot_embed_cat(k[:, :, 1:, :], rope)], dim=2).type_as(v)

        if self.fused_attn:
            x = nn.functional.scaled_dot_product_attention(q, k, v)  # 'torch.nn.functional.scaled_dot_product_attention' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N + 1, -1)
        x = self.drop(x)
        if pre_logits:
            x = self._pool(x, H, W)
            return x
        x = self.proj(x)
        x = self._pool(x, H, W)
        return x


class AttentionPool2d(msnn.Cell):
    """ Attention based 2D feature pooling w/ learned (absolute) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    It was based on impl in CLIP by OpenAI
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: This requires feature size upon construction and well prevent adaptive sizing of the network.
    """
    fused_attn: torch.jit.Final[bool]  # 'torch.jit.Final' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def __init__(
            self,
            in_features: int,
            feat_size: Union[int, Tuple[int, int]] = 7,
            out_features: Optional[int] = None,
            embed_dim: Optional[int] = None,
            head_dim: Optional[int] = 64,
            num_heads: Optional[int] = None,
            qkv_bias: bool = True,
            qkv_separate: bool = False,
            pool_type: str = 'token',
            class_token: bool = False,
            drop_rate: float = 0.,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        assert pool_type in ('', 'token')
        self.embed_dim = embed_dim = embed_dim or in_features
        self.in_features = in_features
        self.out_features = out_features or in_features
        if num_heads is not None:
            assert embed_dim % num_heads == 0
            head_dim = embed_dim // num_heads
        else:
            assert embed_dim % head_dim == 0
            num_heads = embed_dim // head_dim
        self.feat_size = to_2tuple(feat_size)
        self.seq_len = self.feat_size[0] * self.feat_size[1]
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pool_type = pool_type
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        if class_token:
            self.cls_token = ms.Parameter(mint.zeros(1, embed_dim, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.cls_token = None

        if qkv_separate:
            self.q = nn.Linear(in_features, embed_dim, bias=qkv_bias, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
            self.k = nn.Linear(in_features, embed_dim, bias=qkv_bias, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
            self.v = nn.Linear(in_features, embed_dim, bias=qkv_bias, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
            self.qkv = None
        else:
            self.q = self.k = self.v = None
            self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(embed_dim, self.out_features, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.pos_embed = ms.Parameter(mint.zeros(self.seq_len + 1, in_features, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        self.init_weights()

    def init_weights(self, zero_init_last: bool = False):
        if self.qkv is None:
            in_features = self.q.in_features
            trunc_normal_(self.q.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.q.bias)  # 'torch.nn.init.zeros_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            trunc_normal_(self.k.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.k.bias)  # 'torch.nn.init.zeros_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            trunc_normal_(self.v.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.v.bias)  # 'torch.nn.init.zeros_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        else:
            in_features = self.qkv.in_features
            trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.qkv.bias)  # 'torch.nn.init.zeros_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        trunc_normal_(self.pos_embed, std=in_features ** -0.5)

    def reset(self, num_classes: Optional[int] = None, pool_type: Optional[str] = None):
        # NOTE: this module is being used as a head, so need compatible reset()
        if pool_type is not None:
            assert pool_type in ('', 'token')
            self.pool_type = pool_type
        if num_classes is not None:
            self.proj = nn.Linear(self.in_features, num_classes) if num_classes > 0 else msnn.Identity()
            self.out_features = num_classes if num_classes > 0 else self.embed_dim

    def _pool(self, x: ms.Tensor, H: int, W: int) -> ms.Tensor:
        if self.pool_type == 'token':
            x = x[:, 0]
        else:
            # if not pooled, return spatial output without token
            x = x[:, 1:].reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
        return x

    def construct(self, x, pre_logits: bool = False):
        B, _, H, W = x.shape
        N = H * W
        x = x.flatten(2).transpose(1, 2)
        if self.cls_token is None:
            x = mint.cat([x.mean(1, keepdim=True), x], dim=1)
        else:
            x = mint.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        pos_embed = resample_abs_pos_embed(self.pos_embed.unsqueeze(0), (H, W), num_prefix_tokens=1)
        x = x + pos_embed

        if self.qkv is None:
            q = self.q(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            x = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = x.unbind(0)

        if self.fused_attn:
            x = nn.functional.scaled_dot_product_attention(q, k, v)  # 'torch.nn.functional.scaled_dot_product_attention' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N + 1, -1)
        x = self.drop(x)
        if pre_logits:
            x = self._pool(x, H, W)
            return x
        x = self.proj(x)
        x = self._pool(x, H, W)
        return x
