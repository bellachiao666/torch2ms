import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from typing import List, Optional, Type, Union

# import torch
# from torch import nn as nn
# from torch.nn import functional as F

from .config import use_fused_attn
from .create_conv2d import create_conv2d
from .helpers import to_2tuple
from .pool2d_same import create_pool2d


class MultiQueryAttentionV2(msnn.Cell):
    """Multi Query Attention.

    Fast Transformer Decoding: One Write-Head is All You Need
    https://arxiv.org/pdf/1911.02150.pdf

    This is an acceletor optimized version - removing multiple unnecessary
    tensor transpose by re-arranging indices according to the following rules: 1)
    contracted indices are at the end, 2) other indices have the same order in the
    input and output tensores.

    Compared to V1, this gives 3x speed up.
    """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            num_heads: int = 8,
            key_dim: int = 64,
            value_dim: int = 64,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            device=None,
            dtype=None,
    ):
        """Initializer."""
        dd = {'dtype': dtype}
        super().__init__()
        dim_out = dim_out or dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.scale = key_dim ** -0.5

        self.query_proj = ms.Parameter(mint.empty((self.num_heads, self.key_dim, dim), **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.key_proj = ms.Parameter(mint.empty((dim, self.key_dim), **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.value_proj = ms.Parameter(mint.empty((dim, self.value_dim), **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = ms.Parameter(mint.empty((dim_out, self.num_heads, self.value_dim), **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.proj_drop = nn.Dropout(proj_drop)

        self.reset_parameters()

    def reset_parameters(self):
        scale = self.key_proj.shape[0] ** -0.5
        nn.init.normal_(self.query_proj, std=scale)  # 'torch.nn.init.normal_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        nn.init.normal_(self.key_proj, std=scale)  # 'torch.nn.init.normal_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        nn.init.normal_(self.value_proj, std=scale)  # 'torch.nn.init.normal_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        nn.init.normal_(self.out_proj, std=self.out_proj.shape[0] ** -0.5)  # 'torch.nn.init.normal_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def _reshape_input(self, t):
        """Reshapes a tensor to three dimensions, keeping the first and last."""
        s = t.shape
        # Propagate the shape statically where possible.
        #num = t.shape[1:-1].numel()
        #return t.reshape(s[0], num, s[-1])
        return t.reshape(s[0], s[1], -1).transpose(1, 2)

    def construct(self, x, m: Optional[ms.Tensor] = None):
        """Run layer computation."""
        b, _, h, w = x.shape
        m = m if m is not None else x

        reshaped_x = self._reshape_input(x)
        reshaped_m = self._reshape_input(m)

        q = mint.einsum('bnd,hkd->bnhk', reshaped_x, self.query_proj)
        k = mint.einsum('bmd,dk->bmk', reshaped_m, self.key_proj)

        attn = mint.einsum('bnhk,bmk->bnhm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = mint.einsum('bmd,dv->bmv', reshaped_m, self.value_proj)
        o = mint.einsum('bnhm,bmv->bnhv', attn, v)
        result = mint.einsum('bnhv,dhv->bdn', o, self.out_proj)
        result = self.proj_drop(result)
        return result.reshape(b, -1, h, w)


class MultiQueryAttention2d(msnn.Cell):
    """Multi Query Attention with spatial downsampling.

     3 parameters are introduced for the spatial downsampling:
     1. kv_stride: downsampling factor on Key and Values only.
     2. query_strides: horizontal & vertical strides on Query only.

    This is an optimized version.
    1. Projections in Attention is explicit written out as 1x1 Conv2D.
    2. Additional reshapes are introduced to bring a up to 3x speed up.
    """
    fused_attn: bool  # 'torch.jit.Final' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            num_heads: int = 8,
            key_dim: Optional[int] = None,
            value_dim: Optional[int] = None,
            query_strides: int = 1,
            kv_stride: int = 1,
            dw_kernel_size: int = 3,
            dilation: int = 1,
            padding: Union[str, int, List[int]] = '',
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[msnn.Cell] = nn.BatchNorm2d,
            use_bias: bool = False,
            device=None,
            dtype=None,
    ):
        """Initializer.

        Args:
          num_heads: Number of attention heads.
          key_dim: Size of the attention key dimension.
          value_dim: Size of the attention value dimension.
          query_strides: Vertical stride size for query only.
          kv_stride: Key and value stride size.
          dw_kernel_size: Spatial dimension of the depthwise kernel.
        """
        dd = {'dtype': dtype}
        super().__init__()
        dim_out = dim_out or dim
        self.num_heads = num_heads
        self.key_dim = key_dim or dim // num_heads
        self.value_dim = value_dim or dim // num_heads
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.has_query_strides = any([s > 1 for s in self.query_strides])
        self.scale = self.key_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.drop = attn_drop

        self.query = msnn.SequentialCell()
        if self.has_query_strides:
            # FIXME dilation
            if padding == 'same':
                self.query.add_module('down_pool', create_pool2d(
                    'avg',
                    kernel_size=self.query_strides,
                    padding='same',
                ))
            else:
                # no pad if not 'same' as kern=stride=even
                self.query.add_module('down_pool', nn.AvgPool2d(kernel_size = query_strides))
            self.query.add_module('norm', norm_layer(dim, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.query.add_module('proj', create_conv2d(
            dim,
            self.num_heads * self.key_dim,
            kernel_size=1,
            bias=use_bias,
            **dd,
        ))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        self.key = msnn.SequentialCell()
        if kv_stride > 1:
            self.key.add_module('down_conv', create_conv2d(
                dim,
                dim,
                kernel_size=dw_kernel_size,
                stride=kv_stride,
                dilation=dilation,
                padding=padding,
                depthwise=True,
                **dd,
            ))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
            self.key.add_module('norm', norm_layer(dim, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.key.add_module('proj', create_conv2d(
            dim,
            self.key_dim,
            kernel_size=1,
            padding=padding,
            bias=use_bias,
            **dd,
        ))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        self.value = msnn.SequentialCell()
        if kv_stride > 1:
            self.value.add_module('down_conv', create_conv2d(
                dim,
                dim,
                kernel_size=dw_kernel_size,
                stride=kv_stride,
                dilation=dilation,
                padding=padding,
                depthwise=True,
                **dd,
            ))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
            self.value.add_module('norm', norm_layer(dim, **dd))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.value.add_module('proj', create_conv2d(
            dim,
            self.value_dim,
            kernel_size=1,
            bias=use_bias,
            **dd,
        ))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        self.attn_drop = nn.Dropout(attn_drop)

        self.output = msnn.SequentialCell()
        if self.has_query_strides:
            self.output.add_module('upsample', nn.Upsample(
                scale_factor = self.query_strides, mode = 'bilinear', align_corners = False))
        self.output.add_module('proj', create_conv2d(
            self.value_dim * self.num_heads,
            dim_out,
            kernel_size=1,
            bias=use_bias,
            **dd,
        ))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.output.add_module('drop', nn.Dropout(proj_drop))

        self.einsum = False
        self.init_weights()

    def init_weights(self):
        # using xavier appeared to improve stability for mobilenetv4 hybrid w/ this layer
        nn.init.xavier_uniform_(self.query.proj.weight)  # 'torch.nn.init.xavier_uniform_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        nn.init.xavier_uniform_(self.key.proj.weight)  # 'torch.nn.init.xavier_uniform_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        nn.init.xavier_uniform_(self.value.proj.weight)  # 'torch.nn.init.xavier_uniform_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if self.kv_stride > 1:
            nn.init.xavier_uniform_(self.key.down_conv.weight)  # 'torch.nn.init.xavier_uniform_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            nn.init.xavier_uniform_(self.value.down_conv.weight)  # 'torch.nn.init.xavier_uniform_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        nn.init.xavier_uniform_(self.output.proj.weight)  # 'torch.nn.init.xavier_uniform_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def _reshape_input(self, t: ms.Tensor):
        """Reshapes a tensor to three dimensions, keeping the batch and channels."""
        s = t.shape
        t = t.reshape(s[0], s[1], -1).transpose(1, 2)
        if self.einsum:
            return t
        else:
            return t.unsqueeze(1).contiguous()

    def _reshape_projected_query(self, t: ms.Tensor, num_heads: int, key_dim: int):
        """Reshapes projected query: [b, n, n, h x k] -> [b, n x n, h, k]."""
        s = t.shape
        t = t.reshape(s[0], num_heads, key_dim, -1)
        if self.einsum:
            return t.permute(0, 3, 1, 2).contiguous()
        else:
            return t.transpose(-1, -2).contiguous()

    def _reshape_output(self, t: ms.Tensor, num_heads: int, h_px: int, w_px: int):
        """Reshape output:[b, n x n x h, k] -> [b, n, n, hk]."""
        s = t.shape
        feat_dim = s[-1] * num_heads
        if not self.einsum:
            t = t.transpose(1, 2)
        return t.reshape(s[0], h_px, w_px, feat_dim).permute(0, 3, 1, 2).contiguous()

    def construct(self, x, attn_mask: Optional[ms.Tensor] = None):
        """Run layer computation."""
        B, C, H, W = s = x.shape

        q = self.query(x)
        # desired q shape: [b, h, k, n x n] - [b, l, h, k]
        q = self._reshape_projected_query(q, self.num_heads, self.key_dim)

        k = self.key(x)
        # output shape of k: [b, k, p], p = m x m
        k = self._reshape_input(k)

        v = self.value(x)
        # output shape of v: [ b, p, k], p = m x m
        v = self._reshape_input(v)

        # desired q shape: [b, n x n, h, k]
        # desired k shape: [b, m x m, k]
        # desired logits shape: [b, n x n, h, m x m]
        if self.einsum:
            attn = mint.einsum('blhk,bpk->blhp', q, k) * self.scale
            if attn_mask is not None:
                # NOTE: assumes mask is float and in correct shape
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            o = mint.einsum('blhp,bpk->blhk', attn, v)
        else:
            if self.fused_attn:
                o = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.
                )  # 'torch.nn.functional.scaled_dot_product_attention' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            else:
                q = q * self.scale
                attn = q @ k.transpose(-1, -2)
                if attn_mask is not None:
                    # NOTE: assumes mask is float and in correct shape
                    attn = attn + attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                o = attn @ v

        # reshape o into [b, hk, n, n,]
        o = self._reshape_output(o, self.num_heads, H // self.query_strides[0], W // self.query_strides[1])
        x = self.output(o)
        return x


class Attention2d(msnn.Cell):
    fused_attn: torch.jit.Final[bool]  # 'torch.jit.Final' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    """ multi-head attention for 2D NCHW tensors"""
    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            num_heads: int = 32,
            bias: bool = True,
            expand_first: bool = False,
            head_first: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            device=None,
            dtype=None,
    ):
        dd = {'dtype': dtype}
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first else dim
        self.num_heads = num_heads
        self.dim_head = dim_attn // num_heads
        self.head_first = head_first
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Conv2d(dim, dim_attn * 3, 1, bias=bias, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim_attn, dim_out, 1, bias=bias, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.proj_drop = nn.Dropout(proj_drop)

    def construct(self, x, attn_mask: Optional[ms.Tensor] = None):
        B, C, H, W = x.shape

        if self.head_first:
            q, k, v = self.qkv(x).view(B, self.num_heads, self.dim_head * 3, -1).chunk(3, dim=2)
        else:
            q, k, v = self.qkv(x).reshape(B, 3, self.num_heads, self.dim_head, -1).unbind(1)

        if self.fused_attn:
            x = mint.transpose(-1, -2).reshape(B, -1, H, W)  # 'torch.nn.functional.scaled_dot_product_attention' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        else:
            q = q.transpose(-1, -2)
            v = v.transpose(-1, -2)
            attn = q @ k * q.size(-1) ** -0.5
            if attn_mask is not None:
                # NOTE: assumes mask is float and in correct shape
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(-1, -2).reshape(B, -1, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
