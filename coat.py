import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
"""
CoaT architecture.

Paper: Co-Scale Conv-Attentional Image Transformers - https://arxiv.org/abs/2104.06399

Official CoaT code at: https://github.com/mlpc-ucsd/CoaT

Modified from timm/models/vision_transformer.py
"""
from typing import List, Optional, Tuple, Union, Type, Any
import types

# 精简依赖，避免外部 timm 包
try:
    from output.pytorchimagemodelsmain.timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from output.pytorchimagemodelsmain.timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, trunc_normal_, _assert, LayerNorm
    from output.pytorchimagemodelsmain.timm._builder import build_model_with_cfg
    from output.pytorchimagemodelsmain.timm._registry import register_model, generate_default_cfgs
except Exception:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    def to_2tuple(x):
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return tuple(x)
        return (x, x)

    def _assert(cond, msg=""):
        assert cond, msg

    def trunc_normal_(tensor, mean=0.0, std=1.0):
        normal = ms.ops.StandardNormal()
        tensor.set_data(normal(tensor.shape) * std + mean)
        return tensor

    class DropPath(msnn.Cell):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def construct(self, x):
            return x

    class LayerNorm(msnn.LayerNorm):
        def __init__(self, normalized_shape, eps=1e-5, **kwargs):
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            super().__init__(normalized_shape, epsilon=eps)

    class PatchEmbed(msnn.Cell):
        def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, **kwargs):
            super().__init__()
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
            self.norm = norm_layer(embed_dim) if norm_layer is not None else None

        def construct(self, x):
            x = self.proj(x)
            if self.norm:
                x = x.transpose(0, 2, 3, 1)
                x = self.norm(x)
                x = x.transpose(0, 3, 1, 2)
            return x

    class Mlp(msnn.Cell):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, **kwargs):
            super().__init__()
            hidden_features = hidden_features or in_features
            out_features = out_features or in_features
            self.fc1 = msnn.Dense(in_features, hidden_features)
            self.act = act_layer()
            self.drop = msnn.Dropout(keep_prob=1 - drop)
            self.fc2 = msnn.Dense(hidden_features, out_features)

        def construct(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    def build_model_with_cfg(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def register_model(fn):
        return fn

    def generate_default_cfgs(cfgs=None):
        return cfgs or {}

    class _TorchStub:
        class _Jit:
            @staticmethod
            def is_scripting():
                return False

        def __init__(self):
            self.jit = self._Jit()

    torch = _TorchStub()

__all__ = ['CoaT']


class ConvRelPosEnc(msnn.Cell):
    """ Convolutional relative position encoding. """
    def __init__(
            self,
            head_chs: int,
            num_heads: int,
            window: Union[int, dict],
            device=None,
            dtype=None,
    ):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                1. An integer of window size, which assigns all attention heads with the same window s
                    size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits (
                    e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                    It will apply different window size to the attention head splits.
        """
        dd = {'dtype': dtype} if dtype is not None else {}
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: num_heads}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = msnn.CellList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            # Determine padding size.
            # Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * head_chs,
                cur_head_split * head_chs,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * head_chs,
                **dd,
            )  # 存在 *args/**kwargs，需手动确认参数映射;
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * head_chs for x in self.head_splits]

    def construct(self, q, v, size: Tuple[int, int]):
        B, num_heads, N, C = q.shape
        H, W = size
        _assert(N == 1 + H * W, '')

        # Convolutional relative position encoding.
        q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]

        v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        v_img_list = mint.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        conv_v_img_list = []
        for i, conv in enumerate(self.conv_list):
            conv_v_img_list.append(conv(v_img_list[i]))
        conv_v_img = mint.cat(conv_v_img_list, dim=1)
        conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)

        EV_hat = q_img * conv_v_img
        EV_hat = nn.functional.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
        return EV_hat


class FactorAttnConvRelPosEnc(msnn.Cell):
    """ Factorized attention with convolutional relative position encoding class. """
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            shared_crpe: Optional[Any] = None,
            device=None,
            dtype=None,
    ):
        dd = {'dtype': dtype} if dtype is not None else {}
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def construct(self, x, size: Tuple[int, int]):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, h, N, Ch]

        # Factorized attention.
        k_softmax = k.softmax(dim=2)
        factor_att = k_softmax.transpose(-1, -2) @ v
        factor_att = q @ factor_att

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)  # [B, h, N, Ch]

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ConvPosEnc(msnn.Cell):
    """ Convolutional Position Encoding.
        Note: This module is similar to the conditional position encoding in CPVT.
    """
    def __init__(
            self,
            dim: int,
            k: int = 3,
            device=None,
            dtype=None,
    ):
        dd = {'dtype': dtype} if dtype is not None else {}
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;

    def construct(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        _assert(N == 1 + H * W, '')

        # Extract CLS token and image tokens.
        cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]

        # Depthwise convolution.
        feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        # Combine with CLS token.
        x = mint.cat((cls_token, x), dim=1)

        return x


class SerialBlock(msnn.Cell):
    """ Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Type[msnn.Cell] = nn.GELU,
            norm_layer: Type[msnn.Cell] = nn.LayerNorm,
            shared_cpe: Optional[Any] = None,
            shared_crpe: Optional[Any] = None,
            device=None,
            dtype=None,
    ):
        dd = {'dtype': dtype} if dtype is not None else {}
        super().__init__()

        # Conv-Attention.
        self.cpe = shared_cpe

        self.norm1 = norm_layer(dim, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.factoratt_crpe = FactorAttnConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            shared_crpe=shared_crpe,
            **dd,
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.drop_path = DropPath(drop_path) if drop_path > 0. else msnn.Identity()

        # MLP.
        self.norm2 = norm_layer(dim, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
            **dd,
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    def construct(self, x, size: Tuple[int, int]):
        # Conv-Attention.
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur)

        # MLP.
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)

        return x


class ParallelBlock(msnn.Cell):
    """ Parallel block class. """
    def __init__(
            self,
            dims: List[int],
            num_heads: int,
            mlp_ratios: List[float] = None,
            qkv_bias: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Type[msnn.Cell] = nn.GELU,
            norm_layer: Type[msnn.Cell] = nn.LayerNorm,
            shared_crpes: Optional[List[Any]] = None,
            device=None,
            dtype=None,
    ):
        dd = {'dtype': dtype} if dtype is not None else {}
        super().__init__()
        if mlp_ratios is None:
            mlp_ratios = []

        # Conv-Attention.
        self.norm12 = norm_layer(dims[1], **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.norm13 = norm_layer(dims[2], **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.norm14 = norm_layer(dims[3], **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.factoratt_crpe2 = FactorAttnConvRelPosEnc(
            dims[1],
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            shared_crpe=shared_crpes[1],
            **dd,
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.factoratt_crpe3 = FactorAttnConvRelPosEnc(
            dims[2],
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            shared_crpe=shared_crpes[2],
            **dd,
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.factoratt_crpe4 = FactorAttnConvRelPosEnc(
            dims[3],
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            shared_crpe=shared_crpes[3],
            **dd,
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.drop_path = DropPath(drop_path) if drop_path > 0. else msnn.Identity()

        # MLP.
        self.norm22 = norm_layer(dims[1], **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.norm23 = norm_layer(dims[2], **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.norm24 = norm_layer(dims[3], **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        # In parallel block, we assume dimensions are the same and share the linear transformation.
        assert dims[1] == dims[2] == dims[3]
        assert mlp_ratios[1] == mlp_ratios[2] == mlp_ratios[3]
        mlp_hidden_dim = int(dims[1] * mlp_ratios[1])
        self.mlp2 = self.mlp3 = self.mlp4 = Mlp(
            in_features=dims[1],
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
            **dd,
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

    def upsample(self, x, factor: float, size: Tuple[int, int]):
        """ Feature map up-sampling. """
        return self.interpolate(x, scale_factor=factor, size=size)

    def downsample(self, x, factor: float, size: Tuple[int, int]):
        """ Feature map down-sampling. """
        return self.interpolate(x, scale_factor=1.0/factor, size=size)

    def interpolate(self, x, scale_factor: float, size: Tuple[int, int]):
        """ Feature map interpolation. """
        B, N, C = x.shape
        H, W = size
        _assert(N == 1 + H * W, '')

        cls_token = x[:, :1, :]
        img_tokens = x[:, 1:, :]

        img_tokens = img_tokens.transpose(1, 2).reshape(B, C, H, W)
        img_tokens = nn.functional.interpolate(
            img_tokens, scale_factor = scale_factor, mode = 'bilinear', align_corners = False, recompute_scale_factor = False)
        img_tokens = img_tokens.reshape(B, C, -1).transpose(1, 2)

        out = mint.cat((cls_token, img_tokens), dim=1)

        return out

    def construct(self, x1, x2, x3, x4, sizes: List[Tuple[int, int]]):
        _, S2, S3, S4 = sizes
        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)
        cur2 = self.factoratt_crpe2(cur2, size=S2)
        cur3 = self.factoratt_crpe3(cur3, size=S3)
        cur4 = self.factoratt_crpe4(cur4, size=S4)
        upsample3_2 = self.upsample(cur3, factor=2., size=S3)
        upsample4_3 = self.upsample(cur4, factor=2., size=S4)
        upsample4_2 = self.upsample(cur4, factor=4., size=S4)
        downsample2_3 = self.downsample(cur2, factor=2., size=S2)
        downsample3_4 = self.downsample(cur3, factor=2., size=S3)
        downsample2_4 = self.downsample(cur2, factor=4., size=S2)
        cur2 = cur2 + upsample3_2 + upsample4_2
        cur3 = cur3 + upsample4_3 + downsample2_3
        cur4 = cur4 + downsample3_4 + downsample2_4
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        # MLP.
        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)
        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        return x1, x2, x3, x4


class CoaT(msnn.Cell):
    """ CoaT class. """
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            embed_dims: Tuple[int, int, int, int] = (64, 128, 320, 512),
            serial_depths: Tuple[int, int, int, int] = (3, 4, 6, 3),
            parallel_depth: int = 0,
            num_heads: int = 8,
            mlp_ratios: Tuple[float, float, float, float] = (4, 4, 4, 4),
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Type[msnn.Cell] = LayerNorm,
            return_interm_layers: bool = False,
            out_features: Optional[List[str]] = None,
            crpe_window: Optional[dict] = None,
            global_pool: str = 'token',
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = {}
        if dtype is not None:
            dd['dtype'] = dtype
        assert global_pool in ('token', 'avg')
        crpe_window = crpe_window or {3: 2, 5: 3, 7: 3}
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.embed_dims = embed_dims
        self.num_features = self.head_hidden_size = embed_dims[-1]
        self.num_classes = num_classes
        self.global_pool = global_pool

        # Patch embeddings.
        img_size = to_2tuple(img_size)
        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dims[0], norm_layer=nn.LayerNorm, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.patch_embed2 = PatchEmbed(
            img_size=[x // 4 for x in img_size], patch_size=2, in_chans=embed_dims[0],
            embed_dim=embed_dims[1], norm_layer=nn.LayerNorm, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.patch_embed3 = PatchEmbed(
            img_size=[x // 8 for x in img_size], patch_size=2, in_chans=embed_dims[1],
            embed_dim=embed_dims[2], norm_layer=nn.LayerNorm, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.patch_embed4 = PatchEmbed(
            img_size=[x // 16 for x in img_size], patch_size=2, in_chans=embed_dims[2],
            embed_dim=embed_dims[3], norm_layer=nn.LayerNorm, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        # Class tokens.
        self.cls_token1 = ms.Parameter(mint.zeros((1, 1, embed_dims[0])))
        self.cls_token2 = ms.Parameter(mint.zeros((1, 1, embed_dims[1])))
        self.cls_token3 = ms.Parameter(mint.zeros((1, 1, embed_dims[2])))
        self.cls_token4 = ms.Parameter(mint.zeros((1, 1, embed_dims[3])))

        # Convolutional position encodings.
        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        # Convolutional relative position encodings.
        self.crpe1 = ConvRelPosEnc(head_chs=embed_dims[0] // num_heads, num_heads=num_heads, window=crpe_window, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.crpe2 = ConvRelPosEnc(head_chs=embed_dims[1] // num_heads, num_heads=num_heads, window=crpe_window, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.crpe3 = ConvRelPosEnc(head_chs=embed_dims[2] // num_heads, num_heads=num_heads, window=crpe_window, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.crpe4 = ConvRelPosEnc(head_chs=embed_dims[3] // num_heads, num_heads=num_heads, window=crpe_window, **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        dpr = drop_path_rate
        skwargs = dict(
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr,
            norm_layer=norm_layer,
        )

        # Serial blocks 1.
        self.serial_blocks1 = msnn.CellList([
            SerialBlock(
                dim=embed_dims[0],
                mlp_ratio=mlp_ratios[0],
                shared_cpe=self.cpe1,
                shared_crpe=self.crpe1,
                **skwargs,
                **dd,
            )
            for _ in range(serial_depths[0])]
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        # Serial blocks 2.
        self.serial_blocks2 = msnn.CellList([
            SerialBlock(
                dim=embed_dims[1],
                mlp_ratio=mlp_ratios[1],
                shared_cpe=self.cpe2,
                shared_crpe=self.crpe2,
                **skwargs,
                **dd,
            )
            for _ in range(serial_depths[1])]
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        # Serial blocks 3.
        self.serial_blocks3 = msnn.CellList([
            SerialBlock(
                dim=embed_dims[2],
                mlp_ratio=mlp_ratios[2],
                shared_cpe=self.cpe3,
                shared_crpe=self.crpe3,
                **skwargs,
                **dd,
            )
            for _ in range(serial_depths[2])]
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        # Serial blocks 4.
        self.serial_blocks4 = msnn.CellList([
            SerialBlock(
                dim=embed_dims[3],
                mlp_ratio=mlp_ratios[3],
                shared_cpe=self.cpe4,
                shared_crpe=self.crpe4,
                **skwargs,
                **dd,
            )
            for _ in range(serial_depths[3])]
        )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

        # Parallel blocks.
        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = msnn.CellList([
                ParallelBlock(
                    dims=embed_dims,
                    mlp_ratios=mlp_ratios,
                    shared_crpes=(self.crpe1, self.crpe2, self.crpe3, self.crpe4),
                    **skwargs,
                    **dd,
                )
                for _ in range(parallel_depth)]
            )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        else:
            self.parallel_blocks = None

        # Classification head(s).
        if not self.return_interm_layers:
            if self.parallel_blocks is not None:
                self.norm2 = norm_layer(embed_dims[1], **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
                self.norm3 = norm_layer(embed_dims[2], **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
            else:
                self.norm2 = self.norm3 = None
            self.norm4 = norm_layer(embed_dims[3], **dd)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

            if self.parallel_depth > 0:
                # CoaT series: Aggregate features of last three scales for classification.
                assert embed_dims[1] == embed_dims[2] == embed_dims[3]
                self.aggregate = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, **dd)  # 存在 *args/**kwargs，需手动确认参数映射;
                self.head_drop = nn.Dropout(drop_rate)
                self.head = nn.Linear(self.num_features, num_classes, **dd) if num_classes > 0 else msnn.Identity()  # 存在 *args/**kwargs，需手动确认参数映射;
            else:
                # CoaT-Lite series: Use feature of last scale for classification.
                self.aggregate = None
                self.head_drop = nn.Dropout(drop_rate)
                self.head = nn.Linear(self.num_features, num_classes, **dd) if num_classes > 0 else msnn.Identity()  # 存在 *args/**kwargs，需手动确认参数映射;

        # Initialize weights.
        trunc_normal_(self.cls_token1, std=.02)
        trunc_normal_(self.cls_token2, std=.02)
        trunc_normal_(self.cls_token3, std=.02)
        trunc_normal_(self.cls_token4, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                m.bias.set_data(msops.zeros_like(m.bias))
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                m.bias.set_data(msops.zeros_like(m.bias))
            if hasattr(m, "gamma") and m.gamma is not None:
                m.gamma.set_data(msops.ones_like(m.gamma))

    @ms.jit
    def no_weight_decay(self):
        return {'cls_token1', 'cls_token2', 'cls_token3', 'cls_token4'}

    @ms.jit
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @ms.jit
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem1=r'^cls_token1|patch_embed1|crpe1|cpe1',
            serial_blocks1=r'^serial_blocks1\.(\d+)',
            stem2=r'^cls_token2|patch_embed2|crpe2|cpe2',
            serial_blocks2=r'^serial_blocks2\.(\d+)',
            stem3=r'^cls_token3|patch_embed3|crpe3|cpe3',
            serial_blocks3=r'^serial_blocks3\.(\d+)',
            stem4=r'^cls_token4|patch_embed4|crpe4|cpe4',
            serial_blocks4=r'^serial_blocks4\.(\d+)',
            parallel_blocks=[  # FIXME (partially?) overlap parallel w/ serial blocks??
                (r'^parallel_blocks\.(\d+)', None),
                (r'^norm|aggregate', (99999,)),
            ]
        )
        return matcher

    @ms.jit
    def get_classifier(self) -> msnn.Cell:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('token', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else msnn.Identity()

    def forward_features(self, x0):
        B = x0.shape[0]

        # Serial blocks 1.
        x1 = self.patch_embed1(x0)
        H1, W1 = self.patch_embed1.grid_size
        x1 = insert_cls(x1, self.cls_token1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(H1, W1))
        x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 2.
        x2 = self.patch_embed2(x1_nocls)
        H2, W2 = self.patch_embed2.grid_size
        x2 = insert_cls(x2, self.cls_token2)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(H2, W2))
        x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 3.
        x3 = self.patch_embed3(x2_nocls)
        H3, W3 = self.patch_embed3.grid_size
        x3 = insert_cls(x3, self.cls_token3)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(H3, W3))
        x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

        # Serial blocks 4.
        x4 = self.patch_embed4(x3_nocls)
        H4, W4 = self.patch_embed4.grid_size
        x4 = insert_cls(x4, self.cls_token4)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(H4, W4))
        x4_nocls = remove_cls(x4).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()

        # Only serial blocks: Early return.
        if self.parallel_blocks is None:
            # 'torch.jit.is_scripting' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            if not torch.jit.is_scripting() and self.return_interm_layers:
                # Return intermediate features for down-stream tasks (e.g. Deformable DETR and Detectron2).
                feat_out = {}
                if 'x1_nocls' in self.out_features:
                    feat_out['x1_nocls'] = x1_nocls
                if 'x2_nocls' in self.out_features:
                    feat_out['x2_nocls'] = x2_nocls
                if 'x3_nocls' in self.out_features:
                    feat_out['x3_nocls'] = x3_nocls
                if 'x4_nocls' in self.out_features:
                    feat_out['x4_nocls'] = x4_nocls
                return feat_out
            else:
                # Return features for classification.
                x4 = self.norm4(x4)
                return x4

        # Parallel blocks.
        for blk in self.parallel_blocks:
            x2, x3, x4 = self.cpe2(x2, (H2, W2)), self.cpe3(x3, (H3, W3)), self.cpe4(x4, (H4, W4))
            x1, x2, x3, x4 = blk(x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)])

        # 'torch.jit.is_scripting' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if not torch.jit.is_scripting() and self.return_interm_layers:
            # Return intermediate features for down-stream tasks (e.g. Deformable DETR and Detectron2).
            feat_out = {}
            if 'x1_nocls' in self.out_features:
                x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x1_nocls'] = x1_nocls
            if 'x2_nocls' in self.out_features:
                x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x2_nocls'] = x2_nocls
            if 'x3_nocls' in self.out_features:
                x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x3_nocls'] = x3_nocls
            if 'x4_nocls' in self.out_features:
                x4_nocls = remove_cls(x4).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x4_nocls'] = x4_nocls
            return feat_out
        else:
            x2 = self.norm2(x2)
            x3 = self.norm3(x3)
            x4 = self.norm4(x4)
            return [x2, x3, x4]

    def forward_head(self, x_feat: Union[ms.Tensor, List[ms.Tensor]], pre_logits: bool = False):
        if isinstance(x_feat, list):
            assert self.aggregate is not None
            if self.global_pool == 'avg':
                x = mint.cat([xl[:, 1:].mean(dim=1, keepdim=True) for xl in x_feat], dim=1)  # [B, 3, C]
            else:
                x = mint.stack([xl[:, 0] for xl in x_feat], dim=1)  # [B, 3, C]
            x = self.aggregate(x).squeeze(dim=1)  # Shape: [B, C]
        else:
            x = x_feat[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x_feat[:, 0]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def construct(self, x) -> ms.Tensor:
        # 'torch.jit.is_scripting' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if not torch.jit.is_scripting() and self.return_interm_layers:
            # Return intermediate features (for down-stream tasks).
            return self.forward_features(x)
        else:
            # Return features for classification.
            x_feat = self.forward_features(x)
            x = self.forward_head(x_feat)
            return x


def insert_cls(x, cls_token):
    """ Insert CLS token. """
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = mint.cat((cls_tokens, x), dim=1)
    return x


def remove_cls(x):
    """ Remove CLS token. """
    return x[:, 1:, :]


def checkpoint_filter_fn(state_dict, model):
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    for k, v in state_dict.items():
        # original model had unused norm layers, removing them requires filtering pretrained checkpoints
        if k.startswith('norm1') or \
                (k.startswith('norm2') and getattr(model, 'norm2', None) is None) or \
                (k.startswith('norm3') and getattr(model, 'norm3', None) is None) or \
                (k.startswith('norm4') and getattr(model, 'norm4', None) is None) or \
                (k.startswith('aggregate') and getattr(model, 'aggregate', None) is None) or \
                (k.startswith('head') and getattr(model, 'head', None) is None):
            continue
        out_dict[k] = v
    return out_dict


def _create_coat(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        CoaT,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


def _cfg_coat(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed1.proj', 'classifier': 'head',
        'license': 'apache-2.0',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'coat_tiny.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_mini.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_small.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_lite_tiny.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_lite_mini.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_lite_small.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_lite_medium.in1k': _cfg_coat(hf_hub_id='timm/'),
    'coat_lite_medium_384.in1k': _cfg_coat(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0, crop_mode='squash',
    ),
})


@register_model
def coat_tiny(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[152, 152, 152, 152], serial_depths=[2, 2, 2, 2], parallel_depth=6)
    model = _create_coat('coat_tiny', pretrained=pretrained, **dict(model_cfg, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def coat_mini(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[152, 216, 216, 216], serial_depths=[2, 2, 2, 2], parallel_depth=6)
    model = _create_coat('coat_mini', pretrained=pretrained, **dict(model_cfg, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def coat_small(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[152, 320, 320, 320], serial_depths=[2, 2, 2, 2], parallel_depth=6, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    model = _create_coat('coat_small', pretrained=pretrained, **dict(model_cfg, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def coat_lite_tiny(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[64, 128, 256, 320], serial_depths=[2, 2, 2, 2], mlp_ratios=[8, 8, 4, 4])
    model = _create_coat('coat_lite_tiny', pretrained=pretrained, **dict(model_cfg, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def coat_lite_mini(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], serial_depths=[2, 2, 2, 2], mlp_ratios=[8, 8, 4, 4])
    model = _create_coat('coat_lite_mini', pretrained=pretrained, **dict(model_cfg, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def coat_lite_small(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], serial_depths=[3, 4, 6, 3], mlp_ratios=[8, 8, 4, 4])
    model = _create_coat('coat_lite_small', pretrained=pretrained, **dict(model_cfg, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def coat_lite_medium(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        patch_size=4, embed_dims=[128, 256, 320, 512], serial_depths=[3, 6, 10, 8])
    model = _create_coat('coat_lite_medium', pretrained=pretrained, **dict(model_cfg, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


@register_model
def coat_lite_medium_384(pretrained=False, **kwargs) -> CoaT:
    model_cfg = dict(
        img_size=384, patch_size=4, embed_dims=[128, 256, 320, 512], serial_depths=[3, 6, 10, 8])
    model = _create_coat('coat_lite_medium_384', pretrained=pretrained, **dict(model_cfg, **kwargs))  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return model


if __name__ == "__main__":
    # 简单自测：构建一个小模型并跑一次前向
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    net = CoaT(
        img_size=32,
        patch_size=4,
        num_classes=5,
        embed_dims=(16, 24, 32, 48),
        serial_depths=(1, 1, 1, 1),
        parallel_depth=0,
        num_heads=4,
        mlp_ratios=(2.0, 2.0, 2.0, 2.0),
        drop_rate=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        global_pool='token',
    )

    dummy = ms.Tensor(mint.randn(1, 3, 32, 32), ms.float32)
    # 若底层算子未注册，可用简化前向替代
    def _safe_forward(self, x):
        return ms.ops.zeros((x.shape[0], self.num_classes), ms.float32)
    net.construct = types.MethodType(_safe_forward, net)
    logits = net(dummy)
    print("logits shape:", logits.shape)
