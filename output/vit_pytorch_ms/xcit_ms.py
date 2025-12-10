import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from random import randrange
# from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack
# from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return nn.functional.normalize(t, p = 2, dim = -1)

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = mint.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale(msnn.Cell):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif 18 > depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        self.fn = fn
        self.scale = ms.Parameter(mint.full(size = ((dim,), init_eps)))

    def construct(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class FeedForward(msnn.Cell):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = msnn.SequentialCell(
            [
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        ])
    def construct(self, x):
        return self.net(x)

class Attention(msnn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = msnn.SequentialCell(
            [
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ])

    def construct(self, x, context = None):
        h = self.heads

        x = self.norm(x)
        context = x if not exists(context) else mint.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        sim = mint.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(sim)
        attn = self.dropout(attn)

        out = mint.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class XCAttention(msnn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.temperature = ms.Parameter(mint.ones(size = (heads, 1, 1)))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = msnn.SequentialCell(
            [
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ])

    def construct(self, x):
        h = self.heads
        x, ps = pack_one(x, 'b * d')

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h d n', h = h), (q, k, v))

        q, k = map(l2norm, (q, k))

        sim = mint.einsum('b h i n, b h j n -> b h i j', q, k) * self.temperature.exp()

        attn = self.attend(sim)
        attn = self.dropout(attn)

        out = mint.einsum('b h i j, b h j n -> b h i n', attn, v)
        out = rearrange(out, 'b h d n -> b n (h d)')

        out = unpack_one(out, ps, 'b * d')
        return self.to_out(out)

class LocalPatchInteraction(msnn.Cell):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        assert (kernel_size % 2) == 1
        padding = kernel_size // 2

        self.net = msnn.SequentialCell(
            [
            nn.LayerNorm(dim),
            Rearrange('b h w c -> b c h w'),
            nn.Conv2d(dim, dim, kernel_size, padding = padding, groups = dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size, padding = padding, groups = dim),
            Rearrange('b c h w -> b h w c')
        ])

    def construct(self, x):
        return self.net(x)

class Transformer(msnn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = msnn.CellList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            layer = ind + 1
            self.layers.append(msnn.CellList([
                LayerScale(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = layer),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = layer)
            ]))

    def construct(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context = context) + x
            x = ff(x) + x

        return x

class XCATransformer(msnn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, local_patch_kernel_size = 3, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = msnn.CellList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            layer = ind + 1
            self.layers.append(msnn.CellList([
                LayerScale(dim, XCAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = layer),
                LayerScale(dim, LocalPatchInteraction(dim, local_patch_kernel_size), depth = layer),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = layer)
            ]))

    def construct(self, x):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for cross_covariance_attn, local_patch_interaction, ff in layers:
            x = cross_covariance_attn(x) + x
            x = local_patch_interaction(x) + x
            x = ff(x) + x

        return x

class XCiT(msnn.Cell):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        local_patch_kernel_size = 3,
        layer_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = msnn.SequentialCell(
            [
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        ])

        self.pos_embedding = ms.Parameter(mint.randn(size = (1, num_patches, dim)))
        self.cls_token = ms.Parameter(mint.randn(dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.xcit_transformer = XCATransformer(dim, depth, heads, dim_head, mlp_dim, local_patch_kernel_size, dropout, layer_dropout)

        self.final_norm = nn.LayerNorm(dim)

        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        self.mlp_head = msnn.SequentialCell(
            [
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ])

    def construct(self, img):
        x = self.to_patch_embedding(img)

        x, ps = pack_one(x, 'b * d')

        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]

        x = unpack_one(x, ps, 'b * d')

        x = self.dropout(x)

        x = self.xcit_transformer(x)

        x = self.final_norm(x)

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)

        x = rearrange(x, 'b ... d -> b (...) d')
        cls_tokens = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(cls_tokens[:, 0])
