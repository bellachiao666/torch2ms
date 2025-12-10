import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from random import randrange
# from torch import nn, einsum

from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

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
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = mint.zeros(size = (1, 1, dim)).fill_(init_eps)
        self.scale = ms.Parameter(scale)
        self.fn = fn
    def construct(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class FeedForward(msnn.Cell):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = msnn.SequentialCell(
            [nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)])
    def construct(self, x):
        return self.net(x)

class Attention(msnn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.mix_heads_pre_attn = ms.Parameter(mint.randn(size = (heads, heads)))
        self.mix_heads_post_attn = ms.Parameter(mint.randn(size = (heads, heads)))

        self.to_out = msnn.SequentialCell(
            [nn.Linear(inner_dim, dim), nn.Dropout(dropout)])

    def construct(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        context = x if not exists(context) else mint.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = mint.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = mint.einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax

        attn = self.attend(dots)
        attn = self.dropout(attn)

        attn = mint.einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = mint.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(msnn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = msnn.CellList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(msnn.CellList([
                LayerScale(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = ind + 1),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = ind + 1)
            ]))
    def construct(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class CaiT(msnn.Cell):
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
        layer_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = msnn.SequentialCell(
            [Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim)])

        self.pos_embedding = ms.Parameter(mint.randn(size = (1, num_patches, dim)))
        self.cls_token = ms.Parameter(mint.randn(size = (1, 1, dim)))

        self.dropout = nn.Dropout(emb_dropout)

        self.patch_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        self.mlp_head = msnn.SequentialCell(
            [nn.LayerNorm(dim), nn.Linear(dim, num_classes)])

    def construct(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.patch_transformer(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(x[:, 0])
