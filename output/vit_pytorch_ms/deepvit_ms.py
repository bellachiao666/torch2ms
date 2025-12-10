import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# from torch import nn, einsum

from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

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
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.dropout = nn.Dropout(dropout)

        self.reattn_weights = ms.Parameter(mint.randn(size = (heads, heads)))

        self.reattn_norm = msnn.SequentialCell(
            [Rearrange('b h i j -> b i j h'), nn.LayerNorm(heads), Rearrange('b i j h -> b h i j')])

        self.to_out = msnn.SequentialCell(
            [nn.Linear(inner_dim, dim), nn.Dropout(dropout)])

    def construct(self, x):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = mint.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # re-attention

        attn = mint.einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = mint.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(msnn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = msnn.CellList([])
        for _ in range(depth):
            self.layers.append(msnn.CellList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def construct(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DeepViT(msnn.Cell):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = msnn.SequentialCell(
            [Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim)])

        self.pos_embedding = ms.Parameter(mint.randn(size = (1, num_patches + 1, dim)))
        self.cls_token = ms.Parameter(mint.randn(size = (1, 1, dim)))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = msnn.Identity()

        self.mlp_head = msnn.SequentialCell(
            [nn.LayerNorm(dim), nn.Linear(dim, num_classes)])

    def construct(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = mint.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
