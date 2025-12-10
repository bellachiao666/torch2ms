import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# from torch import nn, einsum

from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

# helper methods

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class Residual(msnn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNorm(msnn.Cell):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = ms.Parameter(mint.ones(size = (1, dim, 1, 1)))
        self.b = ms.Parameter(mint.zeros(size = (1, dim, 1, 1)))

    def construct(self, x):
        var = mint.var(x, dim = 1, keepdim = True)
        mean = mint.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class FeedForward(msnn.Cell):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = msnn.SequentialCell(
            [LayerNorm(dim), nn.Conv2d(dim, dim * mult, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv2d(dim * mult, dim, 1), nn.Dropout(dropout)])
    def construct(self, x):
        return self.net(x)

class PatchEmbedding(msnn.Cell):
    def __init__(self, *, dim, dim_out, patch_size):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.patch_size = patch_size

        self.proj = msnn.SequentialCell(
            [LayerNorm(patch_size ** 2 * dim), nn.Conv2d(patch_size ** 2 * dim, dim_out, 1), LayerNorm(dim_out)])

    def construct(self, fmap):
        p = self.patch_size
        fmap = rearrange(fmap, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = p, p2 = p)
        return self.proj(fmap)

class PEG(msnn.Cell):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = Residual(nn.Conv2d(dim, dim, kernel_size = kernel_size, stride = 1, padding = kernel_size // 2, groups = dim))

    def construct(self, x):
        return self.proj(x)

class LocalAttention(msnn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., patch_size = 7):
        super().__init__()
        inner_dim = dim_head *  heads
        self.patch_size = patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)

        self.to_out = msnn.SequentialCell(
            [nn.Conv2d(inner_dim, dim, 1), nn.Dropout(dropout)])

    def construct(self, fmap):
        fmap = self.norm(fmap)

        shape, p = fmap.shape, self.patch_size
        b, n, x, y, h = *shape, self.heads
        x, y = map(lambda t: t // p, (x, y))

        fmap = rearrange(fmap, 'b c (x p1) (y p2) -> (b x y) c p1 p2', p1 = p, p2 = p)

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) p1 p2 -> (b h) (p1 p2) d', h = h), (q, k, v))

        dots = mint.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim = - 1)

        out = mint.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b x y h) (p1 p2) d -> b (h d) (x p1) (y p2)', h = h, x = x, y = y, p1 = p, p2 = p)
        return self.to_out(out)

class GlobalAttention(msnn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., k = 7):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, k, stride = k, bias = False)

        self.dropout = nn.Dropout(dropout)

        self.to_out = msnn.SequentialCell(
            [nn.Conv2d(inner_dim, dim, 1), nn.Dropout(dropout)])

    def construct(self, x):
        x = self.norm(x)

        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = mint.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        out = mint.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(msnn.Cell):
    def __init__(self, dim, depth, heads = 8, dim_head = 64, mlp_mult = 4, local_patch_size = 7, global_k = 7, dropout = 0., has_local = True):
        super().__init__()
        self.layers = msnn.CellList([])
        for _ in range(depth):
            self.layers.append(msnn.CellList([
                Residual(LocalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, patch_size = local_patch_size)) if has_local else msnn.Identity(),
                Residual(FeedForward(dim, mlp_mult, dropout = dropout)) if has_local else msnn.Identity(),
                Residual(GlobalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, k = global_k)),
                Residual(FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def construct(self, x):
        for local_attn, ff1, global_attn, ff2 in self.layers:
            x = local_attn(x)
            x = ff1(x)
            x = global_attn(x)
            x = ff2(x)
        return x

class TwinsSVT(msnn.Cell):
    def __init__(
        self,
        *,
        num_classes,
        s1_emb_dim = 64,
        s1_patch_size = 4,
        s1_local_patch_size = 7,
        s1_global_k = 7,
        s1_depth = 1,
        s2_emb_dim = 128,
        s2_patch_size = 2,
        s2_local_patch_size = 7,
        s2_global_k = 7,
        s2_depth = 1,
        s3_emb_dim = 256,
        s3_patch_size = 2,
        s3_local_patch_size = 7,
        s3_global_k = 7,
        s3_depth = 5,
        s4_emb_dim = 512,
        s4_patch_size = 2,
        s4_local_patch_size = 7,
        s4_global_k = 7,
        s4_depth = 4,
        peg_kernel_size = 3,
        dropout = 0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = 3
        layers = []

        for prefix in ('s1', 's2', 's3', 's4'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            is_last = prefix == 's4'

            dim_next = config['emb_dim']

            layers.append(msnn.SequentialCell(
                [PatchEmbedding(dim = dim, dim_out = dim_next, patch_size = config['patch_size']), Transformer(dim = dim_next, depth = 1, local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last), PEG(dim = dim_next, kernel_size = peg_kernel_size), Transformer(dim = dim_next, depth = config['depth'],  local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last)]))

            dim = dim_next

        self.layers = msnn.SequentialCell(
            [layers, nn.AdaptiveAvgPool2d(1), Rearrange('... () () -> ...'), nn.Linear(dim, num_classes)])

    def construct(self, x):
        return self.layers(x)
