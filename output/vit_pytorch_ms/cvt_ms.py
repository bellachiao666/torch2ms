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

class LayerNorm(msnn.Cell): # layernorm, but done in the channel dimension #1
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

class DepthWiseConv2d(msnn.Cell):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = msnn.SequentialCell(
            [nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, stride = stride, padding = padding, groups = dim_in, bias = bias), nn.BatchNorm2d(dim_in), nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)])
    def construct(self, x):
        return self.net(x)

class Attention(msnn.Cell):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = msnn.SequentialCell(
            [nn.Conv2d(inner_dim, dim, 1), nn.Dropout(dropout)])

    def construct(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = mint.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = mint.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(msnn.Cell):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = msnn.CellList([])
        for _ in range(depth):
            self.layers.append(msnn.CellList([
                Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_mult, dropout = dropout)
            ]))
    def construct(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CvT(msnn.Cell):
    def __init__(
        self,
        *,
        num_classes,
        s1_emb_dim = 64,
        s1_emb_kernel = 7,
        s1_emb_stride = 4,
        s1_proj_kernel = 3,
        s1_kv_proj_stride = 2,
        s1_heads = 1,
        s1_depth = 1,
        s1_mlp_mult = 4,
        s2_emb_dim = 192,
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        s3_emb_dim = 384,
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 6,
        s3_depth = 10,
        s3_mlp_mult = 4,
        dropout = 0.,
        channels = 3
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = channels
        layers = []

        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            layers.append(msnn.SequentialCell(
                [nn.Conv2d(dim, config['emb_dim'], kernel_size = config['emb_kernel'], stride = config['emb_stride'], padding = (config['emb_kernel'] // 2)), LayerNorm(config['emb_dim']), Transformer(dim = config['emb_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout)]))

            dim = config['emb_dim']

        self.layers = msnn.SequentialCell([layers])

        self.to_logits = msnn.SequentialCell(
            [nn.AdaptiveAvgPool2d(1), Rearrange('... () () -> ...'), nn.Linear(dim, num_classes)])

    def construct(self, x):
        latents = self.layers(x)
        return self.to_logits(latents)
