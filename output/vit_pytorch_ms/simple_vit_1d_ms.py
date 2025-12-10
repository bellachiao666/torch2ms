import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# from torch import nn

from einops import rearrange
# from einops.layers.torch import Rearrange

# helpers

def posemb_sincos_1d(patches, temperature = 10000, dtype = ms.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = mint.arange(n)  # 'torch.arange':没有对应的mindspore参数 'device' (position 6);
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = mint.arange(dim // 2) / (dim // 2 - 1)  # 'torch.arange':没有对应的mindspore参数 'device' (position 6);
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = mint.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(msnn.Cell):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = msnn.SequentialCell(
            [nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)])
    def construct(self, x):
        return self.net(x)

class Attention(msnn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def construct(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = mint.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = mint.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(msnn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = msnn.CellList([])
        for _ in range(depth):
            self.layers.append(msnn.CellList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def construct(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(msnn.Cell):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()

        assert seq_len % patch_size == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = msnn.SequentialCell(
            [Rearrange('b c (n p) -> b n (p c)', p = patch_size), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim)])

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = msnn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def construct(self, series):
        *_, n, dtype = *series.shape, series.dtype

        x = self.to_patch_embedding(series)
        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

if __name__ == '__main__':

    v = SimpleViT(
        seq_len = 256,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048
    )

    time_series = mint.randn(size = (4, 3, 256))
    logits = v(time_series) # (4, 1000)
