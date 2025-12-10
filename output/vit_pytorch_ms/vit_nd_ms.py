import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from __future__ import annotations
# from torch import nn

from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

# helpers

def join(arr, delimiter = ' '):
    return delimiter.join(arr)

def ensure_tuple(t, length):
    if isinstance(t, (tuple, list)):
        assert len(t) == length, f'Expected tuple of length {length}, got {len(t)}'
        return tuple(t)
    return (t,) * length

# classes

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
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_out = msnn.SequentialCell(
            [nn.Linear(inner_dim, dim), nn.Dropout(dropout)]) if project_out else msnn.Identity()
    
    def construct(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        dots = mint.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = mint.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(msnn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
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
        return self.norm(x)

class ViTND(msnn.Cell):
    def __init__(
        self,
        *,
        ndim: int,
        input_shape: int | tuple[int, ...],
        patch_size: int | tuple[int, ...],
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = 'cls',
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.
    ):
        super().__init__()
        
        assert 1 <= ndim <= 7, 'ndim must be between 1 and 7'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.ndim = ndim
        self.pool = pool
        
        input_shape = ensure_tuple(input_shape, ndim)
        patch_size = ensure_tuple(patch_size, ndim)
        
        for i, (inp_dim, patch_dim) in enumerate(zip(input_shape, patch_size)):
            assert inp_dim % patch_dim == 0, f'Input dimension {i} ({inp_dim}) must be divisible by patch size ({patch_dim})'
        
        num_patches_per_dim = [inp_dim // patch_dim for inp_dim, patch_dim in zip(input_shape, patch_size)]
        num_patches = 1
        for n in num_patches_per_dim:
            num_patches *= n
        
        patch_dim = channels
        for p in patch_size:
            patch_dim *= p
        
        dim_names = 'fghijkl'[:ndim]
        
        input_dims = [f'({d} p{i})' for i, d in enumerate(dim_names)]
        patch_dims = [f'p{i}' for i in range(ndim)]
        
        input_pattern = f'b c {join(input_dims)}'
        output_pattern = f'b ({join(dim_names)}) ({join(patch_dims)} c)'
        rearrange_str = f'{input_pattern} -> {output_pattern}'
        
        rearrange_kwargs = {f'p{i}': p for i, p in enumerate(patch_size)}
        
        self.to_patch_embedding = msnn.SequentialCell(
            [Rearrange(rearrange_str, **rearrange_kwargs), nn.Linear(patch_dim, dim), nn.LayerNorm(dim)])
        
        self.pos_embedding = ms.Parameter(mint.randn(size = (1, num_patches + 1, dim)))
        self.cls_token = ms.Parameter(mint.randn(size = (1, 1, dim)))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.to_latent = msnn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)
    
    def construct(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = mint.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = x[:, 1:].mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    
    model = ViTND(
        ndim = 4,
        input_shape = (8, 16, 32, 64),
        patch_size = (2, 4, 4, 8),
        num_classes = 1000,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        channels = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    
    occupancy_time = mint.randn(size = (2, 3, 8, 16, 32, 64))
    
    logits = model(occupancy_time)
