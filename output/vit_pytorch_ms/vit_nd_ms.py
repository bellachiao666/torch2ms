from __future__ import annotations
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from mindspore.mint import nn, ops

# helpers

def join(arr, delimiter = ' '):
    return delimiter.join(arr)

def ensure_tuple(t, length):
    if isinstance(t, (tuple, list)):
        assert len(t) == length, f'Expected tuple of length {length}, got {len(t)}'
        return tuple(t)
    return (t,) * length

# classes

class FeedForward(nn.Cell):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.LayerNorm(normalized_shape = dim),
            nn.Linear(in_features = dim, out_features = hidden_dim),
            nn.GELU(),
            nn.Dropout(p = dropout),
            nn.Linear(in_features = hidden_dim, out_features = dim),
            nn.Dropout(p = dropout)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout)
        
        self.to_qkv = nn.Linear(in_features = dim, out_features = inner_dim * 3, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
        
        self.to_out = nn.SequentialCell(
            nn.Linear(in_features = inner_dim, out_features = dim),
            nn.Dropout(p = dropout)
        ) if project_out else nn.Identity()  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
    
    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        dots = ops.matmul(input = q, other = k.transpose(-1, -2)) * self.scale  # 'torch.matmul':没有对应的mindspore参数 'out';
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = ops.matmul(input = attn, other = v)  # 'torch.matmul':没有对应的mindspore参数 'out';
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(nn.CellList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViTND(nn.Cell):
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
        
        self.to_patch_embedding = nn.SequentialCell(
            Rearrange(rearrange_str, **rearrange_kwargs),
            nn.Linear(in_features = patch_dim, out_features = dim),
            nn.LayerNorm(normalized_shape = dim),
        )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';; 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        
        self.pos_embedding = mindspore.Parameter(ops.randn(size = 1, generator = num_patches + 1))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.cls_token = mindspore.Parameter(ops.randn(size = 1, generator = 1))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.dropout = nn.Dropout(p = emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(in_features = dim, out_features = num_classes)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
    
    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = ops.cat(tensors = (cls_tokens, x), dim = 1)  # 'torch.cat':没有对应的mindspore参数 'out';
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
    
    occupancy_time = ops.randn(size = 2, generator = 3, dtype = 16)  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
    
    logits = model(occupancy_time)
