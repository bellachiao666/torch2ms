import torch
from torch import nn
from torch.nn import ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange
from mindspore.mint import nn, ops

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = ops.meshgrid(tensors = ops.arange(start = h), indexing = "ij")  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = ops.arange(start = dim // 4) / (dim // 4 - 1)  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = ops.cat(tensors = (x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)  # 'torch.cat':没有对应的mindspore参数 'out';
    return pe.type(dtype)

# classes

def FeedForward(dim, hidden_dim):
    return nn.Sequential(
        nn.LayerNorm(normalized_shape = dim),
        nn.Linear(in_features = dim, out_features = hidden_dim),
        nn.GELU(),
        nn.Linear(in_features = hidden_dim, out_features = dim),
    )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

class Attention(nn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, learned_value_residual_mix = False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(in_features = dim, out_features = inner_dim * 3, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
        self.to_out = nn.Linear(in_features = inner_dim, out_features = dim, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.to_residual_mix = nn.Sequential(
            nn.Linear(in_features = dim, out_features = heads),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
        ) if learned_value_residual_mix else (lambda _: 0.5)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, x, value_residual = None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if exists(value_residual):
            mix = self.to_residual_mix(x)
            v = v * mix + value_residual * (1. - mix)

        dots = ops.matmul(input = q, other = k.transpose(-1, -2)) * self.scale  # 'torch.matmul':没有对应的mindspore参数 'out';

        attn = self.attend(dots)

        out = ops.matmul(input = attn, other = v)  # 'torch.matmul':没有对应的mindspore参数 'out';
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), v

class Transformer(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.layers = ModuleList([])
        for i in range(depth):
            is_first = i == 0
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, learned_value_residual_mix = not is_first),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        value_residual = None

        for attn, ff in self.layers:

            attn_out, values = attn(x, value_residual = value_residual)
            value_residual = default(value_residual, values)

            x = attn_out + x
            x = ff(x) + x

        return self.norm(x)

class SimpleViT(nn.Cell):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(normalized_shape = patch_dim),
            nn.Linear(in_features = patch_dim, out_features = dim),
            nn.LayerNorm(normalized_shape = dim),
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(in_features = dim, out_features = num_classes)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

# quick test

if __name__ == '__main__':
    v = SimpleViT(
        num_classes = 1000,
        image_size = 256,
        patch_size = 8,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
    )

    images = ops.randn(size = 2, generator = 3, dtype = 256)  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

    logits = v(images)
