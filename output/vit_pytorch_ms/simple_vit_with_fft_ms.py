import torch
from torch.fft import fft2
from torch import nn

from einops import rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange
from mindspore.mint import nn, ops

# helpers

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

class FeedForward(nn.Cell):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.LayerNorm(normalized_shape = dim),
            nn.Linear(in_features = dim, out_features = hidden_dim),
            nn.GELU(),
            nn.Linear(in_features = hidden_dim, out_features = dim),
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';
    def forward(self, x):
        return self.net(x)

class Attention(nn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(in_features = dim, out_features = inner_dim * 3, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
        self.to_out = nn.Linear(in_features = inner_dim, out_features = dim, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = ops.matmul(input = q, other = k.transpose(-1, -2)) * self.scale  # 'torch.matmul':没有对应的mindspore参数 'out';

        attn = self.attend(dots)

        out = ops.matmul(input = attn, other = v)  # 'torch.matmul':没有对应的mindspore参数 'out';
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(nn.CellList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Cell):
    def __init__(self, *, image_size, patch_size, freq_patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        freq_patch_height, freq_patch_width = pair(freq_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_height % freq_patch_height == 0 and image_width % freq_patch_width == 0, 'Image dimensions must be divisible by the freq patch size.'

        patch_dim = channels * patch_height * patch_width
        freq_patch_dim = channels * 2 * freq_patch_height * freq_patch_width

        self.to_patch_embedding = nn.SequentialCell(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(normalized_shape = patch_dim),
            nn.Linear(in_features = patch_dim, out_features = dim),
            nn.LayerNorm(normalized_shape = dim),
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.to_freq_embedding = nn.SequentialCell(
            Rearrange("b c (h p1) (w p2) ri -> b (h w) (p1 p2 ri c)", p1 = freq_patch_height, p2 = freq_patch_width),
            nn.LayerNorm(normalized_shape = freq_patch_dim),
            nn.Linear(in_features = freq_patch_dim, out_features = dim),
            nn.LayerNorm(normalized_shape = dim)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        self.freq_pos_embedding = posemb_sincos_2d(
            h = image_height // freq_patch_height,
            w = image_width // freq_patch_width,
            dim = dim
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(in_features = dim, out_features = num_classes)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, img):
        device, dtype = img.device, img.dtype

        x = self.to_patch_embedding(img)
        freqs = torch.view_as_real(fft2(img))

        f = self.to_freq_embedding(freqs)

        x += self.pos_embedding.to(device, dtype = dtype)
        f += self.freq_pos_embedding.to(device, dtype = dtype)

        x, ps = pack((f, x), 'b * d')

        x = self.transformer(x)

        _, x = unpack(x, ps, 'b * d')
        x = reduce(x, 'b n d -> b d', 'mean')

        x = self.to_latent(x)
        return self.linear_head(x)

if __name__ == '__main__':
    vit = SimpleViT(
        num_classes = 1000,
        image_size = 256,
        patch_size = 8,
        freq_patch_size = 8,
        dim = 1024,
        depth = 1,
        heads = 8,
        mlp_dim = 2048,
    )

    images = ops.randn(size = 8, generator = 3, dtype = 256)  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

    logits = vit(images)
