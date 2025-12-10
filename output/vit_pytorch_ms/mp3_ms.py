import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# from torch import nn, einsum
# import torch.nn.functional as F

from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# positional embedding

def posemb_sincos_2d(patches, temperature = 10000, dtype = ms.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = mint.meshgrid(mint.arange(h), mint.arange(w), indexing = 'ij')  # 'torch.arange':没有对应的mindspore参数 'device' (position 6);
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = mint.arange(dim // 4) / (dim // 4 - 1)  # 'torch.arange':没有对应的mindspore参数 'device' (position 6);
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = mint.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# feedforward

class FeedForward(msnn.Cell):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = msnn.SequentialCell(
            [nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)])
    def construct(self, x):
        return self.net(x)

# (cross)attention

class Attention(msnn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = msnn.SequentialCell(
            [nn.Linear(inner_dim, dim), nn.Dropout(dropout)])

    def construct(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)

        context = self.norm(context) if exists(context) else x

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = mint.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = mint.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(msnn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = msnn.CellList([])
        for _ in range(depth):
            self.layers.append(msnn.CellList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def construct(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class ViT(msnn.Cell):
    def __init__(self, *, num_classes, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.dim = dim
        self.num_patches = num_patches

        self.to_patch_embedding = msnn.SequentialCell(
            [Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim)])

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = msnn.Identity()
        self.linear_head = msnn.SequentialCell(
            [nn.LayerNorm(dim), nn.Linear(dim, num_classes)])

    def construct(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

# Masked Position Prediction Pre-Training

class MP3(msnn.Cell):
    def __init__(self, vit: ViT, masking_ratio):
        super().__init__()
        self.vit = vit

        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        dim = vit.dim
        self.mlp_head = msnn.SequentialCell(
            [nn.LayerNorm(dim), nn.Linear(dim, vit.num_patches)])

    def construct(self, img):
        device = img.device
        tokens = self.vit.to_patch_embedding(img)
        tokens = rearrange(tokens, 'b ... d -> b (...) d')

        batch, num_patches, *_ = tokens.shape

        # Masking
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = mint.rand(size = (batch, num_patches)).argsort(dim = -1)  # 'torch.rand':没有对应的mindspore参数 'device' (position 5);
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        batch_range = mint.arange(batch)[:, None]  # 'torch.arange':没有对应的mindspore参数 'device' (position 6);
        tokens_unmasked = tokens[batch_range, unmasked_indices]

        attended_tokens = self.vit.transformer(tokens, tokens_unmasked)
        logits = rearrange(self.mlp_head(attended_tokens), 'b n d -> (b n) d')
        
        # Define labels
        labels = repeat(mint.arange(num_patches), 'n -> (b n)', b = batch)  # 'torch.arange':没有对应的mindspore参数 'device' (position 6);
        loss = F.cross_entropy(logits, labels)

        return loss
