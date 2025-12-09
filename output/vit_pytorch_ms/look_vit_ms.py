from torch import nn

from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from mindspore.mint import nn, ops

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

# simple vit sinusoidal pos emb

def posemb_sincos_2d(t, temperature = 10000):
    h, w, d, device = *t.shape[1:], t.device
    y, x = ops.meshgrid(tensors = ops.arange(start = h), indexing = 'ij')  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
    assert (d % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = ops.arange(start = d // 4) / (d // 4 - 1)  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
    omega = temperature ** -omega

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pos = ops.cat(tensors = (x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)  # 'torch.cat':没有对应的mindspore参数 'out';

    return pos.float()

# bias-less layernorm with unit offset trick (discovered by Ohad Rubin)

class LayerNorm(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape = dim, elementwise_affine = False)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.gamma = mindspore.Parameter(ops.zeros(size = dim))  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';

    def forward(self, x):
        normed = self.ln(x)
        return normed * (self.gamma + 1)

# mlp

def MLP(dim, factor = 4, dropout = 0.):
    hidden_dim = int(dim * factor)
    return nn.SequentialCell(
        LayerNorm(dim),
        nn.Linear(in_features = dim, out_features = hidden_dim),
        nn.GELU(),
        nn.Dropout(p = dropout),
        nn.Linear(in_features = hidden_dim, out_features = dim),
        nn.Dropout(p = dropout)
    )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

# attention

class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        cross_attend = False,
        reuse_attention = False
    ):
        super().__init__()
        inner_dim = dim_head *  heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.reuse_attention = reuse_attention
        self.cross_attend = cross_attend

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.norm = LayerNorm(dim) if not reuse_attention else nn.Identity()
        self.norm_context = LayerNorm(dim) if cross_attend else nn.Identity()

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout)

        self.to_q = nn.Linear(in_features = dim, out_features = inner_dim, bias = False) if not reuse_attention else None  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
        self.to_k = nn.Linear(in_features = dim, out_features = inner_dim, bias = False) if not reuse_attention else None  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
        self.to_v = nn.Linear(in_features = dim, out_features = inner_dim, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.to_out = nn.SequentialCell(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(in_features = inner_dim, out_features = dim, bias = False),
            nn.Dropout(p = dropout)
        )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(
        self,
        x,
        context = None,
        return_qk_sim = False,
        qk_sim = None
    ):
        x = self.norm(x)

        assert not (exists(context) ^ self.cross_attend)

        if self.cross_attend:
            context = self.norm_context(context)
        else:
            context = x

        v = self.to_v(context)
        v = self.split_heads(v)

        if not self.reuse_attention:
            qk = (self.to_q(x), self.to_k(context))
            q, k = tuple(self.split_heads(t) for t in qk)

            q = q * self.scale
            qk_sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        else:
            assert exists(qk_sim), 'qk sim matrix must be passed in for reusing previous attention'

        attn = self.attend(qk_sim)
        attn = self.dropout(attn)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = self.to_out(out)

        if not return_qk_sim:
            return out

        return out, qk_sim

# LookViT

class LookViT(nn.Cell):
    def __init__(
        self,
        *,
        dim,
        image_size,
        num_classes,
        depth = 3,
        patch_size = 16,
        heads = 8,
        mlp_factor = 4,
        dim_head = 64,
        highres_patch_size = 12,
        highres_mlp_factor = 4,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        patch_conv_kernel_size = 7,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert divisible_by(image_size, highres_patch_size)
        assert divisible_by(image_size, patch_size)
        assert patch_size > highres_patch_size, 'patch size of the main vision transformer should be smaller than the highres patch sizes (that does the `lookup`)'
        assert not divisible_by(patch_conv_kernel_size, 2)

        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size

        kernel_size = patch_conv_kernel_size
        patch_dim = (highres_patch_size * highres_patch_size) * channels

        self.to_patches = nn.SequentialCell(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = highres_patch_size, p2 = highres_patch_size),
            nn.Conv2d(in_channels = patch_dim, out_channels = dim, kernel_size = kernel_size, padding = kernel_size // 2),
            Rearrange('b c h w -> b h w c'),
            LayerNorm(dim),
        )  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';

        # absolute positions

        num_patches = (image_size // highres_patch_size) ** 2
        self.pos_embedding = mindspore.Parameter(ops.randn(size = num_patches, generator = dim))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

        # lookvit blocks

        layers = nn.CellList([])

        for _ in range(depth):
            layers.append(nn.CellList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
                MLP(dim = dim, factor = mlp_factor, dropout = dropout),
                Attention(dim = dim, dim_head = cross_attn_dim_head, heads = cross_attn_heads, dropout = dropout, cross_attend = True),
                Attention(dim = dim, dim_head = cross_attn_dim_head, heads = cross_attn_heads, dropout = dropout, cross_attend = True, reuse_attention = True),
                LayerNorm(dim),
                MLP(dim = dim, factor = highres_mlp_factor, dropout = dropout)
            ]))

        self.layers = layers

        self.norm = LayerNorm(dim)
        self.highres_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(in_features = dim, out_features = num_classes, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, img):
        assert img.shape[-2:] == (self.image_size, self.image_size)

        # to patch tokens and positions

        highres_tokens = self.to_patches(img)
        size = highres_tokens.shape[-2]

        pos_emb = posemb_sincos_2d(highres_tokens)
        highres_tokens = highres_tokens + rearrange(pos_emb, '(h w) d -> h w d', h = size)

        tokens = nn.functional.interpolate(
            input = rearrange(highres_tokens, 'b h w d -> b d h w'), size = img.shape[-1] // self.patch_size, mode = 'bilinear')  # 'torch.nn.functional.interpolate':没有对应的mindspore参数 'antialias';

        tokens = rearrange(tokens, 'b c h w -> b (h w) c')
        highres_tokens = rearrange(highres_tokens, 'b h w c -> b (h w) c')

        # attention and feedforwards

        for attn, mlp, lookup_cross_attn, highres_attn, highres_norm, highres_mlp in self.layers:

            # main tokens cross attends (lookup) on the high res tokens

            lookup_out, qk_sim = lookup_cross_attn(tokens, highres_tokens, return_qk_sim = True)  # return attention as they reuse the attention matrix
            tokens = lookup_out + tokens

            tokens = attn(tokens) + tokens
            tokens = mlp(tokens) + tokens

            # attention-reuse

            qk_sim = rearrange(qk_sim, 'b h i j -> b h j i') # transpose for reverse cross attention

            highres_tokens = highres_attn(highres_tokens, tokens, qk_sim = qk_sim) + highres_tokens
            highres_tokens = highres_norm(highres_tokens)

            highres_tokens = highres_mlp(highres_tokens) + highres_tokens

        # to logits

        tokens = self.norm(tokens)
        highres_tokens = self.highres_norm(highres_tokens)

        tokens = reduce(tokens, 'b n d -> b d', 'mean')
        highres_tokens = reduce(highres_tokens, 'b n d -> b d', 'mean')

        return self.to_logits(tokens + highres_tokens)

# main

if __name__ == '__main__':
    v = LookViT(
        image_size = 256,
        num_classes = 1000,
        dim = 512,
        depth = 2,
        heads = 8,
        dim_head = 64,
        patch_size = 32,
        highres_patch_size = 8,
        highres_mlp_factor = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        dropout = 0.1
    ).cuda()

    img = ops.randn(size = 2, generator = 3, dtype = 256).cuda()  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
    pred = v(img)

    assert pred.shape == (2, 1000)
