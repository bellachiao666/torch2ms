"""
ViT + Hyper-Connections + Register Tokens
https://arxiv.org/abs/2409.19606
"""

import torch
from torch import nn, tensor
from torch.nn import ModuleList

from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange
from mindspore.mint import nn, ops

# b - batch, h - heads, n - sequence, e - expansion rate / residual streams, d - feature dimension

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

# hyper connections

class HyperConnection(nn.Cell):
    def __init__(
        self,
        dim,
        num_residual_streams,
        layer_index
    ):
        """ Appendix J - Algorithm 2, Dynamic only """
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape = dim, bias = False)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';

        self.num_residual_streams = num_residual_streams
        self.layer_index = layer_index

        self.static_beta = nn.Parameter(ops.ones(size = num_residual_streams))  # 'torch.ones':没有对应的mindspore参数 'out';; 'torch.ones':没有对应的mindspore参数 'layout';; 'torch.ones':没有对应的mindspore参数 'device';; 'torch.ones':没有对应的mindspore参数 'requires_grad';

        init_alpha0 = ops.zeros(size = (num_residual_streams, 1))  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';
        init_alpha0[layer_index % num_residual_streams, 0] = 1.

        self.static_alpha = nn.Parameter(ops.cat(tensors = [init_alpha0, ops.eye(n = num_residual_streams)], dim = 1))  # 'torch.eye':没有对应的mindspore参数 'out';; 'torch.eye':没有对应的mindspore参数 'layout';; 'torch.eye':没有对应的mindspore参数 'device';; 'torch.eye':没有对应的mindspore参数 'requires_grad';; 'torch.cat':没有对应的mindspore参数 'out';

        self.dynamic_alpha_fn = nn.Parameter(ops.zeros(size = dim))  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';
        self.dynamic_alpha_scale = nn.Parameter(tensor(1e-2))
        self.dynamic_beta_fn = nn.Parameter(ops.zeros(size = dim))  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';
        self.dynamic_beta_scale = nn.Parameter(tensor(1e-2))

    def width_connection(self, residuals):
        normed = self.norm(residuals)

        wc_weight = (normed @ self.dynamic_alpha_fn).tanh()
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha

        dc_weight = (normed @ self.dynamic_beta_fn).tanh()
        dynamic_beta = dc_weight * self.dynamic_beta_scale
        beta = dynamic_beta + self.static_beta

        # width connection
        mix_h = einsum(alpha, residuals, '... e1 e2, ... e1 d -> ... e2 d')

        branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]

        return branch_input, residuals, beta

    def depth_connection(
        self,
        branch_output,
        residuals,
        beta
    ):
        return einsum(branch_output, beta, "b n d, b n e -> b n e d") + residuals

# classes

class FeedForward(nn.Cell):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_residual_streams):
        super().__init__()

        self.num_residual_streams = num_residual_streams

        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.layers = ModuleList([])

        for layer_index in range(depth):
            self.layers.append(nn.ModuleList([
                HyperConnection(dim, num_residual_streams, layer_index),
                Attention(dim, heads = heads, dim_head = dim_head),
                HyperConnection(dim, num_residual_streams, layer_index),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):

        x = repeat(x, 'b n d -> b n e d', e = self.num_residual_streams)

        for attn_hyper_conn, attn, ff_hyper_conn, ff in self.layers:

            x, attn_res, beta = attn_hyper_conn.width_connection(x)

            x = attn(x)

            x = attn_hyper_conn.depth_connection(x, attn_res, beta)

            x, ff_res, beta = ff_hyper_conn.width_connection(x)

            x = ff(x)

            x = ff_hyper_conn.depth_connection(x, ff_res, beta)

        x = reduce(x, 'b n e d -> b n d', 'sum')

        return self.norm(x)

class SimpleViT(nn.Cell):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, num_residual_streams, num_register_tokens = 4, channels = 3, dim_head = 64):
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

        self.register_tokens = nn.Parameter(ops.randn(size = num_register_tokens, generator = dim))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, num_residual_streams)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(in_features = dim, out_features = num_classes)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, img):
        batch, device = img.shape[0], img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(x)

        r = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        x, ps = pack([x, r], 'b * d')

        x = self.transformer(x)

        x, _ = unpack(x, ps, 'b * d')

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

# main

if __name__ == '__main__':
    vit = SimpleViT(
        num_classes = 1000,
        image_size = 256,
        patch_size = 8,
        dim = 1024,
        depth = 12,
        heads = 8,
        mlp_dim = 2048,
        num_residual_streams = 8
    )

    images = ops.randn(size = 3, generator = 3, dtype = 256)  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

    logits = vit(images)
