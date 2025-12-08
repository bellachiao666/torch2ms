# https://arxiv.org/abs/2510.14657
# but instead of their decorr module updated with SGD, remove all projections and just return a decorrelation auxiliary loss

import torch
from torch import nn, stack, tensor
from torch.nn import ModuleList

from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange
from mindspore.mint import nn, ops

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# decorr loss

class DecorrelationLoss(nn.Cell):
    def __init__(
        self,
        sample_frac = 1.,
        soft_validate_num_sampled = False
    ):
        super().__init__()
        assert 0. <= sample_frac <= 1.
        self.need_sample = sample_frac < 1.
        self.sample_frac = sample_frac

        self.soft_validate_num_sampled = soft_validate_num_sampled
        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        tokens
    ):
        batch, seq_len, dim, device = *tokens.shape[-3:], tokens.device

        if self.need_sample:
            num_sampled = int(seq_len * self.sample_frac)

            assert self.soft_validate_num_sampled or num_sampled >= 2.

            if num_sampled <= 1:
                return self.zero

            tokens, packed_shape = pack([tokens], '* n d e')

            indices = ops.randn(size = tokens.shape[:2]).argsort(dim = -1)[..., :num_sampled, :]  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

            batch_arange = ops.arange(start = tokens.shape[0])  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
            batch_arange = rearrange(batch_arange, 'b -> b 1')

            tokens = tokens[batch_arange, indices]
            tokens, = unpack(tokens, packed_shape, '* n d e')

        dist = einsum(tokens, tokens, '... n d, ... n e -> ... d e') / tokens.shape[-2]
        eye = ops.eye(n = dim)  # 'torch.eye':没有对应的mindspore参数 'out';; 'torch.eye':没有对应的mindspore参数 'layout';; 'torch.eye':没有对应的mindspore参数 'device';; 'torch.eye':没有对应的mindspore参数 'requires_grad';

        loss = dist.pow(2) * (1. - eye) / ((dim - 1) * dim)

        loss = reduce(loss, '... b d e -> b', 'sum')
        return loss.mean()

# classes

class FeedForward(nn.Cell):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';

        self.net = nn.Sequential(
            nn.Linear(in_features = dim, out_features = hidden_dim),
            nn.GELU(),
            nn.Dropout(p = dropout),
            nn.Linear(in_features = hidden_dim, out_features = dim),
            nn.Dropout(p = dropout)
        )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, x):
        normed = self.norm(x)
        return self.net(x), normed

class Attention(nn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout)

        self.to_qkv = nn.Linear(in_features = dim, out_features = inner_dim * 3, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.to_out = nn.Sequential(
            nn.Linear(in_features = inner_dim, out_features = dim),
            nn.Dropout(p = dropout)
        ) if project_out else nn.Identity()  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, x):
        normed = self.norm(x)

        qkv = self.to_qkv(normed).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = ops.matmul(input = q, other = k.transpose(-1, -2)) * self.scale  # 'torch.matmul':没有对应的mindspore参数 'out';

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = ops.matmul(input = attn, other = v)  # 'torch.matmul':没有对应的mindspore参数 'out';
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), normed

class Transformer(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):

        normed_inputs = []

        for attn, ff in self.layers:
            attn_out, attn_normed_inp = attn(x)
            x = attn_out + x

            ff_out, ff_normed_inp = ff(x)
            x = ff_out + x

            normed_inputs.append(attn_normed_inp)
            normed_inputs.append(ff_normed_inp)

        return self.norm(x), ops.stack(tensors = normed_inputs)  # 'torch.stack':没有对应的mindspore参数 'out';

class ViT(nn.Cell):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., decorr_sample_frac = 1.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(normalized_shape = patch_dim),
            nn.Linear(in_features = patch_dim, out_features = dim),
            nn.LayerNorm(normalized_shape = dim),
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.pos_embedding = nn.Parameter(ops.randn(size = 1, generator = num_patches + 1))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.cls_token = nn.Parameter(ops.randn(size = 1, generator = 1))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.dropout = nn.Dropout(p = emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(in_features = dim, out_features = num_classes)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        # decorrelation loss related

        self.has_decorr_loss = decorr_sample_frac > 0.

        if self.has_decorr_loss:
            self.decorr_loss = DecorrelationLoss(decorr_sample_frac)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    def forward(
        self,
        img,
        return_decorr_aux_loss = None
    ):
        return_decorr_aux_loss = default(return_decorr_aux_loss, self.training) and self.has_decorr_loss

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = ops.cat(tensors = (cls_tokens, x), dim = 1)  # 'torch.cat':没有对应的mindspore参数 'out';
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, normed_layer_inputs = self.transformer(x)

        # maybe return decor loss

        decorr_aux_loss = self.zero

        if return_decorr_aux_loss:
            decorr_aux_loss = self.decorr_loss(normed_layer_inputs)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x), decorr_aux_loss

# quick test

if __name__ == '__main__':
    decorr_loss = DecorrelationLoss(0.1)

    hiddens = ops.randn(size = 6, generator = 2, dtype = 256)  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

    decorr_loss(hiddens)
    decorr_loss(hiddens[0])

    decorr_loss = DecorrelationLoss(0.0001, soft_validate_num_sampled = True)
    out = decorr_loss(hiddens)
    assert out.item() == 0