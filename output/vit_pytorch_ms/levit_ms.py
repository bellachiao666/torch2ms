from math import ceil
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from mindspore.mint import nn, ops

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, l = 3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))

def always(val):
    return lambda *args, **kwargs: val

# classes

class FeedForward(nn.Cell):
    def __init__(self, dim, mult, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = dim, out_channels = dim * mult, kernel_size = 1),
            nn.Hardswish(),
            nn.Dropout(p = dropout),
            nn.Conv2d(in_channels = dim * mult, out_channels = dim, kernel_size = 1),
            nn.Dropout(p = dropout)
        )  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';; 'torch.nn.Hardswish':没有对应的mindspore参数 'inplace';
    def forward(self, x):
        return self.net(x)

class Attention(nn.Cell):
    def __init__(self, dim, fmap_size, heads = 8, dim_key = 32, dim_value = 64, dropout = 0., dim_out = None, downsample = False):
        super().__init__()
        inner_dim_key = dim_key *  heads
        inner_dim_value = dim_value *  heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key ** -0.5

        self.to_q = nn.Sequential(nn.Conv2d(in_channels = dim, out_channels = inner_dim_key, kernel_size = 1, stride = (2 if downsample else 1), bias = False), nn.BatchNorm2d(num_features = inner_dim_key))  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';; 'torch.nn.BatchNorm2d':没有对应的mindspore参数 'device';
        self.to_k = nn.Sequential(nn.Conv2d(in_channels = dim, out_channels = inner_dim_key, kernel_size = 1, bias = False), nn.BatchNorm2d(num_features = inner_dim_key))  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';; 'torch.nn.BatchNorm2d':没有对应的mindspore参数 'device';
        self.to_v = nn.Sequential(nn.Conv2d(in_channels = dim, out_channels = inner_dim_value, kernel_size = 1, bias = False), nn.BatchNorm2d(num_features = inner_dim_value))  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';; 'torch.nn.BatchNorm2d':没有对应的mindspore参数 'device';

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout)

        out_batch_norm = nn.BatchNorm2d(num_features = dim_out)  # 'torch.nn.BatchNorm2d':没有对应的mindspore参数 'device';
        nn.init.zeros_(out_batch_norm.weight)

        self.to_out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_channels = inner_dim_value, out_channels = dim_out, kernel_size = 1),
            out_batch_norm,
            nn.Dropout(p = dropout)
        )  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';

        # positional bias

        self.pos_bias = nn.Embedding(num_embeddings = fmap_size * fmap_size, embedding_dim = heads)  # 'torch.nn.Embedding':没有对应的mindspore参数 'device';

        q_range = ops.arange(start = 0, end = fmap_size, step = (2 if downsample else 1))  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
        k_range = ops.arange(start = fmap_size)  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';

        q_pos = ops.stack(tensors = ops.meshgrid(tensors = q_range, indexing = 'ij'), dim = -1)  # 'torch.stack':没有对应的mindspore参数 'out';
        k_pos = ops.stack(tensors = ops.meshgrid(tensors = k_range, indexing = 'ij'), dim = -1)  # 'torch.stack':没有对应的mindspore参数 'out';

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim = -1)
        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer('pos_indices', pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)

    def forward(self, x):
        b, n, *_, h = *x.shape, self.heads

        q = self.to_q(x)
        y = q.shape[2]

        qkv = (q, self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = h), qkv)

        dots = ops.einsum(equation = 'b h i d, b h j d -> b h i j', operands = q) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = ops.einsum(equation = 'b h i j, b h j d -> b h i d', operands = attn)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Cell):
    def __init__(self, dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult = 2, dropout = 0., dim_out = None, downsample = False):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.layers = nn.ModuleList([])
        self.attn_residual = (not downsample) and dim == dim_out

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, fmap_size = fmap_size, heads = heads, dim_key = dim_key, dim_value = dim_value, dropout = dropout, downsample = downsample, dim_out = dim_out),
                FeedForward(dim_out, mlp_mult, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            attn_res = (x if self.attn_residual else 0)
            x = attn(x) + attn_res
            x = ff(x) + x
        return x

class LeViT(nn.Cell):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_mult,
        stages = 3,
        dim_key = 32,
        dim_value = 64,
        dropout = 0.,
        num_distill_classes = None
    ):
        super().__init__()

        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)

        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), 'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'

        self.conv_embedding = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(in_channels = 128, out_channels = dims[0], kernel_size = 3, stride = 2, padding = 1)
        )  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';

        fmap_size = image_size // (2 ** 4)
        layers = []

        for ind, dim, depth, heads in zip(range(stages), dims, depths, layer_heads):
            is_last = ind == (stages - 1)
            layers.append(Transformer(dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult, dropout))

            if not is_last:
                next_dim = dims[ind + 1]
                layers.append(Transformer(dim, fmap_size, 1, heads * 2, dim_key, dim_value, dim_out = next_dim, downsample = True))
                fmap_size = ceil(fmap_size / 2)

        self.backbone = nn.Sequential(*layers)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size = 1),
            Rearrange('... () () -> ...')
        )

        self.distill_head = nn.Linear(in_features = dim, out_features = num_distill_classes) if exists(num_distill_classes) else always(None)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
        self.mlp_head = nn.Linear(in_features = dim, out_features = num_classes)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, img):
        x = self.conv_embedding(img)

        x = self.backbone(x)        

        x = self.pool(x)

        out = self.mlp_head(x)
        distill = self.distill_head(x)

        if exists(distill):
            return out, distill

        return out
