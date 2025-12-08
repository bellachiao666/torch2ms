from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from mindspore.mint import nn, ops

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class Residual(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class FeedForward(nn.Cell):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = dim),
            nn.Linear(in_features = dim, out_features = inner_dim),
            nn.GELU(),
            nn.Dropout(p = dropout),
            nn.Linear(in_features = inner_dim, out_features = dim),
            nn.Dropout(p = dropout)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';
    def forward(self, x):
        return self.net(x)

# MBConv

class SqueezeExcitation(nn.Cell):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(in_features = dim, out_features = hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(in_features = hidden_dim, out_features = dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Cell):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Cell):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(in_channels = dim_in, out_channels = hidden_dim, kernel_size = 1),
        nn.BatchNorm2d(num_features = hidden_dim),
        nn.GELU(),
        nn.Conv2d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(num_features = hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(in_channels = hidden_dim, out_channels = dim_out, kernel_size = 1),
        nn.BatchNorm2d(num_features = dim_out)
    )  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';; 'torch.nn.BatchNorm2d':没有对应的mindspore参数 'device';

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.to_qkv = nn.Linear(in_features = dim, out_features = dim * 3, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(p = dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(in_features = dim, out_features = dim, bias = False),
            nn.Dropout(p = dropout)
        )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        # relative positional bias

        self.rel_pos_bias = nn.Embedding(num_embeddings = (2 * window_size - 1) ** 2, embedding_dim = self.heads)  # 'torch.nn.Embedding':没有对应的mindspore参数 'device';

        pos = ops.arange(start = window_size)  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
        grid = ops.stack(tensors = ops.meshgrid(tensors = pos, indexing = 'ij'))  # 'torch.stack':没有对应的mindspore参数 'out';
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        x = self.norm(x)

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = ops.einsum(equation = 'b h i d, b h j d -> b h i j', operands = q)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = ops.einsum(equation = 'b h i j, b h j d -> b h i d', operands = attn)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class MaxViT(nn.Cell):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels = channels, out_channels = dim_conv_stem, kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(in_channels = dim_conv_stem, out_channels = dim_conv_stem, kernel_size = 3, padding = 1)
        )  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
                    Residual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
                    Residual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )

                self.layers.append(block)

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(normalized_shape = dims[-1]),
            nn.Linear(in_features = dims[-1], out_features = num_classes)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, x):
        x = self.conv_stem(x)

        for stage in self.layers:
            x = stage(x)

        return self.mlp_head(x)
