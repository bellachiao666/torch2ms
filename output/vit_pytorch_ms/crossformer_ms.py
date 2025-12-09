import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from mindspore.mint import nn, ops

# helpers

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# cross embed layer

class CrossEmbedLayer(nn.Cell):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_sizes,
        stride = 2
    ):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.CellList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(in_channels = dim_in, out_channels = dim_scale, kernel_size = kernel, stride = stride, padding = (kernel - stride) // 2))  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return ops.cat(tensors = fmaps, dim = 1)  # 'torch.cat':没有对应的mindspore参数 'out';

# dynamic positional bias

def DynamicPositionBias(dim):
    return nn.SequentialCell(
        nn.Linear(in_features = 2, out_features = dim),
        nn.LayerNorm(normalized_shape = dim),
        nn.ReLU(),
        nn.Linear(in_features = dim, out_features = dim),
        nn.LayerNorm(normalized_shape = dim),
        nn.ReLU(),
        nn.Linear(in_features = dim, out_features = dim),
        nn.LayerNorm(normalized_shape = dim),
        nn.ReLU(),
        nn.Linear(in_features = dim, out_features = 1),
        Rearrange('... () -> ...')
    )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';; 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.ReLU':没有对应的mindspore参数 'inplace';

# transformer classes

class LayerNorm(nn.Cell):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = mindspore.Parameter(ops.ones(size = 1, dtype = 1))  # 'torch.ones':没有对应的mindspore参数 'out';; 'torch.ones':没有对应的mindspore参数 'layout';; 'torch.ones':没有对应的mindspore参数 'device';; 'torch.ones':没有对应的mindspore参数 'requires_grad';
        self.b = mindspore.Parameter(ops.zeros(size = 1, dtype = 1))  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';

    def forward(self, x):
        var = ops.var(input = x, dim = 1, keepdim = True)  # 'torch.var':没有对应的mindspore参数 'out';
        mean = ops.mean(input = x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.SequentialCell(
        LayerNorm(dim),
        nn.Conv2d(in_channels = dim, out_channels = dim * mult, kernel_size = 1),
        nn.GELU(),
        nn.Dropout(p = dropout),
        nn.Conv2d(in_channels = dim * mult, out_channels = dim, kernel_size = 1)
    )  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';

class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        attn_type,
        window_size,
        dim_head = 32,
        dropout = 0.
    ):
        super().__init__()
        assert attn_type in {'short', 'long'}, 'attention type must be one of local or distant'
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.attn_type = attn_type
        self.window_size = window_size

        self.norm = LayerNorm(dim)

        self.dropout = nn.Dropout(p = dropout)

        self.to_qkv = nn.Conv2d(in_channels = dim, out_channels = inner_dim * 3, kernel_size = 1, bias = False)  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';
        self.to_out = nn.Conv2d(in_channels = inner_dim, out_channels = dim, kernel_size = 1)  # 'torch.nn.Conv2d':没有对应的mindspore参数 'device';

        # positions

        self.dpb = DynamicPositionBias(dim // 4)

        # calculate and store indices for retrieving bias

        pos = ops.arange(start = window_size)  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
        grid = ops.stack(tensors = ops.meshgrid(tensors = pos, indexing = 'ij'))  # 'torch.stack':没有对应的mindspore参数 'out';
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = grid[:, None] - grid[None, :]
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        *_, height, width, heads, wsz, device = *x.shape, self.heads, self.window_size, x.device

        # prenorm

        x = self.norm(x)

        # rearrange for short or long distance attention

        if self.attn_type == 'short':
            x = rearrange(x, 'b d (h s1) (w s2) -> (b h w) d s1 s2', s1 = wsz, s2 = wsz)
        elif self.attn_type == 'long':
            x = rearrange(x, 'b d (l1 h) (l2 w) -> (b h w) d l1 l2', l1 = wsz, l2 = wsz)

        # queries / keys / values

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))
        q = q * self.scale

        sim = ops.einsum(equation = 'b h i d, b h j d -> b h i j', operands = q)

        # add dynamic positional bias

        pos = ops.arange(start = -wsz, end = wsz + 1)  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
        rel_pos = ops.stack(tensors = ops.meshgrid(tensors = pos, indexing = 'ij'))  # 'torch.stack':没有对应的mindspore参数 'out';
        rel_pos = rearrange(rel_pos, 'c i j -> (i j) c')
        biases = self.dpb(rel_pos.float())
        rel_pos_bias = biases[self.rel_pos_indices]

        sim = sim + rel_pos_bias

        # attend

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # merge heads

        out = ops.einsum(equation = 'b h i j, b h j d -> b h i d', operands = attn)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = wsz, y = wsz)
        out = self.to_out(out)

        # rearrange back for long or short distance attention

        if self.attn_type == 'short':
            out = rearrange(out, '(b h w) d s1 s2 -> b d (h s1) (w s2)', h = height // wsz, w = width // wsz)
        elif self.attn_type == 'long':
            out = rearrange(out, '(b h w) d l1 l2 -> b d (l1 h) (l2 w)', h = height // wsz, w = width // wsz)

        return out

class Transformer(nn.Cell):
    def __init__(
        self,
        dim,
        *,
        local_window_size,
        global_window_size,
        depth = 4,
        dim_head = 32,
        attn_dropout = 0.,
        ff_dropout = 0.,
    ):
        super().__init__()
        self.layers = nn.CellList([])

        for _ in range(depth):
            self.layers.append(nn.CellList([
                Attention(dim, attn_type = 'short', window_size = local_window_size, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
                Attention(dim, attn_type = 'long', window_size = global_window_size, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout)
            ]))

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x

        return x

# classes

class CrossFormer(nn.Cell):
    def __init__(
        self,
        *,
        dim = (64, 128, 256, 512),
        depth = (2, 2, 8, 2),
        global_window_size = (8, 4, 2, 1),
        local_window_size = 7,
        cross_embed_kernel_sizes = ((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides = (4, 2, 2, 2),
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        channels = 3
    ):
        super().__init__()

        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)

        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size) == 4
        assert len(cross_embed_kernel_sizes) == 4
        assert len(cross_embed_strides) == 4

        # dimensions

        last_dim = dim[-1]
        dims = [channels, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

        # layers

        self.layers = nn.CellList([])

        for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides):
            self.layers.append(nn.CellList([
                CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride = cel_stride),
                Transformer(dim_out, local_window_size = local_wsz, global_window_size = global_wsz, depth = layers, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
            ]))

        # final logits

        self.to_logits = nn.SequentialCell(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(in_features = last_dim, out_features = num_classes)
        )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, x):
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)

        return self.to_logits(x)
