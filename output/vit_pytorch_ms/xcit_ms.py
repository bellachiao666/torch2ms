from random import randrange
from torch import nn, einsum
from torch.nn import ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from mindspore.mint import nn, ops

# helpers

def exists(val):
    return val is not None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return nn.functional.normalize(input = t, p = 2, dim = -1)  # 'torch.nn.functional.normalize':没有对应的mindspore参数 'out';

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = ops.zeros(size = num_layers).uniform_(0., 1.) < dropout  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale(nn.Cell):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif 18 > depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        self.fn = fn
        self.scale = nn.Parameter(ops.full(size = (dim,), fill_value = init_eps))  # 'torch.full':没有对应的mindspore参数 'out';; 'torch.full':没有对应的mindspore参数 'layout';; 'torch.full':没有对应的mindspore参数 'device';; 'torch.full':没有对应的mindspore参数 'requires_grad';

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class FeedForward(nn.Cell):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
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
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.to_q = nn.Linear(in_features = dim, out_features = inner_dim, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
        self.to_kv = nn.Linear(in_features = dim, out_features = inner_dim * 2, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout)

        self.to_out = nn.Sequential(
            nn.Linear(in_features = inner_dim, out_features = dim),
            nn.Dropout(p = dropout)
        )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, x, context = None):
        h = self.heads

        x = self.norm(x)
        context = x if not exists(context) else ops.cat(tensors = (x, context), dim = 1)  # 'torch.cat':没有对应的mindspore参数 'out';

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        sim = ops.einsum(equation = 'b h i d, b h j d -> b h i j', operands = q) * self.scale

        attn = self.attend(sim)
        attn = self.dropout(attn)

        out = ops.einsum(equation = 'b h i j, b h j d -> b h i d', operands = attn)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class XCAttention(nn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';

        self.to_qkv = nn.Linear(in_features = dim, out_features = inner_dim * 3, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.temperature = nn.Parameter(ops.ones(size = heads, dtype = 1))  # 'torch.ones':没有对应的mindspore参数 'out';; 'torch.ones':没有对应的mindspore参数 'layout';; 'torch.ones':没有对应的mindspore参数 'device';; 'torch.ones':没有对应的mindspore参数 'requires_grad';

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout)

        self.to_out = nn.Sequential(
            nn.Linear(in_features = inner_dim, out_features = dim),
            nn.Dropout(p = dropout)
        )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, x):
        h = self.heads
        x, ps = pack_one(x, 'b * d')

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h d n', h = h), (q, k, v))

        q, k = map(l2norm, (q, k))

        sim = ops.einsum(equation = 'b h i n, b h j n -> b h i j', operands = q) * self.temperature.exp()

        attn = self.attend(sim)
        attn = self.dropout(attn)

        out = ops.einsum(equation = 'b h i j, b h j n -> b h i n', operands = attn)
        out = rearrange(out, 'b h d n -> b n (h d)')

        out = unpack_one(out, ps, 'b * d')
        return self.to_out(out)

class LocalPatchInteraction(nn.Cell):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        assert (kernel_size % 2) == 1
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = dim),
            Rearrange('b h w c -> b c h w'),
            nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = kernel_size, padding = padding, groups = dim),
            nn.BatchNorm2d(num_features = dim),
            nn.GELU(),
            nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = kernel_size, padding = padding, groups = dim),
            Rearrange('b c h w -> b h w c'),
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Conv2d':没有对应的mindspore参数 'device';; 'torch.nn.BatchNorm2d':没有对应的mindspore参数 'device';

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            layer = ind + 1
            self.layers.append(ModuleList([
                LayerScale(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = layer),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = layer)
            ]))

    def forward(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context = context) + x
            x = ff(x) + x

        return x

class XCATransformer(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, local_patch_kernel_size = 3, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            layer = ind + 1
            self.layers.append(ModuleList([
                LayerScale(dim, XCAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = layer),
                LayerScale(dim, LocalPatchInteraction(dim, local_patch_kernel_size), depth = layer),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = layer)
            ]))

    def forward(self, x):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for cross_covariance_attn, local_patch_interaction, ff in layers:
            x = cross_covariance_attn(x) + x
            x = local_patch_interaction(x) + x
            x = ff(x) + x

        return x

class XCiT(nn.Cell):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        local_patch_kernel_size = 3,
        layer_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(normalized_shape = patch_dim),
            nn.Linear(in_features = patch_dim, out_features = dim),
            nn.LayerNorm(normalized_shape = dim)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.pos_embedding = nn.Parameter(ops.randn(size = 1, generator = num_patches))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.cls_token = nn.Parameter(ops.randn(size = dim))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

        self.dropout = nn.Dropout(p = emb_dropout)

        self.xcit_transformer = XCATransformer(dim, depth, heads, dim_head, mlp_dim, local_patch_kernel_size, dropout, layer_dropout)

        self.final_norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';

        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape = dim),
            nn.Linear(in_features = dim, out_features = num_classes)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, img):
        x = self.to_patch_embedding(img)

        x, ps = pack_one(x, 'b * d')

        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]

        x = unpack_one(x, ps, 'b * d')

        x = self.dropout(x)

        x = self.xcit_transformer(x)

        x = self.final_norm(x)

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)

        x = rearrange(x, 'b ... d -> b (...) d')
        cls_tokens = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(cls_tokens[:, 0])
