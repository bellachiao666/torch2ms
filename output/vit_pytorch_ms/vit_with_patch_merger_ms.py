from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from mindspore.mint import nn, ops

# helpers

def exists(val):
    return val is not None

def default(val ,d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# patch merger class

class PatchMerger(nn.Cell):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.queries = mindspore.Parameter(ops.randn(size = num_tokens_out, generator = dim))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

    def forward(self, x):
        x = self.norm(x)
        sim = ops.matmul(input = self.queries, other = x.transpose(-1, -2)) * self.scale  # 'torch.matmul':没有对应的mindspore参数 'out';
        attn = sim.softmax(dim = -1)
        return ops.matmul(input = attn, other = x)  # 'torch.matmul':没有对应的mindspore参数 'out';

# classes

class FeedForward(nn.Cell):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.SequentialCell(
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
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout)

        self.to_qkv = nn.Linear(in_features = dim, out_features = inner_dim * 3, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.to_out = nn.SequentialCell(
            nn.Linear(in_features = inner_dim, out_features = dim),
            nn.Dropout(p = dropout)
        ) if project_out else nn.Identity()  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = ops.matmul(input = q, other = k.transpose(-1, -2)) * self.scale  # 'torch.matmul':没有对应的mindspore参数 'out';

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = ops.matmul(input = attn, other = v)  # 'torch.matmul':没有对应的mindspore参数 'out';
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., patch_merge_layer = None, patch_merge_num_tokens = 8):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.layers = nn.CellList([])

        self.patch_merge_layer_index = default(patch_merge_layer, depth // 2) - 1 # default to mid-way through transformer, as shown in paper
        self.patch_merger = PatchMerger(dim = dim, num_tokens_out = patch_merge_num_tokens)

        for _ in range(depth):
            self.layers.append(nn.CellList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for index, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x

            if index == self.patch_merge_layer_index:
                x = self.patch_merger(x)

        return self.norm(x)

class ViT(nn.Cell):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, patch_merge_layer = None, patch_merge_num_tokens = 8, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.SequentialCell(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(normalized_shape = patch_dim),
            nn.Linear(in_features = patch_dim, out_features = dim),
            nn.LayerNorm(normalized_shape = dim)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.pos_embedding = mindspore.Parameter(ops.randn(size = 1, generator = num_patches + 1))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.dropout = nn.Dropout(p = emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, patch_merge_layer, patch_merge_num_tokens)

        self.mlp_head = nn.SequentialCell(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(in_features = dim, out_features = num_classes)
        )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        return self.mlp_head(x)
