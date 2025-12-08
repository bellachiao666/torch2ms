# from torch import nn
import einops
from einops import rearrange, repeat, pack, unpack
# from einops.layers.torch import Rearrange
from mindspore.mint import nn, ops
import mindspore
from mindspore import ops
# classes
class PatchEmbed(nn.Cell):
    def __init__(self, patch_size, patch_dim, dim):
        super().__init__()
        self.patch_size = patch_size
        self.ln1 = nn.LayerNorm([patch_dim])
        self.fc  = nn.Dense(patch_dim, dim)
        self.ln2 = nn.LayerNorm([dim])

    def construct(self, x):
        p=self.patch_size
        x = einops.rearrange(x, 'b c (n p) -> b n (p c)', p=p)
        x = self.ln1(x)
        x = self.fc(x)
        x = self.ln2(x)
        return x

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
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout)

        self.to_qkv = nn.Linear(in_features = dim, out_features = inner_dim * 3, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.to_out = nn.Sequential(
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Cell):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size
        self.to_patch_embedding = mindspore.nn.SequentialCell([
                nn.LayerNorm([patch_dim]),
                mindspore.nn.Dense(patch_dim, dim),
                nn.LayerNorm([dim]),
            ])
        
    def construct(self, x):
        x = einops.rearrange(x, 'b c (n p) -> b n (p c)', p=patch_size)
        x = self.to_patch_embedding(x)
        return x

        # self.to_patch_embedding = nn.Sequential(
        #     lambda x: einops.rearrange(x, 'b c (n p) -> b n (p c)', p=patch_size),
        #     nn.LayerNorm(normalized_shape = patch_dim),
        #     nn.Linear(in_features = patch_dim, out_features = dim),
        #     nn.LayerNorm(normalized_shape = dim),
        # )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.pos_embedding = nn.Parameter(ops.randn(size = 1, generator = num_patches + 1))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.cls_token = nn.Parameter(ops.randn(size = dim))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.dropout = nn.Dropout(p = emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape = dim),
            nn.Linear(in_features = dim, out_features = num_classes)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)

if __name__ == '__main__':

    v = ViT(
        seq_len = 256,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    time_series = ops.randn(4, 256)  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
    logits = v(time_series) # (4, 1000)
