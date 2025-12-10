import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# import torch
# from torch import nn

from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# controlling freezing of layers

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

# classes

class FeedForward(msnn.Cell):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = msnn.SequentialCell(
            [nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)])
    def construct(self, x):
        return self.net(x)

class Attention(msnn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = msnn.SequentialCell(
            [nn.Linear(inner_dim, dim), nn.Dropout(dropout)])

    def construct(self, x, attn_mask = None, memories = None):
        x = self.norm(x)

        x_kv = x # input for key / values projection

        if exists(memories):
            # add memories to key / values if it is passed in
            memories = repeat(memories, 'n d -> b n d', b = x.shape[0]) if memories.ndim == 2 else memories
            x_kv = mint.cat((x_kv, memories), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(x_kv).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = mint.matmul(q, k.transpose(-1, -2)) * self.scale

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = mint.matmul(attn, v)
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

    def construct(self, x, attn_mask = None, memories = None):
        for ind, (attn, ff) in enumerate(self.layers):
            layer_memories = memories[ind] if exists(memories) else None

            x = attn(x, attn_mask = attn_mask, memories = layer_memories) + x
            x = ff(x) + x
        return x

class ViT(msnn.Cell):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = msnn.SequentialCell(
            [Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim)])

        self.pos_embedding = ms.Parameter(mint.randn(size = (1, num_patches + 1, dim)))
        self.cls_token = ms.Parameter(mint.randn(size = (1, 1, dim)))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = msnn.SequentialCell(
            [nn.LayerNorm(dim), nn.Linear(dim, num_classes)])

    def img_to_tokens(self, img):
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = x.shape[0])
        x = mint.cat((cls_tokens, x), dim = 1)

        x += self.pos_embedding
        x = self.dropout(x)
        return x

    def construct(self, img):
        x = self.img_to_tokens(img)        

        x = self.transformer(x)

        cls_tokens = x[:, 0]
        return self.mlp_head(cls_tokens)

# adapter with learnable memories per layer, memory CLS token, and learnable adapter head

class Adapter(msnn.Cell):
    def __init__(
        self,
        *,
        vit,
        num_memories_per_layer = 10,
        num_classes = 2,   
    ):
        super().__init__()
        assert isinstance(vit, ViT)

        # extract some model variables needed

        dim = vit.cls_token.shape[-1]
        layers = len(vit.transformer.layers)
        num_patches = vit.pos_embedding.shape[-2]

        self.vit = vit

        # freeze ViT backbone - only memories will be finetuned

        freeze_all_layers_(vit)

        # learnable parameters

        self.memory_cls_token = ms.Parameter(mint.randn(dim))
        self.memories_per_layer = ms.Parameter(mint.randn(size = (layers, num_memories_per_layer, dim)))

        self.mlp_head = msnn.SequentialCell(
            [nn.LayerNorm(dim), nn.Linear(dim, num_classes)])

        # specialized attention mask to preserve the output of the original ViT
        # it allows the memory CLS token to attend to all other tokens (and the learnable memory layer tokens), but not vice versa        

        attn_mask = mint.ones((num_patches, num_patches), dtype = ms.bool)
        attn_mask = nn.functional.pad(attn_mask, (1, num_memories_per_layer), value = False)  # main tokens cannot attend to learnable memories per layer
        attn_mask = nn.functional.pad(attn_mask, (0, 0, 1, 0), value = True)                  # memory CLS token can attend to everything
        self.register_buffer('attn_mask', attn_mask)

    def construct(self, img):
        b = img.shape[0]

        tokens = self.vit.img_to_tokens(img)

        # add task specific memory tokens

        memory_cls_tokens = repeat(self.memory_cls_token, 'd -> b 1 d', b = b)
        tokens = mint.cat((memory_cls_tokens, tokens), dim = 1)        

        # pass memories along with image tokens through transformer for attending

        out = self.vit.transformer(tokens, memories = self.memories_per_layer, attn_mask = self.attn_mask)

        # extract memory CLS tokens

        memory_cls_tokens = out[:, 0]

        # pass through task specific adapter head

        return self.mlp_head(memory_cls_tokens)
