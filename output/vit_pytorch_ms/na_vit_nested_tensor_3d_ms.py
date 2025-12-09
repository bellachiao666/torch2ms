from __future__ import annotations

from typing import List
from functools import partial

import torch
import packaging.version as pkg_version

from torch import nn, Tensor
import torch.nn.functional as F
from torch.nested import nested_tensor

from einops import rearrange
from einops.layers.torch import Rearrange
from mindspore.mint import nn, ops

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# feedforward

def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.SequentialCell(
        nn.LayerNorm(normalized_shape = dim, bias = False),
        nn.Linear(in_features = dim, out_features = hidden_dim),
        nn.GELU(),
        nn.Dropout(p = dropout),
        nn.Linear(in_features = hidden_dim, out_features = dim),
        nn.Dropout(p = dropout)
    )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

class Attention(nn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., qk_norm = True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = dim, bias = False)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';

        dim_inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        self.to_queries = nn.Linear(in_features = dim, out_features = dim_inner, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
        self.to_keys = nn.Linear(in_features = dim, out_features = dim_inner, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
        self.to_values = nn.Linear(in_features = dim, out_features = dim_inner, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        # in the paper, they employ qk rmsnorm, a way to stabilize attention
        # will use layernorm in place of rmsnorm, which has been shown to work in certain papers. requires l2norm on non-ragged dimension to be supported in nested tensors

        self.query_norm = nn.LayerNorm(normalized_shape = dim_head, bias = False) if qk_norm else nn.Identity()  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.key_norm = nn.LayerNorm(normalized_shape = dim_head, bias = False) if qk_norm else nn.Identity()  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';

        self.dropout = dropout

        self.to_out = nn.Linear(in_features = dim_inner, out_features = dim, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(
        self, 
        x,
        context: Tensor | None = None
    ):

        x = self.norm(x)

        # for attention pooling, one query pooling to entire sequence

        context = default(context, x)

        # queries, keys, values

        query = self.to_queries(x)
        key = self.to_keys(context)
        value = self.to_values(context)

        # split heads

        def split_heads(t):
            return t.unflatten(-1, (self.heads, self.dim_head))

        def transpose_head_seq(t):
            return t.transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))

        # qk norm for attention stability

        query = self.query_norm(query)
        key = self.key_norm(key)

        query, key, value = map(transpose_head_seq, (query, key, value))

        # attention

        out = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p = self.dropout if self.training else 0.
        )

        # merge heads

        out = out.transpose(1, 2).flatten(-2)

        return self.to_out(out)

class Transformer(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., qk_norm = True):
        super().__init__()
        self.layers = nn.CellList([])

        for _ in range(depth):
            self.layers.append(nn.CellList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, qk_norm = qk_norm),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.norm = nn.LayerNorm(normalized_shape = dim, bias = False)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class NaViT(nn.Cell):
    def __init__(
        self,
        *,
        image_size,
        max_frames,
        patch_size,
        frame_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        num_registers = 4,
        qk_rmsnorm = True,
        token_dropout_prob: float | None = None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        if pkg_version.parse(torch.__version__) < pkg_version.parse('2.5'):
            print('nested tensor NaViT was tested on pytorch 2.5')

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width

        self.token_dropout_prob = token_dropout_prob

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'
        assert divisible_by(max_frames, frame_patch_size)

        patch_frame_dim, patch_height_dim, patch_width_dim = (max_frames // frame_patch_size), (image_height // patch_size), (image_width // patch_size)

        patch_dim = channels * (patch_size ** 2) * frame_patch_size

        self.channels = channels
        self.patch_size = patch_size
        self.to_patches = Rearrange('c (f pf) (h p1) (w p2) -> f h w (c pf p1 p2)', p1 = patch_size, p2 = patch_size, pf = frame_patch_size)

        self.to_patch_embedding = nn.SequentialCell(
            nn.LayerNorm(normalized_shape = patch_dim),
            nn.Linear(in_features = patch_dim, out_features = dim),
            nn.LayerNorm(normalized_shape = dim),
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.pos_embed_frame = mindspore.Parameter(ops.zeros(size = patch_frame_dim))  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';
        self.pos_embed_height = mindspore.Parameter(ops.zeros(size = patch_height_dim))  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';
        self.pos_embed_width = mindspore.Parameter(ops.zeros(size = patch_width_dim))  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';

        # register tokens

        self.register_tokens = mindspore.Parameter(ops.zeros(size = num_registers))  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';

        nn.init.normal_(self.pos_embed_frame, std = 0.02)
        nn.init.normal_(self.pos_embed_height, std = 0.02)
        nn.init.normal_(self.pos_embed_width, std = 0.02)
        nn.init.normal_(self.register_tokens, std = 0.02)

        self.dropout = nn.Dropout(p = emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, qk_rmsnorm)

        # final attention pooling queries

        self.attn_pool_queries = mindspore.Parameter(ops.randn(size = dim))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # output to logits

        self.to_latent = nn.Identity()

        self.mlp_head = nn.SequentialCell(
            nn.LayerNorm(normalized_shape = dim, bias = False),
            nn.Linear(in_features = dim, out_features = num_classes, bias = False)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        volumes: List[Tensor], # different resolution images / CT scans
    ):
        batch, device = len(volumes), self.device
        arange = partial(torch.arange, device = device)

        assert all([volume.ndim == 4 and volume.shape[0] == self.channels for volume in volumes]), f'all volumes must have {self.channels} channels and number of dimensions of {self.channels} (channels, frame, height, width)'

        all_patches = [self.to_patches(volume) for volume in volumes]

        # prepare factorized positional embedding height width indices

        positions = []

        for patches in all_patches:
            patch_frame, patch_height, patch_width = patches.shape[:3]
            fhw_indices = ops.stack(tensors = ops.meshgrid(tensors = (arange(patch_frame), arange(patch_height), arange(patch_width)), indexing = 'ij'), dim = -1)  # 'torch.stack':没有对应的mindspore参数 'out';
            fhw_indices = rearrange(fhw_indices, 'f h w c -> (f h w) c')

            positions.append(fhw_indices)

        # need the sizes to compute token dropout + positional embedding

        tokens = [rearrange(patches, 'f h w d -> (f h w) d') for patches in all_patches]

        # handle token dropout

        seq_lens = torch.tensor([i.shape[0] for i in tokens], device = device)

        if self.training and self.token_dropout_prob > 0:

            keep_seq_lens = ((1. - self.token_dropout_prob) * seq_lens).int().clamp(min = 1)

            kept_tokens = []
            kept_positions = []

            for one_image_tokens, one_image_positions, seq_len, num_keep in zip(tokens, positions, seq_lens, keep_seq_lens):
                keep_indices = ops.randn(size = (seq_len,)).topk(num_keep, dim = -1).indices  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

                one_image_kept_tokens = one_image_tokens[keep_indices]
                one_image_kept_positions = one_image_positions[keep_indices]

                kept_tokens.append(one_image_kept_tokens)
                kept_positions.append(one_image_kept_positions)

            tokens, positions, seq_lens = kept_tokens, kept_positions, keep_seq_lens

        # add all height and width factorized positions


        frame_indices, height_indices, width_indices = ops.cat(tensors = positions).unbind(dim = -1)  # 'torch.cat':没有对应的mindspore参数 'out';
        frame_embed, height_embed, width_embed = self.pos_embed_frame[frame_indices], self.pos_embed_height[height_indices], self.pos_embed_width[width_indices]

        pos_embed = frame_embed + height_embed + width_embed

        tokens = ops.cat(tensors = tokens)  # 'torch.cat':没有对应的mindspore参数 'out';

        # linear projection to patch embeddings

        tokens = self.to_patch_embedding(tokens)

        # absolute positions

        tokens = tokens + pos_embed

        # add register tokens

        tokens = tokens.split(seq_lens.tolist())

        tokens = [ops.cat(tensors = (self.register_tokens, one_tokens)) for one_tokens in tokens]  # 'torch.cat':没有对应的mindspore参数 'out';

        # use nested tensor for transformers and save on padding computation

        tokens = nested_tensor(tokens, layout = torch.jagged, device = device)

        # embedding dropout

        tokens = self.dropout(tokens)

        # transformer

        tokens = self.transformer(tokens)

        # attention pooling
        # will use a jagged tensor for queries, as SDPA requires all inputs to be jagged, or not

        attn_pool_queries = [rearrange(self.attn_pool_queries, '... -> 1 ...')] * batch

        attn_pool_queries = nested_tensor(attn_pool_queries, layout = torch.jagged)

        pooled = self.attn_pool(attn_pool_queries, tokens)

        # back to unjagged

        logits = ops.stack(tensors = pooled.unbind())  # 'torch.stack':没有对应的mindspore参数 'out';

        logits = rearrange(logits, 'b 1 d -> b d')

        logits = self.to_latent(logits)

        return self.mlp_head(logits)

# quick test

if __name__ == '__main__':

    # works for torch 2.5

    v = NaViT(
        image_size = 256,
        max_frames = 8,
        patch_size = 32,
        frame_patch_size = 2,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.,
        emb_dropout = 0.,
        token_dropout_prob = 0.1
    )

    # 5 volumetric data (videos or CT scans) of different resolutions - List[Tensor]

    volumes = [
        ops.randn(size = 3, generator = 2, dtype = 256), ops.randn(size = 3, generator = 8, dtype = 128),
        ops.randn(size = 3, generator = 4, dtype = 256), ops.randn(size = 3, generator = 2, dtype = 128),
        ops.randn(size = 3, generator = 4, dtype = 256)
    ]  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

    assert v(volumes).shape == (5, 1000)

    v(volumes).sum().backward()
