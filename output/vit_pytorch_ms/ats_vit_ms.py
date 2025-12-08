import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from mindspore.mint import nn, ops

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# adaptive token sampling functions and classes

def log(t, eps = 1e-6):
    return ops.log(input = t + eps)  # 'torch.log':没有对应的mindspore参数 'out';

def sample_gumbel(shape, device, dtype, eps = 1e-6):
    u = ops.empty(size = shape, dtype = dtype, device = device).uniform_(0, 1)  # 'torch.empty':没有对应的mindspore参数 'out';; 'torch.empty':没有对应的mindspore参数 'layout';; 'torch.empty':没有对应的mindspore参数 'requires_grad';; 'torch.empty':没有对应的mindspore参数 'memory_format';
    return -log(-log(u, eps), eps)

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

class AdaptiveTokenSampling(nn.Cell):
    def __init__(self, output_num_tokens, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.output_num_tokens = output_num_tokens

    def forward(self, attn, value, mask):
        heads, output_num_tokens, eps, device, dtype = attn.shape[1], self.output_num_tokens, self.eps, attn.device, attn.dtype

        # first get the attention values for CLS token to all other tokens

        cls_attn = attn[..., 0, 1:]

        # calculate the norms of the values, for weighting the scores, as described in the paper

        value_norms = value[..., 1:, :].norm(dim = -1)

        # weigh the attention scores by the norm of the values, sum across all heads

        cls_attn = ops.einsum(equation = 'b h n, b h n -> b n', operands = cls_attn)

        # normalize to 1

        normed_cls_attn = cls_attn / (cls_attn.sum(dim = -1, keepdim = True) + eps)

        # instead of using inverse transform sampling, going to invert the softmax and use gumbel-max sampling instead

        pseudo_logits = log(normed_cls_attn)

        # mask out pseudo logits for gumbel-max sampling

        mask_without_cls = mask[:, 1:]
        mask_value = -torch.finfo(attn.dtype).max / 2
        pseudo_logits = pseudo_logits.masked_fill(~mask_without_cls, mask_value)

        # expand k times, k being the adaptive sampling number

        pseudo_logits = repeat(pseudo_logits, 'b n -> b k n', k = output_num_tokens)
        pseudo_logits = pseudo_logits + sample_gumbel(pseudo_logits.shape, device = device, dtype = dtype)

        # gumble-max and add one to reserve 0 for padding / mask

        sampled_token_ids = pseudo_logits.argmax(dim = -1) + 1

        # calculate unique using torch.unique and then pad the sequence from the right

        unique_sampled_token_ids_list = [ops.unique(input = t, sorted = True) for t in ops.unbind(input = sampled_token_ids)]
        unique_sampled_token_ids = pad_sequence(unique_sampled_token_ids_list, batch_first = True)

        # calculate the new mask, based on the padding

        new_mask = unique_sampled_token_ids != 0

        # CLS token never gets masked out (gets a value of True)

        new_mask = nn.functional.pad(input = new_mask, pad = (1, 0), value = True)

        # prepend a 0 token id to keep the CLS attention scores

        unique_sampled_token_ids = nn.functional.pad(input = unique_sampled_token_ids, pad = (1, 0), value = 0)
        expanded_unique_sampled_token_ids = repeat(unique_sampled_token_ids, 'b n -> b h n', h = heads)

        # gather the new attention scores

        new_attn = batched_index_select(attn, expanded_unique_sampled_token_ids, dim = 2)

        # return the sampled attention scores, new mask (denoting padding), as well as the sampled token indices (for the residual)
        return new_attn, new_mask, unique_sampled_token_ids

# classes

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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., output_num_tokens = None):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(normalized_shape = dim)  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout)

        self.to_qkv = nn.Linear(in_features = dim, out_features = inner_dim * 3, bias = False)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.output_num_tokens = output_num_tokens
        self.ats = AdaptiveTokenSampling(output_num_tokens) if exists(output_num_tokens) else None

        self.to_out = nn.Sequential(
            nn.Linear(in_features = inner_dim, out_features = dim),
            nn.Dropout(p = dropout)
        )  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, x, *, mask):
        num_tokens = x.shape[1]

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = ops.matmul(input = q, other = k.transpose(-1, -2)) * self.scale  # 'torch.matmul':没有对应的mindspore参数 'out';

        if exists(mask):
            dots_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(mask, 'b j -> b 1 1 j')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~dots_mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        sampled_token_ids = None

        # if adaptive token sampling is enabled
        # and number of tokens is greater than the number of output tokens
        if exists(self.output_num_tokens) and (num_tokens - 1) > self.output_num_tokens:
            attn, mask, sampled_token_ids = self.ats(attn, v, mask = mask)

        out = ops.matmul(input = attn, other = v)  # 'torch.matmul':没有对应的mindspore参数 'out';
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), mask, sampled_token_ids

class Transformer(nn.Cell):
    def __init__(self, dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        assert len(max_tokens_per_depth) == depth, 'max_tokens_per_depth must be a tuple of length that is equal to the depth of the transformer'
        assert sorted(max_tokens_per_depth, reverse = True) == list(max_tokens_per_depth), 'max_tokens_per_depth must be in decreasing order'
        assert min(max_tokens_per_depth) > 0, 'max_tokens_per_depth must have at least 1 token at any layer'

        self.layers = nn.ModuleList([])
        for _, output_num_tokens in zip(range(depth), max_tokens_per_depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, output_num_tokens = output_num_tokens, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        b, n, device = *x.shape[:2], x.device

        # use mask to keep track of the paddings when sampling tokens
        # as the duplicates (when sampling) are just removed, as mentioned in the paper
        mask = ops.ones(size = (b, n), dtype = torch.bool)  # 'torch.ones':没有对应的mindspore参数 'out';; 'torch.ones':没有对应的mindspore参数 'layout';; 'torch.ones':没有对应的mindspore参数 'device';; 'torch.ones':没有对应的mindspore参数 'requires_grad';

        token_ids = ops.arange(start = n)  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
        token_ids = repeat(token_ids, 'n -> b n', b = b)

        for attn, ff in self.layers:
            attn_out, mask, sampled_token_ids = attn(x, mask = mask)

            # when token sampling, one needs to then gather the residual tokens with the sampled token ids
            if exists(sampled_token_ids):
                x = batched_index_select(x, sampled_token_ids, dim = 1)
                token_ids = batched_index_select(token_ids, sampled_token_ids, dim = 1)

            x = x + attn_out

            x = ff(x) + x

        return x, token_ids

class ViT(nn.Cell):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, max_tokens_per_depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(normalized_shape = patch_dim),
            nn.Linear(in_features = patch_dim, out_features = dim),
            nn.LayerNorm(normalized_shape = dim)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

        self.pos_embedding = nn.Parameter(ops.randn(size = 1, generator = num_patches + 1))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.cls_token = nn.Parameter(ops.randn(size = 1, generator = 1))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.dropout = nn.Dropout(p = emb_dropout)

        self.transformer = Transformer(dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape = dim),
            nn.Linear(in_features = dim, out_features = num_classes)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, img, return_sampled_token_ids = False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = ops.cat(tensors = (cls_tokens, x), dim = 1)  # 'torch.cat':没有对应的mindspore参数 'out';
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, token_ids = self.transformer(x)

        logits = self.mlp_head(x[:, 0])

        if return_sampled_token_ids:
            # remove CLS token and decrement by 1 to make -1 the padding
            token_ids = token_ids[:, 1:] - 1
            return logits, token_ids

        return logits
