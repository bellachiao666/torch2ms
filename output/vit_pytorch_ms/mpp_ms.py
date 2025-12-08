from mindspore.mint import nn, ops
import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

# helpers

def exists(val):
    return val is not None

def prob_mask_like(t, prob):
    batch, seq_length, _ = t.shape
    return ops.zeros(size = (batch, seq_length)).float().uniform_(0, 1) < prob  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';

def get_mask_subset_with_prob(patched_input, prob):
    batch, seq_len, _, device = *patched_input.shape, patched_input.device
    max_masked = math.ceil(prob * seq_len)

    rand = ops.rand(size = (batch, seq_len))  # 'torch.rand':没有对应的mindspore参数 'out';; 'torch.rand':没有对应的mindspore参数 'layout';; 'torch.rand':没有对应的mindspore参数 'device';; 'torch.rand':没有对应的mindspore参数 'requires_grad';; 'torch.rand':没有对应的mindspore参数 'pin_memory';
    _, sampled_indices = rand.topk(max_masked, dim=-1)

    new_mask = ops.zeros(size = (batch, seq_len))  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';
    new_mask.scatter_(1, sampled_indices, 1)
    return new_mask.bool()


# mpp loss


class MPPLoss(nn.Module):
    def __init__(
        self,
        patch_size,
        channels,
        output_channel_bits,
        max_pixel_val,
        mean,
        std
    ):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.output_channel_bits = output_channel_bits
        self.max_pixel_val = max_pixel_val

        self.mean = torch.tensor(mean).view(-1, 1, 1) if mean else None
        self.std = torch.tensor(std).view(-1, 1, 1) if std else None

    def forward(self, predicted_patches, target, mask):
        p, c, mpv, bits, device = self.patch_size, self.channels, self.max_pixel_val, self.output_channel_bits, target.device
        bin_size = mpv / (2 ** bits)

        # un-normalize input
        if exists(self.mean) and exists(self.std):
            target = target * self.std + self.mean

        # reshape target to patches
        target = target.clamp(max = mpv) # clamp just in case
        avg_target = reduce(target, 'b c (h p1) (w p2) -> b (h w) c', 'mean', p1 = p, p2 = p).contiguous()

        channel_bins = ops.arange(start = bin_size, end = mpv, step = bin_size)  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
        discretized_target = torch.bucketize(avg_target, channel_bins)

        bin_mask = (2 ** bits) ** ops.arange(start = 0, end = c).long()  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
        bin_mask = rearrange(bin_mask, 'c -> () () c')

        target_label = ops.sum(input = bin_mask * discretized_target)

        loss = F.cross_entropy(predicted_patches[mask], target_label[mask])
        return loss


# main class


class MPP(nn.Module):
    def __init__(
        self,
        transformer,
        patch_size,
        dim,
        output_channel_bits=3,
        channels=3,
        max_pixel_val=1.0,
        mask_prob=0.15,
        replace_prob=0.5,
        random_patch_prob=0.5,
        mean=None,
        std=None
    ):
        super().__init__()
        self.transformer = transformer
        self.loss = MPPLoss(patch_size, channels, output_channel_bits,
                            max_pixel_val, mean, std)

        # extract patching function
        self.patch_to_emb = nn.Sequential(transformer.to_patch_embedding[1:])

        # output transformation
        self.to_bits = nn.Linear(in_features = dim, out_features = 2**(output_channel_bits * channels))  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

        # vit related dimensions
        self.patch_size = patch_size

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob

        # token ids
        self.mask_token = nn.Parameter(ops.randn(size = 1, generator = 1))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

    def forward(self, input, **kwargs):
        transformer = self.transformer
        # clone original image for loss
        img = input.clone().detach()

        # reshape raw image to patches
        p = self.patch_size
        input = rearrange(input,
                          'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=p,
                          p2=p)

        mask = get_mask_subset_with_prob(input, self.mask_prob)

        # mask input with mask patches with probability of `replace_prob` (keep patches the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mpp
        if self.random_patch_prob > 0:
            random_patch_sampling_prob = self.random_patch_prob / (
                1 - self.replace_prob)
            random_patch_prob = prob_mask_like(input,
                                               random_patch_sampling_prob).to(mask.device)

            bool_random_patch_prob = mask * (random_patch_prob == True)
            random_patches = ops.randint(low = 0, high = input.shape[1], size = (input.shape[0], input.shape[1]))  # 'torch.randint':没有对应的mindspore参数 'out';; 'torch.randint':没有对应的mindspore参数 'layout';; 'torch.randint':没有对应的mindspore参数 'device';; 'torch.randint':没有对应的mindspore参数 'requires_grad';
            randomized_input = masked_input[
                ops.arange(start = masked_input.shape[0]).unsqueeze(-1),
                random_patches]  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
            masked_input[bool_random_patch_prob] = randomized_input[
                bool_random_patch_prob]

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob).to(mask.device)
        bool_mask_replace = (mask * replace_prob) == True
        masked_input[bool_mask_replace] = self.mask_token

        # linear embedding of patches
        masked_input = self.patch_to_emb(masked_input)

        # add cls token to input sequence
        b, n, _ = masked_input.shape
        cls_tokens = repeat(transformer.cls_token, '() n d -> b n d', b=b)
        masked_input = ops.cat(tensors = (cls_tokens, masked_input), dim = 1)  # 'torch.cat':没有对应的mindspore参数 'out';

        # add positional embeddings to input
        masked_input += transformer.pos_embedding[:, :(n + 1)]
        masked_input = transformer.dropout(masked_input)

        # get generator output and get mpp loss
        masked_input = transformer.transformer(masked_input, **kwargs)
        cls_logits = self.to_bits(masked_input)
        logits = cls_logits[:, 1:, :]

        mpp_loss = self.loss(logits, img, mask)

        return mpp_loss
