import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# from torch import nn
from einops import repeat

class SimMIM(msnn.Cell):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = msnn.SequentialCell([encoder.to_patch_embedding[1:]])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # simple linear head

        self.mask_token = ms.Parameter(mint.randn(encoder_dim))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

    def construct(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = mint.arange(batch)[:, None]  # 'torch.arange':没有对应的mindspore参数 'device' (position 6);

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb

        # prepare mask tokens

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = mint.rand(size = (batch, num_patches)).topk(k = num_masked, dim = -1).indices  # 'torch.rand':没有对应的mindspore参数 'device' (position 5);
        masked_bool_mask = mint.zeros((batch, num_patches)).scatter_(-1, masked_indices, 1).bool()  # 'torch.zeros':没有对应的mindspore参数 'device' (position 4);

        # mask tokens

        tokens = mint.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens)

        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = nn.functional.l1_loss(pred_pixel_values, masked_patches) / num_masked
        return recon_loss
