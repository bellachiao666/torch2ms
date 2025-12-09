from torch import nn
from mindspore.mint import nn, ops
from einops import repeat

class SimMIM(nn.Cell):
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
        self.patch_to_emb = nn.SequentialCell(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # simple linear head

        self.mask_token = mindspore.Parameter(ops.randn(size = encoder_dim))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.to_pixels = nn.Linear(in_features = encoder_dim, out_features = pixel_values_per_patch)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = ops.arange(start = batch)[:, None]  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';

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
        masked_indices = ops.rand(size = batch, generator = num_patches).topk(k = num_masked, dim = -1).indices  # 'torch.rand':没有对应的mindspore参数 'out';; 'torch.rand':没有对应的mindspore参数 'layout';; 'torch.rand':没有对应的mindspore参数 'device';; 'torch.rand':没有对应的mindspore参数 'requires_grad';; 'torch.rand':没有对应的mindspore参数 'pin_memory';
        masked_bool_mask = ops.zeros(size = (batch, num_patches)).scatter_(-1, masked_indices, 1).bool()  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';

        # mask tokens

        tokens = ops.where(condition = masked_bool_mask[..., None], input = mask_tokens, other = tokens)  # 'torch.where':没有对应的mindspore参数 'out';

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens)

        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = nn.functional.l1_loss(input = pred_pixel_values, target = masked_patches) / num_masked  # 'torch.nn.functional.l1_loss':没有对应的mindspore参数 'size_average';; 'torch.nn.functional.l1_loss':没有对应的mindspore参数 'reduce';
        return recon_loss
