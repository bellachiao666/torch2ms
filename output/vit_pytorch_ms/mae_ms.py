from torch import nn
from einops import repeat

from vit_pytorch.vit import Transformer
from mindspore.mint import nn, ops

class MAE(nn.Cell):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
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

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(in_features = encoder_dim, out_features = decoder_dim) if encoder_dim != decoder_dim else nn.Identity()  # 'torch.nn.Linear':没有对应的mindspore参数 'device';
        self.mask_token = mindspore.Parameter(ops.randn(size = decoder_dim))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_embeddings = num_patches, embedding_dim = decoder_dim)  # 'torch.nn.Embedding':没有对应的mindspore参数 'device';
        self.to_pixels = nn.Linear(in_features = decoder_dim, out_features = pixel_values_per_patch)  # 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = ops.rand(size = batch, generator = num_patches).argsort(dim = -1)  # 'torch.rand':没有对应的mindspore参数 'out';; 'torch.rand':没有对应的mindspore参数 'layout';; 'torch.rand':没有对应的mindspore参数 'device';; 'torch.rand':没有对应的mindspore参数 'requires_grad';; 'torch.rand':没有对应的mindspore参数 'pin_memory';
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = ops.arange(start = batch)[:, None]  # 'torch.arange':没有对应的mindspore参数 'out';; 'torch.arange':没有对应的mindspore参数 'layout';; 'torch.arange':没有对应的mindspore参数 'device';; 'torch.arange':没有对应的mindspore参数 'requires_grad';
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = ops.zeros(size = batch, dtype = self.decoder_dim)  # 'torch.zeros':没有对应的mindspore参数 'out';; 'torch.zeros':没有对应的mindspore参数 'layout';; 'torch.zeros':没有对应的mindspore参数 'device';; 'torch.zeros':没有对应的mindspore参数 'requires_grad';
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        recon_loss = nn.functional.mse_loss(input = pred_pixel_values, target = masked_patches)  # 'torch.nn.functional.mse_loss':没有对应的mindspore参数 'size_average';; 'torch.nn.functional.mse_loss':没有对应的mindspore参数 'reduce';
        return recon_loss
