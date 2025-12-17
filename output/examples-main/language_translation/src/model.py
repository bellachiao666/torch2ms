import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import math
# from torch import nn

class PositionalEncoding(msnn.Cell):
    def __init__(
        self,
        emb_size,
        dropout,
        maxlen=5000
    ):
        super(PositionalEncoding, self).__init__()
        den = mint.exp(- mint.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = mint.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = mint.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = mint.sin(pos * den)
        pos_embedding[:, 1::2] = mint.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def construct(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class Translator(msnn.Cell):
    def __init__(
            self,
            num_encoder_layers,
            num_decoder_layers,
            embed_size,
            num_heads,
            src_vocab_size,
            tgt_vocab_size,
            dim_feedforward,
            dropout
        ):
        super(Translator, self).__init__()

        # Output of embedding must be equal (embed_size)
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)

        self.pos_enc = PositionalEncoding(embed_size, dropout)

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )  # 'torch.nn.Transformer' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        self.ff = nn.Linear(embed_size, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 'torch.nn.init.xavier_uniform_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):

        src_emb = self.pos_enc(self.src_embedding(src))
        tgt_emb = self.pos_enc(self.tgt_embedding(trg))

        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask
        )

        return self.ff(outs)

    def encode(self, src, src_mask):

        embed = self.src_embedding(src)

        pos_enc = self.pos_enc(embed)

        return self.transformer.encoder(pos_enc, src_mask)

    def decode(self, tgt, memory, tgt_mask):
        
        embed = self.tgt_embedding(tgt)

        pos_enc = self.pos_enc(embed)

        return self.transformer.decoder(pos_enc, memory, tgt_mask)
