from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from model.utils import PositionalEncoding
from .distilbert import DistilbertEncoderBase


class DistilbertTransformerEncoder(DistilbertEncoderBase):
    def __init__(self,
                 modelpath: str,
                 finetune: bool = False,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 embed_size: int = 1024,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)

        encoded_dim = self.text_encoded_dim

        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(encoded_dim, latent_dim))

        self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)
        self.fc = nn.Linear(latent_dim, embed_size)

    def forward(self, texts: List[str]) -> Tuple[Tensor, None]:
        text_encoded, mask = self.get_last_hidden_state(
            texts, return_mask=True)

        x = self.projection(text_encoded)
        bs, nframes, _ = x.shape
        # bs, nframes, totjoints, nfeats = x.shape
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

        # adding the embedding token for all sequences
        xseq = torch.cat((emb_token[None], x), 0)

        # create a bigger mask, to allow attend to emb
        token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        # normalization in the motion embedding space
        final = F.normalize(final[0], dim=1)
        # normalization in the joint embedding space
        return F.normalize(self.fc(final), dim=1), None
