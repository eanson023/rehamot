from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from model.data.tools import lengths_to_mask
from model.utils import PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(self,
                 nfeats: int,
                 latent_dim: int = 256,
                 embed_size: int = 1024,
                 ff_size: int = 1024,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 **kwargs) -> None:
        super(TransformerEncoder, self).__init__()

        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)

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

        self.learning_rates_x = []

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None) -> Tuple[Tensor, None]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

        # adding the embedding token for all sequences
        xseq = torch.cat((emb_token[None], x), 0)

        # create a bigger mask, to allow attend to emb
        token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        features = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        # normalization in the motion embedding space
        features = F.normalize(features[0], dim=1)
        # normalization in the joint embedding space
        return F.normalize(self.fc(features), dim=1), None
