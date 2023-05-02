from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class RNNTypeEncoder(nn.Module):
    RNN_MODULES = {
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU
    }

    def __init__(self, nfeats: int,
                 latent_dim: int = 256,
                 embed_size: int = 1024,
                 dropout: float = 0.1,
                 num_layers: int = 4,
                 rnn_type: str = 'gru',
                 **kwargs) -> None:
        super(RNNTypeEncoder, self).__init__()

        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)

        rnn_module = self.RNN_MODULES.get(rnn_type)
        if rnn_module is None:
            raise ValueError(f"Invalid rnn_type:{rnn_type}")

        self.rnn = rnn_module(latent_dim, latent_dim,
                              num_layers=num_layers, dropout=dropout)

        self.final = nn.Linear(latent_dim, embed_size)

        self.learning_rates_x = []

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None) -> Tuple[Tensor, None]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Get all the output of the rnn
        x = self.rnn(x)[0]

        # Put back the batch dimention first
        x = x.permute(1, 0, 2)  # now it is [bs, nframes, latent_dim]

        # Extract the last valid input
        x = x[tuple(torch.stack((torch.arange(bs, device=x.device),
                                 torch.tensor(lengths, device=x.device) - 1)))]
        # normalization in the motion embedding space
        x = F.normalize(x, dim=1)
        # normalization in the joint embedding space
        return F.normalize(self.final(x), dim=1), None
