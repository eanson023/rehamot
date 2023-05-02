from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from model.data.tools import lengths_to_mask
from model.utils import PositionalEncoding


class TransformerMomentumEncoder(nn.Module):
    def __init__(self,
                 nfeats: int,
                 latent_dim: int = 256,
                 embed_size: int = 1024,
                 ff_size: int = 1024,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 momentum: float = 0.995,
                 queue_size: int = 65536,
                 **kwargs) -> None:
        super(TransformerMomentumEncoder, self).__init__()

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

        # create momentum models
        self.momentum = momentum
        self.queue_size = queue_size
        self.skel_embedding_m = nn.Linear(input_feats, latent_dim)
        self.emb_token_m = nn.Parameter(torch.randn(latent_dim))
        self.seqTransEncoder_m = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                       num_layers=num_layers)
        self.fc_m = nn.Linear(latent_dim, embed_size)

        self.model_pairs = [[self.skel_embedding, self.skel_embedding_m],
                            [self.seqTransEncoder, self.seqTransEncoder_m],
                            [self.fc, self.fc_m],
                            ]
        self.copy_params()

        # create the queue
        self.register_buffer("queue", torch.randn(embed_size, queue_size))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.learning_rates_x = []

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient
        self.emb_token_m.data.copy_(self.emb_token.data)
        self.emb_token_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + \
                               param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # array out of bounds check
        if ptr + batch_size >= self.queue_size:
            split = self.queue_size - ptr
            prefix = batch_size - split
            self.queue[:, ptr:] = keys[:split].T
            self.queue[:, :prefix] = keys[split:].T
        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None) -> Tuple[Tensor, Tensor]:

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
        x = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        x = F.normalize(x[0], dim=1)
        q = F.normalize(self.fc(x), dim=1)

        # compute key features
        with torch.no_grad():
            self._momentum_update()
            bs, nframes, nfeats = features.shape
            mask = lengths_to_mask(lengths, features.device)
            x = self.skel_embedding_m(features)
            x = x.permute(1, 0, 2)
            emb_token = torch.tile(self.emb_token_m, (bs,)).reshape(bs, -1)
            xseq = torch.cat((emb_token[None], x), 0)
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
            xseq = self.sequence_pos_encoding(xseq)
            x_m = self.seqTransEncoder_m(
                xseq, src_key_padding_mask=~aug_mask)
            x_m = F.normalize(x_m[0], dim=1)
            x_m = F.normalize(self.fc(x_m), dim=1)
            keys = torch.cat([x_m, self.queue.clone().detach().T], dim=0)
            # dequeue and enqueue
            self._dequeue_and_enqueue(x_m)

        return q, keys
