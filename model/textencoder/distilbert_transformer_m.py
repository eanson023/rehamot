from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from model.utils import PositionalEncoding
from .distilbert import DistilbertEncoderBase


class DistilbertTransformerMomentumEncoder(DistilbertEncoderBase):
    def __init__(self,
                 modelpath: str,
                 finetune: bool = False,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 embed_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 momentum: float = 0.995,
                 queue_size: int = 65536,
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
        # create momentum models
        self.momentum = momentum
        self.queue_size = queue_size
        self.projection_m = nn.Sequential(nn.ReLU(),
                                          nn.Linear(encoded_dim, latent_dim))
        self.emb_token_m = nn.Parameter(torch.randn(latent_dim))
        self.seqTransEncoder_m = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                       num_layers=num_layers)
        self.fc_m = nn.Linear(latent_dim, embed_size)

        self.model_pairs = [[self.projection, self.projection_m],
                            [self.seqTransEncoder, self.seqTransEncoder_m],
                            [self.fc, self.fc_m]
                            ]
        self.copy_params()

        # create the queue
        self.register_buffer("queue", torch.randn(embed_size, queue_size))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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

    def enc(self, texts: List[str]) -> Tensor:
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
        final = F.normalize(final[0], dim=1)
        return F.normalize(self.fc(final), dim=1)

    def forward(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
        final = self.enc(texts)

        # compute key features
        with torch.no_grad():
            self._momentum_update()
            text_encoded, mask = self.get_last_hidden_state(
                texts, return_mask=True)
            x = self.projection_m(text_encoded)
            bs, nframes, _ = x.shape
            x = x.permute(1, 0, 2)
            emb_token = torch.tile(self.emb_token_m, (bs,)).reshape(bs, -1)
            xseq = torch.cat((emb_token[None], x), 0)
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
            xseq = self.sequence_pos_encoding(xseq)
            final_m = self.seqTransEncoder_m(
                xseq, src_key_padding_mask=~aug_mask)
            final_m = F.normalize(final_m[0], dim=1)
            final_m = F.normalize(self.fc_m(final_m), dim=1)
            keys = torch.cat([final_m, self.queue.clone().detach().T], dim=0)
            # dequeue and enqueue
            self._dequeue_and_enqueue(final_m)

        return final, keys
