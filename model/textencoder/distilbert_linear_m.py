import copy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .distilbert import DistilbertEncoderBase


class DistilbertLinearMomentumEncoder(DistilbertEncoderBase):
    def __init__(self,
                 modelpath: str,
                 finetune: bool = False,
                 embed_size: int = 1024,
                 momentum: float = 0.995,
                 queue_size: int = 65536,
                 **kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)

        encoded_dim = self.text_encoded_dim

        self.fc = nn.Linear(encoded_dim, embed_size)

        # create momentum models
        self.momentum = momentum
        self.queue_size = queue_size
        self.text_model_m = copy.deepcopy(self.text_model)
        self.fc_m = nn.Linear(encoded_dim, embed_size)
        self.model_pairs = [[self.text_model, self.text_model_m],
                            [self.fc, self.fc_m], ]
        self.copy_params()

        # create the queue
        self.register_buffer("queue", torch.randn(embed_size, queue_size))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.learning_rates_x = ['text_model', 'text_model_m']

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

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
        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
        text_encoded = self.get_last_hidden_state(
            texts, return_mask=False)
        # extract cls token from BERT and norm
        emb_token = F.normalize(text_encoded[:, 0, :], dim=1)
        q = F.normalize(self.fc(emb_token), dim=1)

        with torch.no_grad():
            self._momentum_update()
            encoded_inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True)
            output = self.text_model_m(
                **encoded_inputs.to(self.text_model_m.device))
            text_encoded = output.last_hidden_state
            emb_token_m = F.normalize(text_encoded[:, 0, :], dim=1)
            x_m = F.normalize(self.fc_m(emb_token), dim=1)
            keys = torch.cat([x_m, self.queue.clone().detach().T], dim=0)
            # dequeue and enqueue
            self._dequeue_and_enqueue(x_m)
        return q, keys
