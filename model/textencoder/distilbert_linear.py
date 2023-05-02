from typing import List, Tuple

import torch.nn.functional as F
from torch import nn, Tensor

from .distilbert import DistilbertEncoderBase


class DistilbertLinearEncoder(DistilbertEncoderBase):
    def __init__(self,
                 modelpath: str,
                 finetune: bool = False,
                 embed_size: int = 1024,
                 **kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)

        encoded_dim = self.text_encoded_dim
        self.fc = nn.Linear(encoded_dim, embed_size)

    def forward(self, texts: List[str]) -> Tuple[Tensor, None]:
        text_encoded = self.get_last_hidden_state(
            texts, return_mask=False)
        # extract cls token from BERT and norm
        emb_token = F.normalize(text_encoded[:, 0, :], dim=1)
        return F.normalize(self.fc(emb_token), dim=1), None
