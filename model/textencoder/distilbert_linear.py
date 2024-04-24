from typing import List, Tuple

import torch.nn.functional as F
from torch import nn, Tensor

from .distilbert import DistilbertEncoderBase


class DistilbertLinearEncoder(DistilbertEncoderBase):
    def __init__(self,
                 modelpath: str,
                 finetune: bool = False,
                 pre_norm: bool = False,
                 embed_size: int = 1024,
                 only_return_cls_token:bool = True,
                 **kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)

        self.only_return_cls_token = only_return_cls_token
        encoded_dim = self.text_encoded_dim
        self.fc = nn.Linear(encoded_dim, embed_size)
        self.pre_norm = pre_norm

    def forward(self, texts: List[str]) -> Tuple[Tensor, None]:
        text_encoded, mask = self.get_last_hidden_state(
            texts, return_mask=True)
        
        if self.only_return_cls_token:
            # extract cls token from BERT and norm
            cls_token = F.normalize(text_encoded[:, 0, :], dim=1)
            # normalization in the joint embedding space
            return F.normalize(self.fc(cls_token), dim=1), None
        else:
            # return sequence features
            features = self.fc(text_encoded)
            features = F.normalize(features, dim=-1)
            return features, mask
