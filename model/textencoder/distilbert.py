import os
from typing import List, Union

import torch.nn as nn
from torch import Tensor


class DistilbertEncoderBase(nn.Module):
    def __init__(self, modelpath: str,
                 finetune: bool = False) -> None:
        super(DistilbertEncoderBase, self).__init__()
        self.finetune = finetune

        from transformers import AutoTokenizer, AutoModel
        from transformers import logging
        logging.set_verbosity_error()
        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)
        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        self.learning_rates_x = ['text_model']

        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            # Don't put the model in
            if module == self.text_model and not self.finetune:
                continue
            module.train(mode)
        return self

    def get_last_hidden_state(self, texts: List[str],
                              return_mask: bool = False
                              ):
        encoded_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device))
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, encoded_inputs.attention_mask.to(dtype=bool)
