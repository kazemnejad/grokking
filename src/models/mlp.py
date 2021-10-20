from typing import Any, Optional

import torch
from allennlp.modules import FeedForward
from torch import nn

from common import Lazy
from common.tensor_types import IntT, FloatT
from models.base_model import BaseModel
from models.grokking_model import GrokkingModel, GrokkingModelOutput


@BaseModel.register("mlp")
class MLP(GrokkingModel):
    def __init__(
            self,
            embedding_dim: int,
            num_symbols: int,
            hidden_layers: Lazy[FeedForward],
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.embed = nn.Embedding(num_symbols, embedding_dim)
        self.encoder = hidden_layers.construct(input_dim=2 * embedding_dim)
        self.classifier = nn.Linear(self.encoder.get_output_dim(), num_symbols)

    def forward(
            self, sym_ids_1: IntT, sym_ids_2: IntT, labels: Optional[IntT] = None, **kwargs
    ) -> GrokkingModelOutput:
        sym_embed_1 = self.embed(sym_ids_1)
        sym_embed_2 = self.embed(sym_ids_2)

        inputs = torch.cat([sym_embed_1, sym_embed_2], dim=1)
        hidden = self.encoder(inputs)
        logits = self.classifier(hidden)

        predictions = torch.argmax(logits, dim=1).long()

        loss: Optional[FloatT] = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        output = GrokkingModelOutput(logits=logits, loss=loss, predictions=predictions)

        return output
