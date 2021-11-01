from typing import Any, Optional

import torch
from allennlp.modules.transformer.t5 import T5StackOutput
from allennlp.nn.util import add_positional_features
from torch import nn

from common.tensor_types import FloatT, IntT
from models.base_model import BaseModel
from models.grokking_model import GrokkingModel, GrokkingModelOutput
from modules.embedding import ScaledEmbedding
from modules.transformer_stack import TransformerStack


@BaseModel.register("transformer")
class Transformer(GrokkingModel):
    def __init__(
        self,
        num_symbols: int,
        stack: TransformerStack,
        tie_embeddings: bool = True,
        add_extra_input_tokens: bool = True,
        spectral_decoupling_coeff: float = 0.0,
        l2_regularization: float = 0.0,
        mle_cut_off_steps: int = -1,
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        num_embeddings = num_symbols
        if add_extra_input_tokens:
            opr_tok_id = num_symbols
            self.register_buffer(
                "opr_ids", torch.tensor([opr_tok_id], dtype=torch.long)
            )

            equal_tok_id = num_symbols + 1
            self.register_buffer(
                "equal_ids", torch.tensor([equal_tok_id], dtype=torch.long)
            )

            num_embeddings += 2

        self.encoder = stack
        self.hidden_size = self.encoder.hidden_size
        self.embed = ScaledEmbedding(num_embeddings, self.encoder.hidden_size)

        self.add_extra_input_tokens = add_extra_input_tokens
        self.l2_regularization = l2_regularization
        self.spectral_decoupling_coeff = spectral_decoupling_coeff
        self.mle_cut_off_steps = mle_cut_off_steps

        self.classifier = nn.Linear(
            self.hidden_size,
            num_embeddings,
            bias=False,
        )
        if tie_embeddings:
            self.classifier.weight = self.embed.weight

    def forward(
        self, sym_ids_1: IntT, sym_ids_2: IntT, labels: Optional[IntT] = None, **kwargs
    ) -> GrokkingModelOutput:
        if self.add_extra_input_tokens:
            input_ids = torch.stack(
                [
                    sym_ids_1,
                    torch.broadcast_to(self.opr_ids, sym_ids_1.size()),
                    sym_ids_2,
                    torch.broadcast_to(self.equal_ids, sym_ids_1.size()),
                ],
                dim=1,
            )
        else:
            input_ids = torch.stack([sym_ids_1, sym_ids_2], dim=1)

        input_embeds = self.embed(input_ids)
        input_embeds = add_positional_features(input_embeds)

        enc_output: T5StackOutput = self.encoder(
            inputs_embeds=input_embeds,
            output_attentions=True,
            output_all_hidden_states=True,
        )
        hidden = enc_output.last_hidden_state
        logits = self.classifier(hidden[:, -1, :])

        predictions = torch.argmax(logits, dim=1).long()

        loss: Optional[FloatT] = None
        mle_loss: Optional[FloatT] = None
        l2_loss = self.compute_l2_term()
        sd_loss = self.compute_spectral_decoupling_term(logits, labels)

        if labels is not None:
            if self.mle_cut_off_steps != -1 and self.global_step > self.mle_cut_off_steps:
                mle_loss = 0
            else:
                mle_loss = self.compute_loss(logits, labels)

            loss = sum([mle_loss, l2_loss, sd_loss])

        output = GrokkingModelOutput(
            logits=logits,
            loss=loss,
            mle_loss=mle_loss,
            l2_loss=l2_loss,
            sd_loss=sd_loss,
            predictions=predictions,
        )

        return output

    def compute_l2_term(self) -> FloatT:
        if self.l2_regularization == 0:
            return 0.0

        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += (param ** 2).sum()

        l2_loss *= self.l2_regularization

        return l2_loss

    def compute_spectral_decoupling_term(self, logits: FloatT, labels: IntT) -> FloatT:
        if self.spectral_decoupling_coeff == 0:
            return 0.

        norms = torch.norm(logits, p=2, dim=1)
        sd_term = torch.mean(norms)
        sd_term *= self.spectral_decoupling_coeff

        return sd_term
