from dataclasses import dataclass
from typing import Optional, Any, Dict

import torch
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT

from common.tensor_types import FloatT, IntT
from models.base_model import BaseModel

from torch import nn, Tensor


@dataclass
class GrokkingModelOutput:
    logits: Optional[FloatT] = None
    loss: Optional[FloatT] = None
    predictions: Optional[IntT] = None
    mle_loss: Optional[FloatT] = None
    l2_loss: Optional[FloatT] = None
    sd_loss: Optional[FloatT] = None


class GrokkingModel(BaseModel):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.loss_fct = nn.CrossEntropyLoss()

    def register_metrics(self):
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.Accuracy(),
            },
            prefix="train_",
        )

        self.valid_metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.Accuracy(),
            },
            prefix="valid_",
        )

        self.test_metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.Accuracy(),
            },
            prefix="test_",
        )

    def compute_loss(self, logits: FloatT, labels: IntT):
        return self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    def forward(
            self, sym_ids_1: IntT, sym_ids_2: IntT, labels: Optional[IntT] = None, **kwargs
    ) -> GrokkingModelOutput:
        raise NotImplemented

    def training_step(
            self, batch: Dict[str, torch.Tensor], batch_idx, *args, **kwargs
    ) -> STEP_OUTPUT:
        outputs: GrokkingModelOutput = self(
            sym_ids_1=batch["sym_ids_1"],
            sym_ids_2=batch["sym_ids_2"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if outputs.mle_loss is not None:
            self.log(
                "train_mle_loss",
                outputs.mle_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        if outputs.l2_loss is not None:
            self.log(
                "train_l2_loss",
                outputs.l2_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        if outputs.sd_loss is not None:
            self.log(
                "train_sd_loss",
                outputs.sd_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        preds = outputs.predictions
        self.train_metrics(preds, batch["labels"])
        self.log_dict(
            self.train_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(
            self, batch: Dict[str, torch.Tensor], batch_idx, *args, **kwargs
    ) -> Optional[STEP_OUTPUT]:
        outputs: GrokkingModelOutput = self(
            sym_ids_1=batch["sym_ids_1"],
            sym_ids_2=batch["sym_ids_2"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        preds = outputs.predictions

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if outputs.mle_loss is not None:
            self.log(
                "valid_mle_loss",
                outputs.mle_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.valid_metrics(preds, batch["labels"])
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, prog_bar=True)

        return None

    def test_step(
            self, batch: Dict[str, Tensor], batch_idx, *args, **kwargs
    ) -> Optional[STEP_OUTPUT]:
        outputs: GrokkingModelOutput = self(
            sym_ids_1=batch["sym_ids_1"],
            sym_ids_2=batch["sym_ids_2"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = outputs.predictions
        self.test_metrics(preds, batch["labels"])
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

        return None

    def predict_step(
            self,
            batch: Dict[str, Tensor],
            batch_idx: int,
            dataloader_idx: Optional[int] = None,
    ) -> Any:
        outputs: GrokkingModelOutput = self(
            sym_ids_1=batch["sym_ids_1"],
            sym_ids_2=batch["sym_ids_2"],
            labels=batch.get("labels", None),
        )

        return outputs
