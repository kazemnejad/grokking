import abc
from dataclasses import dataclass
from typing import Optional, Any, Dict, Callable

import torch
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
from torch.utils.data import Dataset

from common.tensor_types import FloatT, IntT
from models.base_model import BaseModel


@dataclass
class GrokkingModelOutput:
    logits: Optional[FloatT] = None
    predictions: Optional[IntT] = None
    # loss: Optional[FloatT] = None
    mle_loss: Optional[FloatT] = None
    l2_loss: Optional[FloatT] = None
    sd_loss: Optional[FloatT] = None

    @property
    def loss(self) -> Optional[FloatT]:
        if self.mle_loss is None:
            return None

        return sum(
            l for l in [self.mle_loss, self.l2_loss, self.sd_loss] if l is not None
        )


class GrokkingModel(BaseModel):
    def __init__(
        self,
        spectral_decoupling_coeff: float = 0.0,
        l2_regularization: float = 0.0,
        mle_cut_off_steps: int = -1,
        log_grads: bool = False,
        log_loss_histogram: bool = False,
        log_loss_stats: bool = False,
        train_ds: Optional[Dataset] = None,
        collate_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.loss_fct = nn.CrossEntropyLoss()
        self.l2_regularization = l2_regularization
        self.spectral_decoupling_coeff = spectral_decoupling_coeff
        self.mle_cut_off_steps = mle_cut_off_steps
        self.log_grads = log_grads
        self.train_ds = train_ds
        self.collate_fn = collate_fn
        self.log_loss_histogram = log_loss_histogram
        self.log_loss_stats = log_loss_stats

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

    def compute_mle_loss(self, logits: FloatT, labels: IntT):
        return self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    def compute_l2_term(self) -> Optional[FloatT]:
        if self.l2_regularization == 0:
            return None

        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += (param ** 2).sum()

        l2_loss *= self.l2_regularization

        return l2_loss

    def compute_spectral_decoupling_term(
        self, logits: FloatT, labels: IntT
    ) -> Optional[FloatT]:
        if self.spectral_decoupling_coeff == 0:
            return None

        norms = torch.norm(logits, p=2, dim=1)
        sd_term = torch.mean(norms)
        sd_term *= self.spectral_decoupling_coeff

        return sd_term

    @abc.abstractmethod
    def compute_logits(
        self, sym_ids_1: IntT, sym_ids_2: IntT, labels: Optional[IntT] = None, **kwargs
    ) -> FloatT:
        raise NotImplemented

    def forward(
        self, sym_ids_1: IntT, sym_ids_2: IntT, labels: Optional[IntT] = None, **kwargs
    ) -> GrokkingModelOutput:
        logits = self.compute_logits(
            sym_ids_1=sym_ids_1, sym_ids_2=sym_ids_2, labels=labels, **kwargs
        )

        predictions = torch.argmax(logits, dim=1).long()

        l2_loss = self.compute_l2_term()
        sd_loss = self.compute_spectral_decoupling_term(logits, labels)
        mle_loss: Optional[FloatT] = None
        if labels is not None:
            mle_loss = self.compute_mle_loss(logits, labels)

        return GrokkingModelOutput(
            logits=logits,
            predictions=predictions,
            mle_loss=mle_loss,
            l2_loss=l2_loss,
            sd_loss=sd_loss,
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx, *args, **kwargs
    ) -> STEP_OUTPUT:
        outputs: GrokkingModelOutput = self(
            sym_ids_1=batch["sym_ids_1"],
            sym_ids_2=batch["sym_ids_2"],
            labels=batch["labels"],
        )

        if self.mle_cut_off_steps != -1 and self.global_step > self.mle_cut_off_steps:
            outputs.mle_loss *= 0

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

        if self.log_grads:
            losses_names = [n for n in dir(outputs) if n.endswith("_loss")]
            for l_name in losses_names:
                l = getattr(outputs, l_name)
                if l is not None:
                    self.optimizers().zero_grad()
                    l.backward(retain_graph=True)

                    norms = []
                    for p in self.parameters():
                        if p.grad is not None:
                            norms.append(torch.abs(p.grad).mean().detach().item())

                    norm = sum(norms) / len(norms)
                    self.log(
                        f"{l_name}_avg_grad_norm",
                        norm,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                    )
            self.optimizers().zero_grad()

        preds = outputs.predictions
        self.train_metrics(preds, batch["labels"])
        self.log_dict(
            self.train_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        if self.log_loss_histogram and (self.trainer.current_epoch + 1) % 10 == 0:
            old_mode = self.training
            self.eval()

            with torch.autograd.no_grad():
                losses = []
                for data_instance in self.train_ds:
                    batch = self.collate_fn([data_instance])
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    output: GrokkingModelOutput = self(**batch)
                    losses.append(output.mle_loss.detach().item())

                self.logger.experiment.log_histogram_3d(
                    losses,
                    name="loss",
                    step=self.global_step,
                )

            self.train(old_mode)

        if self.log_loss_stats and (self.trainer.current_epoch + 1) % 10 == 0:
            old_mode = self.training
            self.eval()

            with torch.autograd.no_grad():
                losses = []
                for data_instance in self.train_ds:
                    batch = self.collate_fn([data_instance])
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    output: GrokkingModelOutput = self(**batch)
                    l = output.mle_loss.detach().item()
                    losses.append((data_instance, l, output))

                non_zero_losses = [d for d in losses if d[1] != 0]
                self.log("num_non_zero_losses", len(non_zero_losses), on_epoch=True)
                avg = sum(map(lambda x: x[1], non_zero_losses)) / len(non_zero_losses)
                self.log("avg_non_zero_losses", avg, on_epoch=True)

                incorrect_losses = [
                    d
                    for d in losses
                    if d[0]["labels"] != d[2].predictions[0].detach().item()
                ]
                self.log("num_incorrect", len(incorrect_losses), on_epoch=True)

                avg = sum(map(lambda x: x[1], incorrect_losses)) / len(non_zero_losses)
                self.log("avg_incorrect_losses", avg, on_epoch=True)

                k = 20
                top_k = sorted(losses, key=lambda x: x[1], reverse=True)[:k]

                log_str = ""
                for i, (data_instance, loss, output) in enumerate(top_k):
                    template = (
                        f"# {i + 1}\n"
                        f"Loss= {loss}\n"
                        f"Input: {data_instance['sym1']} o {data_instance['sym2']} =\n"
                        f"Gold:  {data_instance['labels']}\n"
                        f"Model: {output.predictions[0].detach().item()}\n"
                        f"\n\n"
                    )
                    log_str += template

                self.logger.experiment.log_text(
                    log_str,
                    step=self.global_step,
                )

            self.train(old_mode)

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
