from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from common import Registrable, Lazy
from modules.lr_scheduler import LearningRateScheduler
from modules.optimizer import BaseOptimizer


class BaseModel(pl.LightningModule, Registrable):
    def __init__(
        self,
        optimizer: Optional[Lazy[BaseOptimizer]] = None,
        lr_scheduler: Optional[Lazy[LearningRateScheduler]] = None,
        lr_schedule_interval: str = "step",
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        self.register_metrics()

        self.lazy_optimizer = optimizer or Lazy(BaseOptimizer, constructor_extras={})
        self.lazy_lr_scheduler = lr_scheduler
        self.lr_schedule_interval = lr_schedule_interval

    def configure_optimizers(self):
        optim = self.lazy_optimizer.construct(model_params=self.parameters())
        output = {
            "optimizer": optim,
        }
        if self.lazy_lr_scheduler:
            scheduler = self.lazy_lr_scheduler.construct(optimizer=optim)
            output["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": self.lr_schedule_interval,
            }

        return output

    def configure_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [lr_monitor]
