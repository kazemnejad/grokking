import math

import torch.optim
from torch.optim.lr_scheduler import LambdaLR

from common import Registrable

class LearningRateScheduler(LambdaLR, Registrable):
    pass

@LearningRateScheduler.register("inverse_root_with_warmup")
class InverseRootWithWarmupScheduler(LearningRateScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        hidden_size: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """Initialize configuration of the learning rate schedule.
        Args:
          initial_learning_rate: A float, the initial learning rate.
          hidden_size: An integer, the model dimension in the hidden layers.
          warmup_steps: An integer, the number of steps required for linear warmup.
        """
        self.hidden_size = hidden_size
        self.warmup_steps = warmup_steps
        self.warmup_steps_tensor = warmup_steps

        def lr_fn(global_step):
            global_step = global_step
            lr_coeff = self.hidden_size ** -0.5

            # Apply linear warmup
            if global_step < self.warmup_steps:
                lr_coeff *= global_step / self.warmup_steps_tensor
            if global_step >= self.warmup_steps:
                lr_coeff /= math.sqrt(global_step)

            return lr_coeff

        super(InverseRootWithWarmupScheduler, self).__init__(
            optimizer, lr_fn, last_epoch
        )


@LearningRateScheduler.register("constant_with_warmup")
class ConstantWithWarmupScheduler(LearningRateScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """Initialize configuration of the learning rate schedule.
        Args:
          warmup_steps: An integer, the number of steps required for linear warmup.
        """
        self.warmup_steps = warmup_steps

        def lr_fn(global_step):
            lr_coeff = 1

            # Apply linear warmup
            if global_step < self.warmup_steps:
                lr_coeff = global_step / self.warmup_steps

            return lr_coeff

        super(ConstantWithWarmupScheduler, self).__init__(
            optimizer, lr_fn, last_epoch
        )


@LearningRateScheduler.register("kazemink_with_warmup")
class KazeminkLRScheduler(LearningRateScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        saturation_step: int,
        kazemink_coeff: int,
        last_epoch: int = -1,
    ):
        """Initialize configuration of the learning rate schedule.
        Args:
          warmup_steps: An integer, the number of steps required for linear warmup.
        """
        self.saturation_step = saturation_step
        self.warmup_steps = warmup_steps
        self.kazemink_coeff = kazemink_coeff

        def lr_fn(global_step):
            lr_coeff = 1

            # Apply linear warmup
            if global_step < self.warmup_steps:
                lr_coeff = global_step / self.warmup_steps

            if global_step > self.saturation_step:
                lr_coeff = self.kazemink_coeff

            return lr_coeff

        super(KazeminkLRScheduler, self).__init__(
            optimizer, lr_fn, last_epoch
        )