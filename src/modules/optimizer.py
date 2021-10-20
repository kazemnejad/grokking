from typing import Any

import torch
from common import Registrable

class BaseOptimizer(Registrable):
    pass


@BaseOptimizer.register("adam")
class DIAdam(torch.optim.Adam, BaseOptimizer):
    def __init__(self, model_params: Any, **kwargs):
        super(DIAdam, self).__init__(params=model_params, **kwargs)

@BaseOptimizer.register("adamW")
class DIAdamW(torch.optim.AdamW, BaseOptimizer):
    def __init__(self, model_params: Any, **kwargs):
        super(DIAdamW, self).__init__(params=model_params, **kwargs)