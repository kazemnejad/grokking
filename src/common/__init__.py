from enum import Enum


from allennlp.common import FromParams, Lazy, Params, Registrable
from allennlp.common.checks import ConfigurationError

# from .from_params import FromParams, ConfigurationError
# from .lazy import Lazy
# from .params import Params
# from .registrable import Registrable

assert FromParams
assert Lazy
assert Params
assert Registrable
assert ConfigurationError

class ExperimentStage(Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2
    PREDICTION = 3
