from comet_ml import ExistingExperiment
assert ExistingExperiment

import logging
import sys

import fire

from common import py_utils
from common.config import HoconConfig
from experiment import Experiment

logger = logging.getLogger("app")
LOG_FORMAT = "%(levelname)s:%(name)-5s %(message)s"
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(py_utils.NewLineFormatter(LOG_FORMAT))
logger.addHandler(handler)

pl_logger = logging.getLogger("pytorch_lightning")

pl_logger.removeHandler(pl_logger.handlers[0])
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(py_utils.NewLineFormatter("%(levelname)-9s %(message)s"))
pl_logger.addHandler(handler)



class EntryPoint(object):
    def __init__(self, configs: str):
        filenames = list(map(lambda x: x.strip(), configs.split(",")))
        config = HoconConfig.from_files(filenames)

        logger.info(f"# configs: {filenames}")
        logger.info(f"----Config----\n{config}\n--------------")

        config.put("config_filenames", filenames)

        self._config = config

        self._exp = Experiment.from_hocon_config(config)

    def generate_exp_name(self):
        print("==== Unique Experiment Name: ====")
        print(py_utils.unique_experiment_name(self._config))

    def __getattr__(self, attr):
        if attr in self.__class__.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self._exp, attr)

    def __dir__(self):
        return sorted(set(super().__dir__() + self._exp.__dir__()))


if __name__ == "__main__":
    fire.Fire(EntryPoint)
