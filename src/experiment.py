
from pytorch_lightning.utilities.cloud_io import get_filesystem

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any, List, Union

from pytorch_lightning import seed_everything, Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from common import Params, Lazy, gpu_utils, py_utils, FromParams, ExperimentStage
from common.config import HoconConfig
from data.base_dataset import BaseDataLoaderFactory
from models.base_model import BaseModel

logger = logging.getLogger("app")


class DIModelCheckpoint(ModelCheckpoint, FromParams):
    def _is_valid_monitor_key(self, metrics) -> bool:
        return (
            self.monitor in metrics
            or self.monitor in ("step", "epoch")
            or len(metrics) == 0
        )

    def __init_ckpt_dir(self, dirpath: Optional[Union[str, Path]], filename: Optional[str]) -> None:
        self._fs = get_filesystem(str(dirpath) if dirpath else "")

        # if self.save_top_k != 0 and dirpath is not None and self._fs.isdir(dirpath) and len(self._fs.ls(dirpath)) > 0:
        #     rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")

        if dirpath and self._fs.protocol == "file":
            dirpath = os.path.realpath(dirpath)

        self.dirpath = dirpath
        self.filename = filename


class DICommetLogger(CometLogger, FromParams):
    pass


class DITrainer(Trainer, FromParams):
    pass


class Experiment(FromParams):
    def __init__(
        self,
        model: Lazy[BaseModel],
        dataset: Lazy[BaseDataLoaderFactory],
        exp_name: str,
        project_name: str,
        trainer: Optional[Lazy[DITrainer]] = None,
        directory: Optional[str] = "experiments",
        global_vars: Optional[Dict[str, Any]] = None,
        checkpoint: Optional[Lazy[DIModelCheckpoint]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        config_filenames: Optional[List[str]] = None,
    ):
        self.lazy_model = model
        self.lazy_dataset = dataset
        self.exp_name = exp_name
        self.project_name = project_name

        exp_root = Path(directory) / self.exp_name
        exp_root.mkdir(parents=True, exist_ok=True)
        self.exp_root = exp_root

        self.global_vars = global_vars or {"seed": 123}
        self.debug_mode = self.global_vars.get("debug_mode", False)
        self.config_dict = config_dict

        self.lazy_checkpoint = checkpoint or Lazy(
            DIModelCheckpoint,
            contructor_extras={
                "every_n_train_steps": 10,
                "save_top_k": 10,
            },
        )
        self.lazy_trainer = trainer or Lazy(
            DITrainer,
            constructor_extras={
                "max_steps": 100,
                "log_every_n_steps": 1,
                "val_check_interval": 10,
                "auto_select_gpus": True,
                "profiler": "advanced",
            },
        )

        # os.environ["PL_FAULT_TOLERANT_TRAINING"] = "1"

        seed_everything(self.global_vars["seed"])

        self.dl_factory = self.lazy_dataset.construct()
        self.write_meta_data()
        self.logger.log_hyperparams(config_dict)

    @classmethod
    def from_hocon_config(cls, config: HoconConfig) -> "Experiment":
        exp_name = py_utils.unique_experiment_name(config)
        exp_root = Path(config.get("directory", "experiments")) / exp_name
        exp_root.mkdir(parents=True, exist_ok=True)
        config.put("exp_name", exp_name)

        config.to_file(str(exp_root / "config.conf"))
        config.to_file(str(exp_root / "config.json"), "json")

        params = Params({"config_dict": config.to_dict(), **config.to_dict()})
        return Experiment.from_params(params)

    def write_meta_data(self):
        metadata = {"exp_name": self.exp_name, "gpus_info": gpu_utils.get_cuda_info()}
        with open(self.exp_root / "metadata.json", "w") as f:
            f.write(json.dumps(metadata, indent=4, sort_keys=True))

    def create_checkpoint_callback(self) -> ModelCheckpoint:
        checkpoint_dir = self.exp_root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.lazy_checkpoint.construct(
            dirpath=str(checkpoint_dir), filename="{step}"
        )

    def get_last_checkpoint_path(self) -> Optional[Path]:
        checkpoint_dir = self.exp_root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoints = list(checkpoint_dir.iterdir())

        def ckpt_num(p: Path):
            return int(p.name.split("step=")[1].split(".ckpt")[0])

        try:
            last_ckpt = max(checkpoints, key=ckpt_num)
            return last_ckpt
        except:
            return None

    @property
    def logger(self) -> CometLogger:
        if not hasattr(self, "_logger"):
            self._logger = CometLogger(
                api_key=os.environ.get("COMET_API_KEY", None),
                workspace=os.environ.get("COMET_WORKSPACE", None),
                save_dir=str(self.exp_root / "logs"),
                project_name=self.project_name,
                experiment_name=self.exp_name,
                experiment_key=os.environ.get("COMET_EXPERIMENT_KEY", None),
                offline=self.debug_mode,
                log_code=False,
                log_graph=True,
                log_env_cpu=True,
                log_env_gpu=True,
                log_git_patch=False,
                log_env_details=True,
                log_git_metadata=False,
                log_env_host=True,
                parse_args=False,
            )

        return self._logger

    def create_trainer(self, restore_last_ckpt: bool) -> Trainer:
        if restore_last_ckpt:
            resume_ckpt = self.get_last_checkpoint_path()
            if resume_ckpt is None:
                logger.warning(
                    f'No checkpoint found in {str(self.exp_root / "checkpoints")}. Initializing from scratch...'
                )
        else:
            resume_ckpt = None

        ckpt_callback = self.create_checkpoint_callback()
        trainer = self.lazy_trainer.construct(
            callbacks=[ckpt_callback],
            logger=self.logger,
            resume_from_checkpoint=resume_ckpt,
        )
        return trainer

    def create_dl_factory(self):
        return self.lazy_dataset.construct()

    def train(self, from_scratch: bool = False):
        model = self.lazy_model.construct()
        ModelSummary(model)
        self.logger.log_graph(model)

        train_dl = self.dl_factory.build(ExperimentStage.TRAINING)
        try:
            valid_dl = self.dl_factory.build(ExperimentStage.VALIDATION)
        except:
            valid_dl = None

        trainer = self.create_trainer(restore_last_ckpt=not from_scratch)
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    def validate(self, split="valid"):
        model = self.lazy_model.construct()

        import torch
        ckpt_path = self.get_last_checkpoint_path()
        if ckpt_path:
            ckpt = torch.load(str(ckpt_path))
            model.load_state_dict(ckpt["state_dict"])
            logger.info(f"Checkpoint loaded from {str(ckpt_path)}.")
        ModelSummary(model)

        if split == "test":
            stage = ExperimentStage.TEST
        elif split == "valid":
            stage = ExperimentStage.VALIDATION
        else:
            raise ValueError(f"Invalid split: {split}")

        dl = self.dl_factory.build(stage)
        trainer = self.create_trainer(restore_last_ckpt=True)
        trainer.test(model, dl)
