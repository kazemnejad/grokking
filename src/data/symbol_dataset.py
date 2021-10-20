from pathlib import Path
from typing import Optional, List, Any, Callable, Dict

import torch
from torch.utils.data import Dataset, random_split

from common import Registrable, ExperimentStage, Params
from data.base_dataset import BaseDataLoaderFactory, DatasetType


class Operation(Registrable):
    def __call__(self, sym1: int, sym2: int, prime: int) -> int:
        raise NotImplementedError()


@Operation.register("sum")
class SumOp(Operation):
    def __call__(self, sym1: int, sym2: int, prime: int) -> int:
        return (sym1 + sym2) % prime


class DatasetFromList(Dataset):
    def __init__(self, data: List[Any]):
        super(DatasetFromList, self).__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


@BaseDataLoaderFactory.register("symbols")
class SymbolsDataLoaderFactory(BaseDataLoaderFactory):
    def __init__(
        self,
        operation: Operation,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
        prime: int,
        random_split_seed: int,
        train_valid_percent: Optional[float] = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._generate_data(
            operation,
            prime,
            random_split_seed,
            train_valid_percent,
            x_end,
            x_start,
            y_end,
            y_start,
        )

    def _generate_data(
        self,
        operation,
        prime,
        random_split_seed,
        train_valid_percent,
        x_end,
        x_start,
        y_end,
        y_start,
    ):
        X = list(range(x_start, x_end))
        Y = list(range(y_start, y_end))
        XY = []
        for x in X:
            for y in Y:
                XY.append(
                    {
                        "sym1": x,
                        "sym2": y,
                        "labels": operation(x, y, prime),
                    }
                )
        train_len = int(train_valid_percent * len(XY))
        train_data, valid_data = random_split(
            XY,
            [train_len, len(XY) - train_len],
            generator=torch.Generator().manual_seed(random_split_seed),
        )
        self._train_data = list(train_data)
        self._valid_data = list(valid_data)

    def build_dataset(self, path: Path, stage: ExperimentStage) -> DatasetType:
        if stage == ExperimentStage.TRAINING:
            data = self._train_data
        elif stage == ExperimentStage.VALIDATION:
            data = self._valid_data
        elif stage == ExperimentStage.TEST:
            data = self._valid_data

        return DatasetFromList(data)

    @staticmethod
    def _collate(examples: List[Dict[str, Any]]) -> Dict[str, torch.LongTensor]:
        sym1 = [e["sym1"] for e in examples]
        sym2 = [e["sym2"] for e in examples]
        labels = [e["labels"] for e in examples]

        return {
            "sym_ids_1": torch.LongTensor(sym1),
            "sym_ids_2": torch.LongTensor(sym2),
            "labels": torch.LongTensor(labels),
        }

    def get_collate_fn(self, state: ExperimentStage) -> Callable:
        return self._collate

    def transform_line_to_instance(self, line: str, stage: ExperimentStage) -> Any:
        parts = line.strip().split()
        sym1 = int(parts[0])
        sym2 = int(parts[1])
        labels = None
        if len(parts) == 3:
            labels = int(parts[2])

        instance = {"sym1": sym1, "sym2": sym2}

        if labels is not None:
            instance["labels"] = labels

        return instance


if __name__ == "__main__":
    prime = 97
    dl_factory = SymbolsDataLoaderFactory.from_params(
        Params(
            {
                "operation": {"type": "sum"},
                "x_start": 0,
                "x_end": prime,
                "y_start": 0,
                "y_end": prime,
                "prime": prime,
                "random_split_seed": 384,
                "train_valid_percent": 0.5,
                "data_root": "data",
                "name": "symbol-sum",
                "split": "random",
            }
        )
    )

    stage = ExperimentStage.TRAINING
    ds = dl_factory.get_dataset(stage)
    print(ds)
    print(ds[0])

    dataloader = dl_factory.build(stage)
    dataloader = iter(dataloader)

    batch = next(dataloader)

    print(batch)
