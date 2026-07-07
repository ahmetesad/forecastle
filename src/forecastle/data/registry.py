from __future__ import annotations

from collections.abc import Callable

from forecastle.config import DatasetConfig, TrainingConfig
from forecastle.data.csv_dataset import build_csv_datamodule
from forecastle.data.types import DataModule

DatasetFactory = Callable[[DatasetConfig, TrainingConfig, int], DataModule]

DATASET_REGISTRY: dict[str, DatasetFactory] = {
    "csv": build_csv_datamodule,
}


def register_dataset(source: str, factory: DatasetFactory) -> None:
    DATASET_REGISTRY[source] = factory


def build_datamodule(
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    seed: int,
) -> DataModule:
    try:
        factory = DATASET_REGISTRY[dataset_config.source]
    except KeyError as error:
        available = ", ".join(sorted(DATASET_REGISTRY))
        msg = f"Unknown dataset source '{dataset_config.source}'. Available sources: {available}"
        raise ValueError(msg) from error
    return factory(dataset_config, training_config, seed)
