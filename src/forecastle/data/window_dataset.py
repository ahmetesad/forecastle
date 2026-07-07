from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import numpy as np


class WindowedTimeSeriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if len(features) != len(targets):
            msg = "Features and targets must contain the same number of windows."
            raise ValueError(msg)
        self._features = torch.as_tensor(features, dtype=torch.float32)
        self._targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._features[index], self._targets[index]
