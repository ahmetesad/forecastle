from __future__ import annotations

import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        feature_count: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 64]
        input_size = sequence_length * feature_count
        layers: list[nn.Module] = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(current_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_size = hidden_size
        layers.append(nn.Linear(current_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        return self.network(inputs.reshape(batch_size, -1))
