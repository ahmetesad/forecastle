from __future__ import annotations

import torch
from torch import nn


class CNN1DRegressor(nn.Module):
    def __init__(
        self,
        feature_count: int,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64]
        layers: list[nn.Module] = []
        input_channels = feature_count
        padding = kernel_size // 2
        for output_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(input_channels, output_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_channels = output_channels
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(input_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Conv1d expects channels first: [batch, features, time].
        encoded = self.encoder(inputs.transpose(1, 2))
        pooled = self.pool(encoded).squeeze(-1)
        return self.head(pooled)
