from __future__ import annotations

import torch
from torch import nn


class LSTMGRURegressor(nn.Module):
    """Serial LSTM -> GRU -> linear regression head."""

    def __init__(
        self,
        feature_count: int,
        lstm_hidden_size: int = 64,
        gru_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        gru_num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_count,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.between_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=lstm_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_outputs, _ = self.lstm(inputs)
        gru_outputs, _ = self.gru(self.between_dropout(lstm_outputs))
        return self.head(self.head_dropout(gru_outputs[:, -1, :]))


class CNNLSTMRegressor(nn.Module):
    """Same-length Conv1d encoder -> LSTM -> linear regression head."""

    def __init__(
        self,
        feature_count: int,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            msg = "cnn_lstm kernel_size must be a positive odd integer."
            raise ValueError(msg)
        channels = channels or [32, 64]
        if not channels or any(channel < 1 for channel in channels):
            msg = "cnn_lstm channels must contain positive integers."
            raise ValueError(msg)

        convolution_layers: list[nn.Module] = []
        input_channels = feature_count
        padding = kernel_size // 2
        for output_channels in channels:
            convolution_layers.extend(
                [
                    nn.Conv1d(input_channels, output_channels, kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_channels = output_channels
        self.convolution = nn.Sequential(*convolution_layers)
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.convolution(inputs.transpose(1, 2)).transpose(1, 2)
        outputs, _ = self.lstm(encoded)
        return self.head(self.head_dropout(outputs[:, -1, :]))
