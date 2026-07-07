from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable


RecurrentKind = Literal["rnn", "lstm", "gru"]


class RecurrentRegressor(nn.Module):
    def __init__(
        self,
        kind: RecurrentKind,
        feature_count: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        recurrent_cls: Callable[..., nn.Module]
        if kind == "rnn":
            recurrent_cls = nn.RNN
        elif kind == "lstm":
            recurrent_cls = nn.LSTM
        elif kind == "gru":
            recurrent_cls = nn.GRU
        else:
            msg = f"Unknown recurrent model kind: {kind}"
            raise ValueError(msg)

        self.recurrent = recurrent_cls(
            input_size=feature_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(inputs)
        last_output = outputs[:, -1, :]
        return self.head(last_output)


class RNNRegressor(RecurrentRegressor):
    def __init__(
        self,
        feature_count: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__("rnn", feature_count, hidden_size, num_layers, dropout)


class LSTMRegressor(RecurrentRegressor):
    def __init__(
        self,
        feature_count: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__("lstm", feature_count, hidden_size, num_layers, dropout)


class GRURegressor(RecurrentRegressor):
    def __init__(
        self,
        feature_count: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__("gru", feature_count, hidden_size, num_layers, dropout)
