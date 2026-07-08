from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from forecastle.models.cnn import CNN1DRegressor
from forecastle.models.dnfs import DNFSRegressor
from forecastle.models.mlp import MLPRegressor
from forecastle.models.recurrent import GRURegressor, LSTMRegressor, RNNRegressor

ModelFactory = Callable[[int, int, dict[str, Any]], nn.Module]


def _build_mlp(sequence_length: int, feature_count: int, params: dict[str, Any]) -> nn.Module:
    return MLPRegressor(sequence_length=sequence_length, feature_count=feature_count, **params)


def _build_rnn(_sequence_length: int, feature_count: int, params: dict[str, Any]) -> nn.Module:
    return RNNRegressor(feature_count=feature_count, **params)


def _build_lstm(_sequence_length: int, feature_count: int, params: dict[str, Any]) -> nn.Module:
    return LSTMRegressor(feature_count=feature_count, **params)


def _build_gru(_sequence_length: int, feature_count: int, params: dict[str, Any]) -> nn.Module:
    return GRURegressor(feature_count=feature_count, **params)


def _build_cnn1d(_sequence_length: int, feature_count: int, params: dict[str, Any]) -> nn.Module:
    return CNN1DRegressor(feature_count=feature_count, **params)


def _build_dnfs(sequence_length: int, feature_count: int, params: dict[str, Any]) -> nn.Module:
    return DNFSRegressor(sequence_length=sequence_length, feature_count=feature_count, **params)


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "mlp": _build_mlp,
    "rnn": _build_rnn,
    "lstm": _build_lstm,
    "gru": _build_gru,
    "cnn1d": _build_cnn1d,
    "dnfs": _build_dnfs,
}


def build_model(
    name: str,
    sequence_length: int,
    feature_count: int,
    params: dict[str, Any] | None = None,
) -> nn.Module:
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as error:
        msg = f"Unknown model '{name}'. Available models: {', '.join(list_models())}"
        raise ValueError(msg) from error
    return factory(sequence_length, feature_count, params or {})


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY)
