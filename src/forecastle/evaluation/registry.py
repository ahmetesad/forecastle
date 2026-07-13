from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from forecastle.config import validate_recursive_features
from forecastle.evaluation.holdout import run_holdout
from forecastle.evaluation.walk_forward import run_walk_forward

if TYPE_CHECKING:
    from forecastle.config import AppConfig
    from forecastle.evaluation.types import ExperimentResult

EvaluationRunner = Callable[["AppConfig"], "ExperimentResult"]

EVALUATION_REGISTRY: dict[str, EvaluationRunner] = {
    "holdout": run_holdout,
    "walk_forward": run_walk_forward,
}


def run_evaluation(config: AppConfig) -> ExperimentResult:
    validate_recursive_features(config.dataset, config.forecasting)
    try:
        runner = EVALUATION_REGISTRY[config.evaluation.strategy]
    except KeyError as error:
        available = ", ".join(sorted(EVALUATION_REGISTRY))
        msg = f"Unknown evaluation strategy '{config.evaluation.strategy}': {available}"
        raise ValueError(msg) from error
    return runner(config)
