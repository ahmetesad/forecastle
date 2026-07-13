from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forecastle.config import AppConfig
    from forecastle.evaluation.types import ExperimentResult


def run_holdout(config: AppConfig) -> ExperimentResult:
    if config.forecasting.strategy == "recursive":
        from forecastle.evaluation.recursive_holdout import run_recursive_holdout

        return run_recursive_holdout(config)

    from forecastle.evaluation.direct_holdout import run_direct_holdout

    return run_direct_holdout(config)
