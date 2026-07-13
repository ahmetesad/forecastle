from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ExperimentResult:
    run_dir: Path
    comparison_rows: list[dict[str, Any]]


@dataclass
class FitSummary:
    model: str
    fold: int
    best_val_loss: float | None = None
    epochs_ran: int = 0
    training_time_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    checkpoint_path: str = ""


@dataclass(frozen=True)
class ForecastRecord:
    model: str
    fold: int
    forecast_origin: str
    target_date: str
    horizon_step: int
    actual: float
    prediction: float
    actual_price: float
    prediction_price: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
