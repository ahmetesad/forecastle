from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=True)


def write_predictions(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_true: np.ndarray | None = None,
    price_pred: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"actual": y_true.reshape(-1), "prediction": y_pred.reshape(-1)})
    if price_true is not None and price_pred is not None:
        frame["actual_price"] = price_true.reshape(-1)
        frame["prediction_price"] = price_pred.reshape(-1)
    frame.to_csv(path, index=False)


def plot_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(y_true.reshape(-1), label="Actual", linewidth=1.5)
    axis.plot(y_pred.reshape(-1), label="Prediction", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Test window")
    axis.set_ylabel("Target")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_comparison(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows).sort_values("rmse")
    frame.to_csv(run_dir / "comparison.csv", index=False)
    (run_dir / "comparison.md").write_text(dataframe_to_markdown(frame), encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [_format_markdown_value(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
