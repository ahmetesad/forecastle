from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from forecastle.artifacts import write_dataframe
from forecastle.plotting import plt

if TYPE_CHECKING:
    from pathlib import Path

    from forecastle.evaluation.forecasters import FittedForecaster
    from forecastle.evaluation.types import ForecastRecord


def drain_rule_activation_rows(
    forecaster: FittedForecaster,
    records: list[ForecastRecord],
) -> list[dict[str, Any]]:
    """Attach pending one-sample DNFS diagnostics to their dated forecast records."""
    drain = getattr(forecaster, "drain_rule_diagnostics", None)
    if not callable(drain):
        return []
    diagnostics = drain()
    if not diagnostics:
        return []
    if len(diagnostics) != len(records):
        msg = "DNFS diagnostic count does not match its generated forecast count."
        raise ValueError(msg)

    rows = []
    for record, diagnostic in zip(records, diagnostics, strict=True):
        weights = diagnostic["rule_weights"].reshape(-1)
        strengths = diagnostic["log_strengths"].reshape(-1)
        consequents = diagnostic["consequent_outputs"].reshape(-1)
        entropy = float(diagnostic["rule_entropy"].reshape(-1)[0])
        maximum = float(diagnostic["max_activation"].reshape(-1)[0])
        dominant_fraction = float(diagnostic["dominant_rule_fraction"].reshape(-1)[0])
        unused_count = int(diagnostic["unused_rule_count"].item())
        for rule, (weight, strength, consequent) in enumerate(
            zip(weights, strengths, consequents, strict=True)
        ):
            rows.append(
                {
                    "model": record.model,
                    "fold": record.fold,
                    "forecast_origin": record.forecast_origin,
                    "target_date": record.target_date,
                    "horizon_step": record.horizon_step,
                    "rule": rule,
                    "weight": float(weight),
                    "log_strength": float(strength),
                    "consequent_output": float(consequent),
                    "rule_entropy": entropy,
                    "max_activation": maximum,
                    "unused_rule_count": unused_count,
                    "dominant_rule_fraction": dominant_fraction,
                }
            )
    return rows


def write_rule_activation_artifacts(
    run_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(
        ["model", "fold", "forecast_origin", "horizon_step", "rule"]
    )
    write_dataframe(run_dir / "rule_analysis" / "rule_activations.csv", frame)

    aggregated = (
        frame.assign(target_date=pd.to_datetime(frame["target_date"]))
        .groupby(["model", "target_date", "rule"], as_index=False)["weight"]
        .mean()
    )
    for model, group in aggregated.groupby("model"):
        figure, axis = plt.subplots(figsize=(12, 5))
        for rule, rule_group in group.groupby("rule"):
            axis.plot(rule_group["target_date"], rule_group["weight"], label=f"Rule {rule}")
        axis.set_title(f"{model} fuzzy-rule activation over time")
        axis.set_xlabel("Target date")
        axis.set_ylabel("Normalized rule weight")
        axis.grid(alpha=0.25)
        if group["rule"].nunique() <= 16:
            axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        path = run_dir / "plots" / f"{model}_rule_activations.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
        plt.close(figure)
