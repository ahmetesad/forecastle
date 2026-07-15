from __future__ import annotations


class RecursiveForecastDivergence(ValueError):
    def __init__(
        self,
        *,
        model: str,
        fold: int,
        forecast_origin: str,
        horizon_step: int,
        previous_price: float,
        predicted_target: float,
        reconstructed_price: float,
    ) -> None:
        self.model = model
        self.fold = fold
        self.forecast_origin = forecast_origin
        self.horizon_step = horizon_step
        self.previous_price = previous_price
        self.predicted_target = predicted_target
        self.reconstructed_price = reconstructed_price
        super().__init__(
            "Recursive forecast diverged: "
            f"model={model}, fold={fold}, forecast_origin={forecast_origin}, "
            f"horizon_step={horizon_step}, previous_price={previous_price!r}, "
            f"predicted_target={predicted_target!r}, "
            f"reconstructed_price={reconstructed_price!r}."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "divergence": True,
            "divergence_model": self.model,
            "divergence_fold": self.fold,
            "divergence_forecast_origin": self.forecast_origin,
            "divergence_horizon_step": self.horizon_step,
            "divergence_predicted_target": self.predicted_target,
            "divergence_previous_price": self.previous_price,
            "divergence_reconstructed_price": self.reconstructed_price,
            "divergence_reason": str(self),
        }
