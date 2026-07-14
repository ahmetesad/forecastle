from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna
from optuna.importance import get_param_importances

from forecastle.artifacts import dataframe_to_markdown, write_yaml
from forecastle.data import build_datamodule
from forecastle.experiment import make_run_dir, reconstruct_prices, resolve_device, unscale
from forecastle.models import build_model
from forecastle.plotting import plt
from forecastle.training import Trainer, compute_metrics
from forecastle.utils.seed import seed_everything

if TYPE_CHECKING:
    from forecastle.config import AppConfig, ModelRunConfig


NEURAL_MODELS = {"mlp", "rnn", "lstm", "gru", "cnn1d"}


def run_tuning(config: AppConfig) -> Path:
    tuning = config.tuning
    seed = tuning.seed if tuning.seed is not None else config.experiment.seed
    seed_everything(seed)
    model_config = select_model(config)
    run_dir = make_run_dir(config.experiment.output_dir, f"{config.experiment.name}_tuning")

    storage = tuning.storage or f"sqlite:///{run_dir / 'study.db'}"
    ensure_sqlite_parent_exists(storage)
    study_name = tuning.study_name or f"{config.experiment.name}_{model_config.name}"
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner: optuna.pruners.BasePruner
    if tuning.use_pruner:
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(1, config.training.patience // 2))
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    objective = TuningObjective(
        config=config,
        model_config=model_config,
        run_dir=run_dir,
        seed=seed,
    )
    study.optimize(objective, n_trials=tuning.trials, n_jobs=tuning.n_jobs, show_progress_bar=False)

    write_tuning_artifacts(
        config=config,
        model_config=model_config,
        study=study,
        run_dir=run_dir,
        storage=storage,
        validation_baseline=objective.validation_baseline,
    )
    return run_dir


class TuningObjective:
    def __init__(
        self,
        config: AppConfig,
        model_config: ModelRunConfig,
        run_dir: Path,
        seed: int,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.run_dir = run_dir
        self.seed = seed
        self.device = resolve_device(config.experiment.device)
        self.validation_baseline: dict[str, float] | None = None

    def __call__(self, trial: optuna.Trial) -> float:
        trial_seed = self.seed + trial.number
        seed_everything(trial_seed)
        dataset_config = replace(
            self.config.dataset,
            sequence_length=suggest_sequence_length(trial, self.config.tuning.sequence_lengths),
        )
        training_config = replace(
            self.config.training,
            batch_size=suggest_batch_size(trial, self.config.tuning.batch_sizes),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
            models=[
                replace(
                    self.model_config,
                    params=suggest_model_params(trial, self.model_config.name),
                )
            ],
        )
        datamodule = build_datamodule(dataset_config, training_config, trial_seed)
        persistence_metrics = validation_persistence_metrics(datamodule)
        if self.validation_baseline is None:
            self.validation_baseline = persistence_metrics

        model = build_model(
            self.model_config.name,
            sequence_length=datamodule.sequence_length,
            feature_count=datamodule.feature_count,
            params=training_config.models[0].params,
        )
        checkpoint_path = (
            self.run_dir / "checkpoints" / f"trial_{trial.number:04d}_{self.model_config.name}.pt"
        )
        trainer = Trainer(model, training_config, self.device, checkpoint_path)

        def report(epoch: int, val_loss: float) -> None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        fit_result = trainer.fit(datamodule.train_loader, datamodule.val_loader, report)
        actual_scaled, predicted_scaled, inference_time = trainer.predict(datamodule.val_loader)
        actual = unscale(actual_scaled.numpy(), datamodule.target_mean, datamodule.target_std)
        predicted = unscale(predicted_scaled.numpy(), datamodule.target_mean, datamodule.target_std)
        metrics = compute_metrics(actual, predicted)
        price_predicted = reconstruct_prices(
            datamodule.val_previous_prices,
            predicted,
            datamodule.target_transform,
        )
        price_metrics = compute_metrics(datamodule.val_target_prices, price_predicted)

        trial.set_user_attr("val_rmse", metrics.rmse)
        trial.set_user_attr("val_price_rmse", price_metrics.rmse)
        trial.set_user_attr("best_val_loss", fit_result.best_val_loss)
        trial.set_user_attr("epochs_ran", fit_result.epochs_ran)
        trial.set_user_attr("training_time_seconds", fit_result.training_time_seconds)
        trial.set_user_attr("inference_time_seconds", inference_time)
        trial.set_user_attr("checkpoint_path", str(fit_result.checkpoint_path))
        trial.set_user_attr("validation_persistence_rmse", persistence_metrics["rmse"])
        trial.set_user_attr(
            "validation_persistence_price_rmse",
            persistence_metrics["price_rmse"],
        )

        if self.config.tuning.metric == "price_rmse":
            return price_metrics.rmse
        return metrics.rmse


def select_model(config: AppConfig) -> ModelRunConfig:
    requested_name = config.tuning.model
    if requested_name is not None:
        matching = [model for model in config.training.models if model.name == requested_name]
        if not matching:
            msg = f"tuning.model '{requested_name}' is not present in training.models."
            raise ValueError(msg)
        model_config = matching[0]
    else:
        model_config = config.training.models[0]

    if model_config.name not in NEURAL_MODELS:
        msg = f"Optuna tuning only supports neural models: {', '.join(sorted(NEURAL_MODELS))}."
        raise ValueError(msg)
    return model_config


def ensure_sqlite_parent_exists(storage: str) -> None:
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return
    path = Path(storage.removeprefix(prefix))
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def suggest_sequence_length(trial: optuna.Trial, values: list[int]) -> int:
    return int(trial.suggest_categorical("sequence_length", sorted(set(values))))


def suggest_batch_size(trial: optuna.Trial, values: list[int]) -> int:
    return int(trial.suggest_categorical("batch_size", sorted(set(values))))


def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    hidden_size = int(trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256]))
    num_layers = int(trial.suggest_int("num_layers", 1, 3))
    dropout = float(trial.suggest_float("dropout", 0.0, 0.5))

    if model_name in {"rnn", "lstm", "gru"}:
        return {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }
    if model_name == "mlp":
        return {
            "hidden_sizes": [hidden_size for _ in range(num_layers)],
            "dropout": dropout,
        }
    if model_name == "cnn1d":
        kernel_size = int(trial.suggest_categorical("kernel_size", [3, 5, 7]))
        return {
            "channels": [hidden_size for _ in range(num_layers)],
            "kernel_size": kernel_size,
            "dropout": dropout,
        }
    msg = f"Unsupported model for tuning: {model_name}"
    raise ValueError(msg)


def validation_persistence_metrics(datamodule: Any) -> dict[str, float]:
    if datamodule.target_transform == "price":
        predicted = datamodule.val_previous_prices
    else:
        predicted = datamodule.val_actuals * 0.0
    metrics = compute_metrics(datamodule.val_actuals, predicted)
    price_predicted = reconstruct_prices(
        datamodule.val_previous_prices,
        predicted,
        datamodule.target_transform,
    )
    price_metrics = compute_metrics(datamodule.val_target_prices, price_predicted)
    return {"rmse": metrics.rmse, "price_rmse": price_metrics.rmse}


def write_tuning_artifacts(
    config: AppConfig,
    model_config: ModelRunConfig,
    study: optuna.Study,
    run_dir: Path,
    storage: str,
    validation_baseline: dict[str, float] | None,
) -> None:
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials.to_csv(run_dir / "optimization_history.csv", index=False)
    (run_dir / "optimization_history.md").write_text(
        dataframe_to_markdown(trials),
        encoding="utf-8",
    )
    write_top_trials(trials, run_dir)

    best_trial = study.best_trial
    best_config = build_tuned_config(config, model_config, best_trial)
    write_yaml(run_dir / "best_params.yaml", dict(best_trial.params))
    write_yaml(
        run_dir / "best_summary.yaml",
        {
            "model": model_config.name,
            "metric": config.tuning.metric,
            "best_validation_score": study.best_value,
            "best_trial_number": best_trial.number,
            "storage": storage,
            "validation_persistence": validation_baseline,
            "best_trial_validation_persistence": {
                "rmse": best_trial.user_attrs.get("validation_persistence_rmse"),
                "price_rmse": best_trial.user_attrs.get("validation_persistence_price_rmse"),
            },
            "best_trial_attrs": dict(best_trial.user_attrs),
        },
    )
    write_yaml(run_dir / "tuned_config.yaml", config_to_dict(best_config))
    write_param_importance(study, run_dir)
    write_plots(study, run_dir)


def build_tuned_config(
    config: AppConfig,
    model_config: ModelRunConfig,
    trial: optuna.trial.FrozenTrial,
) -> AppConfig:
    params = dict(trial.params)
    sequence_length = int(params.pop("sequence_length"))
    batch_size = int(params.pop("batch_size"))
    learning_rate = float(params.pop("learning_rate"))
    weight_decay = float(params.pop("weight_decay"))

    tuned_model = replace(
        model_config,
        params=trial_params_to_model_params(model_config.name, params),
    )
    return replace(
        config,
        experiment=replace(config.experiment, name=f"{config.experiment.name}_tuned"),
        dataset=replace(config.dataset, sequence_length=sequence_length),
        training=replace(
            config.training,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            models=[tuned_model],
        ),
    )


def trial_params_to_model_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    hidden_size = int(params["hidden_size"])
    num_layers = int(params["num_layers"])
    dropout = float(params["dropout"])
    if model_name in {"rnn", "lstm", "gru"}:
        return {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }
    if model_name == "mlp":
        return {
            "hidden_sizes": [hidden_size for _ in range(num_layers)],
            "dropout": dropout,
        }
    if model_name == "cnn1d":
        return {
            "channels": [hidden_size for _ in range(num_layers)],
            "kernel_size": int(params["kernel_size"]),
            "dropout": dropout,
        }
    msg = f"Unsupported model for tuning: {model_name}"
    raise ValueError(msg)


def write_top_trials(trials: Any, run_dir: Path, top_n: int = 10) -> None:
    complete_trials = trials[trials["state"] == "COMPLETE"].sort_values("value").head(top_n)
    preferred_columns = [
        "number",
        "value",
        "params_sequence_length",
        "params_batch_size",
        "params_hidden_size",
        "params_num_layers",
        "params_dropout",
        "params_learning_rate",
        "params_weight_decay",
        "user_attrs_val_rmse",
        "user_attrs_val_price_rmse",
        "user_attrs_validation_persistence_rmse",
        "user_attrs_validation_persistence_price_rmse",
        "user_attrs_epochs_ran",
        "user_attrs_training_time_seconds",
    ]
    columns = [column for column in preferred_columns if column in complete_trials.columns]
    top_trials = complete_trials[columns]
    top_trials.to_csv(run_dir / "top_trials.csv", index=False)
    (run_dir / "top_trials.md").write_text(
        dataframe_to_markdown(top_trials),
        encoding="utf-8",
    )


def write_param_importance(study: optuna.Study, run_dir: Path) -> None:
    try:
        importances = {name: float(value) for name, value in get_param_importances(study).items()}
    except (RuntimeError, ValueError):
        importances = {}
    write_yaml(run_dir / "parameter_importance.yaml", importances)


def write_plots(study: optuna.Study, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("optimization_history.png", optuna.visualization.matplotlib.plot_optimization_history),
        ("parameter_importance.png", optuna.visualization.matplotlib.plot_param_importances),
    ]
    for filename, plotter in plot_specs:
        try:
            axis = plotter(study)
            figure = axis.figure
            figure.tight_layout()
            figure.savefig(run_dir / filename, dpi=150)
            plt.close(figure)
        except (RuntimeError, ValueError, ImportError):
            plt.close("all")


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    raw = asdict(config)
    return normalize_for_yaml(raw)


def normalize_for_yaml(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: normalize_for_yaml(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_for_yaml(item) for item in value]
    return value
