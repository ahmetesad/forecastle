from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

EncoderType = Literal["gru", "lstm", "cnn1d", "flatten"]
StrengthReduction = Literal["mean", "sum"]
ConsequentType = Literal["zero_order", "first_order", "mlp"]
RuleInitialization = Literal["random", "kmeans"]
GatingType = Literal["softmax", "topk"]
ResidualMode = Literal["none", "persistence", "linear"]


class CausalConv1d(nn.Module):
    """A left-padded convolution whose output has the same temporal length as its input."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1:
            msg = "encoder_kernel_size must be positive."
            raise ValueError(msg)
        self.left_padding = kernel_size - 1
        self.convolution = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.convolution(F.pad(inputs, (self.left_padding, 0)))


class DNFSRegressor(nn.Module):
    """Temporal deep neuro-fuzzy regressor with explicit Gaussian TSK rules.

    A configurable temporal encoder produces a normalized latent state. Gaussian fuzzy
    antecedents gate rule-specific Takagi-Sugeno consequents; their normalized weighted sum is
    the prediction, or a correction when residual prediction is enabled.
    """

    def __init__(
        self,
        sequence_length: int,
        feature_count: int,
        num_rules: int = 16,
        dropout: float | None = None,
        encoder_type: EncoderType = "gru",
        encoder_hidden_size: int = 64,
        encoder_num_layers: int = 1,
        encoder_dropout: float = 0.0,
        bidirectional: bool = False,
        latent_size: int | None = None,
        encoder_kernel_size: int = 3,
        strength_reduction: StrengthReduction = "mean",
        rule_temperature: float = 1.0,
        min_width: float = 0.05,
        max_width: float = 5.0,
        consequent_type: ConsequentType = "first_order",
        consequent_hidden_size: int = 32,
        consequent_dropout: float = 0.0,
        antecedent_dropout: float = 0.0,
        share_dropout_representation: bool = False,
        rule_initialization: RuleInitialization = "random",
        usage_regularization: float = 0.0,
        gating: GatingType = "softmax",
        top_k_rules: int | None = None,
        residual_mode: ResidualMode = "none",
        legacy_mode: bool = False,
    ) -> None:
        super().__init__()
        self._validate_configuration(
            sequence_length=sequence_length,
            feature_count=feature_count,
            num_rules=num_rules,
            encoder_type=encoder_type,
            encoder_hidden_size=encoder_hidden_size,
            encoder_num_layers=encoder_num_layers,
            bidirectional=bidirectional,
            latent_size=latent_size,
            strength_reduction=strength_reduction,
            rule_temperature=rule_temperature,
            min_width=min_width,
            max_width=max_width,
            consequent_type=consequent_type,
            consequent_hidden_size=consequent_hidden_size,
            rule_initialization=rule_initialization,
            usage_regularization=usage_regularization,
            gating=gating,
            top_k_rules=top_k_rules,
            residual_mode=residual_mode,
        )

        if dropout is not None:
            # Historical DNFS configs used one dropped flattened representation for both paths.
            antecedent_dropout = dropout
            consequent_dropout = dropout
            share_dropout_representation = True

        self.sequence_length = sequence_length
        self.feature_count = feature_count
        self.num_rules = num_rules
        self.encoder_type = encoder_type
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.encoder_dropout_rate = encoder_dropout
        self.bidirectional = bidirectional
        self.encoder_kernel_size = encoder_kernel_size
        self.strength_reduction = strength_reduction
        self.rule_temperature = rule_temperature
        self.min_width = min_width
        self.max_width = max_width
        self.consequent_type = consequent_type
        self.consequent_hidden_size = consequent_hidden_size
        self.rule_initialization = rule_initialization
        self.usage_regularization = usage_regularization
        self.gating = gating
        self.top_k_rules = top_k_rules
        self.residual_mode = residual_mode
        self.legacy_mode = legacy_mode
        self.share_dropout_representation = share_dropout_representation

        encoder_output_size = self._build_encoder()
        self.latent_size = latent_size or encoder_output_size
        self.latent_projection: nn.Module
        if self.latent_size == encoder_output_size:
            self.latent_projection = nn.Identity()
        else:
            self.latent_projection = nn.Linear(encoder_output_size, self.latent_size)
        self.latent_normalization: nn.Module = (
            nn.Identity() if legacy_mode else nn.LayerNorm(self.latent_size)
        )
        self.encoder_output_dropout = nn.Dropout(encoder_dropout)
        self.antecedent_dropout = nn.Dropout(antecedent_dropout)
        self.consequent_dropout = nn.Dropout(consequent_dropout)

        self.centers = nn.Parameter(torch.empty(num_rules, self.latent_size))
        self.raw_widths = nn.Parameter(torch.empty(num_rules, self.latent_size))
        self.consequent_weights: nn.Parameter | None = None
        self.consequent_bias: nn.Parameter | None = None
        self.mlp_consequents = nn.ModuleList()
        if consequent_type == "zero_order":
            self.consequent_bias = nn.Parameter(torch.empty(num_rules))
        elif consequent_type == "first_order":
            self.consequent_weights = nn.Parameter(torch.empty(num_rules, self.latent_size))
            self.consequent_bias = nn.Parameter(torch.empty(num_rules))
        else:
            self.mlp_consequents = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.latent_size, consequent_hidden_size),
                        nn.GELU(),
                        nn.Linear(consequent_hidden_size, 1),
                    )
                    for _ in range(num_rules)
                ]
            )

        flattened_size = sequence_length * feature_count
        self.linear_residual = nn.Linear(flattened_size, 1) if residual_mode == "linear" else None
        self._last_rule_weights: torch.Tensor | None = None
        self._rules_initialized = False
        self._linear_residual_initialized = False
        self._target_feature_index = feature_count - 1
        self._target_transform = "log_return"
        self._feature_target_mean = 0.0
        self._feature_target_std = 1.0
        self._target_mean = 0.0
        self._target_std = 1.0
        self.reset_parameters()

    def _build_encoder(self) -> int:
        if self.encoder_type == "flatten":
            self.encoder = nn.Identity()
            return self.sequence_length * self.feature_count

        if self.encoder_type in {"gru", "lstm"}:
            recurrent_class = nn.GRU if self.encoder_type == "gru" else nn.LSTM
            recurrent_dropout = self.encoder_dropout_rate if self.encoder_num_layers > 1 else 0.0
            self.encoder = recurrent_class(
                input_size=self.feature_count,
                hidden_size=self.encoder_hidden_size,
                num_layers=self.encoder_num_layers,
                dropout=recurrent_dropout,
                bidirectional=self.bidirectional,
                batch_first=True,
            )
            directions = 2 if self.bidirectional else 1
            return self.encoder_hidden_size * directions

        layers: list[nn.Module] = []
        input_channels = self.feature_count
        for _ in range(self.encoder_num_layers):
            layers.extend(
                [
                    CausalConv1d(
                        input_channels,
                        self.encoder_hidden_size,
                        self.encoder_kernel_size,
                    ),
                    nn.GELU(),
                    nn.Dropout(self.encoder_dropout_rate),
                ]
            )
            input_channels = self.encoder_hidden_size
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.encoder = nn.Sequential(*layers)
        return self.encoder_hidden_size

    def reset_parameters(self) -> None:
        """Reset every trainable encoder, antecedent, consequent, and residual parameter."""
        for module in self.modules():
            if module is self:
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

        nn.init.normal_(self.centers, mean=0.0, std=0.1)
        initial_width = torch.full_like(self.raw_widths, 1.0)
        with torch.no_grad():
            self.raw_widths.copy_(self._width_to_raw(initial_width))
        if self.consequent_weights is not None:
            nn.init.normal_(self.consequent_weights, mean=0.0, std=0.02)
        if self.consequent_bias is not None:
            nn.init.zeros_(self.consequent_bias)
        self._last_rule_weights = None
        self._rules_initialized = self.rule_initialization == "random"
        self._linear_residual_initialized = False
        if self.linear_residual is not None:
            self.linear_residual.requires_grad_(True)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            msg = "DNFS inputs must have shape [batch, sequence_length, feature_count]."
            raise ValueError(msg)
        if inputs.shape[1:] != (self.sequence_length, self.feature_count):
            msg = (
                "DNFS input dimensions do not match its configured sequence length and feature "
                f"count: expected (*, {self.sequence_length}, {self.feature_count}), "
                f"received {tuple(inputs.shape)}."
            )
            raise ValueError(msg)

        if self.encoder_type == "flatten":
            encoded = inputs.reshape(inputs.shape[0], -1)
        elif self.encoder_type in {"gru", "lstm"}:
            recurrent_output = self.encoder(inputs)
            hidden = recurrent_output[1]
            if self.encoder_type == "lstm":
                hidden = hidden[0]
            if self.bidirectional:
                encoded = torch.cat((hidden[-2], hidden[-1]), dim=-1)
            else:
                encoded = hidden[-1]
        else:
            encoded = self.encoder(inputs.transpose(1, 2)).squeeze(-1)

        latent = self.latent_projection(encoded)
        latent = self.latent_normalization(latent)
        return self.encoder_output_dropout(latent)

    def effective_widths(self) -> torch.Tensor:
        width_range = self.max_width - self.min_width
        return self.min_width + width_range * torch.sigmoid(self.raw_widths)

    def forward_with_diagnostics(
        self,
        inputs: torch.Tensor,
        unused_rule_threshold: float = 1e-3,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode(inputs)
        antecedent_latent = self.antecedent_dropout(latent)
        consequent_latent = (
            antecedent_latent
            if self.share_dropout_representation
            else self.consequent_dropout(latent)
        )
        widths = self.effective_widths()
        distances = (antecedent_latent.unsqueeze(1) - self.centers.unsqueeze(0)) / widths
        squared_distances = distances.square()
        if self.strength_reduction == "mean":
            reduced_distance = squared_distances.mean(dim=-1)
        else:
            reduced_distance = squared_distances.sum(dim=-1)
        log_strengths = -0.5 * reduced_distance
        rule_weights = self._normalize_rule_weights(log_strengths)
        consequent_outputs = self._consequent_outputs(consequent_latent)
        correction = torch.sum(rule_weights * consequent_outputs, dim=-1, keepdim=True)
        prediction = self._residual_prediction(inputs, latent) + correction
        self._last_rule_weights = rule_weights

        mean_activation = rule_weights.mean(dim=0)
        entropy = -(rule_weights * rule_weights.clamp_min(1e-12).log()).sum(dim=-1)
        return {
            "prediction": prediction,
            "latent": latent,
            "rule_weights": rule_weights,
            "log_strengths": log_strengths,
            "consequent_outputs": consequent_outputs,
            "centers": self.centers,
            "widths": widths,
            "rule_entropy": entropy,
            "mean_activation": mean_activation,
            "max_activation": rule_weights.max(dim=-1).values,
            "unused_rule_count": (mean_activation < unused_rule_threshold).sum(),
            "dominant_rule_fraction": mean_activation.max(),
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_diagnostics(inputs)["prediction"]

    def regularization_loss(self) -> torch.Tensor:
        if self.usage_regularization <= 0.0 or self._last_rule_weights is None:
            return self.centers.new_zeros(())
        return self.usage_regularization * rule_usage_balance_loss(self._last_rule_weights)

    def initialize_from_training_data(
        self,
        train_loader: DataLoader,
        device: torch.device,
        seed: int,
    ) -> None:
        """Initialize training-only linear residuals and optional latent K-means rules."""
        initialize_rules = self.rule_initialization == "kmeans" and not self._rules_initialized
        initialize_linear = (
            self.linear_residual is not None and not self._linear_residual_initialized
        )
        if not initialize_rules and not initialize_linear:
            return
        was_training = self.training
        self.eval()
        latent_batches = []
        feature_batches = []
        target_batches = []
        with torch.no_grad():
            for features, targets in train_loader:
                if initialize_rules:
                    latent_batches.append(self.encode(features.to(device)).cpu())
                if initialize_linear:
                    feature_batches.append(features.reshape(features.shape[0], -1).cpu())
                    target_batches.append(targets.cpu())
        self.train(was_training)

        if initialize_linear:
            assert self.linear_residual is not None
            flat_features = torch.cat(feature_batches).numpy()
            targets = torch.cat(target_batches).reshape(-1).numpy()
            regression = LinearRegression().fit(flat_features, targets)
            with torch.no_grad():
                self.linear_residual.weight.copy_(
                    torch.as_tensor(
                        regression.coef_[None],
                        device=device,
                        dtype=self.linear_residual.weight.dtype,
                    )
                )
                self.linear_residual.bias.copy_(
                    torch.as_tensor(
                        [regression.intercept_],
                        device=device,
                        dtype=self.linear_residual.bias.dtype,
                    )
                )
            self.linear_residual.requires_grad_(False)
            self._linear_residual_initialized = True

        if not initialize_rules:
            return
        latent = torch.cat(latent_batches).numpy()
        if len(latent) < self.num_rules:
            msg = "K-means DNFS initialization requires at least num_rules training windows."
            raise ValueError(msg)

        kmeans = KMeans(n_clusters=self.num_rules, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(latent)
        centers = kmeans.cluster_centers_.astype(np.float32)
        global_spread = np.std(latent, axis=0)
        positive_global = global_spread[global_spread > 1e-6]
        global_fallback = float(np.mean(positive_global)) if len(positive_global) else 1.0
        widths = np.empty_like(centers)
        for rule in range(self.num_rules):
            cluster = latent[labels == rule]
            spread = np.std(cluster, axis=0) if len(cluster) > 1 else np.zeros(latent.shape[1])
            nearest_distance = _nearest_center_distance(centers, rule)
            fallback = max(nearest_distance / np.sqrt(self.latent_size), global_fallback)
            widths[rule] = np.where(spread > 1e-6, spread, fallback)

        center_tensor = torch.as_tensor(centers, device=device, dtype=self.centers.dtype)
        width_tensor = torch.as_tensor(widths, device=device, dtype=self.raw_widths.dtype)
        width_tensor = width_tensor.clamp(self.min_width, self.max_width)
        with torch.no_grad():
            self.centers.copy_(center_tensor)
            self.raw_widths.copy_(self._width_to_raw(width_tensor))
        self._rules_initialized = True

    def configure_residual(
        self,
        *,
        target_feature_index: int,
        target_transform: str,
        feature_target_mean: float,
        feature_target_std: float,
        target_mean: float,
        target_std: float,
    ) -> None:
        self._target_feature_index = target_feature_index
        self._target_transform = target_transform
        self._feature_target_mean = feature_target_mean
        self._feature_target_std = feature_target_std
        self._target_mean = target_mean
        self._target_std = target_std

    def _normalize_rule_weights(self, log_strengths: torch.Tensor) -> torch.Tensor:
        logits = log_strengths / self.rule_temperature
        weights = torch.softmax(logits, dim=-1)
        if self.gating == "softmax":
            return weights
        top_k = min(self.top_k_rules or self.num_rules, self.num_rules)
        retained_weights, retained_indices = torch.topk(weights, top_k, dim=-1)
        sparse_weights = torch.zeros_like(weights).scatter(-1, retained_indices, retained_weights)
        return sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _consequent_outputs(self, latent: torch.Tensor) -> torch.Tensor:
        if self.consequent_type == "zero_order":
            assert self.consequent_bias is not None
            return self.consequent_bias.unsqueeze(0).expand(latent.shape[0], -1)
        if self.consequent_type == "first_order":
            assert self.consequent_weights is not None
            assert self.consequent_bias is not None
            return latent @ self.consequent_weights.transpose(0, 1) + self.consequent_bias
        return torch.cat([network(latent) for network in self.mlp_consequents], dim=-1)

    def _residual_prediction(
        self,
        inputs: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.residual_mode == "none":
            return latent.new_zeros((latent.shape[0], 1))
        if self.residual_mode == "linear":
            assert self.linear_residual is not None
            return self.linear_residual(inputs.reshape(inputs.shape[0], -1))
        if self._target_transform != "price":
            return latent.new_zeros((latent.shape[0], 1))
        scaled_feature = inputs[:, -1, self._target_feature_index]
        raw_price = scaled_feature * self._feature_target_std + self._feature_target_mean
        scaled_target = (raw_price - self._target_mean) / self._target_std
        return scaled_target.unsqueeze(-1)

    def _width_to_raw(self, widths: torch.Tensor) -> torch.Tensor:
        fraction = (widths - self.min_width) / (self.max_width - self.min_width)
        fraction = fraction.clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(fraction)

    @staticmethod
    def _validate_configuration(**values: Any) -> None:
        positive_names = (
            "sequence_length",
            "feature_count",
            "num_rules",
            "encoder_hidden_size",
            "encoder_num_layers",
            "consequent_hidden_size",
        )
        if any(int(values[name]) < 1 for name in positive_names):
            msg = "DNFS dimensions, layer counts, and num_rules must be positive."
            raise ValueError(msg)
        if values["latent_size"] is not None and int(values["latent_size"]) < 1:
            msg = "latent_size must be positive when provided."
            raise ValueError(msg)
        if values["encoder_type"] not in {"gru", "lstm", "cnn1d", "flatten"}:
            msg = "encoder_type must be one of: gru, lstm, cnn1d, flatten."
            raise ValueError(msg)
        if values["bidirectional"] and values["encoder_type"] not in {"gru", "lstm"}:
            msg = "bidirectional is supported only for GRU and LSTM DNFS encoders."
            raise ValueError(msg)
        if values["strength_reduction"] not in {"mean", "sum"}:
            msg = "strength_reduction must be one of: mean, sum."
            raise ValueError(msg)
        if float(values["rule_temperature"]) <= 0:
            msg = "rule_temperature must be positive."
            raise ValueError(msg)
        if float(values["min_width"]) <= 0 or float(values["max_width"]) <= float(
            values["min_width"]
        ):
            msg = "DNFS widths require 0 < min_width < max_width."
            raise ValueError(msg)
        if values["consequent_type"] not in {"zero_order", "first_order", "mlp"}:
            msg = "consequent_type must be one of: zero_order, first_order, mlp."
            raise ValueError(msg)
        if values["rule_initialization"] not in {"random", "kmeans"}:
            msg = "rule_initialization must be one of: random, kmeans."
            raise ValueError(msg)
        if float(values["usage_regularization"]) < 0:
            msg = "usage_regularization must be non-negative."
            raise ValueError(msg)
        if values["gating"] not in {"softmax", "topk"}:
            msg = "gating must be one of: softmax, topk."
            raise ValueError(msg)
        top_k = values["top_k_rules"]
        if values["gating"] == "topk" and (
            top_k is None or not 1 <= int(top_k) <= values["num_rules"]
        ):
            msg = "top_k_rules must be between 1 and num_rules when top-k gating is enabled."
            raise ValueError(msg)
        if values["residual_mode"] not in {"none", "persistence", "linear"}:
            msg = "residual_mode must be one of: none, persistence, linear."
            raise ValueError(msg)


def rule_usage_balance_loss(rule_weights: torch.Tensor) -> torch.Tensor:
    """KL divergence from uniform average usage; zero is balanced, without forcing it."""
    usage = rule_weights.mean(dim=0).clamp_min(1e-12)
    return torch.sum(usage * torch.log(usage * rule_weights.shape[-1]))


def estimate_average_rule_usage(
    model: DNFSRegressor,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    usage_sum = torch.zeros(model.num_rules, device=device)
    sample_count = 0
    with torch.no_grad():
        for features, _targets in loader:
            diagnostics = model.forward_with_diagnostics(features.to(device))
            weights = diagnostics["rule_weights"]
            usage_sum += weights.sum(dim=0)
            sample_count += len(features)
    model.train(was_training)
    return (usage_sum / max(sample_count, 1)).cpu()


def identify_unused_rules(usage: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold < 0:
        msg = "Rule pruning threshold must be non-negative."
        raise ValueError(msg)
    return torch.nonzero(usage < threshold, as_tuple=False).flatten()


def prune_rules(
    model: DNFSRegressor,
    usage: torch.Tensor,
    threshold: float,
) -> DNFSRegressor:
    """Return an exact-parameter copy containing only rules meeting the usage threshold."""
    if usage.shape != (model.num_rules,):
        msg = "Rule usage must contain one value per DNFS rule."
        raise ValueError(msg)
    retained = torch.nonzero(usage >= threshold, as_tuple=False).flatten()
    if not len(retained):
        retained = torch.argmax(usage).reshape(1)
    retained_list = retained.tolist()
    pruned = copy.deepcopy(model)
    pruned.num_rules = len(retained_list)
    with torch.no_grad():
        pruned.centers = nn.Parameter(model.centers[retained].detach().clone())
        pruned.raw_widths = nn.Parameter(model.raw_widths[retained].detach().clone())
        if model.consequent_weights is not None:
            pruned.consequent_weights = nn.Parameter(
                model.consequent_weights[retained].detach().clone()
            )
        if model.consequent_bias is not None:
            pruned.consequent_bias = nn.Parameter(model.consequent_bias[retained].detach().clone())
        if model.consequent_type == "mlp":
            pruned.mlp_consequents = nn.ModuleList(
                [copy.deepcopy(model.mlp_consequents[index]) for index in retained_list]
            )
    if pruned.gating == "topk":
        pruned.top_k_rules = min(pruned.top_k_rules or pruned.num_rules, pruned.num_rules)
    pruned._last_rule_weights = None
    return pruned


def _nearest_center_distance(centers: np.ndarray, rule: int) -> float:
    if len(centers) == 1:
        return 1.0
    distances = np.linalg.norm(centers[rule] - centers, axis=1)
    distances[rule] = np.inf
    return float(np.min(distances))
