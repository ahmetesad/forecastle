from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from forecastle.models.dnfs import (
    DNFSRegressor,
    estimate_average_rule_usage,
    identify_unused_rules,
    prune_rules,
    rule_usage_balance_loss,
)


@pytest.mark.parametrize("encoder_type", ["gru", "lstm", "cnn1d", "flatten"])
def test_dnfs_encoder_output_shapes(encoder_type: str) -> None:
    model = DNFSRegressor(
        sequence_length=9,
        feature_count=3,
        encoder_type=encoder_type,  # type: ignore[arg-type]
        encoder_hidden_size=8,
        latent_size=6,
        num_rules=4,
    )

    assert model(torch.randn(5, 9, 3)).shape == (5, 1)


@pytest.mark.parametrize("consequent_type", ["zero_order", "first_order", "mlp"])
def test_dnfs_consequent_variants(consequent_type: str) -> None:
    model = DNFSRegressor(
        sequence_length=7,
        feature_count=2,
        encoder_hidden_size=6,
        latent_size=5,
        num_rules=3,
        consequent_type=consequent_type,  # type: ignore[arg-type]
        consequent_hidden_size=4,
    )
    diagnostics = model.forward_with_diagnostics(torch.randn(4, 7, 2))

    assert diagnostics["prediction"].shape == (4, 1)
    assert diagnostics["consequent_outputs"].shape == (4, 3)


def test_dnfs_rule_weights_are_normalized_and_topk_is_sparse() -> None:
    inputs = torch.randn(6, 8, 2)
    dense = DNFSRegressor(8, 2, num_rules=5, encoder_hidden_size=7)
    sparse = DNFSRegressor(
        8,
        2,
        num_rules=5,
        encoder_hidden_size=7,
        gating="topk",
        top_k_rules=2,
    )

    dense_weights = dense.forward_with_diagnostics(inputs)["rule_weights"]
    sparse_weights = sparse.forward_with_diagnostics(inputs)["rule_weights"]

    assert torch.allclose(dense_weights.sum(dim=-1), torch.ones(6))
    assert torch.allclose(sparse_weights.sum(dim=-1), torch.ones(6))
    assert torch.equal((sparse_weights > 0).sum(dim=-1), torch.full((6,), 2))


def test_dnfs_widths_remain_within_bounds() -> None:
    model = DNFSRegressor(6, 2, min_width=0.2, max_width=1.7)
    with torch.no_grad():
        model.raw_widths.copy_(
            torch.linspace(-100, 100, model.raw_widths.numel()).reshape_as(model.raw_widths)
        )

    widths = model.effective_widths()

    assert torch.all(widths >= 0.2)
    assert torch.all(widths <= 1.7)


def test_dnfs_gradients_reach_encoder_antecedents_and_consequents() -> None:
    model = DNFSRegressor(
        8,
        3,
        encoder_type="gru",
        encoder_hidden_size=7,
        latent_size=6,
        num_rules=4,
        consequent_type="first_order",
        usage_regularization=0.01,
    )
    loss = model(torch.randn(5, 8, 3)).square().mean() + model.regularization_loss()
    loss.backward()

    encoder_gradients = [parameter.grad for parameter in model.encoder.parameters()]
    assert encoder_gradients and all(gradient is not None for gradient in encoder_gradients)
    assert model.centers.grad is not None
    assert model.raw_widths.grad is not None
    assert model.consequent_weights is not None
    assert model.consequent_weights.grad is not None
    assert all(
        torch.isfinite(gradient).all()
        for gradient in [
            *encoder_gradients,
            model.centers.grad,
            model.raw_widths.grad,
            model.consequent_weights.grad,
        ]
        if gradient is not None
    )


def test_dnfs_reset_parameters_resets_every_trainable_component() -> None:
    model = DNFSRegressor(
        6,
        2,
        encoder_type="lstm",
        encoder_hidden_size=5,
        latent_size=4,
        num_rules=3,
        consequent_type="mlp",
        residual_mode="linear",
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(7.0)

    model.reset_parameters()

    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert all(not torch.all(parameter == 7.0) for parameter in model.parameters())


def test_kmeans_initialization_matches_training_latents_only() -> None:
    torch.manual_seed(4)
    features = torch.cat(
        (
            torch.randn(12, 6, 2) * 0.05 - 1.0,
            torch.randn(12, 6, 2) * 0.05 + 1.0,
        )
    )
    loader = DataLoader(TensorDataset(features, torch.zeros(24, 1)), batch_size=5)
    model = DNFSRegressor(
        6,
        2,
        encoder_type="flatten",
        latent_size=3,
        num_rules=2,
        rule_initialization="kmeans",
    )
    model.eval()
    with torch.no_grad():
        training_latents = model.encode(features).numpy()
    expected = KMeans(n_clusters=2, random_state=19, n_init=10).fit(training_latents)

    model.initialize_from_training_data(loader, torch.device("cpu"), seed=19)

    assert np.allclose(model.centers.detach().numpy(), expected.cluster_centers_, atol=1e-6)
    # Validation and test loaders cannot leak: the initialization API receives only this loader.
    assert model._rules_initialized


def test_balance_loss_prefers_balanced_usage() -> None:
    balanced = torch.full((8, 4), 0.25)
    collapsed = torch.zeros(8, 4)
    collapsed[:, 0] = 1.0

    assert rule_usage_balance_loss(balanced) < rule_usage_balance_loss(collapsed)


def test_persistence_residual_converts_feature_scale_to_target_scale() -> None:
    model = DNFSRegressor(4, 2, num_rules=3, residual_mode="persistence")
    model.configure_residual(
        target_feature_index=1,
        target_transform="price",
        feature_target_mean=100.0,
        feature_target_std=10.0,
        target_mean=90.0,
        target_std=5.0,
    )
    assert model.consequent_weights is not None
    assert model.consequent_bias is not None
    with torch.no_grad():
        model.consequent_weights.zero_()
        model.consequent_bias.zero_()
    inputs = torch.zeros(2, 4, 2)
    inputs[:, -1, 1] = 2.0

    assert torch.allclose(model(inputs), torch.full((2, 1), 6.0))


def test_linear_residual_is_training_only_frozen_ols() -> None:
    features = torch.randn(20, 4, 2)
    flat = features.reshape(20, -1)
    targets = (flat[:, 0] * 0.5 - flat[:, 3] * 0.2 + 0.1).unsqueeze(-1)
    loader = DataLoader(TensorDataset(features, targets), batch_size=5)
    model = DNFSRegressor(
        4,
        2,
        encoder_type="flatten",
        num_rules=2,
        residual_mode="linear",
    )

    model.initialize_from_training_data(loader, torch.device("cpu"), seed=3)
    assert model.consequent_weights is not None
    assert model.consequent_bias is not None
    with torch.no_grad():
        model.consequent_weights.zero_()
        model.consequent_bias.zero_()

    assert torch.allclose(model(features), targets, atol=1e-5)
    assert model.linear_residual is not None
    assert not model.linear_residual.weight.requires_grad


@pytest.mark.parametrize("encoder_type", ["gru", "lstm"])
def test_temporal_dnfs_is_sensitive_to_sequence_order(encoder_type: str) -> None:
    torch.manual_seed(8)
    model = DNFSRegressor(
        10,
        2,
        encoder_type=encoder_type,  # type: ignore[arg-type]
        encoder_hidden_size=8,
        num_rules=4,
    ).eval()
    inputs = torch.randn(3, 10, 2)

    assert not torch.allclose(model(inputs), model(inputs.flip(1)))


def test_flatten_legacy_mode_remains_compatible() -> None:
    model = DNFSRegressor(
        5,
        3,
        encoder_type="flatten",
        legacy_mode=True,
        dropout=0.0,
        num_rules=4,
    )
    diagnostics = model.forward_with_diagnostics(torch.randn(2, 5, 3))

    assert model.latent_size == 15
    assert diagnostics["latent"].shape == (2, 15)
    assert diagnostics["prediction"].shape == (2, 1)


def test_dnfs_diagnostic_shapes() -> None:
    model = DNFSRegressor(7, 2, latent_size=5, num_rules=4)
    diagnostics = model.forward_with_diagnostics(torch.randn(3, 7, 2))

    expected_shapes = {
        "prediction": (3, 1),
        "latent": (3, 5),
        "rule_weights": (3, 4),
        "log_strengths": (3, 4),
        "consequent_outputs": (3, 4),
        "centers": (4, 5),
        "widths": (4, 5),
        "rule_entropy": (3,),
        "mean_activation": (4,),
        "max_activation": (3,),
        "unused_rule_count": (),
        "dominant_rule_fraction": (),
    }
    assert {name: tuple(value.shape) for name, value in diagnostics.items()} == expected_shapes


def test_rule_usage_estimation_and_pruning_copy_retained_rules() -> None:
    model = DNFSRegressor(5, 2, num_rules=3, encoder_hidden_size=4)
    loader = DataLoader(
        TensorDataset(torch.randn(10, 5, 2), torch.zeros(10, 1)),
        batch_size=4,
    )
    usage = estimate_average_rule_usage(model, loader, torch.device("cpu"))
    forced_usage = torch.tensor([0.6, 0.001, 0.399])

    pruned = prune_rules(model, forced_usage, threshold=0.01)

    assert usage.shape == (3,)
    assert identify_unused_rules(forced_usage, 0.01).tolist() == [1]
    assert pruned.num_rules == 2
    assert torch.equal(pruned.centers, model.centers[[0, 2]])
    assert torch.equal(pruned.raw_widths, model.raw_widths[[0, 2]])
