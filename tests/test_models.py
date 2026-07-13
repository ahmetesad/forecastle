from __future__ import annotations

import torch

from forecastle.models import build_model, list_models


def test_all_registered_models_forward_shape() -> None:
    batch_size = 4
    sequence_length = 12
    feature_count = 5
    inputs = torch.randn(batch_size, sequence_length, feature_count)

    for model_name in list_models():
        model = build_model(model_name, sequence_length, feature_count)
        output = model(inputs)
        assert output.shape == (batch_size, 1)


def test_registry_builds_dnfs() -> None:
    model = build_model(
        "dnfs",
        sequence_length=8,
        feature_count=3,
        params={"num_rules": 4, "dropout": 0.0},
    )
    inputs = torch.randn(2, 8, 3)

    output = model(inputs)

    assert output.shape == (2, 1)


def test_hybrid_models_support_forward_and_backward() -> None:
    inputs = torch.randn(3, 10, 5)
    for model_name in ["lstm_gru", "cnn_lstm"]:
        model = build_model(model_name, sequence_length=10, feature_count=5)
        output = model(inputs)
        output.sum().backward()

        assert output.shape == (3, 1)
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_cnn_lstm_rejects_even_kernel_size() -> None:
    try:
        build_model(
            "cnn_lstm",
            sequence_length=10,
            feature_count=5,
            params={"kernel_size": 4},
        )
    except ValueError as error:
        assert "positive odd" in str(error)
    else:
        raise AssertionError("Expected an even CNN-LSTM kernel to be rejected.")
