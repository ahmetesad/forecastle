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
