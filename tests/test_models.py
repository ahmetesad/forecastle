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
