from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DNFSRegressor(nn.Module):
    """Compact Takagi-Sugeno-style neuro-fuzzy regressor.

    The model flattens a time-series window, evaluates each fuzzy rule with a learned
    Gaussian membership function, normalizes rule strengths, and combines learned linear
    consequents into a scalar forecast.
    """

    def __init__(
        self,
        sequence_length: int,
        feature_count: int,
        num_rules: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_rules < 1:
            msg = "num_rules must be positive."
            raise ValueError(msg)

        input_size = sequence_length * feature_count
        self.num_rules = num_rules
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.centers = nn.Parameter(torch.empty(num_rules, input_size))
        self.raw_widths = nn.Parameter(torch.zeros(num_rules, input_size))
        self.consequent_weights = nn.Parameter(torch.empty(num_rules, input_size))
        self.consequent_bias = nn.Parameter(torch.zeros(num_rules))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.centers, mean=0.0, std=0.1)
        nn.init.normal_(self.consequent_weights, mean=0.0, std=0.02)
        nn.init.zeros_(self.consequent_bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        flattened = inputs.reshape(batch_size, self.input_size)
        flattened = self.dropout(flattened)

        widths = F.softplus(self.raw_widths) + 1e-6
        normalized_distance = (flattened.unsqueeze(1) - self.centers.unsqueeze(0)) / widths
        log_rule_strengths = -0.5 * normalized_distance.square().mean(dim=-1)
        rule_weights = torch.softmax(log_rule_strengths, dim=-1)

        rule_outputs = flattened @ self.consequent_weights.transpose(0, 1)
        rule_outputs = rule_outputs + self.consequent_bias
        output = torch.sum(rule_weights * rule_outputs, dim=-1, keepdim=True)
        return output
