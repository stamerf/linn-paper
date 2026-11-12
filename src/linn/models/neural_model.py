from torch import nn
from .base_model import BaseModel


class NeuralRegressor(BaseModel):
    """Process-agnostic MLP: no time semantics, positivity activation, or physics."""
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 16)):
        super().__init__()
        if min(input_dim, output_dim, *hidden_dims) <= 0:
            raise ValueError("Layer dimensions must be positive")
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.Tanh()])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)
