from abc import abstractmethod
from torch import nn


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self, features):
        """Map feature tensor [batch, input_dim] to unconstrained outputs."""
        raise NotImplementedError
