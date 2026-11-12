from abc import ABC, abstractmethod


class ConstraintSet(ABC):
    @abstractmethod
    def compute(self, prediction, batch, context):
        """Return named scalar penalties; batch contains only permitted constraint inputs."""
        raise NotImplementedError


class NoConstraints(ConstraintSet):
    def compute(self, prediction, batch, context):
        return {}
