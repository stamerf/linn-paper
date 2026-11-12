from dataclasses import dataclass
from typing import Tuple
from linn.constraints.base_constraint import ConstraintSet


@dataclass
class ProcessSpec:
    name: str
    target_names: Tuple[str, ...]
    constraints: ConstraintSet
    context: dict
