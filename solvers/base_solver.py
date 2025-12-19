from dataclasses import dataclass
from typing import List, Optional, Any
import numpy as np

@dataclass
class SolutionStep:
    """Represents a single step in a mathematical derivation."""
    title: str        # Brief title, e.g., "B Noktasının Hızı"
    formula: str      # LaTeX Formula, e.g., "V_B = \omega_2 \cdot |O_2B|"
    substitution: str # LaTeX with numbers, e.g., "V_B = 95 \cdot 0.050"
    result_str: str   # Final string, e.g., "V_B = 4.75 m/s"
    result_val: float # Numeric value for validation
    description: str = "" # Additional context
    category: str = "Genel" # Grouping category for UI (e.g. Kinematics, Dynamics)

@dataclass
class MechanismState:
    """Holds the full state of the mechanism at a specific angle."""
    theta2: float
    joints: dict
    velocities: dict
    accelerations: dict
    forces: dict
    torque: float
    steps: List[SolutionStep]

class BaseSolver:
    def solve(self, inputs) -> MechanismState:
        raise NotImplementedError
