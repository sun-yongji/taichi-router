"""
Expert implementations for the TaiChi-Router MoE engine.

Provides base expert classes and common expert templates that map
the original TaiChi S-Field three-expert architecture onto generic
AI infrastructure primitives.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Abstract Expert
# ---------------------------------------------------------------------------

class Expert(ABC):
    """Abstract base class for a routable expert module.

    Each expert has two core methods:
    - :meth:`condition`: Evaluate how well this expert matches the input.
    - :meth:`forward`: Execute the expert's computation on the input.
    """

    def __init__(self, name: str, priority: float = 0.5):
        self.name = name
        self.priority = priority

    @abstractmethod
    def condition(self, input_data: Any) -> float:
        """Score how relevant this expert is for the given input.

        Returns:
            Float in [0, 1]. 0 = completely irrelevant, 1 = perfectly matched.
        """
        ...

    @abstractmethod
    def forward(self, input_data: Any) -> Any:
        """Execute this expert's computation.

        Args:
            input_data: The input routed to this expert.

        Returns:
            Expert output (tensor, dict, or any structure).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, priority={self.priority})"


# ---------------------------------------------------------------------------
# Built-in Expert Implementations
# ---------------------------------------------------------------------------

class ObservationExpert(Expert):
    """E1: Steady-state observation expert.

    Best suited for stable, low-variance inputs. This expert focuses on
    extracting long-term patterns and baseline signals. Maps to the
    'astronomical observation expert' in the TaiChi S-Field model.

    The condition score is inversely proportional to input variance:
    a perfectly steady input (variance=0) scores 1.0; a highly variable
    input scores close to 0.
    """

    def __init__(
        self,
        name: str = "E1_observation",
        forward_fn: Optional[Callable[[Any], Any]] = None,
        priority: float = 0.7,
        sensitivity: float = 1.0,
    ):
        super().__init__(name=name, priority=priority)
        self._forward_fn = forward_fn or (lambda x: x)
        self.sensitivity = sensitivity

    def condition(self, input_data: Any) -> float:
        arr = _to_numpy(input_data).reshape(-1)
        if len(arr) < 2:
            return 0.5
        std = float(np.std(arr))
        mean_abs = float(np.mean(np.abs(arr))) + 1e-8
        cv = std / mean_abs
        # score = 1 / (1 + sensitivity * cv)
        # Low CV -> high score
        score = 1.0 / (1.0 + self.sensitivity * cv)
        return float(np.clip(score, 0.0, 1.0))

    def forward(self, input_data: Any) -> Any:
        return self._forward_fn(input_data)


class DynamicsExpert(Expert):
    """E2: Transitional/轨道 dynamics expert.

    Tracks periodic fluctuations and transitional patterns. Maps to the
    'orbital dynamics expert' in the TaiChi S-Field model.

    Condition score peaks at intermediate variance levels — too steady
    or too chaotic both reduce relevance.
    """

    def __init__(
        self,
        name: str = "E2_dynamics",
        forward_fn: Optional[Callable[[Any], Any]] = None,
        priority: float = 0.5,
        optimal_cv: float = 0.5,  # optimal coefficient of variation
    ):
        super().__init__(name=name, priority=priority)
        self._forward_fn = forward_fn or (lambda x: x)
        self.optimal_cv = optimal_cv

    def condition(self, input_data: Any) -> float:
        arr = _to_numpy(input_data).reshape(-1)
        if len(arr) < 2:
            return 0.5
        std = float(np.std(arr))
        mean_abs = float(np.mean(np.abs(arr))) + 1e-8
        cv = std / mean_abs
        # Gaussian-style peak around optimal_cv
        score = math.exp(-((cv - self.optimal_cv) ** 2) / 0.1)
        return float(np.clip(score, 0.0, 1.0))

    def forward(self, input_data: Any) -> Any:
        return self._forward_fn(input_data)


class VortexExpert(Expert):
    """E3: Strong-perturbation vortex expert.

    Activates under high-variance, strongly perturbed inputs. Maps to the
    'earth-axis vortex expert' in the TaiChi S-Field model (地轴涡旋专家).

    This expert handles inputs where the coupling strength exceeds the
    perturbation threshold — chaotic, high-energy regimes requiring
    specialized processing.
    """

    def __init__(
        self,
        name: str = "E3_vortex",
        forward_fn: Optional[Callable[[Any], Any]] = None,
        priority: float = 0.3,
        threshold: float = 1.5,
    ):
        super().__init__(name=name, priority=priority)
        self._forward_fn = forward_fn or (lambda x: x)
        self.threshold = threshold

    def condition(self, input_data: Any) -> float:
        arr = _to_numpy(input_data).reshape(-1)
        if len(arr) < 2:
            return 0.1
        std = float(np.std(arr))
        mean_abs = float(np.mean(np.abs(arr))) + 1e-8
        cv = std / mean_abs

        # Activation grows with cv above the threshold
        # Using a sigmoid centered at threshold
        score = 1.0 / (1.0 + math.exp(-2.0 * (cv - self.threshold)))
        return float(np.clip(score, 0.0, 1.0))

    def forward(self, input_data: Any) -> Any:
        return self._forward_fn(input_data)


# ---------------------------------------------------------------------------
# PyTorch Expert Wrappers (optional)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class TorchExpert(Expert, nn.Module):
        """Expert backed by a PyTorch nn.Module.

        Wraps any nn.Module as a routable expert, adding the condition
        interface on top of the module's forward pass.
        """

        def __init__(
            self,
            module: nn.Module,
            name: str,
            priority: float = 0.5,
            condition_fn: Optional[Callable[[torch.Tensor], float]] = None,
        ):
            Expert.__init__(self, name=name, priority=priority)
            nn.Module.__init__(self)
            self.module = module
            self._condition_fn = condition_fn or self._default_condition

        def condition(self, input_data: Any) -> float:
            if isinstance(input_data, torch.Tensor):
                return self._condition_fn(input_data)
            return self._default_condition(torch.as_tensor(input_data, dtype=torch.float32))

        @staticmethod
        def _default_condition(x: torch.Tensor) -> float:
            std = float(x.std().item())
            mean_abs = float(x.abs().mean().item()) + 1e-8
            cv = std / mean_abs
            return float(1.0 / (1.0 + cv))

        def forward(self, input_data: Any) -> Any:
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.as_tensor(input_data, dtype=torch.float32)
            return self.module(input_data)


# ---------------------------------------------------------------------------
# Expert Factory
# ---------------------------------------------------------------------------

def create_default_experts(
    observation_fn: Optional[Callable] = None,
    dynamics_fn: Optional[Callable] = None,
    vortex_fn: Optional[Callable] = None,
) -> List[Expert]:
    """Create the default three-expert set (E1, E2, E3).

    This convenience factory instantiates the three canonical experts
    from the TaiChi S-Field model for quick setup.

    Args:
        observation_fn: Custom forward for E1 (default: identity).
        dynamics_fn: Custom forward for E2 (default: identity).
        vortex_fn: Custom forward for E3 (default: identity).

    Returns:
        List of three Expert instances.
    """
    return [
        ObservationExpert(forward_fn=observation_fn),
        DynamicsExpert(forward_fn=dynamics_fn),
        VortexExpert(forward_fn=vortex_fn),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(data: Any) -> np.ndarray:
    """Convert various input types to numpy array."""
    if HAS_TORCH and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data, dtype=np.float64)
