"""
Routing strategies for the TaiChi-Router MoE engine.

Provides pluggable strategies for:
- Expert weight allocation patterns
- Mode-specific blending functions
- Custom gating logic
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .router import (
    DEFAULT_ENTROPY_COMP,
    LUOSHU_CORRECTION,
    NormalizationMethod,
    normalize_weights,
)


# ---------------------------------------------------------------------------
# Routing Strategy Interface
# ---------------------------------------------------------------------------

class RoutingStrategy(ABC):
    """Abstract base for a pluggable routing strategy.

    A strategy encapsulates the full logic for computing expert weights
    given input data, expert registry, and coupling strength.
    """

    @abstractmethod
    def compute_weights(
        self,
        input_data: Any,
        coupling_strength: float,
        expert_scores: Dict[str, float],
        expert_priorities: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute normalized expert weights.

        Args:
            input_data: Original input.
            coupling_strength: Pre-computed coupling strength.
            expert_scores: {name: condition_score} for each expert.
            expert_priorities: {name: priority} for each expert.

        Returns:
            {name: normalized_weight} summing to 1.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in Strategies
# ---------------------------------------------------------------------------

class ThreeRegimeStrategy(RoutingStrategy):
    """The canonical three-regime routing from the TaiChi S-Field model.

    Uses three weight templates mapped to coupling regimes:
      - STEADY:        E1=0.5  E2=0.3  E3=0.2
      - TRANSITIONAL:  E1=0.4  E2=0.4  E3=0.2
      - PERTURBATION:  E1=0.3  E2=0.3  E3=0.4

    These templates are blended with individual expert condition scores
    using a 60/40 split (condition/template), derived from the C6 hexagonal
    symmetry group's sin(60°) = 0.866 → 60% condition-driven.
    """

    def __init__(
        self,
        templates: Optional[Dict[str, Dict[str, float]]] = None,
        condition_blend: float = 0.6,
        normalization: NormalizationMethod = NormalizationMethod.ENTROPY_BALANCED,
        entropy_comp: float = DEFAULT_ENTROPY_COMP,
    ):
        self.templates = templates or {
            "steady": {"E1": 0.5, "E2": 0.3, "E3": 0.2},
            "transitional": {"E1": 0.4, "E2": 0.4, "E3": 0.2},
            "perturbation": {"E1": 0.3, "E2": 0.3, "E3": 0.4},
        }
        self.condition_blend = condition_blend
        self.normalization = normalization
        self.entropy_comp = entropy_comp

    def compute_weights(
        self,
        input_data: Any,
        coupling_strength: float,
        expert_scores: Dict[str, float],
        expert_priorities: Dict[str, float],
    ) -> Dict[str, float]:
        # Select template by regime
        if coupling_strength < 0.5:
            regime = "steady"
        elif coupling_strength < 1.5:
            regime = "transitional"
        else:
            regime = "perturbation"

        template = self.templates.get(regime, self.templates["transitional"])

        # Blend template with condition scores
        raw: Dict[str, float] = {}
        for name in expert_scores:
            tpl_w = template.get(name, expert_priorities.get(name, 0.3))
            cond_w = expert_scores.get(name, 0.5)
            raw[name] = self.condition_blend * cond_w + (1 - self.condition_blend) * tpl_w

        return normalize_weights(raw, method=self.normalization, entropy_comp=self.entropy_comp)


class EntropyMaximizingStrategy(RoutingStrategy):
    """Strategy that maximizes expert utilization entropy.

    Prevents expert collapse by ensuring all experts receive meaningful
    weight allocation, using a combination of condition scoring and
    an entropy bonus term.
    """

    def __init__(self, temperature: float = 1.0, min_weight: float = 0.05):
        self.temperature = temperature
        self.min_weight = min_weight

    def compute_weights(
        self,
        input_data: Any,
        coupling_strength: float,
        expert_scores: Dict[str, float],
        expert_priorities: Dict[str, float],
    ) -> Dict[str, float]:
        n = len(expert_scores)
        if n == 0:
            return {}

        names = list(expert_scores.keys())
        scores = np.array([expert_scores[name] for name in names], dtype=np.float64)

        # Temperature-scaled softmax
        scores = scores / max(self.temperature, 1e-8)
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()

        # Add minimum weight floor and re-normalize
        floor = np.ones(n) * self.min_weight
        probs = probs + floor
        probs = probs / probs.sum()

        return {name: float(w) for name, w in zip(names, probs)}


class HexagonalGatingStrategy(RoutingStrategy):
    """Gating strategy incorporating C6 hexagonal topology constraints.

    This strategy maps expert scores through a hexagonal lattice
    projection, enforcing that the allocation pattern respects the
    60-degree rotational symmetry of the C6 group. The hexagonal
    constraint acts as a structured sparsity prior.
    """

    def __init__(
        self,
        hex_angle_deg: float = 60.0,
        luoshu_correction: float = LUOSHU_CORRECTION,
    ):
        self.hex_angle_rad = np.radians(hex_angle_deg)
        self.luoshu_correction = luoshu_correction

    def compute_weights(
        self,
        input_data: Any,
        coupling_strength: float,
        expert_scores: Dict[str, float],
        expert_priorities: Dict[str, float],
    ) -> Dict[str, float]:
        n = len(expert_scores)
        if n == 0:
            return {}

        names = list(expert_scores.keys())
        scores = np.array([expert_scores[name] for name in names], dtype=np.float64)

        # Hexagonal projection:
        # Each expert is placed on a 60-degree-spaced point on the unit circle.
        # The projection score is: s_i * cos(angle_i - coupling_phase)
        # where angle_i = i * 60° and coupling_phase = coupling_strength * 60°
        coupling_phase = coupling_strength * self.hex_angle_rad

        projected = np.zeros(n, dtype=np.float64)
        for i in range(n):
            angle_i = i * self.hex_angle_rad
            # Cosine similarity between expert angle and coupling phase
            cos_sim = np.cos(angle_i - coupling_phase)
            projected[i] = scores[i] * (0.5 + 0.5 * cos_sim)

        # Apply Luo-Shu correction as regularization
        projected = projected * (1.0 - self.luoshu_correction)

        # Normalize
        total = projected.sum()
        if total > 0:
            probs = projected / total
        else:
            probs = np.ones(n) / n

        return {name: float(w) for name, w in zip(names, probs)}


class AdaptiveThresholdStrategy(RoutingStrategy):
    """Strategy with dynamic threshold gating.

    Experts are gated by a threshold that adapts based on coupling strength:
    higher coupling → lower threshold (more experts activated).
    """

    def __init__(
        self,
        base_threshold: float = 0.3,
        min_threshold: float = 0.1,
        min_active: int = 1,
    ):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.min_active = min_active

    def compute_weights(
        self,
        input_data: Any,
        coupling_strength: float,
        expert_scores: Dict[str, float],
        expert_priorities: Dict[str, float],
    ) -> Dict[str, float]:
        n = len(expert_scores)
        if n == 0:
            return {}

        # Threshold decreases with coupling — perturbation mode allows more experts
        threshold = max(
            self.min_threshold,
            self.base_threshold / (1.0 + coupling_strength),
        )

        names = list(expert_scores.keys())
        scores = np.array([expert_scores[name] for name in names], dtype=np.float64)

        mask = scores >= threshold
        if mask.sum() < self.min_active:
            # Ensure at least min_active experts
            idx = np.argpartition(scores, -self.min_active)[-self.min_active:]
            mask[:] = False
            mask[idx] = True

        gated = scores * mask.astype(np.float64)
        total = gated.sum()
        if total > 0:
            probs = gated / total
        else:
            probs = np.ones(n) / n

        return {name: float(w) for name, w in zip(names, probs)}


# ---------------------------------------------------------------------------
# Strategy Registry
# ---------------------------------------------------------------------------

# Mapping of strategy names to their constructors
STRATEGY_REGISTRY: Dict[str, type] = {
    "three_regime": ThreeRegimeStrategy,
    "entropy_max": EntropyMaximizingStrategy,
    "hexagonal": HexagonalGatingStrategy,
    "adaptive_threshold": AdaptiveThresholdStrategy,
}


def get_strategy(name: str, **kwargs) -> RoutingStrategy:
    """Instantiate a routing strategy by name.

    Args:
        name: Strategy name from STRATEGY_REGISTRY.
        **kwargs: Passed to the strategy constructor.

    Returns:
        Instantiated RoutingStrategy.

    Raises:
        KeyError: If strategy name is not recognized.
    """
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {available}"
        )
    return STRATEGY_REGISTRY[name](**kwargs)
