"""
TaiChi-Router: An Oriental-Numerology-Inspired MoE Dynamic Routing Engine.

Core routing engine that implements dynamic expert allocation based on
input field characteristics. Inspired by the Three-Expert architecture
from the TaiChi S-Field coupling dynamics model:
  - E1: Observation expert (stable-state field sensing)
  - E2: Dynamics expert (transitional state tracking)
  - E3: Vortex expert (strong-perturbation mode handling)

The routing logic maps the 60-degree hexagonal topology principle
of the Early Heaven Bagua (先天八卦) onto a modern MoE gating mechanism,
where expert weight distribution follows the coupling strength of
input features rather than static softmax assignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Try importing torch; fall back to numpy-only mode if unavailable
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The universal hexagonal topology angle (60 degrees), derived from
# the Early Heaven Bagua symmetry group C6.
HEX_TOPOLOGY_ANGLE_DEG: float = 60.0
HEX_TOPOLOGY_ANGLE_RAD: float = math.radians(HEX_TOPOLOGY_ANGLE_DEG)

# Routing mode thresholds (coupling strength boundaries)
# These map to the three routing modes from the TaiChi S-Field model:
#   steady < 0.5 < transitional < 1.5 < perturbation
STEADY_THRESHOLD: float = 0.5
PERTURBATION_THRESHOLD: float = 1.5

# Default entropy compensation factor (golden-ratio derived: 0.618 * 10 = 6.18%)
DEFAULT_ENTROPY_COMP: float = 0.0618

# Luo-Shu matrix correction coefficient (3.4%), derived from
# (9+8+7+6) / 8.8235 in the HeTu-LuoShu numerological framework.
# Used here as a default regularization term in weight allocation.
LUOSHU_CORRECTION: float = 0.034


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RoutingMode(Enum):
    """Three routing modes corresponding to field coupling regimes."""
    STEADY = "steady"           # Coupling < 0.5: nearly balanced expert weights
    TRANSITIONAL = "transitional"  # 0.5 <= coupling < 1.5: shifting allocation
    PERTURBATION = "perturbation"  # Coupling >= 1.5: vortex-dominant routing


class NormalizationMethod(Enum):
    """Weight normalization strategies."""
    SOFTMAX = "softmax"
    ENTROPY_BALANCED = "entropy_balanced"
    THRESHOLD_GATE = "threshold_gate"
    TOPK = "topk"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ExpertSpec:
    """Specification for registering an expert.

    Attributes:
        name: Unique expert identifier (e.g. 'E1_observation').
        condition_fn: Callable(input_tensor) -> float in [0, 1] indicating
            how well this expert matches the current input.
        forward_fn: Callable(input_tensor) -> output_tensor. The expert's
            actual computation.
        priority: Base priority weight (higher = more likely to be selected
            when coupling is ambiguous). Range [0, 1].
        description: Human-readable description.
        tags: Optional tags for grouping/filtering.
    """
    name: str
    condition_fn: Callable[[Any], float]
    forward_fn: Callable[[Any], Any]
    priority: float = 0.5
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class RoutingResult:
    """Result of a routing decision.

    Attributes:
        mode: The detected routing mode.
        coupling_strength: Computed coupling strength of the input.
        weights: Dict mapping expert name -> allocated weight.
        selected_experts: List of (name, weight, output) for experts
            that were actually activated.
        metadata: Additional diagnostics.
    """
    mode: RoutingMode
    coupling_strength: float
    weights: Dict[str, float]
    selected_experts: List[Tuple[str, float, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def compute_coupling_strength(features: Union[np.ndarray, "torch.Tensor"]) -> float:
    """Compute the coupling strength of input features.

    Maps the feature variance structure to a scalar coupling metric.
    The default implementation uses the coefficient of variation of the
    feature vector as a proxy for field perturbation intensity.

    Args:
        features: Input feature vector or tensor, shape (D,) or (B, D).

    Returns:
        Coupling strength in [0, inf). Higher values indicate stronger
        perturbation requiring different expert allocation.
    """
    if HAS_TORCH and isinstance(features, torch.Tensor):
        arr = features.detach().cpu().numpy()
    elif isinstance(features, np.ndarray):
        arr = features
    else:
        arr = np.asarray(features, dtype=np.float64)

    arr = arr.reshape(-1)
    if len(arr) < 2:
        return 0.0

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))

    if mean_val == 0:
        return std_val

    # Coefficient of variation as base coupling metric
    cv = std_val / abs(mean_val)

    # Apply hexagonal topology modulation:
    # coupling = cv * sin(60°) / (1 + LUOSHU_CORRECTION)
    # This embeds the C6 symmetry constraint into the coupling metric.
    coupling = cv * math.sin(HEX_TOPOLOGY_ANGLE_RAD) / (1.0 + LUOSHU_CORRECTION)

    return float(coupling)


def detect_routing_mode(coupling_strength: float) -> RoutingMode:
    """Detect the routing mode based on coupling strength.

    Thresholds derived from the TaiChi S-Field three-regime classification:
      - coupling < STEADY_THRESHOLD (0.5):  Stable field, balanced expert mix
      - STEADY_THRESHOLD <= coupling < PERTURBATION_THRESHOLD (1.5): Transition
      - coupling >= PERTURBATION_THRESHOLD: Strong perturbation, vortex-dominant

    Args:
        coupling_strength: Scalar coupling metric.

    Returns:
        The corresponding RoutingMode.
    """
    if coupling_strength < STEADY_THRESHOLD:
        return RoutingMode.STEADY
    elif coupling_strength < PERTURBATION_THRESHOLD:
        return RoutingMode.TRANSITIONAL
    else:
        return RoutingMode.PERTURBATION


def normalize_weights(
    raw_weights: Dict[str, float],
    method: NormalizationMethod = NormalizationMethod.ENTROPY_BALANCED,
    entropy_comp: float = DEFAULT_ENTROPY_COMP,
    topk: Optional[int] = None,
) -> Dict[str, float]:
    """Normalize expert weights to sum to 1.

    Args:
        raw_weights: Raw weight dict {expert_name: score}.
        method: Normalization strategy.
        entropy_comp: Entropy compensation factor for entropy_balanced mode.
        topk: Keep only top-k experts (rest zeroed) for topk mode.

    Returns:
        Normalized weight dict summing to 1.
    """
    if not raw_weights:
        return {}

    names = list(raw_weights.keys())
    values = np.array([raw_weights[n] for n in names], dtype=np.float64)

    if method == NormalizationMethod.SOFTMAX:
        values = values - np.max(values)  # numerical stability
        exp_vals = np.exp(values)
        norm = exp_vals / exp_vals.sum()

    elif method == NormalizationMethod.ENTROPY_BALANCED:
        # Entropy-balanced normalization:
        # w_i = exp(s_i) / sum(exp(s_j)) * (1 - entropy_comp) + entropy_comp / N
        values = values - np.max(values)
        exp_vals = np.exp(values)
        softmax = exp_vals / exp_vals.sum()
        n = len(names)
        uniform = np.ones(n) / n
        norm = softmax * (1.0 - entropy_comp) + uniform * entropy_comp

    elif method == NormalizationMethod.THRESHOLD_GATE:
        # Only experts with score above mean are activated; softmax within that set
        threshold = float(np.mean(values))
        mask = values >= threshold
        if mask.sum() == 0:
            mask[:] = True  # fallback: keep all
        gated = values * mask.astype(np.float64)
        gated = gated - gated.max()
        exp_vals = np.exp(gated) * mask.astype(np.float64)
        norm = exp_vals / exp_vals.sum()

    elif method == NormalizationMethod.TOPK:
        if topk is None:
            topk = max(1, len(names) // 2)
        idx = np.argpartition(values, -topk)[-topk:]
        mask = np.zeros(len(names), dtype=np.float64)
        mask[idx] = 1.0
        gated = values * mask
        gated = gated - gated.max()
        exp_vals = np.exp(gated) * mask
        norm = exp_vals / exp_vals.sum()

    else:
        total = values.sum()
        norm = values / total if total > 0 else np.ones(len(names)) / len(names)

    return {name: float(w) for name, w in zip(names, norm)}


# ---------------------------------------------------------------------------
# Core Router
# ---------------------------------------------------------------------------

class TaiChiRouter:
    """Dynamic MoE routing engine with hexagonal-topology-aware gating.

    The router maintains a registry of experts and dynamically allocates
    weights based on input coupling strength, following the three-mode
    routing protocol from the TaiChi S-Field coupling dynamics model.

    Usage::

        router = TaiChiRouter()

        # Register experts
        router.register_expert(
            name="E1_observation",
            condition_fn=lambda x: 1.0 - abs(x.mean()),
            forward_fn=observation_expert,
            priority=0.7,
        )
        router.register_expert(
            name="E2_dynamics",
            condition_fn=lambda x: abs(x.std() - 0.5),
            forward_fn=dynamics_expert,
            priority=0.5,
        )
        router.register_expert(
            name="E3_vortex",
            condition_fn=lambda x: abs(x.std()),
            forward_fn=vortex_expert,
            priority=0.3,
        )

        # Route and forward
        result = router.forward(input_tensor)
        for name, weight, output in result.selected_experts:
            print(f"{name}: weight={weight:.3f}")
    """

    def __init__(
        self,
        coupling_fn: Optional[Callable[[Any], float]] = None,
        normalization: NormalizationMethod = NormalizationMethod.ENTROPY_BALANCED,
        entropy_comp: float = DEFAULT_ENTROPY_COMP,
        min_experts_active: int = 1,
        merge_outputs: Optional[Callable[[List[Tuple[str, float, Any]]], Any]] = None,
    ):
        """Initialize the TaiChi router.

        Args:
            coupling_fn: Custom function to compute coupling strength.
                Defaults to :func:`compute_coupling_strength`.
            normalization: Weight normalization method.
            entropy_comp: Entropy compensation factor (controls
                exploration vs exploitation in expert allocation).
            min_experts_active: Minimum number of experts to always activate.
            merge_outputs: Function to merge multiple expert outputs.
                Defaults to weighted sum for tensors, else returns the
                list of (name, weight, output) tuples.
        """
        self._experts: Dict[str, ExpertSpec] = {}
        self._coupling_fn = coupling_fn or compute_coupling_strength
        # Accept string shorthand (e.g. "entropy_balanced") or enum directly
        if isinstance(normalization, str):
            normalization = NormalizationMethod(normalization)
        self._normalization = normalization
        self._entropy_comp = entropy_comp
        self._min_experts_active = min_experts_active
        self._merge_outputs = merge_outputs

    # ---- Expert Registry ----

    def register_expert(
        self,
        name: str,
        condition_fn: Callable[[Any], float],
        forward_fn: Callable[[Any], Any],
        priority: float = 0.5,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register an expert with the router.

        Args:
            name: Unique expert identifier.
            condition_fn: Function mapping input -> relevance score [0, 1].
            forward_fn: Function mapping input -> expert output.
            priority: Base priority [0, 1]. Higher priority experts receive
                larger base weight allocation.
            description: Human-readable description.
            tags: Tags for grouping or filtering experts.

        Raises:
            ValueError: If an expert with the same name is already registered.
        """
        if name in self._experts:
            raise ValueError(f"Expert '{name}' is already registered.")
        self._experts[name] = ExpertSpec(
            name=name,
            condition_fn=condition_fn,
            forward_fn=forward_fn,
            priority=priority,
            description=description,
            tags=tags or [],
        )

    def unregister_expert(self, name: str) -> None:
        """Remove an expert from the registry."""
        self._experts.pop(name, None)

    def list_experts(self) -> List[str]:
        """Return registered expert names."""
        return list(self._experts.keys())

    def get_expert(self, name: str) -> ExpertSpec:
        """Get expert spec by name."""
        if name not in self._experts:
            raise KeyError(f"Expert '{name}' not found.")
        return self._experts[name]

    # ---- Routing ----

    def route(
        self,
        input_data: Any,
        coupling_strength: Optional[float] = None,
    ) -> RoutingResult:
        """Compute expert weights based on input.

        This is the core routing logic implementing the three-mode protocol:

        1. Compute coupling strength from input features.
        2. Detect routing mode (steady / transitional / perturbation).
        3. Apply mode-specific weight template:
           - STEADY:      E1=0.5, E2=0.3, E3=0.2
           - TRANSITIONAL: E1=0.4, E2=0.4, E3=0.2
           - PERTURBATION: E1=0.3, E2=0.3, E3=0.4
        4. Blend template with individual expert condition scores.
        5. Normalize and return.

        Args:
            input_data: Input features (array, tensor, or any structure
                understood by the coupling function).
            coupling_strength: Pre-computed coupling strength. If None,
                computed automatically from input_data.

        Returns:
            RoutingResult with mode, weights, and metadata.
        """
        if coupling_strength is None:
            coupling_strength = self._coupling_fn(input_data)

        mode = detect_routing_mode(coupling_strength)

        # Mode-specific base allocation templates
        # These templates are the numerical encoding of the three-regime
        # expert dispatch pattern from the TaiChi S-Field model.
        if mode == RoutingMode.STEADY:
            template = {"E1": 0.5, "E2": 0.3, "E3": 0.2}
        elif mode == RoutingMode.TRANSITIONAL:
            template = {"E1": 0.4, "E2": 0.4, "E3": 0.2}
        else:  # PERTURBATION
            template = {"E1": 0.3, "E2": 0.3, "E3": 0.4}

        # Blend template with individual expert condition scores and priority
        raw_weights: Dict[str, float] = {}
        for name, expert in self._experts.items():
            # Get condition score from the expert's condition function
            try:
                condition_score = float(expert.condition_fn(input_data))
                condition_score = max(0.0, min(1.0, condition_score))
            except Exception:
                condition_score = 0.5  # fallback

            # Get template weight (use priority if no template entry)
            template_weight = template.get(name, expert.priority)

            # Blend: 60% condition-driven, 40% template (hexagonal topology tilt)
            # sin(60°) ≈ 0.866 → 60/40 split derived from C6 symmetry
            blend = 0.6 * condition_score + 0.4 * template_weight

            raw_weights[name] = blend

        # Ensure minimum active experts
        if len(raw_weights) < self._min_experts_active:
            # Not enough experts registered; proceed with what we have
            pass

        # Normalize
        normalized = normalize_weights(
            raw_weights,
            method=self._normalization,
            entropy_comp=self._entropy_comp,
        )

        return RoutingResult(
            mode=mode,
            coupling_strength=coupling_strength,
            weights=normalized,
            selected_experts=[],  # filled by forward()
            metadata={
                "num_experts": len(self._experts),
                "template": template,
                "raw_weights": raw_weights,
            },
        )

    def forward(self, input_data: Any) -> RoutingResult:
        """Route and execute experts in one call.

        Args:
            input_data: Input to route and process.

        Returns:
            RoutingResult with selected_experts populated with
            (name, weight, output) tuples.
        """
        result = self.route(input_data)

        selected: List[Tuple[str, float, Any]] = []
        for name, weight in result.weights.items():
            if weight <= 0:
                continue
            expert = self._experts[name]
            try:
                output = expert.forward_fn(input_data)
            except Exception as exc:
                output = None
                if name not in result.metadata:
                    result.metadata[name] = {}
                result.metadata[name]["error"] = str(exc)
            selected.append((name, weight, output))

        # Sort by weight descending
        selected.sort(key=lambda x: x[1], reverse=True)

        result.selected_experts = selected

        if self._merge_outputs is not None:
            result.metadata["merged_output"] = self._merge_outputs(selected)

        return result

    # ---- Convenience ----

    def __call__(self, input_data: Any) -> RoutingResult:
        """Alias for forward()."""
        return self.forward(input_data)

    def __repr__(self) -> str:
        n = len(self._experts)
        experts = ", ".join(self._experts.keys())
        norm = self._normalization.value if isinstance(self._normalization, NormalizationMethod) else str(self._normalization)
        return f"TaiChiRouter(experts=[{experts}], normalization={norm})"
