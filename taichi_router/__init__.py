"""
TaiChi-Router: A Group-Theory-Driven MoE Dynamic Routing Engine.

Key components:
- :class:`TaiChiRouter` — Core routing engine
- :class:`Expert` — Abstract expert base class
- :class:`RoutingStrategy` — Pluggable routing strategies
- :func:`compute_coupling_strength` — Input field analysis
"""

from .router import (
    TaiChiRouter,
    RoutingResult,
    RoutingMode,
    NormalizationMethod,
    ExpertSpec,
    compute_coupling_strength,
    detect_routing_mode,
    normalize_weights,
    HEX_TOPOLOGY_ANGLE_DEG,
    DEFAULT_ENTROPY_COMP,
    LUOSHU_CORRECTION,
)

from .experts import (
    Expert,
    ObservationExpert,
    DynamicsExpert,
    VortexExpert,
    create_default_experts,
)

from .strategies import (
    RoutingStrategy,
    ThreeRegimeStrategy,
    EntropyMaximizingStrategy,
    HexagonalGatingStrategy,
    AdaptiveThresholdStrategy,
    get_strategy,
    STRATEGY_REGISTRY,
)

# PyTorch integration (optional)
try:
    from .experts import TorchExpert
    _HAS_TORCH_EXPERT = True
except ImportError:
    _HAS_TORCH_EXPERT = False

__all__ = [
    # Core
    "TaiChiRouter",
    "RoutingResult",
    "RoutingMode",
    "NormalizationMethod",
    "ExpertSpec",
    # Utilities
    "compute_coupling_strength",
    "detect_routing_mode",
    "normalize_weights",
    # Experts
    "Expert",
    "ObservationExpert",
    "DynamicsExpert",
    "VortexExpert",
    "create_default_experts",
    # Strategies
    "RoutingStrategy",
    "ThreeRegimeStrategy",
    "EntropyMaximizingStrategy",
    "HexagonalGatingStrategy",
    "AdaptiveThresholdStrategy",
    "get_strategy",
    "STRATEGY_REGISTRY",
    # Constants
    "HEX_TOPOLOGY_ANGLE_DEG",
    "DEFAULT_ENTROPY_COMP",
    "LUOSHU_CORRECTION",
]

__version__ = "0.1.0"
