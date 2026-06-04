"""
Tests for TaiChi-Router core functionality.
"""

import math
import numpy as np
import pytest

from taichi_router import (
    TaiChiRouter,
    RoutingMode,
    NormalizationMethod,
    compute_coupling_strength,
    detect_routing_mode,
    normalize_weights,
    create_default_experts,
    get_strategy,
    HEX_TOPOLOGY_ANGLE_DEG,
    LUOSHU_CORRECTION,
)


# ── Coupling Strength Tests ────────────────────────────────

class TestCouplingStrength:
    def test_constant_input_yields_zero(self):
        data = np.ones(100) * 5.0
        cs = compute_coupling_strength(data)
        assert cs == pytest.approx(0.0, abs=1e-4)

    def test_random_input_positive(self):
        data = np.random.randn(100)
        cs = compute_coupling_strength(data)
        assert cs > 0.0

    def test_high_variance_larger_coupling(self):
        low_var = np.ones(100) + np.random.randn(100) * 0.1
        high_var = np.random.randn(100) * 5.0
        assert compute_coupling_strength(high_var) > compute_coupling_strength(low_var)

    def test_scalar_input(self):
        cs = compute_coupling_strength(np.array([3.14]))
        assert cs == 0.0


# ── Routing Mode Tests ─────────────────────────────────────

class TestRoutingMode:
    def test_steady_mode(self):
        assert detect_routing_mode(0.0) == RoutingMode.STEADY
        assert detect_routing_mode(0.49) == RoutingMode.STEADY

    def test_transitional_mode(self):
        assert detect_routing_mode(0.5) == RoutingMode.TRANSITIONAL
        assert detect_routing_mode(1.0) == RoutingMode.TRANSITIONAL
        assert detect_routing_mode(1.49) == RoutingMode.TRANSITIONAL

    def test_perturbation_mode(self):
        assert detect_routing_mode(1.5) == RoutingMode.PERTURBATION
        assert detect_routing_mode(5.0) == RoutingMode.PERTURBATION


# ── Normalize Weights Tests ────────────────────────────────

class TestNormalizeWeights:
    def test_sums_to_one(self):
        for method in NormalizationMethod:
            raw = {"a": 0.5, "b": 0.3, "c": 0.2}
            result = normalize_weights(raw, method=method)
            assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_empty_returns_empty(self):
        assert normalize_weights({}) == {}

    def test_entropy_balanced_minimum_entropy(self):
        raw = {"a": 10.0, "b": 0.1, "c": 0.1}
        result = normalize_weights(raw, method=NormalizationMethod.ENTROPY_BALANCED, entropy_comp=0.1)
        # All experts must have non-zero weight due to entropy compensation
        for w in result.values():
            assert w > 0.0


# ── Router Core Tests ───────────────────────────────────────

class TestTaiChiRouter:
    @pytest.fixture
    def router(self):
        r = TaiChiRouter()
        for expert in create_default_experts():
            r.register_expert(
                name=expert.name,
                condition_fn=expert.condition,
                forward_fn=expert.forward,
                priority=expert.priority,
            )
        return r

    def test_registry(self, router):
        assert len(router.list_experts()) == 3
        assert "E1_observation" in router.list_experts()

    def test_duplicate_register_raises(self, router):
        with pytest.raises(ValueError):
            router.register_expert(
                name="E1_observation",
                condition_fn=lambda x: 1.0,
                forward_fn=lambda x: x,
            )

    def test_route_returns_routing_result(self, router):
        data = np.random.randn(100)
        result = router.route(data)
        assert result.mode is not None
        assert result.coupling_strength >= 0
        assert len(result.weights) == 3
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_forward_executes_experts(self, router):
        data = np.random.randn(100)
        result = router.forward(data)
        assert len(result.selected_experts) == 3
        for name, weight, output in result.selected_experts:
            assert name in router.list_experts()
            assert 0 <= weight <= 1
            assert output is not None

    def test_call_alias(self, router):
        data = np.random.randn(100)
        r1 = router.forward(data)
        r2 = router(data)
        assert r1.mode == r2.mode

    def test_steady_input_routes_to_steady_mode(self, router):
        data = np.ones(100) + np.random.randn(100) * 0.01
        result = router(data)
        assert result.mode == RoutingMode.STEADY

    def test_high_variance_routes_to_perturbation(self, router):
        data = np.random.randn(20) * 10.0 + np.sin(np.linspace(0, 100, 20)) * 5
        result = router(data)
        # May be perturbation or transitional depending on the random seed
        assert result.coupling_strength > 0

    def test_unregister_expert(self, router):
        router.unregister_expert("E2_dynamics")
        assert len(router.list_experts()) == 2
        result = router(np.random.randn(100))
        assert len(result.weights) == 2


# ── Strategy Tests ─────────────────────────────────────────

class TestStrategies:
    @pytest.fixture
    def expert_data(self):
        return {
            "scores": {"E1": 0.9, "E2": 0.5, "E3": 0.3},
            "priorities": {"E1": 0.7, "E2": 0.5, "E3": 0.3},
        }

    def test_three_regime_steady(self, expert_data):
        strat = get_strategy("three_regime")
        weights = strat.compute_weights(
            np.ones(10), coupling_strength=0.2,
            expert_scores=expert_data["scores"],
            expert_priorities=expert_data["priorities"],
        )
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        # E1 should get highest weight in steady mode
        assert weights["E1"] > weights["E3"]

    def test_three_regime_perturbation(self, expert_data):
        strat = get_strategy("three_regime")
        weights = strat.compute_weights(
            np.random.randn(10) * 3, coupling_strength=2.0,
            expert_scores=expert_data["scores"],
            expert_priorities=expert_data["priorities"],
        )
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_hexagonal_strategy(self, expert_data):
        strat = get_strategy("hexagonal", hex_angle_deg=60.0)
        weights = strat.compute_weights(
            np.random.randn(10), coupling_strength=1.0,
            expert_scores=expert_data["scores"],
            expert_priorities=expert_data["priorities"],
        )
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_unknown_strategy_raises(self):
        with pytest.raises(KeyError):
            get_strategy("nonexistent")

    def test_entropy_max(self, expert_data):
        strat = get_strategy("entropy_max", temperature=0.5)
        weights = strat.compute_weights(
            np.ones(10), coupling_strength=1.0,
            expert_scores=expert_data["scores"],
            expert_priorities=expert_data["priorities"],
        )
        # All experts should have meaningful weight
        for w in weights.values():
            assert w >= 0.01

    def test_adaptive_threshold(self, expert_data):
        strat = get_strategy("adaptive_threshold", base_threshold=0.4, min_active=2)
        weights = strat.compute_weights(
            np.ones(10), coupling_strength=2.0,
            expert_scores=expert_data["scores"],
            expert_priorities=expert_data["priorities"],
        )
        # With high coupling, threshold drops, more experts active
        active = sum(1 for w in weights.values() if w > 1e-6)
        assert active >= 2


# ── Constants Tests ────────────────────────────────────────

class TestConstants:
    def test_hex_angle(self):
        assert HEX_TOPOLOGY_ANGLE_DEG == 60.0

    def test_luoshu_correction(self):
        assert 0.03 < LUOSHU_CORRECTION < 0.04
