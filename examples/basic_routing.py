"""
Basic routing example for TaiChi-Router.

This example demonstrates the three-mode routing protocol with
default experts on synthetic data with varying perturbation levels.
"""

import numpy as np

from taichi_router import (
    TaiChiRouter,
    create_default_experts,
    compute_coupling_strength,
    detect_routing_mode,
    get_strategy,
)


def main():
    print("=" * 60)
    print("  TaiChi-Router: Basic Routing Example")
    print("=" * 60)

    # ── Setup router ──────────────────────────────────────────
    router = TaiChiRouter(
        normalization="entropy_balanced",
        entropy_comp=0.0618,  # golden-ratio derived
    )

    # Register three canonical experts
    for expert in create_default_experts():
        router.register_expert(
            name=expert.name,
            condition_fn=expert.condition,
            forward_fn=expert.forward,
            priority=expert.priority,
        )

    print(f"\nRouter: {router}")
    print(f"Registered experts: {router.list_experts()}")

    # ── Test three coupling regimes ───────────────────────────
    test_cases = {
        "STEADY (low variance)": np.ones(50) * 0.5 + np.random.randn(50) * 0.02,
        "TRANSITIONAL (moderate variance)": np.random.randn(50) * 0.3,
        "PERTURBATION (high variance)": np.random.randn(50) * 1.2 + np.sin(np.linspace(0, 20, 50)),
    }

    for label, data in test_cases.items():
        cs = compute_coupling_strength(data)
        mode = detect_routing_mode(cs)
        result = router(data)

        print(f"\n{'─' * 50}")
        print(f"  {label}")
        print(f"  Coupling: {cs:.4f} → Mode: {mode.value}")
        print(f"  Expert weights:")
        for name, weight, _ in result.selected_experts:
            bar = "█" * int(weight * 40)
            print(f"    {name:20s} {weight:.3f}  {bar}")

    # ── Custom strategy example ───────────────────────────────
    print(f"\n{'─' * 50}")
    print("  Using HexagonalGatingStrategy")
    hex_router = TaiChiRouter()
    for expert in create_default_experts():
        hex_router.register_expert(
            name=expert.name,
            condition_fn=expert.condition,
            forward_fn=expert.forward,
            priority=expert.priority,
        )

    # Override with hexagonal strategy
    # (This is done by subclassing; here we just demo the strategy directly)
    strategy = get_strategy("hexagonal", hex_angle_deg=60.0)

    data = np.random.randn(100)
    cs = compute_coupling_strength(data)
    scores = {e.name: e.condition(data) for e in create_default_experts()}
    priorities = {e.name: e.priority for e in create_default_experts()}
    weights = strategy.compute_weights(data, cs, scores, priorities)

    print(f"  Coupling: {cs:.4f}")
    for name, w in weights.items():
        bar = "█" * int(w * 40)
        print(f"    {name:20s} {w:.3f}  {bar}")


if __name__ == "__main__":
    main()
