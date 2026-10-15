"""Pure regression tests for the offline near-goal latency audit.

These tests exercise only deterministic parsing/metric helpers.  They do not
import the controller, simulator, policy, training, rollout, or environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.experiments import single_uav_near_goal_capture_latency_audit as audit


def _row(step: int, distance_after: float) -> dict[str, float | int]:
    return {"_step": step, "_distance_post": distance_after}


def test_distance_progress_uses_post_step_distance_only() -> None:
    rows = [
        {"_step": 0, "_distance_post": 4.0, "_distance_pre": 99.0},
        {"_step": 1, "_distance_post": 3.0, "_distance_pre": 4.0},
        {"_step": 2, "_distance_post": 2.0, "_distance_pre": 3.0},
    ]

    result = audit._distance_progress(rows)

    assert result["distance_source"] == "goal_distance_after_m_post_step"
    assert result["distance_at_window_start_m"] == 4.0
    assert result["distance_at_window_end_m"] == 2.0
    assert result["net_distance_reduction_m"] == 2.0


def test_linear_slope_and_r2_are_deterministic() -> None:
    result = audit._linear_slope([_row(2, 3.0), _row(0, 5.0), _row(1, 4.0)])

    assert result["n"] == 3
    assert result["slope_m_per_step"] == -1.0
    assert result["r2"] == 1.0


def test_vector_norm_requires_three_components() -> None:
    assert audit._vector_norm("[3, 4, 0]", "vector") == 5.0

    try:
        audit._vector_norm("[3, 4]", "vector")
    except ValueError as exc:
        assert "three components" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("a non-3D vector must be rejected")


def test_audit_module_has_no_runtime_environment_or_controller_imports() -> None:
    path = Path(audit.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden_fragments = (
        "controller",
        "environment",
        "simulator",
        "training",
        "rollout",
    )
    runtime_imports = {name for name in imported_modules | imported_names if name}
    assert not any(
        any(fragment in module.lower() for fragment in forbidden_fragments)
        for module in runtime_imports
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        for node in ast.walk(tree)
    )
