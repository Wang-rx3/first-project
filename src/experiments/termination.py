"""Episode termination classification shared by logs and training figures.

The environment distinguishes true terminal collisions from finite-horizon
truncation.  Keeping this classification in one small module prevents the
plotting layer from inferring a failure merely from a short episode.
"""

from __future__ import annotations

from typing import Any, Mapping


def truthy(value: Any) -> bool:
    """Parse the boolean spellings used by JSONL/CSV diagnostics."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y", "t"}


def classify_episode_outcome(
    row: Mapping[str, Any],
    *,
    max_steps: int,
) -> str:
    """Return a conservative, mutually exclusive episode outcome class.

    ``time_limit`` is a normal finite-horizon truncation when the configured
    horizon is reached.  A short row without a reliable reason is deliberately
    classified as ``unknown_short`` rather than being called an early failure.
    """

    reason = str(row.get("termination_reason", "")).strip().lower()
    collision = truthy(row.get("collision_occurred", row.get("collision", False)))
    terminated = truthy(row.get("terminated", False))
    truncated = truthy(row.get("truncated", False))

    if reason == "collision" or collision:
        return "collision"
    if terminated:
        return "early_termination"
    if reason in {"early_termination", "task_failure", "invalid_state", "failure"}:
        return "early_termination"
    if truncated or reason in {"time_limit", "truncated"}:
        return "time_limit"

    try:
        window_count = int(float(row.get("window_count", row.get("steps", max_steps))))
    except (TypeError, ValueError):
        window_count = max_steps
    if window_count < int(max_steps):
        return "unknown_short"
    return "normal"


def outcome_marker(outcome_class: str) -> str | None:
    """Return the main reward-figure marker for an outcome class."""

    return {
        "collision": "X",
        "early_termination": "^",
    }.get(str(outcome_class))


def outcome_label(outcome_class: str) -> str:
    """Human-readable label used in legends and audit tables."""

    return {
        "collision": "Collision termination",
        "early_termination": "Non-collision early termination",
        "time_limit": "Normal time-limit truncation",
        "unknown_short": "Unknown short episode",
        "normal": "Normal episode",
    }.get(str(outcome_class), "Unknown outcome")


__all__ = [
    "classify_episode_outcome",
    "outcome_label",
    "outcome_marker",
    "truthy",
]
