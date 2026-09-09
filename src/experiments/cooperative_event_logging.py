"""Window-scoped cooperative-update accounting; never changes EKF state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def accepted_counts(environment: Any) -> dict[str, int]:
    return {str(key): int(ekf.valid_neighbor_count) for key, ekf in environment.ekfs.items()}


def collect_window_events(
    environment: Any, before: Mapping[str, int], *, run_root: Path,
    episode_id: str, split: str, step: int,
) -> list[dict[str, Any]]:
    """The environment resets this list on EVERY step, including empty steps.

    Compare with independent cumulative EKF counters, per target, before writing.
    A failure stops the run rather than emitting incomplete quality statistics.
    """
    after = accepted_counts(environment)
    if set(after) != set(before):
        raise ValueError("cooperative counter UAV identities changed")
    observed = dict.fromkeys(after, 0)
    events: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    run_id = hashlib.sha256(str(run_root.resolve()).encode()).hexdigest()[:20]
    for event in environment.relative_range_update_events:
        target, source = str(event["target_uav_id"]), str(event["source_uav_id"])
        identity = (target, source, event.get("measurement_time_s"), event.get("message_send_time_s"))
        if identity in identities:
            raise ValueError(f"duplicate cooperative update in window {step}: {identity}")
        identities.add(identity)
        if target not in observed:
            raise ValueError(f"unknown cooperative target {target}")
        observed[target] += int(bool(event["accepted"]))
        event_id = hashlib.sha256(json.dumps([run_id, split, episode_id, step, identity]).encode()).hexdigest()
        events.append({**dict(event), "run_id": run_id, "event_id": event_id,
                       "episode_id": episode_id, "split": split, "step": int(step)})
    for target in after:
        delta = after[target] - int(before[target])
        if delta < 0 or delta != observed[target]:
            raise ValueError(f"cooperative event/counter mismatch {episode_id}/{step}/{target}: "
                             f"events={observed[target]}, counter_delta={delta}")
    return events


def require_new_episode(episode_root: Path) -> None:
    if episode_root.exists() and any(episode_root.iterdir()):
        raise FileExistsError(f"refusing to append/overwrite an existing episode: {episode_root}")
