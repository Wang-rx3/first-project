"""Pure composition of learned communication and fixed navigation actions.

The composer is deliberately kept outside the simulator.  A learned policy
supplies only the communication portion of an action; this module computes the
requested motion from each UAV's raw local estimate and goal.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from .fixed_navigation import FixedNavigationConfig, fixed_navigation_command


def _copy_command(command: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a controller diagnostic while keeping array values independent."""

    copied: dict[str, Any] = {}
    for key, value in command.items():
        if isinstance(value, np.ndarray):
            copied[str(key)] = value.copy()
        else:
            copied[str(key)] = copy.deepcopy(value)
    return copied


def compose_fixed_navigation_actions(
    raw_observations: Mapping[str, Mapping[str, Any]],
    communication_actions: Mapping[str, Mapping[str, Any]],
    config: FixedNavigationConfig | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Compose communication actions with fixed-navigation commands.

    ``raw_observations`` and ``communication_actions`` must contain exactly the
    same UAV identifiers.  Only ``estimate`` and ``goal`` are read from each
    raw observation.  Communication actions retain their measurement source
    and outgoing bandwidth mapping; a learned ``delta_v`` is rejected so that
    the motion source cannot be ambiguous.
    """

    controller_config = config if config is not None else FixedNavigationConfig()
    if not isinstance(controller_config, FixedNavigationConfig):
        raise TypeError("config must be a FixedNavigationConfig")
    if not isinstance(raw_observations, Mapping) or not isinstance(communication_actions, Mapping):
        raise TypeError("raw_observations and communication_actions must be mappings")

    raw_keys = {str(key) for key in raw_observations}
    communication_keys = {str(key) for key in communication_actions}
    if raw_keys != communication_keys:
        raise ValueError("raw_observations and communication_actions must have identical UAV keys")

    actions: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for uid in sorted(raw_keys):
        observation = raw_observations.get(uid)
        communication = communication_actions.get(uid)
        if not isinstance(observation, Mapping):
            raise TypeError(f"raw observation for {uid} must be a mapping")
        if not isinstance(communication, Mapping):
            raise TypeError(f"communication action for {uid} must be a mapping")
        if "delta_v" in communication:
            raise ValueError(
                f"communication action for {uid} must not contain learned delta_v; "
                "fixed navigation owns motion"
            )

        command = fixed_navigation_command(
            observation["estimate"][:6],
            observation["goal"],
            controller_config,
        )
        measurement_uav_id = communication.get("measurement_uav_id", "none")
        bandwidth = communication.get("bandwidth", communication.get("outgoing_bandwidth", {}))
        if not isinstance(bandwidth, Mapping):
            raise TypeError(f"bandwidth for {uid} must be a mapping")
        copied_bandwidth = copy.deepcopy(dict(bandwidth))
        requested_delta = np.asarray(command["command_delta_v_mps"], dtype=float).reshape(3).copy()
        actions[uid] = {
            "measurement_uav_id": copy.deepcopy(measurement_uav_id),
            "bandwidth": copied_bandwidth,
            "delta_v": requested_delta.tolist(),
        }
        diagnostics[uid] = {
            "motion_action_mode": "fixed_navigation_comm_v1",
            "requested_delta_v": requested_delta.copy(),
            "fixed_navigation_command": _copy_command(command),
        }
        diagnostics[uid].update(_copy_command(command))
    return actions, diagnostics


__all__ = ["compose_fixed_navigation_actions"]
