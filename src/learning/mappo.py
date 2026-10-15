"""A small, environment-compatible MAPPO implementation.

The trainer uses a shared edge-aware GAT actor for all UAVs and a centralized
critic fed by ``info["central_state"]``.  It supports the environment's mixed
action contract:

* a masked categorical measurement-source selection;
* a masked Dirichlet bandwidth allocation over potential outgoing edges;
* a bounded continuous three-dimensional velocity increment.

This is intentionally a research baseline, not a replacement for a large
distributed RL framework.  It is CPU-runnable, deterministic when seeded, and
keeps the environment's physical-unit logs separate from normalized learning
features.  The current action space contains a measurement-source choice, a
simplex bandwidth vector, and a bounded velocity increment; there is no
independent relay action.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical, Dirichlet, Normal, TransformedDistribution, TanhTransform

from .gat import EdgeGATNetwork
from .measurement_proxy import (
    add_information_gain_features,
    add_persistent_information_gain_features,
)
from .measurement_quality import add_quality_features
from .normalization import FeatureNormalizer, ObservationNormalizer


# The legacy 37-dimensional contract remains the default for all historical
# checkpoints.  Safety motion-feature experiments explicitly pass 41 through
# ``MAPPOConfig``; the network itself already supports either dimension.
LEGACY_NODE_FEATURE_DIM = 37
DYNAMIC_OBSTACLE_NODE_FEATURE_DIM = 41
# The v3 measurement head appends a fixed, actor-visible descriptor for each
# candidate source.  Keeping the dimension explicit makes checkpoint/config
# incompatibilities fail early instead of silently changing the action head.
ACTION_CONDITIONED_V3_FEATURE_DIM = 12
ACTION_CONDITIONED_V3_NONE_CONTEXT_DIM = 4


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


def _two_layer_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Build the non-graph encoder used by the architecture ablation.

    The encoder deliberately has one hidden layer.  Its input is a fixed
    role-aware slot table assembled from the actor-visible local graph; no
    graph message passing or attention is performed in this path.
    """

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


def _construct_with_seed(factory: Any, seed: int | None, offset: int) -> Any:
    """Construct a module from an isolated deterministic CPU RNG stream.

    Historical runs leave ``seed`` as ``None`` and therefore retain the
    original global-RNG behavior. Corrected paired experiments provide a seed
    so shared modules do not inherit different initial parameters merely
    because the measurement head has a different input dimension.
    """

    if seed is None:
        return factory()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + int(offset))
        return factory()


def _module_state_hash(module: nn.Module) -> str:
    """Hash a module state deterministically for paired-run provenance."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _fit_length(values: np.ndarray, length: int, *, fill: float = 0.0) -> np.ndarray:
    output = np.full(length, fill, dtype=np.float32)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    output[: min(length, flat.size)] = flat[:length]
    return output


def central_state_vector_dim(
    uav_count: int,
    node_feature_dim: int = LEGACY_NODE_FEATURE_DIM,
    edge_feature_dim: int = 11,
    max_obstacles: int = 16,
    include_remaining_time: bool = False,
    critic_measurement_budget_features: bool = False,
) -> int:
    """Return the fixed centralized-critic input dimension."""

    edge_slots = uav_count * uav_count
    obstacle_dim = 12
    return (
        uav_count * node_feature_dim
        + 3 * edge_slots * edge_feature_dim
        + 3 * edge_slots
        + max_obstacles * obstacle_dim
        + int(include_remaining_time)
        + (uav_count if critic_measurement_budget_features else 0)
    )


def _edge_table(
    graph: Mapping[str, Any],
    name: str,
    uav_count: int,
    edge_feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(graph.get(f"{name}_edge_features", np.empty((0, edge_feature_dim))), dtype=np.float32)
    indices = np.asarray(graph.get(f"{name}_edge_index", np.empty((2, 0), dtype=np.int64)), dtype=np.int64)
    table = np.zeros((uav_count * uav_count, edge_feature_dim), dtype=np.float32)
    present = np.zeros(uav_count * uav_count, dtype=np.float32)
    if indices.ndim == 2 and indices.shape[0] == 2:
        for position in range(min(indices.shape[1], features.shape[0])):
            source = int(indices[0, position])
            target = int(indices[1, position])
            if 0 <= source < uav_count and 0 <= target < uav_count:
                slot = source * uav_count + target
                table[slot, : min(edge_feature_dim, features.shape[1])] = features[position, :edge_feature_dim]
                present[slot] = 1.0
    return table.reshape(-1), present


def _obstacle_table(obstacles: Sequence[Mapping[str, Any]], max_obstacles: int) -> np.ndarray:
    result = np.zeros((max_obstacles, 12), dtype=np.float32)
    for row, obstacle in enumerate(obstacles[:max_obstacles]):
        center = _fit_length(np.asarray(obstacle.get("center_m", ()), dtype=np.float32), 3)
        velocity = _fit_length(np.asarray(obstacle.get("velocity_mps", ()), dtype=np.float32), 3)
        result[row, :3] = center
        result[row, 3:6] = velocity
        result[row, 6] = float(obstacle.get("active", 0.0))
        geometry = str(obstacle.get("geometry_type", obstacle.get("obstacle_type", ""))).lower()
        result[row, 7] = float(geometry in {"box", "aabb", "building"})
        result[row, 8] = float(geometry in {"sphere", "vehicle", "person", "pedestrian"})
        if "size_m" in obstacle:
            result[row, 9:12] = _fit_length(np.asarray(obstacle["size_m"], dtype=np.float32), 3)
        elif "radius_m" in obstacle:
            result[row, 9] = float(obstacle["radius_m"])
    return result.reshape(-1)


def central_state_to_vector(
    central_state: Mapping[str, Any] | None,
    *,
    uav_count: int,
    node_feature_dim: int = LEGACY_NODE_FEATURE_DIM,
    edge_feature_dim: int = 11,
    max_obstacles: int = 16,
    include_remaining_time: bool = False,
    critic_measurement_budget_features: bool = False,
) -> np.ndarray:
    """Convert the critic-visible central state to a fixed-size vector."""

    state = central_state or {}
    graph = state.get("graph", {}) if isinstance(state, Mapping) else {}
    node_features = np.asarray(graph.get("node_features", np.empty((0, node_feature_dim))), dtype=np.float32)
    node_table = np.zeros((uav_count, node_feature_dim), dtype=np.float32)
    if node_features.ndim == 2:
        node_table[: min(uav_count, node_features.shape[0]), : min(node_feature_dim, node_features.shape[1])] = node_features[
            :uav_count, :node_feature_dim
        ]
    else:
        estimates = state.get("estimates", {}) if isinstance(state, Mapping) else {}
        covariances = state.get("covariances", {}) if isinstance(state, Mapping) else {}
        for row, key in enumerate(sorted(estimates)[:uav_count]):
            estimate = np.asarray(estimates[key], dtype=np.float32).reshape(-1)
            covariance = np.asarray(covariances.get(key, np.empty((0, 0))), dtype=np.float32)
            fallback = np.concatenate([estimate, np.diag(covariance) if covariance.ndim == 2 else np.empty(0)])
            node_table[row, : min(node_feature_dim, fallback.size)] = fallback[:node_feature_dim]

    vector_parts: list[np.ndarray] = [node_table.reshape(-1)]
    for name in ("effective", "candidate", "potential"):
        table, present = _edge_table(graph, name, uav_count, edge_feature_dim)
        vector_parts.extend([table, present])
    vector_parts.append(_obstacle_table(state.get("obstacles", []), max_obstacles))
    if include_remaining_time:
        value = float(state["remaining_time_fraction"])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("remaining_time_fraction must be finite and in [0, 1]")
        vector_parts.append(np.asarray([value], dtype=np.float32))
    if critic_measurement_budget_features:
        if not isinstance(state, Mapping) or "measurement_budget_remaining_ratios" not in state:
            raise ValueError(
                "measurement_budget_remaining_ratios is required when critic measurement budget features are enabled"
            )
        ratios = state["measurement_budget_remaining_ratios"]
        if not isinstance(ratios, Mapping):
            raise ValueError("measurement_budget_remaining_ratios must be a mapping")
        expected = {f"uav_{index:02d}" for index in range(uav_count)}
        if {str(key) for key in ratios} != expected:
            raise ValueError(
                "measurement_budget_remaining_ratios must contain exactly all UAV identifiers"
            )
        ordered_ratios: list[float] = []
        for uav_id in (f"uav_{index:02d}" for index in range(uav_count)):
            try:
                ratio = float(ratios[uav_id])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "measurement_budget_remaining_ratios contains an invalid value"
                ) from exc
            if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise ValueError(
                    "measurement_budget_remaining_ratios values must be finite and in [0, 1]"
                )
            ordered_ratios.append(ratio)
        vector_parts.append(np.asarray(ordered_ratios, dtype=np.float32))
    result = np.concatenate(vector_parts).astype(np.float32)
    expected = central_state_vector_dim(
        uav_count,
        node_feature_dim,
        edge_feature_dim,
        max_obstacles,
        include_remaining_time,
        critic_measurement_budget_features,
    )
    if result.size != expected:  # defensive guard for future schema changes
        result = _fit_length(result, expected)
    return result


def _masked_logits(logits: Tensor, mask: Tensor, *, none_index: int) -> tuple[Tensor, Tensor]:
    mask = mask.to(device=logits.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != logits.numel():
        padded = torch.zeros_like(logits, dtype=torch.bool)
        padded[: min(mask.numel(), logits.numel())] = mask[: logits.numel()]
        mask = padded
    if not bool(mask.any()):
        mask = mask.clone()
        mask[min(none_index, mask.numel() - 1)] = True
    return torch.where(mask, logits, torch.full_like(logits, -1e9)), mask


def _mean_features(value: Any, feature_dim: int, device: torch.device) -> Tensor:
    def _fit_tensor(tensor: Tensor) -> Tensor:
        flat = tensor.reshape(-1)
        result = torch.zeros(feature_dim, dtype=torch.float32, device=device)
        result[: min(feature_dim, flat.numel())] = flat[:feature_dim]
        return result

    if isinstance(value, Mapping):
        arrays = [np.asarray(item, dtype=np.float32) for item in value.values() if np.asarray(item).size]
        if not arrays:
            return torch.zeros(feature_dim, dtype=torch.float32, device=device)
        stacked = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
        return _fit_tensor(stacked.mean(dim=0))
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return torch.zeros(feature_dim, dtype=torch.float32, device=device)
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        return _fit_tensor(tensor)
    return _fit_tensor(tensor.mean(dim=0))


@dataclass(frozen=True)
class MAPPOConfig:
    episode_objective: str = "legacy"
    include_remaining_time: bool = False
    uav_count: int = 3
    node_feature_dim: int = LEGACY_NODE_FEATURE_DIM
    edge_feature_dim: int = 11
    gat_hidden_dim: int = 64
    gat_output_dim: int = 64
    gat_heads: int = 2
    gat_layers: int = 2
    gat_dropout: float = 0.0
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    max_obstacles: int = 16
    max_delta_v: float = 0.6
    motion_action_mode: str = "joint"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    policy_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 1e-4
    # New stability protocols may slow only the actor/measurement update
    # while keeping the centralized critic learning rate unchanged.  The
    # default preserves the historical single-rate optimizer behavior.
    actor_learning_rate_scale: float = 1.0
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    initial_delta_std: float = 0.20
    max_grad_norm: float = 0.5
    # Stability controls are opt-in so legacy checkpoints and experiments
    # retain the historical MSE/joint-clipping/update behavior.
    critic_loss_type: str = "mse"
    value_clip_enabled: bool = False
    value_clip_epsilon: float = 0.2
    target_kl: float | None = None
    bootstrap_time_limits: bool = False
    # PPO's old log probability must describe the action sampled by the
    # policy.  ``executed`` is retained for historical checkpoints; new
    # protocols use ``sampled`` and log safety projections separately.
    ppo_action_logprob_mode: str = "executed"
    # Measurement-source scoring. ``global_mean`` is the historical head;
    # ``action_conditioned_v2`` scores every candidate source separately;
    # ``action_conditioned_v3`` adds candidate-level, actor-visible quality
    # features and a candidate summary to the explicit ``none`` head.
    measurement_head_mode: str = "global_mean"
    device: str = "cpu"
    # Optional paired-run initialization. ``None`` preserves historical
    # global-RNG behavior; a concrete seed isolates each module's RNG stream.
    parameter_initialization_seed: int | None = None
    # Local graph encoder.  ``gat`` is the historical default; ``mlp_flat``
    # is a fixed-slot, role-aware architecture ablation.
    encoder_type: str = "gat"
    measurement_proxy_full_packet_cost: bool = False
    auxiliary_gradient_diagnostics: bool = False
    # Only the measurement categorical head receives these actor-visible
    # features; bandwidth, velocity, GAT and critic paths stay unchanged.
    use_measurement_quality_features: bool = False
    # Fixed monotone penalty applied to quality-aware candidate measurement
    # logits.  A zero value preserves the historical baseline behavior.
    measurement_quality_logit_penalty: float = 0.0
    candidate_source_ids: list[str] = field(default_factory=list)
    candidate_visible: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    candidate_action_mask: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    candidate_action_features: np.ndarray = field(
        default_factory=lambda: np.empty((0, ACTION_CONDITIONED_V3_FEATURE_DIM), dtype=np.float32)
    )
    candidate_quality_scores: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    none_context_summary: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    none_logit_raw: float = 0.0
    # The budgeted quality-aware branch adds only the remaining-budget ratio
    # to the measurement head; the bandwidth, velocity, GAT, and critic paths
    # remain unchanged.
    use_measurement_budget_features: bool = False
    # Optional train-only constraint for the rate-controlled quality-aware
    # branch.  The target is fitted from baseline training episodes and is
    # never used by the baseline or by inference-time action selection.
    measurement_rate_constraint_enabled: bool = False
    measurement_rate_constraint_coef: float = 0.5
    measurement_rate_tolerance: float = 0.05
    measurement_rate_target_by_uav: Mapping[str, float] | None = None
    # Optional train-only primal-dual control for the measurement-rate
    # constraint.  It is disabled by default so the historical fixed-coef
    # branch and all old checkpoints keep their exact behavior.
    measurement_rate_dual_enabled: bool = False
    measurement_rate_dual_initial: float = 0.0
    measurement_rate_dual_step_size: float = 0.25
    measurement_rate_dual_max: float = 5.0
    # Optional execution-time information-gain/communication-cost features
    # for the measurement head.  These are added only for the new proxy-
    # guided branch; all legacy branches keep a zero-dimensional block.
    use_measurement_proxy_features: bool = False
    measurement_proxy_guidance_enabled: bool = False
    measurement_proxy_teacher_coef: float = 0.25
    measurement_proxy_temperature: float = 0.25
    measurement_proxy_communication_cost_weight: float = 1.0
    # Separate persistent-proxy block.  Keeping it separate preserves the
    # input dimension and checkpoint compatibility of all historical proxy
    # branches.
    use_measurement_persistent_proxy_features: bool = False
    measurement_persistent_proxy_guidance_enabled: bool = False
    measurement_persistent_proxy_horizon: int = 3
    measurement_persistent_proxy_model: Mapping[str, Any] | None = None
    # When enabled for the persistent teacher, charge only communication
    # cost above the baseline-train target for this UAV.  The default is
    # False so historical checkpoints retain the original gain-minus-cost
    # teacher semantics.
    measurement_persistent_proxy_target_relative_cost: bool = False
    # Equivalent target-relative charging for the legacy proxy teacher.  It
    # is used by the communication-constrained branch so the fixed packet
    # floor is not charged twice: the teacher charges only cost above the
    # baseline-train target, while the real communication constraint still
    # measures the full execution-time cost.
    measurement_proxy_target_relative_cost: bool = False
    # Optional one-sided execution-time communication-cost constraint for the
    # proxy-guided measurement head. Targets are fitted from baseline train.
    measurement_communication_constraint_enabled: bool = False
    measurement_communication_cost_target_by_uav: Mapping[str, float] | None = None
    measurement_communication_cost_tolerance: float = 0.05
    measurement_communication_dual_initial: float = 0.0
    measurement_communication_dual_step_size: float = 0.25
    measurement_communication_dual_max: float = 5.0
    measurement_communication_fixed_packet_fraction: float = 0.5
    # Gradient clipping is opt-in for new protocols.  ``joint`` preserves
    # the historical single global norm clip; ``actor_critic`` keeps a large
    # critic target from suppressing the actor/measurement update.
    gradient_clip_scope: str = "joint"
    # ``all`` preserves the historical fixed-navigation communication policy.
    # ``measurement_only`` learns only the masked measurement categorical and
    # delegates navigation plus outgoing bandwidth to the fixed protocol.
    communication_learning_scope: str = "all"
    # Optional centralized-critic accounting context.  The actor never sees
    # this block; when enabled it is appended after remaining time in the
    # CTDE-only central-state vector.
    critic_measurement_budget_features: bool = False

    def __post_init__(self) -> None:
        if self.episode_objective not in {"legacy", "finite_horizon"}:
            raise ValueError("unknown episode_objective")
        if self.episode_objective == "finite_horizon" and (
            self.bootstrap_time_limits or not self.include_remaining_time
        ):
            raise ValueError("finite_horizon requires remaining time and no time-limit bootstrap")
        if self.uav_count < 1:
            raise ValueError("uav_count must be at least 1")
        if self.node_feature_dim < 1 or self.edge_feature_dim < 1:
            raise ValueError("feature dimensions must be positive")
        if self.max_delta_v <= 0 or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid motion or discount configuration")
        if self.motion_action_mode not in {"joint", "fixed_navigation_comm_v1"}:
            raise ValueError("motion_action_mode must be 'joint' or 'fixed_navigation_comm_v1'")
        if self.motion_action_mode == "fixed_navigation_comm_v1":
            if self.ppo_action_logprob_mode != "sampled":
                raise ValueError("fixed_navigation_comm_v1 requires sampled PPO log probabilities")
            if not math.isclose(float(self.max_delta_v), 0.6, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("fixed_navigation_comm_v1 requires max_delta_v=0.6")
        if self.communication_learning_scope not in {"all", "measurement_only"}:
            raise ValueError("communication_learning_scope must be 'all' or 'measurement_only'")
        if self.communication_learning_scope == "measurement_only":
            if self.motion_action_mode != "fixed_navigation_comm_v1":
                raise ValueError(
                    "measurement_only communication learning requires fixed_navigation_comm_v1"
                )
            if self.ppo_action_logprob_mode != "sampled":
                raise ValueError("measurement_only communication learning requires sampled PPO log probabilities")
        if self.policy_epochs < 1 or self.minibatch_size < 1 or self.learning_rate <= 0 or self.initial_delta_std <= 0:
            raise ValueError("invalid optimizer configuration")
        if not math.isfinite(float(self.max_grad_norm)) or float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be finite and positive")
        if not math.isfinite(float(self.actor_learning_rate_scale)) or float(self.actor_learning_rate_scale) <= 0.0:
            raise ValueError("actor_learning_rate_scale must be finite and positive")
        if self.parameter_initialization_seed is not None:
            if isinstance(self.parameter_initialization_seed, bool) or not isinstance(
                self.parameter_initialization_seed, (int, np.integer)
            ):
                raise ValueError("parameter_initialization_seed must be an integer or None")
        if self.encoder_type not in {"gat", "mlp_flat", "mlp_slots_v2"}:
            raise ValueError("encoder_type must be 'gat', 'mlp_flat', or 'mlp_slots_v2'")
        if self.measurement_head_mode not in {
            "global_mean",
            "action_conditioned_v2",
            "action_conditioned_v3",
        }:
            raise ValueError(
                "measurement_head_mode must be 'global_mean', 'action_conditioned_v2', or "
                "'action_conditioned_v3'"
            )
        if self.gradient_clip_scope not in {"joint", "actor_critic"}:
            raise ValueError("gradient_clip_scope must be 'joint' or 'actor_critic'")
        if self.critic_loss_type not in {"mse", "huber"}:
            raise ValueError("critic_loss_type must be mse or huber")
        if not math.isfinite(float(self.value_clip_epsilon)) or float(self.value_clip_epsilon) <= 0.0:
            raise ValueError("value_clip_epsilon must be finite and positive")
        if self.target_kl is not None and (
            not math.isfinite(float(self.target_kl)) or float(self.target_kl) <= 0.0
        ):
            raise ValueError("target_kl must be None or finite and positive")
        if self.ppo_action_logprob_mode not in {"sampled", "executed"}:
            raise ValueError("ppo_action_logprob_mode must be sampled or executed")
        if not math.isfinite(self.measurement_quality_logit_penalty) or self.measurement_quality_logit_penalty < 0.0:
            raise ValueError("measurement_quality_logit_penalty must be finite and non-negative")
        if (
            not math.isfinite(self.measurement_rate_constraint_coef)
            or self.measurement_rate_constraint_coef < 0.0
        ):
            raise ValueError("measurement_rate_constraint_coef must be finite and non-negative")
        if (
            not math.isfinite(self.measurement_rate_tolerance)
            or not 0.0 <= self.measurement_rate_tolerance <= 1.0
        ):
            raise ValueError("measurement_rate_tolerance must be finite and in [0, 1]")
        if self.measurement_rate_constraint_enabled and self.measurement_rate_target_by_uav is None:
            raise ValueError(
                "measurement_rate_target_by_uav is required when the rate constraint is enabled"
            )
        if self.measurement_rate_target_by_uav is not None:
            expected = {f"uav_{index:02d}" for index in range(self.uav_count)}
            if {str(key) for key in self.measurement_rate_target_by_uav} != expected:
                raise ValueError(
                    "measurement_rate_target_by_uav must contain exactly all UAV identifiers"
                )
            if any(
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in self.measurement_rate_target_by_uav.values()
            ):
                raise ValueError("measurement-rate targets must be finite values in [0, 1]")
        for name, value in (
            ("measurement_rate_dual_initial", self.measurement_rate_dual_initial),
            ("measurement_rate_dual_step_size", self.measurement_rate_dual_step_size),
            ("measurement_rate_dual_max", self.measurement_rate_dual_max),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(self.measurement_rate_dual_initial) > float(self.measurement_rate_dual_max):
            raise ValueError("measurement_rate_dual_initial must not exceed measurement_rate_dual_max")
        if self.measurement_rate_dual_enabled:
            if not self.measurement_rate_constraint_enabled:
                raise ValueError(
                    "measurement_rate_dual_enabled requires measurement_rate_constraint_enabled"
                )
            if float(self.measurement_rate_dual_step_size) <= 0.0:
                raise ValueError("measurement_rate_dual_step_size must be positive when dual control is enabled")
            if float(self.measurement_rate_dual_max) <= 0.0:
                raise ValueError("measurement_rate_dual_max must be positive when dual control is enabled")
        if self.measurement_proxy_guidance_enabled and not self.use_measurement_proxy_features:
            raise ValueError(
                "measurement_proxy_guidance_enabled requires use_measurement_proxy_features"
            )
        if (
            self.measurement_persistent_proxy_guidance_enabled
            and not self.use_measurement_persistent_proxy_features
        ):
            raise ValueError(
                "measurement_persistent_proxy_guidance_enabled requires "
                "use_measurement_persistent_proxy_features"
            )
        if self.use_measurement_proxy_features and self.use_measurement_persistent_proxy_features:
            raise ValueError("legacy and persistent proxy feature blocks are mutually exclusive")
        if self.use_measurement_persistent_proxy_features and not isinstance(
            self.measurement_persistent_proxy_model, Mapping
        ):
            raise ValueError(
                "measurement_persistent_proxy_model is required when persistent proxy features are enabled"
            )
        if self.measurement_persistent_proxy_target_relative_cost:
            if not self.measurement_persistent_proxy_guidance_enabled:
                raise ValueError(
                    "target-relative persistent proxy cost requires persistent proxy guidance"
                )
            if self.measurement_communication_cost_target_by_uav is None:
                raise ValueError(
                    "measurement communication-cost targets are required for target-relative persistent proxy cost"
                )
        if self.measurement_proxy_target_relative_cost:
            if not self.measurement_proxy_guidance_enabled:
                raise ValueError(
                    "target-relative proxy cost requires proxy guidance"
                )
            if not self.measurement_communication_constraint_enabled:
                raise ValueError(
                    "target-relative proxy cost requires the communication constraint"
                )
            if self.measurement_communication_cost_target_by_uav is None:
                raise ValueError(
                    "measurement communication-cost targets are required for target-relative proxy cost"
                )
        if (
            isinstance(self.measurement_persistent_proxy_horizon, bool)
            or int(self.measurement_persistent_proxy_horizon) < 1
        ):
            raise ValueError("measurement_persistent_proxy_horizon must be a positive integer")
        if (
            not math.isfinite(float(self.measurement_proxy_teacher_coef))
            or float(self.measurement_proxy_teacher_coef) < 0.0
        ):
            raise ValueError("measurement_proxy_teacher_coef must be finite and non-negative")
        if (
            not math.isfinite(float(self.measurement_proxy_temperature))
            or float(self.measurement_proxy_temperature) <= 0.0
        ):
            raise ValueError("measurement_proxy_temperature must be finite and positive")
        if (
            not math.isfinite(float(self.measurement_proxy_communication_cost_weight))
            or float(self.measurement_proxy_communication_cost_weight) < 0.0
        ):
            raise ValueError(
                "measurement_proxy_communication_cost_weight must be finite and non-negative"
            )
        if self.measurement_communication_constraint_enabled:
            if not (
                self.use_measurement_proxy_features
                or self.use_measurement_persistent_proxy_features
            ):
                raise ValueError("measurement communication constraint requires proxy features")
            if self.measurement_rate_constraint_enabled or self.measurement_rate_dual_enabled:
                raise ValueError("communication and measurement-rate constraints are mutually exclusive")
            if self.measurement_communication_cost_target_by_uav is None:
                raise ValueError("measurement communication-cost targets are required")
        if self.measurement_communication_cost_target_by_uav is not None:
            expected = {f"uav_{index:02d}" for index in range(self.uav_count)}
            if {str(key) for key in self.measurement_communication_cost_target_by_uav} != expected:
                raise ValueError("measurement communication-cost targets must contain all UAV identifiers")
            if any(
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in self.measurement_communication_cost_target_by_uav.values()
            ):
                raise ValueError("measurement communication-cost targets must be in [0, 1]")
        for name, value in (
            ("measurement_communication_cost_tolerance", self.measurement_communication_cost_tolerance),
            ("measurement_communication_dual_initial", self.measurement_communication_dual_initial),
            ("measurement_communication_dual_step_size", self.measurement_communication_dual_step_size),
            ("measurement_communication_dual_max", self.measurement_communication_dual_max),
            ("measurement_communication_fixed_packet_fraction", self.measurement_communication_fixed_packet_fraction),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= float(self.measurement_communication_cost_tolerance) <= 1.0:
            raise ValueError("measurement communication-cost tolerance must be in [0, 1]")
        if not 0.0 <= float(self.measurement_communication_fixed_packet_fraction) <= 1.0:
            raise ValueError("fixed packet fraction must be in [0, 1]")
        if float(self.measurement_communication_dual_initial) > float(self.measurement_communication_dual_max):
            raise ValueError("measurement communication dual initial must not exceed dual max")
        if self.measurement_communication_constraint_enabled:
            if float(self.measurement_communication_dual_step_size) <= 0.0:
                raise ValueError("measurement communication dual step size must be positive")
            if float(self.measurement_communication_dual_max) <= 0.0:
                raise ValueError("measurement communication dual max must be positive")

    @property
    def critic_input_dim(self) -> int:
        return central_state_vector_dim(
            self.uav_count,
            self.node_feature_dim,
            self.edge_feature_dim,
            self.max_obstacles,
            self.include_remaining_time,
            self.critic_measurement_budget_features,
        )


@dataclass
class PolicySample:
    measurement_index: int
    bandwidth: np.ndarray
    delta_raw: np.ndarray
    delta_v: np.ndarray
    old_log_prob: float
    # Execution-time values are diagnostics only in ``sampled`` mode.  The
    # policy fields above remain the sampled action used by PPO.
    executed_measurement_index: int = -1
    executed_bandwidth: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    executed_delta_v: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    executed_link_selection: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool)
    )
    action_execution_changed: bool = False
    measurement_probabilities: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    selected_quality_risk: float = -1.0
    selected_quality_valid: float = 0.0
    measurement_logits_raw: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    measurement_quality_penalty: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    measurement_logits_adjusted: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    measurement_quality_logit_penalty: float = 0.0
    measurement_proxy_gain: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    measurement_proxy_cost: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    measurement_proxy_valid: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    measurement_proxy_utility: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    persistent_proxy_gain: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    persistent_proxy_cost: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    persistent_proxy_valid: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    persistent_proxy_utility: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    # Candidate-level traces are copied into the sample so that diagnostics
    # remain per-UAV.  Reading the policy's last head trace here would make
    # every UAV inherit the trace of whichever UAV was processed last.
    candidate_source_ids: list[str] = field(default_factory=list)
    candidate_visible: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    candidate_action_mask: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    candidate_action_features: np.ndarray = field(
        default_factory=lambda: np.empty(
            (0, ACTION_CONDITIONED_V3_FEATURE_DIM), dtype=np.float32
        )
    )
    candidate_quality_scores: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    none_context_summary: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    none_logit_raw: float = 0.0
    # Fixed-navigation communication mode metadata.  Empty defaults preserve
    # the historical joint-action sample schema and behavior.
    link_selection: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    sampled_link_probabilities: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )


@dataclass
class PolicyDecision:
    actions: dict[str, dict[str, Any]]
    samples: dict[str, PolicySample]
    observations: dict[str, dict[str, Any]]
    central_vector: np.ndarray
    value: float
    graph_traces: dict[str, dict[str, Any]] = field(default_factory=dict)
    measurement_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RolloutTransition:
    decision: PolicyDecision
    rewards: dict[str, float]
    next_central_state: Mapping[str, Any] | None
    done: bool
    terminated: bool = False
    truncated: bool = False


class MAPPOPolicy(nn.Module):
    """Shared actor plus centralized value network."""

    def __init__(self, config: MAPPOConfig, *, uav_ids: Sequence[str] | None = None) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.uav_ids = tuple(uav_ids or (f"uav_{index:02d}" for index in range(config.uav_count)))
        if len(self.uav_ids) != config.uav_count:
            raise ValueError("uav_ids length must equal config.uav_count")
        init_seed = config.parameter_initialization_seed
        self.gat = _construct_with_seed(
            lambda: EdgeGATNetwork(
                config.node_feature_dim,
                config.edge_feature_dim,
                hidden_dim=config.gat_hidden_dim,
                output_dim=config.gat_output_dim,
                heads=config.gat_heads,
                layers=config.gat_layers,
                dropout=config.gat_dropout,
            ),
            init_seed,
            11,
        )
        # Keep the GAT module in the state dict for a uniform checkpoint
        # schema, but do not execute it for the MLP architecture ablation.
        # The flat encoder uses 2*uav_count role slots (received/self and
        # candidate) plus a presence mask, so missing graph nodes cannot be
        # confused with a real zero-valued node.
        flat_input_dim = 2 * config.uav_count * config.node_feature_dim + 2 * config.uav_count
        if config.encoder_type == "mlp_slots_v2":
            flat_input_dim += ((2 * config.uav_count)**2 + 3 * config.uav_count) * (config.edge_feature_dim + 1)
        self.mlp_flat_encoder = (
            _construct_with_seed(
                lambda: _two_layer_mlp(
                    flat_input_dim,
                    config.gat_hidden_dim,
                    config.gat_output_dim,
                ),
                init_seed,
                101,
            )
            if config.encoder_type in {"mlp_flat", "mlp_slots_v2"}
            else None
        )
        # Four pooled edge groups are followed by one measurement mask and
        # one per-destination bandwidth mask. Relay selection is absent from
        # both the actor input and the action distribution.
        context_dim = config.gat_output_dim + 4 * config.edge_feature_dim + (config.uav_count + 1) + config.uav_count
        context_dim += int(config.include_remaining_time)
        self.actor_body = _construct_with_seed(
            lambda: _mlp(context_dim, config.actor_hidden_dim, config.actor_hidden_dim),
            init_seed,
            23,
        )
        quality_dim = 2 * config.uav_count if config.use_measurement_quality_features else 0
        budget_dim = 1 if config.use_measurement_budget_features else 0
        proxy_dim = 3 * config.uav_count if config.use_measurement_proxy_features else 0
        persistent_proxy_dim = (
            3 * config.uav_count
            if config.use_measurement_persistent_proxy_features
            else 0
        )
        measurement_extra_dim = quality_dim + budget_dim + proxy_dim + persistent_proxy_dim
        if config.measurement_head_mode in {"action_conditioned_v2", "action_conditioned_v3"}:
            # The same scorer is used by GAT and the fixed-slot MLP ablation.
            # For a visible source the GAT embedding is used; for a candidate
            # that is not currently in the effective graph, a zero/presence
            # descriptor and the incoming edge features remain available.
            action_input_dim = (
                config.actor_hidden_dim
                + config.gat_output_dim
                + config.edge_feature_dim
                + 5
                + measurement_extra_dim
            )
            if config.measurement_head_mode == "action_conditioned_v3":
                action_input_dim += ACTION_CONDITIONED_V3_FEATURE_DIM
            self.measurement_head = _construct_with_seed(
                lambda: _two_layer_mlp(
                    action_input_dim,
                    config.actor_hidden_dim,
                    1,
                ),
                init_seed,
                37,
            )
            none_context_dim = 2 + (
                ACTION_CONDITIONED_V3_NONE_CONTEXT_DIM
                if config.measurement_head_mode == "action_conditioned_v3"
                else 0
            )
            self.measurement_none_head = _construct_with_seed(
                lambda: nn.Linear(config.actor_hidden_dim + measurement_extra_dim + none_context_dim, 1),
                init_seed,
                139,
            )
            self.measurement_source_encoder = _construct_with_seed(
                lambda: _two_layer_mlp(
                    config.node_feature_dim + 1,
                    config.gat_hidden_dim,
                    config.gat_output_dim,
                ),
                init_seed,
                151,
            )
        else:
            self.measurement_head = _construct_with_seed(
                lambda: nn.Linear(
                    config.actor_hidden_dim
                    + measurement_extra_dim,
                    config.uav_count + 1,
                ),
                init_seed,
                37,
            )
            self.measurement_none_head = None
            self.measurement_source_encoder = None
        self.bandwidth_head = _construct_with_seed(
            lambda: nn.Linear(config.actor_hidden_dim, config.uav_count),
            init_seed,
            53,
        )
        self.delta_mean_head = _construct_with_seed(
            lambda: nn.Linear(config.actor_hidden_dim, 3),
            init_seed,
            71,
        )
        self.delta_logstd = nn.Parameter(torch.full((3,), math.log(config.initial_delta_std)))
        self.critic = _construct_with_seed(
            lambda: _mlp(config.critic_input_dim, config.critic_hidden_dim, 1),
            init_seed,
            89,
        )
        # The communication-only protocol adds one isolated Bernoulli link
        # head after all historical modules have been constructed.  Keeping
        # this branch last leaves the legacy module graph and RNG stream
        # untouched.
        if config.motion_action_mode == "fixed_navigation_comm_v1":
            self.link_selection_head = _construct_with_seed(
                lambda: nn.Linear(config.actor_hidden_dim, config.uav_count),
                init_seed,
                173,
            )
            self.delta_mean_head.requires_grad_(False)
            self.delta_logstd.requires_grad_(False)
            if config.communication_learning_scope == "measurement_only":
                # Keep the historical fixed-communication modules in the
                # state-dict, but make the delegated protocol explicit in
                # autograd and optimizer construction.
                self.bandwidth_head.requires_grad_(False)
                self.link_selection_head.requires_grad_(False)
        else:
            self.link_selection_head = None
        self._last_graph_trace: dict[str, Any] | None = None
        self._last_measurement_head_trace: dict[str, Any] = {}
        self._last_link_logits: Tensor | None = None
        self._last_link_legal_mask: Tensor | None = None
        self.to(self.device)
        self._initial_parameter_hashes = self.parameter_initialization_hashes()

    def parameter_initialization_hashes(self) -> dict[str, str]:
        """Return hashes used to verify paired branch initialization."""

        delta_hash = hashlib.sha256(
            self.delta_logstd.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        hashes = {
            "gat": _module_state_hash(self.gat),
            "actor_body": _module_state_hash(self.actor_body),
            "measurement_head": _module_state_hash(self.measurement_head),
            "bandwidth_head": _module_state_hash(self.bandwidth_head),
            "delta_mean_head": _module_state_hash(self.delta_mean_head),
            "delta_logstd": delta_hash,
            "critic": _module_state_hash(self.critic),
        }
        if self.measurement_none_head is not None:
            hashes["measurement_none_head"] = _module_state_hash(self.measurement_none_head)
        if self.measurement_source_encoder is not None:
            hashes["measurement_source_encoder"] = _module_state_hash(self.measurement_source_encoder)
        # Preserve the legacy GAT hash schema.  The flat encoder is an
        # additional module only for the MLP ablation, so record its hash
        # there instead of adding a misleading ``disabled`` entry to every
        # default GAT experiment.
        if self.mlp_flat_encoder is not None:
            hashes["mlp_flat_encoder"] = _module_state_hash(self.mlp_flat_encoder)
        if self.link_selection_head is not None:
            hashes["link_selection_head"] = _module_state_hash(self.link_selection_head)
        return hashes

    def initial_parameter_initialization_hashes(self) -> dict[str, str]:
        """Return hashes captured immediately after construction."""

        return dict(self._initial_parameter_hashes)

    def _measurement_quality_features(self, observation: Mapping[str, Any]) -> Tensor:
        """Return fixed-size, finite quality features for the measurement head."""

        if not self.config.use_measurement_quality_features:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        risk = _fit_length(
            np.asarray(observation.get("measurement_quality_risk", []), dtype=np.float32),
            self.config.uav_count,
            fill=1.0,
        )
        valid = _fit_length(
            np.asarray(observation.get("measurement_quality_valid", []), dtype=np.float32),
            self.config.uav_count,
            fill=0.0,
        )
        risk = np.clip(np.nan_to_num(risk, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
        valid = np.clip(np.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        return torch.as_tensor(
            np.concatenate([risk, valid]), dtype=torch.float32, device=self.device
        )

    def _measurement_budget_features(self, observation: Mapping[str, Any]) -> Tensor:
        """Return the finite remaining-budget ratio for the measurement head."""

        if not self.config.use_measurement_budget_features:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        value = observation.get("measurement_budget_remaining_ratio", 1.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        value = float(np.clip(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
        return torch.as_tensor([value], dtype=torch.float32, device=self.device)

    def _measurement_proxy_features(self, observation: Mapping[str, Any]) -> Tensor:
        """Return gain, cost and validity blocks for the measurement head."""

        if not self.config.use_measurement_proxy_features:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        gain = _fit_length(
            np.asarray(observation.get("expected_information_gain_norm", []), dtype=np.float32),
            self.config.uav_count,
            fill=0.0,
        )
        cost = _fit_length(
            np.asarray(observation.get("predicted_communication_cost_norm", []), dtype=np.float32),
            self.config.uav_count,
            fill=1.0,
        )
        valid = _fit_length(
            np.asarray(observation.get("proxy_valid", []), dtype=np.float32),
            self.config.uav_count,
            fill=0.0,
        )
        gain = np.clip(np.nan_to_num(gain, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        cost = np.clip(np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        valid = np.clip(np.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        return torch.as_tensor(
            np.concatenate([gain, cost, valid]), dtype=torch.float32, device=self.device
        )

    def _measurement_persistent_proxy_features(self, observation: Mapping[str, Any]) -> Tensor:
        """Return the fixed-horizon gain, cost and validity blocks."""

        if not self.config.use_measurement_persistent_proxy_features:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        gain = _fit_length(
            np.asarray(observation.get("persistent_gain_norm", observation.get("persistent_information_gain_norm", [])), dtype=np.float32),
            self.config.uav_count,
            fill=0.0,
        )
        cost = _fit_length(
            np.asarray(observation.get("proxy_cost_norm", observation.get("persistent_communication_cost_norm", [])), dtype=np.float32),
            self.config.uav_count,
            fill=1.0,
        )
        valid = _fit_length(
            np.asarray(observation.get("persistent_gain_valid", observation.get("persistent_proxy_valid", [])), dtype=np.float32),
            self.config.uav_count,
            fill=0.0,
        )
        gain = np.clip(np.nan_to_num(gain, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        cost = np.clip(np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        valid = np.clip(np.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        return torch.as_tensor(
            np.concatenate([gain, cost, valid]), dtype=torch.float32, device=self.device
        )

    def _measurement_proxy_target_cost(
        self,
        observation: Mapping[str, Any],
        *,
        uav_id: str | None = None,
        persistent: bool = False,
    ) -> float | None:
        """Resolve the train-fitted target used by a target-relative teacher.

        This helper is deliberately separate from the execution-time
        communication-cost constraint.  The latter continues to use the
        complete cost, including the fixed packet floor.  Only the auxiliary
        teacher removes the already-budgeted portion from its charged cost.
        """

        enabled = (
            bool(self.config.measurement_persistent_proxy_target_relative_cost)
            if persistent
            else bool(self.config.measurement_proxy_target_relative_cost)
        )
        if not enabled:
            return None
        targets = self.config.measurement_communication_cost_target_by_uav or {}
        resolved_uav_id = str(uav_id or observation.get("uav_id", ""))
        if resolved_uav_id in targets:
            return float(targets[resolved_uav_id])
        if len(targets) == 1:
            # Useful for standalone policy tests that omit UAV metadata.
            return float(next(iter(targets.values())))
        raise ValueError(
            "uav_id is required to resolve the target-relative proxy cost"
        )

    def _measurement_branch_name(self) -> str:
        if self.config.use_measurement_persistent_proxy_features:
            return "persistent_proxy_guided_quality_mappo"
        if self.config.measurement_communication_constraint_enabled:
            return "communication_constrained_proxy_guided_mappo"
        if self.config.measurement_proxy_guidance_enabled:
            return "proxy_guided_adaptive_budgeted_quality_aware_mappo"
        if self.config.measurement_rate_dual_enabled:
            return "adaptive_budgeted_quality_aware_mappo"
        if self.config.measurement_rate_constraint_enabled:
            return "rate_controlled_quality_aware_mappo"
        if self.config.use_measurement_budget_features:
            return "budgeted_quality_aware_mappo"
        if self.config.use_measurement_quality_features:
            return "quality_aware_mappo"
        return "baseline_mappo"

    def _capture_graph_trace(
        self,
        *,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        embedding: Tensor,
    ) -> dict[str, Any]:
        """Capture detached diagnostics without changing the actor input.

        ``edge_index`` is the physical effective graph supplied by the local
        observation.  The per-layer attention maps additionally contain the
        virtual self-loops inserted inside ``EdgeGATLayer``.  Keeping these
        two representations separate prevents diagnostics from being
        mistaken for radio-link logs.
        """

        attention_maps: list[dict[str, Any]] = []
        for layer_index, recorded in enumerate(self.gat.attention_maps()):
            if recorded is None:
                continue
            attention_edge_index, alpha, logits = recorded
            attention_maps.append(
                {
                    "layer": int(layer_index),
                    "edge_index": attention_edge_index.detach().cpu().numpy().astype(np.int64),
                    "attention": alpha.detach().cpu().numpy().astype(np.float32),
                    "logits": logits.detach().cpu().numpy().astype(np.float32),
                }
            )
        attention_edge_indices = [item["edge_index"] for item in attention_maps]
        attention_coefficients = [item["attention"] for item in attention_maps]
        return {
            "node_embeddings": embedding.detach().cpu().numpy().astype(np.float32),
            "physical_edge_index": edge_index.detach().cpu().numpy().astype(np.int64),
            "physical_edge_features": edge_features.detach().cpu().numpy().astype(np.float32),
            "attention_maps": attention_maps,
            "attention_edge_index": attention_edge_indices,
            "attention_coefficients": attention_coefficients,
            "layer_count": int(len(self.gat.layers)),
            "virtual_self_loops_added": True,
        }

    def _local_context(
        self, observation: Mapping[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[str]]:
        node = torch.as_tensor(np.asarray(observation["node_features"], dtype=np.float32), device=self.device)
        if node.ndim == 1:
            node = node.reshape(1, -1)
        edge_index = torch.as_tensor(
            np.asarray(observation.get("effective_edge_index", np.empty((2, 0))), dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        )
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        edge_features = torch.as_tensor(
            np.asarray(
                observation.get("effective_edge_features", np.empty((0, self.config.edge_feature_dim))),
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if edge_features.numel() == 0:
            edge_features = torch.empty((0, self.config.edge_feature_dim), dtype=torch.float32, device=self.device)
        node_ids = list(observation.get("node_ids", []))
        if self.config.encoder_type in {"mlp_flat", "mlp_slots_v2"}:
            slot_count = 2 * self.config.uav_count
            slot_table = np.zeros(
                (slot_count, self.config.node_feature_dim), dtype=np.float32
            )
            presence = np.zeros(slot_count, dtype=np.float32)
            if len(node_ids) != int(node.shape[0]):
                # The graph interface is the source of role metadata.  A
                # malformed/missing list is treated as absent nodes rather
                # than allowing row order to become an accidental identity.
                node_ids = []
            for row_index, node_id in enumerate(node_ids):
                text_id = str(node_id)
                candidate = text_id.startswith("candidate:")
                token = text_id.split(":", 1)[1] if candidate else text_id
                try:
                    slot_uav = self.uav_ids.index(token)
                except ValueError:
                    continue
                slot = self.config.uav_count + slot_uav if candidate else slot_uav
                if 0 <= slot < slot_count:
                    slot_table[slot, : min(self.config.node_feature_dim, node.shape[1])] = (
                        node[row_index, : self.config.node_feature_dim]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    presence[slot] = 1.0
            flat_input = np.concatenate([slot_table.reshape(-1), presence])
            if self.config.encoder_type == "mlp_slots_v2":
                if len(node_ids) != int(node.shape[0]):
                    raise ValueError("mlp_slots_v2 requires explicit node identities")
                slots = []
                for node_id in node_ids:
                    token = str(node_id).removeprefix("candidate:")
                    if token not in self.uav_ids:
                        raise ValueError(f"unknown graph identity {node_id}")
                    slots.append(self.uav_ids.index(token) + (self.config.uav_count if str(node_id).startswith("candidate:") else 0))
                edge_table = np.zeros((slot_count, slot_count, self.config.edge_feature_dim + 1), np.float32)
                indices = edge_index.detach().cpu().numpy()
                values = edge_features.detach().cpu().numpy()
                for k in range(indices.shape[1]):
                    a, b = (slots[int(v)] for v in indices[:, k])
                    edge_table[a, b, :-1] = values[k]
                    edge_table[a, b, -1] = 1
                blocks = [flat_input, edge_table.reshape(-1)]
                for key in ("candidate_edge_features", "candidate_incoming_edge_features", "potential_edge_features"):
                    table = np.zeros((self.config.uav_count, self.config.edge_feature_dim + 1), np.float32)
                    mapping = observation.get(key, {})
                    if not isinstance(mapping, Mapping):
                        raise ValueError(f"mlp_slots_v2 requires identity-keyed {key}")
                    for identity, features in mapping.items():
                        index = self.uav_ids.index(str(identity))
                        table[index, :-1] = np.asarray(features, dtype=np.float32)
                        table[index, -1] = 1
                    blocks.append(table.reshape(-1))
                flat_input = np.concatenate(blocks)
                if not np.isfinite(flat_input).all():
                    raise FloatingPointError("non-finite slot encoder input")
            flat_input = np.nan_to_num(flat_input, nan=0.0, posinf=0.0, neginf=0.0)
            if self.mlp_flat_encoder is None:  # defensive guard for malformed config
                raise RuntimeError("mlp_flat encoder is not initialized")
            embedding_root = self.mlp_flat_encoder(
                torch.as_tensor(flat_input, dtype=torch.float32, device=self.device)
            )
            embedding = embedding_root.reshape(1, -1)
            self._last_graph_trace = {
                "encoder_type": self.config.encoder_type,
                "node_embeddings": embedding.detach().cpu().numpy().astype(np.float32),
                "physical_edge_index": edge_index.detach().cpu().numpy().astype(np.int64),
                "physical_edge_features": edge_features.detach().cpu().numpy().astype(np.float32),
                "attention_maps": [],
                "attention_edge_index": [],
                "attention_coefficients": [],
                "layer_count": 0,
                "virtual_self_loops_added": False,
                "flat_slot_count": int(slot_count),
                "flat_presence_mask": presence.tolist(),
            }
        else:
            embedding = self.gat(node, edge_index, edge_features)
            self._last_graph_trace = self._capture_graph_trace(
                node_features=node,
                edge_index=edge_index,
                edge_features=edge_features,
                embedding=embedding,
            )
        root = embedding[0]
        effective = _mean_features(observation.get("effective_edge_features", {}), self.config.edge_feature_dim, self.device)
        candidate = _mean_features(observation.get("candidate_edge_features", {}), self.config.edge_feature_dim, self.device)
        candidate_incoming = _mean_features(
            observation.get("candidate_incoming_edge_features", {}), self.config.edge_feature_dim, self.device
        )
        potential = _mean_features(observation.get("potential_edge_features", {}), self.config.edge_feature_dim, self.device)
        action_mask = torch.as_tensor(
            _fit_length(np.asarray(observation.get("action_mask", []), dtype=np.float32), self.config.uav_count + 1),
            dtype=torch.bool,
            device=self.device,
        )
        bandwidth_mask = torch.as_tensor(
            _fit_length(np.asarray(observation.get("bandwidth_mask", []), dtype=np.float32), self.config.uav_count),
            dtype=torch.bool,
            device=self.device,
        )
        context = torch.cat(
            [
                root,
                effective,
                candidate,
                candidate_incoming,
                potential,
                action_mask.to(dtype=torch.float32),
                bandwidth_mask.to(dtype=torch.float32),
            ]
        )
        if self.config.include_remaining_time:
            remaining = float(observation["remaining_time_fraction"])
            if not np.isfinite(remaining) or not 0.0 <= remaining <= 1.0:
                raise ValueError("remaining_time_fraction must be finite and in [0, 1]")
            context = torch.cat([context, context.new_tensor([remaining])])
        return context, action_mask, bandwidth_mask, effective, potential, embedding, node_ids

    def _candidate_action_features_v3(
        self,
        observation: Mapping[str, Any],
        *,
        source: str,
        source_visible: bool,
        node_by_id: Mapping[str, int],
        node: Tensor,
        candidate_available: bool,
        action_allowed: bool,
        edge: Tensor,
    ) -> tuple[Tensor, float]:
        """Build one finite, actor-visible descriptor for a candidate source.

        The descriptor uses only the current local estimate/covariance,
        received link state, UWB availability and masks.  It deliberately
        excludes simulator truth, future NIS, LOS/NLOS truth and future
        communication outcomes.  The returned scalar is a diagnostic quality
        summary used to form the ``none`` context.
        """

        edge_values = _fit_length(
            np.asarray(edge.detach().cpu(), dtype=np.float32).reshape(-1),
            self.config.edge_feature_dim,
            fill=0.0,
        )
        links = observation.get("candidate_incoming_links", {})
        link = links.get(source, {}) if isinstance(links, Mapping) else {}
        if not isinstance(link, Mapping):
            link = {}

        def finite_float(value: Any, default: float = 0.0) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return float(default)
            return number if np.isfinite(number) else float(default)

        def bounded_quality(value: Any, denominator: float, *, invert: bool = False) -> float:
            number = finite_float(value, 0.0)
            scaled = float(np.clip(number / max(float(denominator), 1e-6), 0.0, 1.0))
            return float(1.0 - scaled if invert else scaled)

        available = observation.get("uwb_available", {})
        uwb_valid = bool(available.get(source, False)) if isinstance(available, Mapping) else False
        ranges = observation.get("uwb_ranges", {})
        measured_range = ranges.get(source, np.nan) if isinstance(ranges, Mapping) else np.nan
        measured_range = finite_float(measured_range, -1.0)
        range_quality = float(
            np.clip(1.0 - measured_range / 100.0, 0.0, 1.0)
            if uwb_valid and measured_range >= 0.0
            else 0.0
        )

        def covariance_confidence(row_index: int | None, source_id: str | None = None) -> float:
            if self.config.communication_learning_scope == "measurement_only":
                # The normalized node row stores covariance diagonals after
                # the 15-state block (columns 15:30).  The historical branch
                # intentionally keeps its original column-12:15 descriptor
                # for checkpoint and audit compatibility.  The new
                # measurement-only protocol reads the actor-visible physical
                # covariance instead, which avoids that layout bias.
                resolved_source_id = source_id
                if resolved_source_id is None and row_index is not None:
                    for candidate_id, candidate_row in node_by_id.items():
                        if candidate_row == row_index:
                            resolved_source_id = candidate_id
                            break
                if resolved_source_id == root_id:
                    covariance_value = observation.get("covariance", None)
                else:
                    messages = observation.get("neighbor_messages", {})
                    message = (
                        messages.get(resolved_source_id, {})
                        if isinstance(messages, Mapping)
                        else {}
                    )
                    covariance_value = message.get("covariance") if isinstance(message, Mapping) else None
                try:
                    covariance = np.asarray(covariance_value, dtype=np.float64)
                except (TypeError, ValueError):
                    return 0.0
                if covariance.ndim != 2 or covariance.shape[0] < 3 or covariance.shape[1] < 3:
                    return 0.0
                covariance = covariance[:3, :3]
                if not np.isfinite(covariance).all() or not np.allclose(
                    covariance, covariance.T, rtol=0.0, atol=1e-6
                ):
                    return 0.0
                try:
                    eigenvalues = np.linalg.eigvalsh(covariance)
                except np.linalg.LinAlgError:
                    return 0.0
                if not np.isfinite(eigenvalues).all() or float(eigenvalues.min()) < -1e-6:
                    return 0.0
                trace_per_axis = float(np.trace(covariance) / 3.0)
                if not np.isfinite(trace_per_axis) or trace_per_axis < 0.0:
                    return 0.0
                return float(np.clip(1.0 / (1.0 + trace_per_axis), 0.0, 1.0))
            if row_index is None or row_index < 0 or row_index >= int(node.shape[0]) or int(node.shape[1]) < 15:
                return 0.0
            diagonal = torch.abs(node[row_index, 12:15])
            value = float(torch.mean(diagonal).detach().cpu())
            if not np.isfinite(value):
                return 0.0
            # Node rows may already be normalized.  A bounded monotone score
            # is therefore safer than assuming physical covariance units.
            return float(np.clip(1.0 / (1.0 + value), 0.0, 1.0))

        root_id = str(observation.get("uav_id", self.uav_ids[0]))
        local_covariance_confidence = covariance_confidence(
            node_by_id.get(root_id, 0), root_id
        )
        neighbor_covariance_confidence = covariance_confidence(
            node_by_id.get(source) if source_visible else None,
            source if source_visible else None,
        )
        packet_quality = bounded_quality(
            link.get("packet_loss", edge_values[7] if edge_values.size > 7 else 0.0),
            1.0,
            invert=True,
        )
        delay_quality = bounded_quality(
            link.get("delay_s", edge_values[8] if edge_values.size > 8 else 0.0),
            1.0,
            invert=True,
        )
        aoi_quality = bounded_quality(
            link.get("aoi_s", edge_values[9] if edge_values.size > 9 else 0.0),
            60.0,
            invert=True,
        )
        queue_quality = bounded_quality(
            link.get("queue_length_packets", edge_values[10] if edge_values.size > 10 else 0.0),
            10.0,
            invert=True,
        )
        rssi_quality = bounded_quality(
            finite_float(link.get("rssi_dbm", edge_values[5] if edge_values.size > 5 else -100.0)) + 100.0,
            100.0,
        )
        snr_quality = bounded_quality(
            link.get("snr_db", edge_values[6] if edge_values.size > 6 else 0.0),
            100.0,
        )
        signal_queue_quality = float(
            np.clip(0.5 * (rssi_quality + snr_quality) + 0.5 * queue_quality, 0.0, 1.0)
        )
        # A current, actor-visible proxy for relative measurement variance.
        # It is deliberately not the post-update covariance or future NIS.
        measurement_variance_quality = float(
            np.clip(0.5 * range_quality + 0.5 * packet_quality, 0.0, 1.0)
            if uwb_valid
            else 0.0
        )
        values = np.asarray(
            [
                float(source_visible),
                float(candidate_available),
                float(uwb_valid),
                range_quality,
                local_covariance_confidence,
                neighbor_covariance_confidence,
                measurement_variance_quality,
                packet_quality,
                delay_quality,
                aoi_quality,
                signal_queue_quality,
                float(action_allowed),
            ],
            dtype=np.float32,
        )
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        candidate_quality = float(
            np.clip(
                np.mean(
                    [
                        values[2], values[4], values[5], values[6],
                        values[7], values[8], values[9], values[10],
                    ]
                ),
                0.0,
                1.0,
            )
        )
        return (
            torch.as_tensor(values, dtype=torch.float32, device=self.device),
            candidate_quality,
        )

    def _action_conditioned_measurement_logits(
        self,
        hidden: Tensor,
        observation: Mapping[str, Any],
        embedding: Tensor,
        node_ids: Sequence[str],
        extra_features: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Score each incoming measurement source with its own visible context.

        The graph encoder produces a representation for effective/visible
        nodes.  A candidate source that is not currently visible receives a
        learned missing-source descriptor plus its current incoming edge
        features.  This keeps the action-to-neighbour mapping explicit while
        avoiding hidden truth, future outcomes, or a fabricated neighbour
        state.
        """

        if self.measurement_head_mode_is_global:
            return self.measurement_head(torch.cat([hidden, extra_features])), {}
        if self.measurement_none_head is None or self.measurement_source_encoder is None:
            raise RuntimeError("action-conditioned measurement modules are not initialized")

        node = torch.as_tensor(
            np.asarray(observation.get("node_features", np.empty((0, self.config.node_feature_dim))), dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        if node.ndim == 1:
            node = node.reshape(1, -1)
        node_by_id: dict[str, int] = {
            str(node_id): int(index)
            for index, node_id in enumerate(node_ids)
            if not str(node_id).startswith("candidate:")
        }
        incoming_mapping = observation.get("candidate_incoming_edge_features", {})
        if not isinstance(incoming_mapping, Mapping):
            incoming_mapping = {}
        candidate_all = {
            str(value)
            for value in observation.get("candidate_incoming_sources", [])
        }
        effective_sources = {
            str(value)
            for value in observation.get("effective_sources", [])
        }
        action_mask = _fit_length(
            np.asarray(observation.get("action_mask", []), dtype=np.float32),
            self.config.uav_count + 1,
        ) > 0.5
        budget_ratio = float(
            np.clip(
                np.nan_to_num(
                    observation.get("measurement_budget_remaining_ratio", 1.0),
                    nan=1.0,
                    posinf=1.0,
                    neginf=0.0,
                ),
                0.0,
                1.0,
            )
        )
        zeros_edge = torch.zeros(
            self.config.edge_feature_dim, dtype=torch.float32, device=self.device
        )
        candidate_logits: list[Tensor] = []
        visible_sources: list[float] = []
        candidate_features: list[Tensor] = []
        candidate_quality_scores: list[float] = []
        for index, source in enumerate(self.uav_ids):
            row = node_by_id.get(source)
            visible = row is not None and row < int(node.shape[0])
            if visible:
                raw_source = torch.zeros(
                    self.config.node_feature_dim, dtype=torch.float32, device=self.device
                )
                raw_source[: min(self.config.node_feature_dim, int(node.shape[1]))] = node[
                    row, : self.config.node_feature_dim
                ]
                source_embedding = (
                    embedding[row]
                    if row < int(embedding.shape[0])
                    else self.measurement_source_encoder(
                        torch.cat([raw_source, torch.ones(1, device=self.device)])
                    )
                )
            else:
                raw_source = torch.zeros(
                    self.config.node_feature_dim, dtype=torch.float32, device=self.device
                )
                source_embedding = self.measurement_source_encoder(
                    torch.cat([raw_source, torch.zeros(1, device=self.device)])
                )
            edge_value = incoming_mapping.get(source, zeros_edge.detach().cpu().numpy())
            edge = torch.as_tensor(
                _fit_length(np.asarray(edge_value, dtype=np.float32), self.config.edge_feature_dim),
                dtype=torch.float32,
                device=self.device,
            )
            edge = torch.nan_to_num(edge, nan=0.0, posinf=0.0, neginf=0.0)
            flags = torch.as_tensor(
                [
                    float(source in candidate_all),
                    float(source in effective_sources),
                    float(action_mask[index]),
                    float(visible),
                    budget_ratio,
                ],
                dtype=torch.float32,
                device=self.device,
            )
            if self.config.measurement_head_mode == "action_conditioned_v3":
                v3_features, candidate_quality = self._candidate_action_features_v3(
                    observation,
                    source=source,
                    source_visible=visible,
                    node_by_id=node_by_id,
                    node=node,
                    candidate_available=source in candidate_all,
                    action_allowed=bool(action_mask[index]),
                    edge=edge,
                )
                candidate_features.append(v3_features)
                candidate_quality_scores.append(candidate_quality)
            else:
                v3_features = torch.empty(0, dtype=torch.float32, device=self.device)
            candidate_input = torch.cat(
                [hidden, source_embedding, edge, flags, extra_features, v3_features]
            )
            candidate_logits.append(self.measurement_head(candidate_input).reshape(()))
            visible_sources.append(float(visible))
        any_candidate = float(any(action_mask[: self.config.uav_count]))
        if self.config.measurement_head_mode == "action_conditioned_v3":
            eligible_quality = [
                score
                for score, allowed in zip(
                    candidate_quality_scores,
                    action_mask[: self.config.uav_count],
                )
                if bool(allowed)
            ]
            eligible_features = [
                feature.detach().cpu().numpy()
                for feature, allowed in zip(
                    candidate_features,
                    action_mask[: self.config.uav_count],
                )
                if bool(allowed)
            ]
            none_context = np.asarray(
                [
                    float(len(eligible_quality) / max(1, self.config.uav_count)),
                    float(max(eligible_quality, default=0.0)),
                    float(np.mean(eligible_quality)) if eligible_quality else 0.0,
                    float(max((row[4] for row in eligible_features), default=0.0)),
                ],
                dtype=np.float32,
            )
            none_context_tensor = torch.as_tensor(
                none_context, dtype=torch.float32, device=self.device
            )
        else:
            none_context = np.empty(0, dtype=np.float32)
            none_context_tensor = torch.empty(0, dtype=torch.float32, device=self.device)
        none_input = torch.cat(
            [
                hidden,
                extra_features,
                torch.as_tensor(
                    [budget_ratio, any_candidate],
                    dtype=torch.float32,
                    device=self.device,
                ),
                none_context_tensor,
            ]
        )
        none_logit = self.measurement_none_head(none_input).reshape(())
        return torch.cat([torch.stack(candidate_logits), none_logit.reshape(1)]), {
            "candidate_source_ids": list(self.uav_ids),
            "candidate_visible": np.asarray(visible_sources, dtype=np.float32),
            "candidate_action_mask": action_mask.astype(np.float32),
            "budget_ratio": budget_ratio,
            "candidate_action_features": (
                np.asarray(
                    [feature.detach().cpu().numpy() for feature in candidate_features],
                    dtype=np.float32,
                )
                if candidate_features
                else np.empty((0, ACTION_CONDITIONED_V3_FEATURE_DIM), dtype=np.float32)
            ),
            "candidate_quality_scores": np.asarray(
                candidate_quality_scores, dtype=np.float32
            ),
            "none_context_summary": none_context,
        }

    @property
    def measurement_head_mode_is_global(self) -> bool:
        return self.config.measurement_head_mode == "global_mean"

    def _fixed_local_uav_id(self, observation: Mapping[str, Any], node_ids: Sequence[str]) -> str:
        """Resolve the local identity used by fixed communication masks."""

        fallback = str(node_ids[0]) if node_ids else self.uav_ids[0]
        candidate = str(observation.get("uav_id", fallback)).removeprefix("candidate:")
        if candidate in self.uav_ids:
            return candidate
        for node_id in node_ids:
            token = str(node_id).removeprefix("candidate:")
            if not str(node_id).startswith("candidate:") and token in self.uav_ids:
                return token
        return self.uav_ids[0]

    def _fixed_potential_targets(
        self, observation: Mapping[str, Any]
    ) -> set[str] | None:
        """Return optional explicit potential-link identities."""

        if "potential_targets" not in observation:
            return None
        value = observation.get("potential_targets")
        if isinstance(value, Mapping):
            return {str(key) for key, allowed in value.items() if bool(allowed)}
        try:
            return {str(item) for item in value}
        except TypeError as exc:
            raise ValueError("potential_targets must be a mapping or iterable") from exc

    def _fixed_action_masks(
        self,
        observation: Mapping[str, Any],
        action_mask: Tensor,
        bandwidth_mask: Tensor,
        node_ids: Sequence[str],
    ) -> tuple[Tensor, Tensor]:
        """Apply explicit self/non-potential masking for the fixed protocol."""

        own_id = self._fixed_local_uav_id(observation, node_ids)
        potential = self._fixed_potential_targets(observation)
        measurement_mask = action_mask.clone().to(dtype=torch.bool)
        measurement_mask[self.config.uav_count] = True
        legal_links = bandwidth_mask.clone().to(dtype=torch.bool)
        for index, uid in enumerate(self.uav_ids):
            allowed = bool(measurement_mask[index]) and uid != own_id
            # Receiver-side measurement legality already comes from the
            # incoming action mask.  Outgoing recipients may be asymmetric.
            measurement_mask[index] = allowed
            legal = bool(legal_links[index]) and uid != own_id
            if potential is not None:
                legal = legal and uid in potential
            legal_links[index] = legal
        if not bool(measurement_mask[: self.config.uav_count].any()):
            measurement_mask[:] = False
            measurement_mask[self.config.uav_count] = True
        return measurement_mask, legal_links

    def _actor_distributions(
        self, observation: Mapping[str, Any]
    ) -> tuple[Categorical, Tensor | None, TransformedDistribution | None, Tensor]:
        context, action_mask, bandwidth_mask, _effective, _potential, embedding, node_ids = self._local_context(observation)
        hidden = self.actor_body(context)
        quality_features = self._measurement_quality_features(observation)
        budget_features = self._measurement_budget_features(observation)
        proxy_features = self._measurement_proxy_features(observation)
        persistent_features = self._measurement_persistent_proxy_features(observation)
        measurement_extra_features = torch.cat(
            [quality_features, budget_features, proxy_features, persistent_features]
        )
        raw_measurement_logits, action_trace = self._action_conditioned_measurement_logits(
            hidden,
            observation,
            embedding,
            node_ids,
            measurement_extra_features,
        )
        quality_penalty = torch.zeros(self.config.uav_count, dtype=torch.float32, device=self.device)
        effective_quality_coefficient = (
            float(self.config.measurement_quality_logit_penalty)
            if self.config.use_measurement_quality_features
            else 0.0
        )
        if self.config.use_measurement_quality_features:
            quality_risk = quality_features[: self.config.uav_count]
            quality_valid = quality_features[self.config.uav_count :]
            quality_penalty = (
                effective_quality_coefficient
                * quality_risk
                * quality_valid
            )
        adjusted_measurement_logits = raw_measurement_logits.clone()
        adjusted_measurement_logits[: self.config.uav_count] = (
            adjusted_measurement_logits[: self.config.uav_count] - quality_penalty
        )
        self._last_measurement_head_trace = {
            "raw_logits": raw_measurement_logits.detach().cpu().numpy().astype(np.float32),
            "quality_penalty": torch.cat(
                [quality_penalty, torch.zeros(1, dtype=torch.float32, device=self.device)]
            ).detach().cpu().numpy().astype(np.float32),
            "adjusted_logits": adjusted_measurement_logits.detach().cpu().numpy().astype(np.float32),
            "coefficient": effective_quality_coefficient,
            "branch": self._measurement_branch_name(),
            "measurement_head_mode": self.config.measurement_head_mode,
            **action_trace,
            "proxy_gain": proxy_features[: self.config.uav_count].detach().cpu().numpy().astype(np.float32)
            if self.config.use_measurement_proxy_features
            else np.empty(0, dtype=np.float32),
            "proxy_cost": proxy_features[
                self.config.uav_count : 2 * self.config.uav_count
            ].detach().cpu().numpy().astype(np.float32)
            if self.config.use_measurement_proxy_features
            else np.empty(0, dtype=np.float32),
            "proxy_valid": proxy_features[2 * self.config.uav_count :].detach().cpu().numpy().astype(np.float32)
            if self.config.use_measurement_proxy_features
            else np.empty(0, dtype=np.float32),
            "proxy_utility": (
                proxy_features[: self.config.uav_count]
                - float(self.config.measurement_proxy_communication_cost_weight)
                * (
                    self._teacher_candidate_cost(proxy_features[self.config.uav_count : 2 * self.config.uav_count])
                    if self._measurement_proxy_target_cost(observation) is None
                    else torch.relu(
                        self._teacher_candidate_cost(proxy_features[self.config.uav_count : 2 * self.config.uav_count])
                        - float(self._measurement_proxy_target_cost(observation))
                    )
                )
            ).detach().cpu().numpy().astype(np.float32)
            if self.config.use_measurement_proxy_features
            else np.empty(0, dtype=np.float32),
            "persistent_gain": persistent_features[: self.config.uav_count].detach().cpu().numpy().astype(np.float32)
            if self.config.use_measurement_persistent_proxy_features
            else np.empty(0, dtype=np.float32),
            "persistent_cost": persistent_features[
                self.config.uav_count : 2 * self.config.uav_count
            ].detach().cpu().numpy().astype(np.float32)
            if self.config.use_measurement_persistent_proxy_features
            else np.empty(0, dtype=np.float32),
            "persistent_valid": persistent_features[2 * self.config.uav_count :].detach().cpu().numpy().astype(np.float32)
            if self.config.use_measurement_persistent_proxy_features
            else np.empty(0, dtype=np.float32),
            "persistent_utility": (
                persistent_features[: self.config.uav_count]
                - float(self.config.measurement_proxy_communication_cost_weight)
                * (
                    persistent_features[self.config.uav_count : 2 * self.config.uav_count]
                    if self._measurement_proxy_target_cost(observation, persistent=True) is None
                    else torch.relu(
                        persistent_features[self.config.uav_count : 2 * self.config.uav_count]
                        - float(self._measurement_proxy_target_cost(observation, persistent=True))
                    )
                )
            ).detach().cpu().numpy().astype(np.float32)
            if self.config.use_measurement_persistent_proxy_features
            else np.empty(0, dtype=np.float32),
            "persistent_horizon": int(self.config.measurement_persistent_proxy_horizon),
        }
        if self.config.motion_action_mode == "fixed_navigation_comm_v1":
            fixed_measurement_mask, legal_links = self._fixed_action_masks(
                observation, action_mask, bandwidth_mask, node_ids
            )
        else:
            fixed_measurement_mask, legal_links = action_mask, bandwidth_mask
        measurement_logits, measurement_mask = _masked_logits(
            adjusted_measurement_logits,
            fixed_measurement_mask[: self.config.uav_count + 1],
            none_index=self.config.uav_count,
        )
        if self.config.communication_learning_scope == "measurement_only":
            # The fixed protocol owns navigation and outgoing bandwidth.  Do
            # not even evaluate those heads: this keeps the branch categorical
            # only and makes accidental learned communication gradients
            # impossible.
            self._last_link_logits = None
            self._last_link_legal_mask = legal_links
            return (
                Categorical(logits=measurement_logits),
                None,
                None,
                measurement_mask,
            )
        if self.config.motion_action_mode == "fixed_navigation_comm_v1":
            if self.link_selection_head is None:  # defensive guard for malformed state
                raise RuntimeError("fixed navigation link-selection head is not initialized")
            self._last_link_logits = self.link_selection_head(hidden)
            self._last_link_legal_mask = legal_links
            return (
                Categorical(logits=measurement_logits),
                self.bandwidth_head(hidden),
                None,
                measurement_mask,
            )
        mean = self.delta_mean_head(hidden)
        std = torch.exp(self.delta_logstd).clamp(min=1e-4, max=10.0)
        delta_distribution = TransformedDistribution(
            Normal(mean, std),
            [TanhTransform(cache_size=1)],
        )
        return (
            Categorical(logits=measurement_logits),
            self.bandwidth_head(hidden),
            delta_distribution,
            measurement_mask,
        )

    def _delta_component_limit(self) -> float:
        """Use an invertible componentwise bound with Euclidean feasibility."""

        # Each component is bounded by max_delta/sqrt(3), which guarantees
        # ||delta_v||_2 <= max_delta without a non-invertible radial clip.
        return float(self.config.max_delta_v / math.sqrt(3.0))

    def _uav_id(self, index: int) -> str:
        return "none" if index >= self.config.uav_count else self.uav_ids[index]

    def _act_communication_only(
        self, observation: Mapping[str, Any], *, deterministic: bool = False
    ) -> tuple[dict[str, Any], PolicySample]:
        """Sample the communication-only action used by fixed navigation."""

        measurement, bandwidth_logits, motion_distribution, _measurement_mask = self._actor_distributions(
            observation
        )
        if motion_distribution is not None or self._last_link_logits is None or self._last_link_legal_mask is None:
            raise RuntimeError("fixed communication distributions were not initialized")
        measurement_index = (
            int(torch.argmax(measurement.probs).item())
            if deterministic
            else int(measurement.sample().item())
        )
        legal_mask = self._last_link_legal_mask.to(device=self.device, dtype=torch.bool)
        link_distribution = Bernoulli(logits=self._last_link_logits[legal_mask])
        link_probabilities = torch.zeros(self.config.uav_count, dtype=torch.float32, device=self.device)
        link_probabilities[legal_mask] = link_distribution.probs
        sampled_links = torch.zeros(self.config.uav_count, dtype=torch.bool, device=self.device)
        sampled_links[legal_mask] = (
            link_distribution.probs >= 0.5
            if deterministic
            else link_distribution.sample().to(dtype=torch.bool)
        )
        link_log_prob = link_distribution.log_prob(sampled_links[legal_mask].to(dtype=torch.float32)).sum()

        bandwidth = torch.zeros(self.config.uav_count, dtype=torch.float32, device=self.device)
        active_mask = sampled_links & legal_mask
        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        active_count = int(active_indices.numel())
        if active_count == 1:
            bandwidth[active_indices[0]] = 1.0
            bandwidth_log_prob = torch.zeros((), dtype=torch.float32, device=self.device)
        elif active_count >= 2:
            concentrations = torch.nn.functional.softplus(bandwidth_logits[active_mask]) + 1.0
            bandwidth_distribution = Dirichlet(concentrations)
            active_bandwidth = (
                concentrations / concentrations.sum()
                if deterministic
                else bandwidth_distribution.sample()
            )
            bandwidth[active_mask] = active_bandwidth
            bandwidth_log_prob = bandwidth_distribution.log_prob(active_bandwidth)
        else:
            bandwidth_log_prob = torch.zeros((), dtype=torch.float32, device=self.device)

        log_prob = (
            measurement.log_prob(torch.tensor(measurement_index, device=self.device))
            + link_log_prob
            + bandwidth_log_prob
        )
        bandwidth_action = {
            self.uav_ids[index]: float(bandwidth[index].detach().cpu())
            for index in torch.nonzero(legal_mask, as_tuple=False).flatten().tolist()
        }
        action = {
            "measurement_uav_id": self._uav_id(measurement_index),
            "bandwidth": bandwidth_action,
        }
        quality_risk = np.asarray(observation.get("measurement_quality_risk", []), dtype=np.float32).reshape(-1)
        quality_valid = np.asarray(observation.get("measurement_quality_valid", []), dtype=np.float32).reshape(-1)
        head_trace = self._last_measurement_head_trace
        sample = PolicySample(
            measurement_index=measurement_index,
            bandwidth=bandwidth.detach().cpu().numpy().astype(np.float32),
            delta_raw=np.empty(0, dtype=np.float32),
            delta_v=np.empty(0, dtype=np.float32),
            old_log_prob=float(log_prob.detach().cpu()),
            measurement_probabilities=measurement.probs.detach().cpu().numpy().astype(np.float32),
            selected_quality_risk=(
                float(quality_risk[measurement_index]) if measurement_index < quality_risk.size else -1.0
            ),
            selected_quality_valid=(
                float(quality_valid[measurement_index]) if measurement_index < quality_valid.size else 0.0
            ),
            measurement_logits_raw=np.asarray(head_trace.get("raw_logits", []), dtype=np.float32).copy(),
            measurement_quality_penalty=np.asarray(head_trace.get("quality_penalty", []), dtype=np.float32).copy(),
            measurement_logits_adjusted=np.asarray(head_trace.get("adjusted_logits", []), dtype=np.float32).copy(),
            measurement_quality_logit_penalty=float(head_trace.get("coefficient", 0.0)),
            measurement_proxy_gain=np.asarray(head_trace.get("proxy_gain", []), dtype=np.float32).copy(),
            measurement_proxy_cost=np.asarray(head_trace.get("proxy_cost", []), dtype=np.float32).copy(),
            measurement_proxy_valid=np.asarray(head_trace.get("proxy_valid", []), dtype=np.float32).copy(),
            measurement_proxy_utility=np.asarray(head_trace.get("proxy_utility", []), dtype=np.float32).copy(),
            persistent_proxy_gain=np.asarray(head_trace.get("persistent_gain", []), dtype=np.float32).copy(),
            persistent_proxy_cost=np.asarray(head_trace.get("persistent_cost", []), dtype=np.float32).copy(),
            persistent_proxy_valid=np.asarray(head_trace.get("persistent_valid", []), dtype=np.float32).copy(),
            persistent_proxy_utility=np.asarray(head_trace.get("persistent_utility", []), dtype=np.float32).copy(),
            candidate_source_ids=[str(value) for value in head_trace.get("candidate_source_ids", [])],
            candidate_visible=np.asarray(head_trace.get("candidate_visible", []), dtype=np.float32).copy(),
            candidate_action_mask=np.asarray(head_trace.get("candidate_action_mask", []), dtype=np.float32).copy(),
            candidate_action_features=np.asarray(head_trace.get("candidate_action_features", []), dtype=np.float32).copy(),
            candidate_quality_scores=np.asarray(head_trace.get("candidate_quality_scores", []), dtype=np.float32).copy(),
            none_context_summary=np.asarray(head_trace.get("none_context_summary", []), dtype=np.float32).copy(),
            none_logit_raw=float(
                head_trace.get("raw_logits", [0.0] * (self.config.uav_count + 1))[-1]
                if len(head_trace.get("raw_logits", [])) > self.config.uav_count else 0.0
            ),
            link_selection=sampled_links.detach().cpu().numpy().astype(bool),
            sampled_link_probabilities=link_probabilities.detach().cpu().numpy().astype(np.float32),
        )
        return action, sample

    def _act_measurement_only(
        self, observation: Mapping[str, Any], *, deterministic: bool = False
    ) -> tuple[dict[str, Any], PolicySample]:
        """Sample only the measurement categorical for fixed navigation.

        Outgoing bandwidth is a fixed execution request for every currently
        legal potential target.  It is intentionally absent from the policy
        sample so PPO's probability ratio contains only the measurement
        categorical term.
        """

        measurement, bandwidth_logits, motion_distribution, _measurement_mask = self._actor_distributions(
            observation
        )
        if bandwidth_logits is not None or motion_distribution is not None:
            raise RuntimeError("measurement-only communication distributions were not initialized")
        measurement_index = (
            int(torch.argmax(measurement.probs).item())
            if deterministic
            else int(measurement.sample().item())
        )
        legal_mask = self._last_link_legal_mask
        if legal_mask is None:
            raise RuntimeError("measurement-only fixed action mask was not initialized")
        legal_mask_np = legal_mask.detach().cpu().numpy().astype(bool)
        bandwidth_action = {
            self.uav_ids[index]: 0.5
            for index in np.flatnonzero(legal_mask_np).tolist()
        }
        action = {
            "measurement_uav_id": self._uav_id(measurement_index),
            "bandwidth": bandwidth_action,
        }
        quality_risk = np.asarray(
            observation.get("measurement_quality_risk", []), dtype=np.float32
        ).reshape(-1)
        quality_valid = np.asarray(
            observation.get("measurement_quality_valid", []), dtype=np.float32
        ).reshape(-1)
        head_trace = self._last_measurement_head_trace
        raw_logits = np.asarray(head_trace.get("raw_logits", []), dtype=np.float32).copy()
        sample = PolicySample(
            measurement_index=measurement_index,
            bandwidth=np.empty(0, dtype=np.float32),
            delta_raw=np.empty(0, dtype=np.float32),
            delta_v=np.empty(0, dtype=np.float32),
            old_log_prob=float(
                measurement.log_prob(
                    torch.tensor(measurement_index, dtype=torch.long, device=self.device)
                ).detach().cpu()
            ),
            measurement_probabilities=measurement.probs.detach().cpu().numpy().astype(np.float32),
            selected_quality_risk=(
                float(quality_risk[measurement_index])
                if measurement_index < quality_risk.size
                else -1.0
            ),
            selected_quality_valid=(
                float(quality_valid[measurement_index])
                if measurement_index < quality_valid.size
                else 0.0
            ),
            measurement_logits_raw=raw_logits,
            measurement_quality_penalty=np.asarray(
                head_trace.get("quality_penalty", []), dtype=np.float32
            ).copy(),
            measurement_logits_adjusted=np.asarray(
                head_trace.get("adjusted_logits", []), dtype=np.float32
            ).copy(),
            measurement_quality_logit_penalty=float(head_trace.get("coefficient", 0.0)),
            measurement_proxy_gain=np.asarray(
                head_trace.get("proxy_gain", []), dtype=np.float32
            ).copy(),
            measurement_proxy_cost=np.asarray(
                head_trace.get("proxy_cost", []), dtype=np.float32
            ).copy(),
            measurement_proxy_valid=np.asarray(
                head_trace.get("proxy_valid", []), dtype=np.float32
            ).copy(),
            measurement_proxy_utility=np.asarray(
                head_trace.get("proxy_utility", []), dtype=np.float32
            ).copy(),
            persistent_proxy_gain=np.asarray(
                head_trace.get("persistent_gain", []), dtype=np.float32
            ).copy(),
            persistent_proxy_cost=np.asarray(
                head_trace.get("persistent_cost", []), dtype=np.float32
            ).copy(),
            persistent_proxy_valid=np.asarray(
                head_trace.get("persistent_valid", []), dtype=np.float32
            ).copy(),
            persistent_proxy_utility=np.asarray(
                head_trace.get("persistent_utility", []), dtype=np.float32
            ).copy(),
            candidate_source_ids=[
                str(value) for value in head_trace.get("candidate_source_ids", [])
            ],
            candidate_visible=np.asarray(
                head_trace.get("candidate_visible", []), dtype=np.float32
            ).copy(),
            candidate_action_mask=np.asarray(
                head_trace.get("candidate_action_mask", []), dtype=np.float32
            ).copy(),
            candidate_action_features=np.asarray(
                head_trace.get("candidate_action_features", []), dtype=np.float32
            ).copy(),
            candidate_quality_scores=np.asarray(
                head_trace.get("candidate_quality_scores", []), dtype=np.float32
            ).copy(),
            none_context_summary=np.asarray(
                head_trace.get("none_context_summary", []), dtype=np.float32
            ).copy(),
            none_logit_raw=float(
                raw_logits[-1] if raw_logits.size > self.config.uav_count else 0.0
            ),
            link_selection=np.empty(0, dtype=bool),
            sampled_link_probabilities=np.empty(0, dtype=np.float32),
        )
        return action, sample

    def act(self, observation: Mapping[str, Any], *, deterministic: bool = False) -> tuple[dict[str, Any], PolicySample]:
        if self.config.motion_action_mode == "fixed_navigation_comm_v1":
            if self.config.communication_learning_scope == "measurement_only":
                return self._act_measurement_only(observation, deterministic=deterministic)
            return self._act_communication_only(observation, deterministic=deterministic)
        measurement, bandwidth_logits, delta_distribution, _measurement_mask = self._actor_distributions(observation)
        measurement_index = int(torch.argmax(measurement.probs).item()) if deterministic else int(measurement.sample().item())
        bandwidth_mask = torch.as_tensor(
            _fit_length(np.asarray(observation.get("bandwidth_mask", []), dtype=np.float32), self.config.uav_count),
            dtype=torch.bool,
            device=self.device,
        )
        bandwidth = torch.zeros(self.config.uav_count, dtype=torch.float32, device=self.device)
        if bool(bandwidth_mask.any()):
            concentrations = torch.nn.functional.softplus(bandwidth_logits[bandwidth_mask]) + 1.0
            distribution = Dirichlet(concentrations)
            active_bandwidth = concentrations / concentrations.sum() if deterministic else distribution.sample()
            bandwidth[bandwidth_mask] = active_bandwidth
            bandwidth_log_prob = distribution.log_prob(active_bandwidth)
        else:
            bandwidth_log_prob = torch.zeros((), device=self.device)
        base_distribution = delta_distribution.base_dist
        raw_delta = base_distribution.mean if deterministic else base_distribution.rsample()
        bounded_delta = torch.tanh(raw_delta)
        delta_v = bounded_delta * self._delta_component_limit()
        log_prob = (
            measurement.log_prob(torch.tensor(measurement_index, device=self.device))
            + bandwidth_log_prob
            + delta_distribution.log_prob(bounded_delta).sum()
        )
        bandwidth_action = {
            self.uav_ids[index]: float(bandwidth[index].detach().cpu())
            for index in torch.nonzero(bandwidth_mask, as_tuple=False).flatten().tolist()
        }
        action = {
            "measurement_uav_id": self._uav_id(measurement_index),
            "bandwidth": bandwidth_action,
            "delta_v": delta_v.detach().cpu().numpy().astype(float).tolist(),
        }
        quality_risk = np.asarray(observation.get("measurement_quality_risk", []), dtype=np.float32).reshape(-1)
        quality_valid = np.asarray(observation.get("measurement_quality_valid", []), dtype=np.float32).reshape(-1)
        head_trace = self._last_measurement_head_trace
        sample = PolicySample(
            measurement_index=measurement_index,
            bandwidth=bandwidth.detach().cpu().numpy().astype(np.float32),
            delta_raw=raw_delta.detach().cpu().numpy().astype(np.float32),
            delta_v=delta_v.detach().cpu().numpy().astype(np.float32),
            old_log_prob=float(log_prob.detach().cpu()),
            measurement_probabilities=measurement.probs.detach().cpu().numpy().astype(np.float32),
            selected_quality_risk=(
                float(quality_risk[measurement_index]) if measurement_index < quality_risk.size else -1.0
            ),
            selected_quality_valid=(
                float(quality_valid[measurement_index]) if measurement_index < quality_valid.size else 0.0
            ),
            measurement_logits_raw=np.asarray(
                head_trace.get("raw_logits", []), dtype=np.float32
            ).copy(),
            measurement_quality_penalty=np.asarray(
                head_trace.get("quality_penalty", []), dtype=np.float32
            ).copy(),
            measurement_logits_adjusted=np.asarray(
                head_trace.get("adjusted_logits", []), dtype=np.float32
            ).copy(),
            measurement_quality_logit_penalty=float(head_trace.get("coefficient", 0.0)),
            candidate_source_ids=[
                str(value) for value in head_trace.get("candidate_source_ids", [])
            ],
            candidate_visible=np.asarray(
                head_trace.get("candidate_visible", []), dtype=np.float32
            ).copy(),
            candidate_action_mask=np.asarray(
                head_trace.get("candidate_action_mask", []), dtype=np.float32
            ).copy(),
            candidate_action_features=np.asarray(
                head_trace.get("candidate_action_features", []), dtype=np.float32
            ).copy(),
            candidate_quality_scores=np.asarray(
                head_trace.get("candidate_quality_scores", []), dtype=np.float32
            ).copy(),
            none_context_summary=np.asarray(
                head_trace.get("none_context_summary", []), dtype=np.float32
            ).copy(),
            none_logit_raw=float(
                head_trace.get("raw_logits", [0.0] * (self.config.uav_count + 1))[-1]
                if len(head_trace.get("raw_logits", [])) > self.config.uav_count
                else 0.0
            ),
            measurement_proxy_gain=np.asarray(
                head_trace.get("proxy_gain", []), dtype=np.float32
            ).copy(),
            measurement_proxy_cost=np.asarray(
                head_trace.get("proxy_cost", []), dtype=np.float32
            ).copy(),
            measurement_proxy_valid=np.asarray(
                head_trace.get("proxy_valid", []), dtype=np.float32
            ).copy(),
            measurement_proxy_utility=np.asarray(
                head_trace.get("proxy_utility", []), dtype=np.float32
            ).copy(),
            persistent_proxy_gain=np.asarray(
                head_trace.get("persistent_gain", []), dtype=np.float32
            ).copy(),
            persistent_proxy_cost=np.asarray(
                head_trace.get("persistent_cost", []), dtype=np.float32
            ).copy(),
            persistent_proxy_valid=np.asarray(
                head_trace.get("persistent_valid", []), dtype=np.float32
            ).copy(),
            persistent_proxy_utility=np.asarray(
                head_trace.get("persistent_utility", []), dtype=np.float32
            ).copy(),
        )
        return action, sample

    def act_with_trace(
        self,
        observation: Mapping[str, Any],
        *,
        deterministic: bool = False,
    ) -> tuple[dict[str, Any], PolicySample, dict[str, Any]]:
        """Sample an action and return detached GAT diagnostics.

        The original :meth:`act` return contract is intentionally unchanged.
        This method is used by the pilot logger and never feeds diagnostics
        back into the actor or critic.
        """

        action, sample = self.act(observation, deterministic=deterministic)
        trace = copy.deepcopy(self._last_graph_trace or {})
        return action, sample, trace

    def _evaluate_communication_only(
        self, observation: Mapping[str, Any], sample: PolicySample
    ) -> tuple[Tensor, Tensor]:
        """Evaluate a fixed-navigation communication sample exactly as stored."""

        measurement, bandwidth_logits, motion_distribution, measurement_mask = self._actor_distributions(observation)
        if motion_distribution is not None or self._last_link_logits is None or self._last_link_legal_mask is None:
            raise RuntimeError("fixed communication distributions were not initialized")
        delta_raw = np.asarray(sample.delta_raw, dtype=np.float32).reshape(-1)
        delta_v = np.asarray(sample.delta_v, dtype=np.float32).reshape(-1)
        if delta_raw.size != 0 or delta_v.size != 0:
            raise ValueError("fixed navigation samples must not contain learned motion values")
        measurement_index = int(sample.measurement_index)
        if not 0 <= measurement_index <= self.config.uav_count:
            raise ValueError("measurement index is outside the fixed action table")
        if not bool(measurement_mask[measurement_index]):
            raise ValueError("sampled measurement source is masked in the fixed action table")

        legal_mask = self._last_link_legal_mask.to(device=self.device, dtype=torch.bool)
        links = np.asarray(sample.link_selection, dtype=bool).reshape(-1)
        if links.size != self.config.uav_count:
            raise ValueError("fixed navigation link_selection must have one entry per UAV")
        if np.any(links & ~legal_mask.detach().cpu().numpy().astype(bool)):
            raise ValueError("sampled fixed-navigation link is not legal")
        link_tensor = torch.as_tensor(links, dtype=torch.float32, device=self.device)
        link_distribution = Bernoulli(logits=self._last_link_logits[legal_mask])
        log_prob = measurement.log_prob(torch.tensor(measurement_index, dtype=torch.long, device=self.device))
        log_prob = log_prob + link_distribution.log_prob(link_tensor[legal_mask]).sum()

        bandwidth = np.asarray(sample.bandwidth, dtype=np.float32).reshape(-1).copy()
        if bandwidth.size != self.config.uav_count or not np.isfinite(bandwidth).all():
            raise ValueError("fixed-navigation bandwidth must be finite with one entry per UAV")
        legal = legal_mask.detach().cpu().numpy().astype(bool)
        if np.any(np.abs(bandwidth[~legal]) > 1e-6):
            raise ValueError("non-legal fixed-navigation bandwidth entries must be zero")
        active = links & legal
        if np.any(bandwidth[~active] != 0.0):
            raise ValueError("inactive fixed-navigation bandwidth entries must be zero")
        active_count = int(active.sum())
        if active_count == 0:
            if np.any(np.abs(bandwidth) > 1e-6):
                raise ValueError("zero-link fixed-navigation action must have zero bandwidth")
            bandwidth_entropy = torch.zeros((), dtype=torch.float32, device=self.device)
        elif active_count == 1:
            if not np.isclose(float(bandwidth[active][0]), 1.0, atol=1e-6, rtol=0.0):
                raise ValueError("single-link fixed-navigation bandwidth must equal one")
            if np.any(np.abs(bandwidth[active] - 1.0) > 1e-6):
                raise ValueError("single-link fixed-navigation bandwidth is invalid")
            bandwidth_entropy = torch.zeros((), dtype=torch.float32, device=self.device)
        else:
            active_values = bandwidth[active]
            if np.any(active_values <= 0.0) or not np.isclose(float(active_values.sum()), 1.0, atol=1e-6, rtol=0.0):
                raise ValueError("multi-link fixed-navigation bandwidth must be a positive simplex")
            concentrations = torch.nn.functional.softplus(bandwidth_logits[torch.as_tensor(active, device=self.device)]) + 1.0
            bandwidth_distribution = Dirichlet(concentrations)
            active_tensor = torch.as_tensor(active_values, dtype=torch.float32, device=self.device)
            log_prob = log_prob + bandwidth_distribution.log_prob(active_tensor)
            bandwidth_entropy = bandwidth_distribution.entropy()
        entropy = measurement.entropy() + link_distribution.entropy().sum() + bandwidth_entropy
        return log_prob, entropy

    def evaluate_actions(self, observation: Mapping[str, Any], sample: PolicySample) -> tuple[Tensor, Tensor]:
        if self.config.motion_action_mode == "fixed_navigation_comm_v1":
            if self.config.communication_learning_scope == "measurement_only":
                for name, values in (
                    ("bandwidth", sample.bandwidth),
                    ("delta_raw", sample.delta_raw),
                    ("delta_v", sample.delta_v),
                    ("link_selection", sample.link_selection),
                    ("sampled_link_probabilities", sample.sampled_link_probabilities),
                ):
                    if np.asarray(values).size:
                        raise ValueError(
                            f"measurement-only samples must not contain learned {name} values"
                        )
                measurement, _bandwidth_logits, _motion_distribution, _measurement_mask = self._actor_distributions(
                    observation
                )
                if not 0 <= int(sample.measurement_index) <= self.config.uav_count or not bool(_measurement_mask[int(sample.measurement_index)]):
                    raise ValueError("measurement-only sample selects an invalid or masked category")
                measurement_index = torch.tensor(
                    sample.measurement_index, dtype=torch.long, device=self.device
                )
                return measurement.log_prob(measurement_index), measurement.entropy()
            return self._evaluate_communication_only(observation, sample)
        measurement, bandwidth_logits, delta_distribution, _measurement_mask = self._actor_distributions(observation)
        measurement_index = torch.tensor(sample.measurement_index, dtype=torch.long, device=self.device)
        log_prob = measurement.log_prob(measurement_index)
        bandwidth_mask = torch.as_tensor(
            _fit_length(np.asarray(observation.get("bandwidth_mask", []), dtype=np.float32), self.config.uav_count),
            dtype=torch.bool,
            device=self.device,
        )
        bandwidth_entropy = torch.zeros((), dtype=torch.float32, device=self.device)
        if bool(bandwidth_mask.any()):
            concentrations = torch.nn.functional.softplus(bandwidth_logits[bandwidth_mask]) + 1.0
            bandwidth_distribution = Dirichlet(concentrations)
            active_bandwidth = torch.as_tensor(sample.bandwidth, dtype=torch.float32, device=self.device)[bandwidth_mask]
            active_bandwidth = torch.clamp(active_bandwidth, min=1e-6)
            active_bandwidth = active_bandwidth / active_bandwidth.sum().clamp(min=1e-6)
            log_prob = log_prob + bandwidth_distribution.log_prob(active_bandwidth)
            bandwidth_entropy = bandwidth_distribution.entropy()
        bounded_delta = torch.as_tensor(sample.delta_v, dtype=torch.float32, device=self.device)
        bounded_delta = bounded_delta / self._delta_component_limit()
        bounded_delta = torch.clamp(bounded_delta, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        log_prob = log_prob + delta_distribution.log_prob(bounded_delta).sum()
        # Tanh-squashed entropy has no simple closed form in this small
        # implementation.  The base Gaussian entropy remains a stable
        # exploration diagnostic; PPO's probability ratio uses the exact
        # transformed log-probability above.
        entropy = measurement.entropy() + bandwidth_entropy + delta_distribution.base_dist.entropy().sum()
        return log_prob, entropy

    def measurement_probability(self, observation: Mapping[str, Any]) -> Tensor:
        """Return the differentiable probability of selecting a neighbour.

        The rate constraint uses this expected action probability rather than
        the sampled action, which keeps the additional training signal
        differentiable and avoids treating individual rollout windows as
        independent statistical observations.
        """

        measurement, _bandwidth_logits, _delta_distribution, _measurement_mask = self._actor_distributions(
            observation
        )
        return 1.0 - measurement.probs[self.config.uav_count]

    def _teacher_candidate_cost(self, risk: Tensor) -> Tensor:
        from .measurement_cost import candidate_communication_cost
        return candidate_communication_cost(risk, self.config.measurement_communication_fixed_packet_fraction) if self.config.measurement_proxy_full_packet_cost else risk

    def expected_measurement_communication_cost(self, observation: Mapping[str, Any]) -> Tensor:
        """Return expected communication cost with gradient only to measurement head."""

        if not self.config.measurement_communication_constraint_enabled:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        context, action_mask, _bandwidth_mask, _effective, _potential, embedding, node_ids = self._local_context(observation)
        hidden = self.actor_body(context).detach()
        quality_features = self._measurement_quality_features(observation)
        budget_features = self._measurement_budget_features(observation)
        proxy_features = self._measurement_proxy_features(observation)
        persistent_features = self._measurement_persistent_proxy_features(observation)
        cost_features = persistent_features if self.config.use_measurement_persistent_proxy_features else proxy_features
        link_risk = cost_features[self.config.uav_count : 2 * self.config.uav_count]
        fixed = float(self.config.measurement_communication_fixed_packet_fraction)
        from .measurement_cost import candidate_communication_cost
        candidate_cost = candidate_communication_cost(link_risk, fixed)
        raw_logits, _action_trace = self._action_conditioned_measurement_logits(
            hidden,
            observation,
            embedding.detach(),
            node_ids,
            torch.cat([quality_features, budget_features, proxy_features, persistent_features]),
        )
        quality_penalty = torch.zeros(self.config.uav_count, dtype=torch.float32, device=self.device)
        if self.config.use_measurement_quality_features:
            quality_penalty = (
                float(self.config.measurement_quality_logit_penalty)
                * quality_features[: self.config.uav_count]
                * quality_features[self.config.uav_count :]
            )
        adjusted = raw_logits.clone()
        adjusted[: self.config.uav_count] -= quality_penalty
        policy_logits, _policy_mask = _masked_logits(
            adjusted, action_mask[: self.config.uav_count + 1], none_index=self.config.uav_count
        )
        probabilities = Categorical(logits=policy_logits).probs
        return torch.sum(probabilities[: self.config.uav_count] * candidate_cost)

    def _measurement_proxy_teacher_terms(
        self,
        observation: Mapping[str, Any],
        *,
        uav_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Build the policy and detached teacher distributions once.

        This helper is shared by the training loss and the read-only audit
        surface.  Keeping the utility construction here prevents the audit
        from silently implementing a second teacher formula.
        """

        guidance_enabled = bool(
            self.config.measurement_proxy_guidance_enabled
            or self.config.measurement_persistent_proxy_guidance_enabled
        )
        if not guidance_enabled:
            return None

        context, action_mask, _bandwidth_mask, _effective, _potential, embedding, node_ids = self._local_context(observation)
        hidden = self.actor_body(context).detach()
        quality_features = self._measurement_quality_features(observation)
        budget_features = self._measurement_budget_features(observation)
        proxy_features = self._measurement_proxy_features(observation)
        persistent_features = self._measurement_persistent_proxy_features(observation)
        teacher_features = (
            persistent_features
            if self.config.measurement_persistent_proxy_guidance_enabled
            else proxy_features
        )
        gain = teacher_features[: self.config.uav_count]
        cost = self._teacher_candidate_cost(
            teacher_features[self.config.uav_count : 2 * self.config.uav_count]
        )
        valid = teacher_features[2 * self.config.uav_count :]
        raw_logits, _action_trace = self._action_conditioned_measurement_logits(
            hidden,
            observation,
            embedding.detach(),
            node_ids,
            torch.cat([quality_features, budget_features, proxy_features, persistent_features]),
        )

        quality_penalty = torch.zeros(
            self.config.uav_count, dtype=torch.float32, device=self.device
        )
        if self.config.use_measurement_quality_features:
            quality_risk = quality_features[: self.config.uav_count]
            quality_valid = quality_features[self.config.uav_count :]
            quality_penalty = (
                float(self.config.measurement_quality_logit_penalty)
                * quality_risk
                * quality_valid
            )
        adjusted = raw_logits.clone()
        adjusted[: self.config.uav_count] -= quality_penalty
        policy_logits, policy_mask = _masked_logits(
            adjusted,
            action_mask[: self.config.uav_count + 1],
            none_index=self.config.uav_count,
        )
        policy_distribution = Categorical(logits=policy_logits)

        target_cost = self._measurement_proxy_target_cost(
            observation,
            uav_id=uav_id,
            persistent=bool(self.config.measurement_persistent_proxy_guidance_enabled),
        )
        charged_cost = cost if target_cost is None else torch.relu(cost - target_cost)
        utilities = gain - float(self.config.measurement_proxy_communication_cost_weight) * charged_cost
        utilities = torch.where(valid > 0.5, utilities, torch.full_like(utilities, -1e9))
        teacher_raw = torch.cat(
            [utilities, torch.zeros(1, dtype=torch.float32, device=self.device)]
        )
        teacher_raw = torch.where(
            policy_mask,
            teacher_raw,
            torch.full_like(teacher_raw, -1e9),
        )
        teacher_distribution = Categorical(
            logits=teacher_raw / float(self.config.measurement_proxy_temperature)
        )
        return {
            "policy_distribution": policy_distribution,
            "teacher_distribution": teacher_distribution,
            "policy_mask": policy_mask,
            "gain": gain,
            "cost": cost,
            "charged_cost": charged_cost,
            "valid": valid,
            "utilities": utilities,
            "target_cost": target_cost,
            "raw_logits": raw_logits,
            "adjusted_logits": adjusted,
        }

    def measurement_proxy_teacher_diagnostics(
        self,
        observation: Mapping[str, Any],
        *,
        uav_id: str | None = None,
    ) -> dict[str, Any]:
        """Return detached teacher diagnostics without changing model state.

        This method is intended for frozen audits and plotting.  It does not
        call an optimizer, update a multiplier, or expose truth-derived
        quantities.  Arrays are returned as NumPy values so callers cannot
        accidentally backpropagate through the diagnostic path.
        """

        with torch.no_grad():
            terms = self._measurement_proxy_teacher_terms(observation, uav_id=uav_id)
            if terms is None:
                return {
                    "enabled": False,
                    "policy_probs": np.empty(0, dtype=np.float32),
                    "teacher_probs": np.empty(0, dtype=np.float32),
                    "teacher_utility": np.empty(0, dtype=np.float32),
                    "teacher_gain": np.empty(0, dtype=np.float32),
                    "teacher_cost": np.empty(0, dtype=np.float32),
                    "teacher_charged_cost": np.empty(0, dtype=np.float32),
                    "teacher_valid": np.empty(0, dtype=np.float32),
                    "target_cost": None,
                }
            policy_probs = terms["policy_distribution"].probs.detach().cpu().numpy().astype(np.float32)
            teacher_probs = terms["teacher_distribution"].probs.detach().cpu().numpy().astype(np.float32)
            utility = terms["utilities"].detach().cpu().numpy().astype(np.float32)
            gain = terms["gain"].detach().cpu().numpy().astype(np.float32)
            cost = terms["cost"].detach().cpu().numpy().astype(np.float32)
            charged = terms["charged_cost"].detach().cpu().numpy().astype(np.float32)
            valid = terms["valid"].detach().cpu().numpy().astype(np.float32)
            return {
                "enabled": True,
                "policy_probs": policy_probs,
                "teacher_probs": teacher_probs,
                "teacher_utility": utility,
                "teacher_gain": gain,
                "teacher_cost": cost,
                "teacher_charged_cost": charged,
                "teacher_valid": valid,
                "target_cost": terms["target_cost"],
            }

    def measurement_proxy_teacher_loss(
        self,
        observation: Mapping[str, Any],
        sample: PolicySample | None = None,
        *,
        uav_id: str | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Return a masked, detached-teacher KL loss for the measurement head.

        The context/graph embedding is detached before the measurement head is
        evaluated.  Consequently the auxiliary loss can update only
        ``measurement_head``; it cannot add a gradient to the GAT, actor body,
        bandwidth head, velocity head, or centralized critic.  The teacher is
        constructed solely from the current proxy blocks and the current
        action mask.  A missing proxy is assigned no candidate utility, while
        the explicit ``none`` action remains available.
        """

        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        terms = self._measurement_proxy_teacher_terms(observation, uav_id=uav_id)
        if terms is None:
            return zero, {"proxy_teacher_valid": 0.0, "proxy_teacher_agreement": 0.0}
        policy_distribution = terms["policy_distribution"]
        teacher_distribution = terms["teacher_distribution"]
        valid = terms["valid"]
        target_cost = terms["target_cost"]
        teacher_probs = teacher_distribution.probs.detach()
        log_teacher = torch.log(teacher_probs.clamp_min(1e-12))
        loss = torch.sum(teacher_probs * (log_teacher - policy_distribution.logits.log_softmax(dim=-1)))
        finite_loss = torch.isfinite(loss)
        loss = torch.where(finite_loss, loss, zero)
        agreement = 0.0
        if sample is not None:
            agreement = float(
                int(torch.argmax(teacher_probs).item()) == int(sample.measurement_index)
            )
        return loss, {
            "proxy_teacher_valid": float(torch.any(valid > 0.5).detach().cpu()),
            "proxy_teacher_agreement": agreement,
            "proxy_teacher_persistent": float(
                bool(self.config.measurement_persistent_proxy_guidance_enabled)
            ),
            "proxy_teacher_target_relative_cost": float(target_cost is not None),
            "proxy_teacher_target_cost": float(target_cost) if target_cost is not None else 0.0,
        }

    def value(self, central_vector: np.ndarray | Tensor) -> Tensor:
        vector = torch.as_tensor(central_vector, dtype=torch.float32, device=self.device)
        if vector.ndim == 1:
            vector = vector.unsqueeze(0)
        return self.critic(vector).squeeze(-1)


class MAPPOTrainer:
    """Shared-policy MAPPO trainer for ``MultiUAVJointEnv``."""

    def __init__(
        self,
        config: MAPPOConfig | None = None,
        *,
        uav_ids: Sequence[str] | None = None,
        normalizer: ObservationNormalizer | None = None,
        central_normalizer: FeatureNormalizer | None = None,
        measurement_quality_model: Mapping[str, Any] | None = None,
        measurement_proxy_model: Mapping[str, Any] | None = None,
        measurement_persistent_proxy_model: Mapping[str, Any] | None = None,
        seed: int = 2026,
    ) -> None:
        self.config = config or MAPPOConfig()
        torch.manual_seed(int(seed))
        self.rng = np.random.default_rng(seed)
        self.normalizer = normalizer or ObservationNormalizer()
        self.central_normalizer = central_normalizer or FeatureNormalizer()
        self.measurement_quality_model = copy.deepcopy(measurement_quality_model)
        self.measurement_proxy_model = copy.deepcopy(measurement_proxy_model)
        self.measurement_persistent_proxy_model = copy.deepcopy(
            measurement_persistent_proxy_model
        )
        self.policy = MAPPOPolicy(self.config, uav_ids=uav_ids)
        # Preserve the historical one-group optimizer exactly when no actor
        # learning-rate override is requested.  The second branch is used by
        # the versioned stability experiments and keeps actor/critic rates
        # explicit without changing old checkpoint semantics.
        if float(self.config.actor_learning_rate_scale) == 1.0:
            optimizer_parameters = (
                [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
                if self.config.communication_learning_scope == "measurement_only"
                else self.policy.parameters()
            )
            self.optimizer = torch.optim.Adam(
                optimizer_parameters, lr=float(self.config.learning_rate)
            )
        else:
            critic_parameters = list(self.policy.critic.parameters())
            critic_parameter_ids = {id(parameter) for parameter in critic_parameters}
            actor_parameters = [
                parameter
                for parameter in self.policy.parameters()
                if id(parameter) not in critic_parameter_ids
                and (self.config.communication_learning_scope != "measurement_only" or parameter.requires_grad)
            ]
            if not actor_parameters or not critic_parameters or (
                len({id(parameter) for parameter in actor_parameters} & critic_parameter_ids) != 0
            ):
                raise RuntimeError("MAPPO actor/critic optimizer parameter groups are invalid")
            self.optimizer = torch.optim.Adam(
                [
                    {
                        "params": actor_parameters,
                        "lr": float(self.config.learning_rate)
                        * float(self.config.actor_learning_rate_scale),
                    },
                    {"params": critic_parameters, "lr": float(self.config.learning_rate)},
                ],
                lr=float(self.config.learning_rate),
            )
        self.rollout: list[RolloutTransition] = []
        self.measurement_rate_dual_by_uav: dict[str, float] = {
            uav_id: float(self.config.measurement_rate_dual_initial)
            for uav_id in self.policy.uav_ids
        }
        self.measurement_communication_dual_by_uav: dict[str, float] = {
            uav_id: float(self.config.measurement_communication_dual_initial)
            for uav_id in self.policy.uav_ids
        }

    def measurement_rate_dual_state_dict(self) -> dict[str, float]:
        """Return the detached per-UAV multiplier state for checkpoints."""

        return {
            str(uav_id): float(value)
            for uav_id, value in sorted(self.measurement_rate_dual_by_uav.items())
        }

    def load_measurement_rate_dual_state_dict(self, state: Mapping[str, Any] | None) -> None:
        """Restore dual state when present, while accepting old checkpoints."""

        if state is None:
            return
        expected = set(self.policy.uav_ids)
        if {str(key) for key in state} != expected:
            raise ValueError("measurement-rate dual state must contain exactly all UAV identifiers")
        restored: dict[str, float] = {}
        for uav_id in self.policy.uav_ids:
            value = float(state[uav_id])
            if not math.isfinite(value) or not 0.0 <= value <= float(self.config.measurement_rate_dual_max):
                raise ValueError("measurement-rate dual state contains an invalid multiplier")
            restored[uav_id] = value
        self.measurement_rate_dual_by_uav = restored

    def measurement_communication_dual_state_dict(self) -> dict[str, float]:
        return {
            str(uav_id): float(value)
            for uav_id, value in sorted(self.measurement_communication_dual_by_uav.items())
        }

    def load_measurement_communication_dual_state_dict(
        self, state: Mapping[str, Any] | None
    ) -> None:
        if state is None:
            return
        expected = set(self.policy.uav_ids)
        if {str(key) for key in state} != expected:
            raise ValueError("measurement communication dual state must contain all UAV identifiers")
        restored: dict[str, float] = {}
        for uav_id in self.policy.uav_ids:
            value = float(state[uav_id])
            if not math.isfinite(value) or not 0.0 <= value <= float(self.config.measurement_communication_dual_max):
                raise ValueError("measurement communication dual state contains an invalid multiplier")
            restored[uav_id] = value
        self.measurement_communication_dual_by_uav = restored

    @classmethod
    def from_environment(
        cls,
        environment: Any,
        *,
        seed: int = 2026,
        normalizer: ObservationNormalizer | None = None,
        central_normalizer: FeatureNormalizer | None = None,
        measurement_quality_model: Mapping[str, Any] | None = None,
        measurement_proxy_model: Mapping[str, Any] | None = None,
        measurement_persistent_proxy_model: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> "MAPPOTrainer":
        motion = environment.config.motion
        defaults = {
            "uav_count": len(environment.uav_ids),
            "max_delta_v": motion.max_acceleration_mps2 * motion.dt_s,
        }
        defaults.update(overrides)
        if defaults.get("motion_action_mode") == "fixed_navigation_comm_v1":
            if not (math.isclose(motion.dt_s, .2) and math.isclose(motion.max_acceleration_mps2, 3.)
                    and math.isclose(motion.max_speed_mps, 12.)
                    and np.array_equal(motion.bounds_min_m, [0., 0., 2.])
                    and np.array_equal(motion.bounds_max_m, [200., 200., 60.])):
                raise ValueError("fixed navigation requires the validated motion contract")
        # MAPPOConfig validates the persistent-proxy feature block before the
        # trainer receives the separately stored model.  Mirror the explicit
        # trainer argument into the config so the validation sees the same
        # train-fitted mapping and old configs remain unchanged.
        if measurement_persistent_proxy_model is not None:
            defaults["measurement_persistent_proxy_model"] = copy.deepcopy(
                measurement_persistent_proxy_model
            )
        return cls(
            MAPPOConfig(**defaults),
            uav_ids=environment.uav_ids,
            normalizer=normalizer,
            central_normalizer=central_normalizer,
            measurement_quality_model=measurement_quality_model,
            measurement_proxy_model=measurement_proxy_model,
            measurement_persistent_proxy_model=measurement_persistent_proxy_model,
            seed=seed,
        )

    def act(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        central_state: Mapping[str, Any] | None,
        *,
        deterministic: bool = False,
    ) -> PolicyDecision:
        working_observations = {
            str(uav_id): dict(local)
            for uav_id, local in observations.items()
            if isinstance(local, Mapping)
        }
        if self.config.use_measurement_proxy_features:
            if self.measurement_proxy_model is None:
                raise RuntimeError(
                    "proxy-guided MAPPO requires a train-fitted measurement proxy model"
                )
            working_observations = {
                uav_id: add_information_gain_features(
                    local,
                    self.policy.uav_ids,
                    self.measurement_proxy_model,
                )
                for uav_id, local in working_observations.items()
            }
        if self.config.use_measurement_persistent_proxy_features:
            if self.measurement_persistent_proxy_model is None:
                raise RuntimeError(
                    "persistent-proxy MAPPO requires a train-fitted persistent proxy model"
                )
            working_observations = {
                uav_id: add_persistent_information_gain_features(
                    local,
                    self.policy.uav_ids,
                    self.measurement_persistent_proxy_model,
                    horizon=self.config.measurement_persistent_proxy_horizon,
                )
                for uav_id, local in working_observations.items()
            }
        # Fixed navigation consumes this pre-normalization copy.  The policy
        # still receives the historical normalized observations for its graph
        # and communication heads.
        raw_navigation_observations = (copy.deepcopy(working_observations)
                                      if self.config.motion_action_mode == "fixed_navigation_comm_v1" else {})
        normalized = self.normalizer.transform(working_observations)
        actions: dict[str, dict[str, Any]] = {}
        samples: dict[str, PolicySample] = {}
        graph_traces: dict[str, dict[str, Any]] = {}
        for uav_id in self.policy.uav_ids:
            local = normalized.get(uav_id, {})
            # Keep the agent identifier as non-numeric metadata.  It is not
            # fed to the actor, but lets train-fitted per-UAV teacher targets
            # be resolved deterministically when diagnostics are recorded.
            local["uav_id"] = uav_id
            if self.config.use_measurement_quality_features:
                if self.measurement_quality_model is None:
                    raise RuntimeError(
                        "quality-aware MAPPO requires a warm-up-fitted measurement quality model"
                    )
                local = add_quality_features(local, self.policy.uav_ids, self.measurement_quality_model)
                normalized[uav_id] = local
            action, sample, trace = self.policy.act_with_trace(local, deterministic=deterministic)
            actions[uav_id] = action
            samples[uav_id] = sample
            graph_traces[uav_id] = trace
        navigation_diagnostics: dict[str, dict[str, Any]] = {}
        if self.config.motion_action_mode == "fixed_navigation_comm_v1":
            from ..control.fixed_navigation_execution import compose_fixed_navigation_actions

            actions, navigation_diagnostics = compose_fixed_navigation_actions(
                raw_navigation_observations,
                actions,
            )
        central_vector = central_state_to_vector(
            central_state,
            uav_count=self.config.uav_count,
            node_feature_dim=self.config.node_feature_dim,
            edge_feature_dim=self.config.edge_feature_dim,
            max_obstacles=self.config.max_obstacles,
            include_remaining_time=self.config.include_remaining_time,
            critic_measurement_budget_features=self.config.critic_measurement_budget_features,
        )
        central_vector = self.central_normalizer.transform(central_vector)
        with torch.no_grad():
            value = float(self.policy.value(central_vector).item())
        measurement_diagnostics = {
            uav_id: {
                "measurement_probabilities": sample.measurement_probabilities.copy(),
                "selected_quality_risk": float(sample.selected_quality_risk),
                "selected_quality_valid": float(sample.selected_quality_valid),
                "measurement_logits_raw": sample.measurement_logits_raw.copy(),
                "measurement_quality_penalty": sample.measurement_quality_penalty.copy(),
                "measurement_logits_adjusted": sample.measurement_logits_adjusted.copy(),
                "candidate_source_ids": list(sample.candidate_source_ids),
                "candidate_visible": sample.candidate_visible.copy(),
                "candidate_action_mask": sample.candidate_action_mask.copy(),
                "candidate_action_features": sample.candidate_action_features.copy(),
                "candidate_quality_scores": sample.candidate_quality_scores.copy(),
                "none_context_summary": sample.none_context_summary.copy(),
                "none_logit_raw": float(sample.none_logit_raw),
                "selected_action_deterministic": bool(deterministic),
                "measurement_quality_logit_penalty": float(
                    sample.measurement_quality_logit_penalty
                ),
                "quality_actor_branch": self.policy._measurement_branch_name(),
                "measurement_proxy_features_enabled": bool(
                    self.config.use_measurement_proxy_features
                ),
                "measurement_proxy_guidance_enabled": bool(
                    self.config.measurement_proxy_guidance_enabled
                ),
                "measurement_proxy_teacher_coef": float(
                    self.config.measurement_proxy_teacher_coef
                ),
                "measurement_proxy_temperature": float(
                    self.config.measurement_proxy_temperature
                ),
                "measurement_proxy_communication_cost_weight": float(
                    self.config.measurement_proxy_communication_cost_weight
                ),
                "measurement_proxy_gain": sample.measurement_proxy_gain.tolist(),
                "measurement_proxy_cost": sample.measurement_proxy_cost.tolist(),
                "measurement_proxy_valid": sample.measurement_proxy_valid.tolist(),
                "measurement_proxy_utility": sample.measurement_proxy_utility.tolist(),
                "measurement_persistent_proxy_features_enabled": bool(
                    self.config.use_measurement_persistent_proxy_features
                ),
                "measurement_persistent_proxy_guidance_enabled": bool(
                    self.config.measurement_persistent_proxy_guidance_enabled
                ),
                "measurement_persistent_proxy_horizon": int(
                    self.config.measurement_persistent_proxy_horizon
                ),
                "persistent_proxy_gain": sample.persistent_proxy_gain.tolist(),
                "persistent_proxy_cost": sample.persistent_proxy_cost.tolist(),
                "persistent_proxy_valid": sample.persistent_proxy_valid.tolist(),
                "persistent_proxy_utility": sample.persistent_proxy_utility.tolist(),
                "quality_feature_enabled": bool(self.config.use_measurement_quality_features),
                "measurement_budget_feature_enabled": bool(
                    self.config.use_measurement_budget_features
                ),
                "measurement_rate_constraint_enabled": bool(
                    self.config.measurement_rate_constraint_enabled
                ),
                "measurement_rate_target": float(
                    (self.config.measurement_rate_target_by_uav or {}).get(uav_id, 0.0)
                ),
                "measurement_budget_remaining_ratio": float(
                    decision_observation.get("measurement_budget_remaining_ratio", 1.0)
                ),
                "candidate_quality_valid_count": float(
                    np.asarray(
                        decision_observation.get("measurement_quality_valid", []), dtype=np.float32
                    ).sum()
                ) if self.config.use_measurement_quality_features else 0.0,
            }
            for uav_id, sample in samples.items()
            for decision_observation in [normalized.get(uav_id, {})]
        }
        for uav_id, navigation in navigation_diagnostics.items():
            measurement_diagnostics.setdefault(uav_id, {}).update(navigation)
            measurement_diagnostics[uav_id].update(
                {
                    "motion_action_mode": "fixed_navigation_comm_v1",
                    "learned_motion_dim": 0,
                    "requested_nav": np.asarray(
                        navigation.get("requested_delta_v", np.empty(0)), dtype=np.float32
                    ).copy(),
                    "link_selection": samples[uav_id].link_selection.copy(),
                    "sampled_link_probabilities": samples[uav_id].sampled_link_probabilities.copy(),
                    "links": samples[uav_id].link_selection.copy(),
                    "probs": samples[uav_id].sampled_link_probabilities.copy(),
                }
            )
        return PolicyDecision(
            actions=actions,
            samples=samples,
            observations=copy.deepcopy(dict(normalized)),
            central_vector=central_vector.copy(),
            value=value,
            graph_traces=graph_traces,
            measurement_diagnostics=measurement_diagnostics,
        )

    def add_transition(
        self,
        decision: PolicyDecision,
        rewards: Mapping[str, float],
        next_central_state: Mapping[str, Any] | None,
        *,
        done: bool = False,
        terminated: bool | None = None,
        truncated: bool = False,
    ) -> None:
        if terminated is None:
            # Backward-compatible callers only supplied ``done``.  A legacy
            # done transition is a true terminal unless it is explicitly
            # marked as a time-limit truncation.
            terminated = bool(done and not truncated)
        done = bool(done or terminated or truncated)
        self.rollout.append(
            RolloutTransition(
                decision=decision,
                rewards={str(key): float(value) for key, value in rewards.items()},
                next_central_state=copy.deepcopy(next_central_state),
                done=bool(done),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )

    def reconcile_action_execution(
        self,
        decision: PolicyDecision,
        info: Mapping[str, Any] | None,
    ) -> None:
        """Record execution projection without corrupting PPO's sampled action.

        In the new ``sampled`` mode the stored categorical/continuous action
        and its old log probability remain exactly those produced by the
        policy.  Safety projection, budget gating, and bandwidth projection
        are execution-time transformations and may be non-invertible, so they
        are recorded as diagnostics instead of being fed back into PPO's
        importance ratio.  The historical ``executed`` mode retains the old
        reconciliation behavior for backward-compatible replay/checkpoints.
        """

        payload = info or {}
        executed_actions = payload.get("actions", {})
        applied_bandwidth = payload.get("applied_bandwidth", {})
        execution = payload.get("action_execution", {})
        if not isinstance(executed_actions, Mapping):
            executed_actions = {}
        if not isinstance(applied_bandwidth, Mapping):
            applied_bandwidth = {}
        if not isinstance(execution, Mapping):
            execution = {}
        if self.config.motion_action_mode == "fixed_navigation_comm_v1":
            if self.config.communication_learning_scope == "measurement_only":
                # In this scope ``PolicySample`` deliberately has empty
                # learned communication arrays.  Reconciliation therefore
                # resolves the requested fixed bandwidth from the concrete
                # decision action mapping and records execution diagnostics
                # without recomputing or changing the categorical log prob.
                def action_bandwidth_vector(action: Mapping[str, Any]) -> np.ndarray:
                    vector = np.zeros(self.config.uav_count, dtype=np.float32)
                    allocations = action.get("bandwidth", action.get("outgoing_bandwidth", {}))
                    if isinstance(allocations, Mapping):
                        for index, target in enumerate(self.policy.uav_ids):
                            try:
                                vector[index] = float(allocations.get(target, 0.0))
                            except (TypeError, ValueError):
                                vector[index] = 0.0
                    return vector

                def navigation_vector(value: Any) -> np.ndarray:
                    if value is None:
                        return np.empty(0, dtype=np.float32)
                    try:
                        vector = np.asarray(value, dtype=np.float32).reshape(-1)
                    except (TypeError, ValueError):
                        return np.empty(0, dtype=np.float32)
                    return vector.copy() if vector.size else np.empty(0, dtype=np.float32)

                for uav_id, sample in decision.samples.items():
                    requested_action = decision.actions.get(uav_id, {})
                    if not isinstance(requested_action, Mapping):
                        requested_action = {}
                    applied = executed_actions.get(uav_id, requested_action)
                    if not isinstance(applied, Mapping):
                        applied = {}
                    selected = str(applied.get("measurement_uav_id", "none"))
                    try:
                        measurement_index = self.policy.uav_ids.index(selected)
                    except ValueError:
                        measurement_index = self.config.uav_count
                    requested_bandwidth = action_bandwidth_vector(requested_action)
                    allocations = applied_bandwidth.get(uav_id, {})
                    if not isinstance(allocations, Mapping):
                        allocations = applied.get("bandwidth", applied.get("outgoing_bandwidth", {}))
                    executed_bandwidth = np.zeros(self.config.uav_count, dtype=np.float32)
                    if isinstance(allocations, Mapping):
                        for index, target in enumerate(self.policy.uav_ids):
                            try:
                                executed_bandwidth[index] = float(allocations.get(target, 0.0))
                            except (TypeError, ValueError):
                                executed_bandwidth[index] = 0.0
                    requested_link = requested_bandwidth > 0.0
                    executed_link = executed_bandwidth > 0.0
                    applied_execution = execution.get(uav_id, {})
                    if not isinstance(applied_execution, Mapping):
                        applied_execution = {}
                    diagnostics = decision.measurement_diagnostics.get(uav_id, {})
                    requested_nav = navigation_vector(
                        diagnostics.get("requested_nav", requested_action.get("delta_v"))
                    )
                    if "delta_v_applied" in applied_execution:
                        executed_nav = navigation_vector(applied_execution["delta_v_applied"])
                    else:
                        executed_nav = navigation_vector(applied.get("delta_v", requested_nav))
                    communication_changed = bool(
                        int(measurement_index) != int(sample.measurement_index)
                        or not np.allclose(
                            requested_bandwidth,
                            executed_bandwidth,
                            atol=1e-6,
                            rtol=1e-6,
                        )
                        or not np.array_equal(requested_link, executed_link)
                    )
                    nav_projection_changed = bool(
                        requested_nav.shape != executed_nav.shape
                        or not np.allclose(
                            requested_nav,
                            executed_nav,
                            atol=1e-6,
                            rtol=1e-6,
                        )
                    )
                    sample.executed_measurement_index = int(measurement_index)
                    sample.executed_bandwidth = executed_bandwidth.copy()
                    sample.executed_delta_v = executed_nav.copy()
                    sample.executed_link_selection = executed_link.copy()
                    sample.action_execution_changed = communication_changed
                    decision.measurement_diagnostics.setdefault(uav_id, {}).update(
                        {
                            "ppo_action_logprob_mode": self.config.ppo_action_logprob_mode,
                            "sampled_measurement_index": int(sample.measurement_index),
                            "executed_measurement_index": int(measurement_index),
                            "sampled_bandwidth": requested_bandwidth.copy(),
                            "executed_bandwidth": executed_bandwidth.copy(),
                            "sampled_link_selection": requested_link.copy(),
                            "executed_link_selection": executed_link.copy(),
                            "sampled_delta_v": np.empty(0, dtype=np.float32),
                            "executed_delta_v": executed_nav.copy(),
                            "action_execution_changed": int(communication_changed),
                            "nav_projection_changed": int(nav_projection_changed),
                        }
                    )
                return
            for uav_id, sample in decision.samples.items():
                applied = executed_actions.get(uav_id, decision.actions.get(uav_id, {}))
                if not isinstance(applied, Mapping):
                    applied = {}
                selected = str(applied.get("measurement_uav_id", "none"))
                try:
                    measurement_index = self.policy.uav_ids.index(selected)
                except ValueError:
                    measurement_index = self.config.uav_count
                allocations = applied_bandwidth.get(uav_id, {})
                if not isinstance(allocations, Mapping):
                    allocations = applied.get("bandwidth", {})
                executed_bandwidth = np.zeros(self.config.uav_count, dtype=np.float32)
                if isinstance(allocations, Mapping):
                    for index, target in enumerate(self.policy.uav_ids):
                        try:
                            executed_bandwidth[index] = float(allocations.get(target, 0.0))
                        except (TypeError, ValueError):
                            executed_bandwidth[index] = 0.0
                executed_link = executed_bandwidth > 0.0
                requested_bandwidth = np.asarray(sample.bandwidth, dtype=np.float32).reshape(-1)
                requested_link = np.asarray(sample.link_selection, dtype=bool).reshape(-1)
                applied_execution = execution.get(uav_id, {})
                if not isinstance(applied_execution, Mapping):
                    applied_execution = {}
                requested_nav = np.asarray(
                    decision.measurement_diagnostics.get(uav_id, {}).get(
                        "requested_nav", decision.actions.get(uav_id, {}).get("delta_v", [0.0, 0.0, 0.0])
                    ),
                    dtype=np.float32,
                ).reshape(3)
                if "delta_v_applied" in applied_execution:
                    executed_delta_v = np.asarray(applied_execution["delta_v_applied"], dtype=np.float32).reshape(3)
                else:
                    executed_delta_v = np.asarray(
                        applied.get("delta_v", requested_nav), dtype=np.float32
                    ).reshape(3)
                communication_changed = bool(
                    int(measurement_index) != int(sample.measurement_index)
                    or requested_bandwidth.shape != executed_bandwidth.shape
                    or not np.allclose(requested_bandwidth, executed_bandwidth, atol=1e-6, rtol=1e-6)
                    or requested_link.shape != executed_link.shape
                    or not np.array_equal(requested_link, executed_link)
                )
                nav_projection_changed = bool(
                    not np.allclose(requested_nav, executed_delta_v, atol=1e-6, rtol=1e-6)
                )
                sample.executed_measurement_index = int(measurement_index)
                sample.executed_bandwidth = executed_bandwidth.copy()
                sample.executed_delta_v = executed_delta_v.copy()
                sample.executed_link_selection = executed_link.copy()
                sample.action_execution_changed = communication_changed
                decision.measurement_diagnostics.setdefault(uav_id, {}).update(
                    {
                        "ppo_action_logprob_mode": self.config.ppo_action_logprob_mode,
                        "sampled_measurement_index": int(sample.measurement_index),
                        "executed_measurement_index": int(measurement_index),
                        "sampled_bandwidth": requested_bandwidth.copy(),
                        "executed_bandwidth": executed_bandwidth.copy(),
                        "sampled_link_selection": requested_link.copy(),
                        "executed_link_selection": executed_link.copy(),
                        "sampled_delta_v": np.empty(0, dtype=np.float32),
                        "executed_delta_v": executed_delta_v.copy(),
                        "action_execution_changed": int(communication_changed),
                        "nav_projection_changed": int(nav_projection_changed),
                    }
                )
            return
        for uav_id, sample in decision.samples.items():
            applied = executed_actions.get(uav_id, decision.actions.get(uav_id, {}))
            if not isinstance(applied, Mapping):
                continue
            selected = str(applied.get("measurement_uav_id", "none"))
            try:
                measurement_index = self.policy.uav_ids.index(selected)
            except ValueError:
                measurement_index = self.config.uav_count
            bandwidth = np.zeros(self.config.uav_count, dtype=np.float32)
            allocations = applied_bandwidth.get(uav_id, {})
            if isinstance(allocations, Mapping):
                for index, target in enumerate(self.policy.uav_ids):
                    try:
                        bandwidth[index] = float(allocations.get(target, 0.0))
                    except (TypeError, ValueError):
                        bandwidth[index] = 0.0
            applied_motion = execution.get(uav_id, {})
            if isinstance(applied_motion, Mapping) and "delta_v_applied" in applied_motion:
                delta_v = np.asarray(applied_motion["delta_v_applied"], dtype=np.float32).reshape(3)
            else:
                delta_v = np.asarray(applied.get("delta_v", sample.delta_v), dtype=np.float32).reshape(3)
            requested_bandwidth = np.asarray(sample.bandwidth, dtype=np.float32).reshape(-1)
            executed_bandwidth = np.asarray(bandwidth, dtype=np.float32).reshape(-1)
            changed = bool(
                int(measurement_index) != int(sample.measurement_index)
                or not np.allclose(requested_bandwidth, executed_bandwidth, atol=1e-6, rtol=1e-6)
                or not np.allclose(np.asarray(sample.delta_v, dtype=np.float32), delta_v, atol=1e-6, rtol=1e-6)
            )
            sample.executed_measurement_index = int(measurement_index)
            sample.executed_bandwidth = executed_bandwidth.copy()
            sample.executed_delta_v = delta_v.copy()
            sample.action_execution_changed = changed
            decision.measurement_diagnostics.setdefault(uav_id, {}).update(
                {
                    "ppo_action_logprob_mode": self.config.ppo_action_logprob_mode,
                    "sampled_measurement_index": int(sample.measurement_index),
                    "executed_measurement_index": int(measurement_index),
                    "action_execution_changed": int(changed),
                    "sampled_bandwidth": requested_bandwidth.tolist(),
                    "executed_bandwidth": executed_bandwidth.tolist(),
                    "sampled_delta_v": np.asarray(sample.delta_v, dtype=np.float32).tolist(),
                    "executed_delta_v": delta_v.tolist(),
                }
            )
            if self.config.ppo_action_logprob_mode == "executed":
                sample.measurement_index = int(measurement_index)
                sample.bandwidth = executed_bandwidth
                sample.delta_v = delta_v
                with torch.no_grad():
                    log_prob, _entropy = self.policy.evaluate_actions(
                        decision.observations[uav_id], sample
                    )
                sample.old_log_prob = float(log_prob.detach().cpu())

    def collect_rollout(
        self,
        environment: Any,
        *,
        horizon: int,
        seed: int | None = None,
        deterministic: bool = False,
    ) -> dict[str, float]:
        observations, info = environment.reset(seed=seed)
        steps = 0
        for _ in range(int(horizon)):
            decision = self.act(observations, info.get("central_state"), deterministic=deterministic)
            next_observations, rewards, terminated, truncated, next_info = environment.step(decision.actions)
            self.reconcile_action_execution(decision, next_info)
            self.add_transition(
                decision,
                next_info.get("learning_rewards", rewards),
                next_info.get("central_state"),
                done=bool(terminated or truncated),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            observations, info = next_observations, next_info
            steps += 1
            if terminated or truncated:
                break
        return {"steps": float(steps), "transitions": float(len(self.rollout))}

    def _next_value(self, transition: RolloutTransition) -> float:
        terminal_for_bootstrap = bool(
            transition.terminated
            or (transition.done and not transition.truncated)
            or (transition.truncated and not self.config.bootstrap_time_limits)
        )
        if terminal_for_bootstrap:
            return 0.0
        vector = central_state_to_vector(
            transition.next_central_state,
            uav_count=self.config.uav_count,
            node_feature_dim=self.config.node_feature_dim,
            edge_feature_dim=self.config.edge_feature_dim,
            max_obstacles=self.config.max_obstacles,
            include_remaining_time=self.config.include_remaining_time,
            critic_measurement_budget_features=self.config.critic_measurement_budget_features,
        )
        vector = self.central_normalizer.transform(vector)
        with torch.no_grad():
            return float(self.policy.value(vector).item())

    def _training_records(self) -> list[dict[str, Any]]:
        if not self.rollout:
            raise ValueError("cannot build MAPPO records from an empty rollout")
        agents = list(self.policy.uav_ids)
        # The environment returns a shared team reward. Keep the public
        # per-agent mapping for compatibility, but compute one GAE stream and
        # replicate its scalar advantage/return for every actor.
        next_advantage = 0.0
        advantages: list[dict[str, float]] = [dict() for _ in self.rollout]
        returns: list[dict[str, float]] = [dict() for _ in self.rollout]
        for index in range(len(self.rollout) - 1, -1, -1):
            transition = self.rollout[index]
            next_value = self._next_value(transition)
            # Time-limit truncation has two different roles here.  When the
            # stable configuration enables bootstrapping, the value at the
            # truncated next state belongs in the one-step TD target.  It
            # must not, however, connect the GAE recursion to the first
            # transition of the next episode in a multi-episode rollout.
            bootstrap_nonterminal = 0.0 if (
                transition.terminated
                or (transition.done and not transition.truncated)
                or (transition.truncated and not self.config.bootstrap_time_limits)
            ) else 1.0
            gae_continuation = 0.0 if transition.done else 1.0
            reward_values = [float(transition.rewards.get(agent, 0.0)) for agent in agents]
            reward = float(np.mean(reward_values)) if reward_values else 0.0
            delta = (
                reward
                + self.config.gamma * next_value * bootstrap_nonterminal
                - transition.decision.value
            )
            advantage = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * gae_continuation
                * next_advantage
            )
            for agent in agents:
                advantages[index][agent] = advantage
                returns[index][agent] = advantage + transition.decision.value
            next_advantage = advantage
        records: list[dict[str, Any]] = []
        all_advantages = np.asarray(
            [advantages[index][agent] for index in range(len(self.rollout)) for agent in agents], dtype=np.float32
        )
        advantage_mean = float(all_advantages.mean())
        advantage_std = float(all_advantages.std())
        for index, transition in enumerate(self.rollout):
            for agent in agents:
                records.append(
                    {
                        "uav_id": agent,
                        "observation": transition.decision.observations[agent],
                        "sample": transition.decision.samples[agent],
                        "central_vector": transition.decision.central_vector,
                        "advantage": (advantages[index][agent] - advantage_mean) / max(advantage_std, 1e-8),
                        "return": returns[index][agent],
                        "old_value": float(transition.decision.value),
                    }
                )
        return records

    def _clip_gradients(self) -> tuple[Tensor, float, float, float, float]:
        """Clip optimizer gradients with an explicit legacy-compatible scope.

        Returns the aggregate pre-clipping norm, actor norm, critic norm, and
        the two applied clipping scales.  In ``joint`` mode the first norm is
        exactly the historical global norm and the two group values are only
        diagnostics.  In ``actor_critic`` mode the actor/GAT/measurement,
        bandwidth and velocity parameters form one group and the centralized
        critic forms the other; no parameter is clipped twice.
        """

        all_parameters = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        if self.config.gradient_clip_scope == "joint":
            norm = nn.utils.clip_grad_norm_(all_parameters, self.config.max_grad_norm)
            norm_value = float(norm.detach().cpu())
            critic_ids = {id(parameter) for parameter in self.policy.critic.parameters()}
            actor_norm_sq = 0.0
            critic_norm_sq = 0.0
            for parameter in all_parameters:
                if parameter.grad is None:
                    continue
                value = float(parameter.grad.detach().square().sum().cpu())
                if id(parameter) in critic_ids:
                    critic_norm_sq += value
                else:
                    actor_norm_sq += value
            return (
                norm,
                float(actor_norm_sq ** 0.5),
                float(critic_norm_sq ** 0.5),
                float(min(1.0, self.config.max_grad_norm / max(actor_norm_sq ** 0.5, 1e-12))),
                float(min(1.0, self.config.max_grad_norm / max(critic_norm_sq ** 0.5, 1e-12))),
            )

        critic_parameters = [parameter for parameter in self.policy.critic.parameters() if parameter.requires_grad]
        critic_ids = {id(parameter) for parameter in critic_parameters}
        actor_parameters = [parameter for parameter in all_parameters if id(parameter) not in critic_ids]
        if len(actor_parameters) + len(critic_parameters) != len(all_parameters):
            raise RuntimeError("actor/critic gradient groups overlap or omit parameters")
        actor_norm = nn.utils.clip_grad_norm_(actor_parameters, self.config.max_grad_norm)
        critic_norm = nn.utils.clip_grad_norm_(critic_parameters, self.config.max_grad_norm)
        actor_value = float(actor_norm.detach().cpu())
        critic_value = float(critic_norm.detach().cpu())
        aggregate = torch.as_tensor(
            math.sqrt(actor_value * actor_value + critic_value * critic_value),
            dtype=torch.float32,
            device=self.policy.device,
        )
        return (
            aggregate,
            actor_value,
            critic_value,
            float(min(1.0, self.config.max_grad_norm / max(actor_value, 1e-12))),
            float(min(1.0, self.config.max_grad_norm / max(critic_value, 1e-12))),
        )

    def update(self) -> dict[str, float]:
        records = self._training_records()
        losses: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        clip_fractions: list[float] = []
        gradient_norms: list[float] = []
        actor_gradient_norms: list[float] = []
        critic_gradient_norms: list[float] = []
        actor_clip_scales: list[float] = []
        critic_clip_scales: list[float] = []
        rate_constraint_losses: list[float] = []
        rate_expected_by_uav: dict[str, list[float]] = {}
        rate_upper_residual_by_uav: dict[str, list[float]] = {}
        rate_lower_residual_by_uav: dict[str, list[float]] = {}
        rate_raw_loss_by_uav: dict[str, list[float]] = {}
        rate_active_by_uav: dict[str, list[float]] = {}
        rate_violation_by_uav: dict[str, list[float]] = {}
        rate_dual_losses: list[float] = []
        proxy_teacher_losses: list[float] = []
        proxy_teacher_valid_values: list[float] = []
        proxy_teacher_agreements: list[float] = []
        communication_expected_by_uav: dict[str, list[float]] = {}
        communication_violation_by_uav: dict[str, list[float]] = {}
        communication_dual_losses: list[float] = []
        component_gradients: dict[str, list[float]] = {name: [] for name in ("ppo", "teacher", "communication")}
        approximate_kls: list[float] = []
        stop_early = False
        for epoch in range(self.config.policy_epochs):
            order = self.rng.permutation(len(records))
            for start in range(0, len(records), self.config.minibatch_size):
                indices = order[start : start + self.config.minibatch_size]
                new_log_probs: list[Tensor] = []
                entropy_values: list[Tensor] = []
                measurement_probability_values: dict[str, list[Tensor]] = {}
                values: list[Tensor] = []
                old_log_probs: list[float] = []
                advantages: list[float] = []
                returns: list[float] = []
                old_values: list[float] = []
                central_vectors: list[np.ndarray] = []
                proxy_losses: list[Tensor] = []
                communication_cost_values: dict[str, list[Tensor]] = {}
                for index in indices:
                    record = records[int(index)]
                    log_prob, entropy = self.policy.evaluate_actions(record["observation"], record["sample"])
                    new_log_probs.append(log_prob)
                    entropy_values.append(entropy)
                    central_vectors.append(record["central_vector"])
                    old_log_probs.append(float(record["sample"].old_log_prob))
                    advantages.append(float(record["advantage"]))
                    returns.append(float(record["return"]))
                    old_values.append(float(record["old_value"]))
                    if self.config.measurement_rate_constraint_enabled:
                        measurement_probability_values.setdefault(str(record["uav_id"]), []).append(
                            self.policy.measurement_probability(record["observation"])
                        )
                    if (
                        self.config.measurement_proxy_guidance_enabled
                        or self.config.measurement_persistent_proxy_guidance_enabled
                    ):
                        proxy_loss, proxy_diagnostics = self.policy.measurement_proxy_teacher_loss(
                            record["observation"],
                            record["sample"],
                            uav_id=str(record["uav_id"]),
                        )
                        proxy_losses.append(proxy_loss)
                        proxy_teacher_valid_values.append(
                            float(proxy_diagnostics.get("proxy_teacher_valid", 0.0))
                        )
                        proxy_teacher_agreements.append(
                            float(proxy_diagnostics.get("proxy_teacher_agreement", 0.0))
                        )
                    if self.config.measurement_communication_constraint_enabled:
                        communication_cost_values.setdefault(str(record["uav_id"]), []).append(
                            self.policy.expected_measurement_communication_cost(record["observation"])
                        )
                if not new_log_probs:
                    continue
                new_log_prob_tensor = torch.stack(new_log_probs)
                entropy_tensor = torch.stack(entropy_values)
                value_tensor = self.policy.value(np.stack(central_vectors))
                old_log_prob_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.policy.device)
                advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.policy.device)
                return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.policy.device)
                old_value_tensor = torch.as_tensor(old_values, dtype=torch.float32, device=self.policy.device)
                ratio = torch.exp(new_log_prob_tensor - old_log_prob_tensor)
                batch_kl = float(
                    (ratio - 1.0 - (new_log_prob_tensor - old_log_prob_tensor))
                    .mean()
                    .detach()
                    .cpu()
                )
                approximate_kls.append(batch_kl)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                actor_loss = -torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor).mean()
                if self.config.critic_loss_type == "huber":
                    critic_per_sample = torch.nn.functional.smooth_l1_loss(
                        value_tensor,
                        return_tensor,
                        reduction="none",
                    )
                else:
                    critic_per_sample = 0.5 * (value_tensor - return_tensor).pow(2)
                if self.config.value_clip_enabled:
                    clipped_values = old_value_tensor + torch.clamp(
                        value_tensor - old_value_tensor,
                        -float(self.config.value_clip_epsilon),
                        float(self.config.value_clip_epsilon),
                    )
                    if self.config.critic_loss_type == "huber":
                        clipped_per_sample = torch.nn.functional.smooth_l1_loss(
                            clipped_values,
                            return_tensor,
                            reduction="none",
                        )
                    else:
                        clipped_per_sample = 0.5 * (clipped_values - return_tensor).pow(2)
                    critic_loss = torch.maximum(critic_per_sample, clipped_per_sample).mean()
                else:
                    critic_loss = critic_per_sample.mean()
                entropy = entropy_tensor.mean()
                rate_constraint_loss = torch.zeros((), dtype=torch.float32, device=self.policy.device)
                rate_dual_loss = torch.zeros((), dtype=torch.float32, device=self.policy.device)
                if measurement_probability_values:
                    targets = self.config.measurement_rate_target_by_uav or {}
                    penalties: list[Tensor] = []
                    dual_terms: list[Tensor] = []
                    for uav_id, probabilities in measurement_probability_values.items():
                        if uav_id not in targets or not probabilities:
                            continue
                        mean_probability = torch.stack(probabilities).mean()
                        target = float(targets[uav_id])
                        tolerance = float(self.config.measurement_rate_tolerance)
                        upper_residual = torch.relu(mean_probability - target - tolerance)
                        lower_residual = torch.relu(target - tolerance - mean_probability)
                        raw_penalty = upper_residual.pow(2) + lower_residual.pow(2)
                        violation = upper_residual + lower_residual
                        penalties.append(raw_penalty)
                        rate_violation_by_uav.setdefault(uav_id, []).append(
                            float(violation.detach().cpu())
                        )
                        if self.config.measurement_rate_dual_enabled:
                            # The multiplier is plain detached state, not an
                            # optimizer parameter and therefore cannot receive
                            # policy gradients.
                            dual = float(self.measurement_rate_dual_by_uav.get(uav_id, 0.0))
                            dual_terms.append(violation * dual)
                        rate_expected_by_uav.setdefault(uav_id, []).append(
                            float(mean_probability.detach().cpu())
                        )
                        rate_upper_residual_by_uav.setdefault(uav_id, []).append(
                            float(upper_residual.detach().cpu())
                        )
                        rate_lower_residual_by_uav.setdefault(uav_id, []).append(
                            float(lower_residual.detach().cpu())
                        )
                        rate_raw_loss_by_uav.setdefault(uav_id, []).append(
                            float(raw_penalty.detach().cpu())
                        )
                        rate_active_by_uav.setdefault(uav_id, []).append(
                            float((raw_penalty.detach() > 0.0).cpu())
                        )
                    if penalties:
                        rate_constraint_loss = torch.stack(penalties).mean()
                    if dual_terms:
                        rate_dual_loss = torch.stack(dual_terms).mean()
                proxy_guidance_loss = torch.zeros((), dtype=torch.float32, device=self.policy.device)
                if proxy_losses:
                    proxy_guidance_loss = torch.stack(proxy_losses).mean()
                communication_dual_loss = torch.zeros((), dtype=torch.float32, device=self.policy.device)
                communication_terms: list[Tensor] = []
                communication_targets = self.config.measurement_communication_cost_target_by_uav or {}
                for uav_id, costs in communication_cost_values.items():
                    if uav_id not in communication_targets or not costs:
                        continue
                    mean_cost = torch.stack(costs).mean()
                    violation = torch.relu(
                        mean_cost - float(communication_targets[uav_id])
                        - float(self.config.measurement_communication_cost_tolerance)
                    )
                    dual = float(self.measurement_communication_dual_by_uav.get(uav_id, 0.0))
                    communication_terms.append(violation * dual)
                    communication_expected_by_uav.setdefault(uav_id, []).append(float(mean_cost.detach().cpu()))
                    communication_violation_by_uav.setdefault(uav_id, []).append(float(violation.detach().cpu()))
                if communication_terms:
                    communication_dual_loss = torch.stack(communication_terms).mean()
                rate_dual_losses.append(float(rate_dual_loss.detach().cpu()))
                proxy_teacher_losses.append(float(proxy_guidance_loss.detach().cpu()))
                communication_dual_losses.append(float(communication_dual_loss.detach().cpu()))
                rate_penalty = rate_dual_loss if self.config.measurement_rate_dual_enabled else (
                    self.config.measurement_rate_constraint_coef * rate_constraint_loss
                )
                total_loss = (
                    actor_loss
                    + self.config.value_coef * critic_loss
                    - self.config.entropy_coef * entropy
                    + rate_penalty
                    + float(self.config.measurement_proxy_teacher_coef) * proxy_guidance_loss
                    + communication_dual_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                if self.config.auxiliary_gradient_diagnostics:
                    measurement_modules = [self.policy.measurement_head]
                    if self.policy.measurement_none_head is not None:
                        measurement_modules.append(self.policy.measurement_none_head)
                    if self.policy.measurement_source_encoder is not None:
                        measurement_modules.append(self.policy.measurement_source_encoder)
                    parameters = tuple(
                        parameter
                        for module in measurement_modules
                        for parameter in module.parameters()
                    )
                    for name, component in (("ppo", actor_loss - self.config.entropy_coef * entropy),
                                            ("teacher", self.config.measurement_proxy_teacher_coef * proxy_guidance_loss),
                                            ("communication", communication_dual_loss)):
                        grads = torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True) if component.requires_grad else ()
                        norm = sum(float(g.detach().square().sum()) for g in grads if g is not None)**0.5
                        if not np.isfinite(norm):
                            raise FloatingPointError(f"non-finite {name} gradient")
                        component_gradients[name].append(norm)
                total_loss.backward()
                (
                    gradient_norm,
                    actor_gradient_norm,
                    critic_gradient_norm,
                    actor_clip_scale,
                    critic_clip_scale,
                ) = self._clip_gradients()
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("MAPPO gradient norm is non-finite")
                if not all(
                    np.isfinite(value)
                    for value in (
                        actor_gradient_norm,
                        critic_gradient_norm,
                        actor_clip_scale,
                        critic_clip_scale,
                    )
                ):
                    raise FloatingPointError("MAPPO grouped gradient diagnostics are non-finite")
                self.optimizer.step()
                losses.append(float(total_loss.detach().cpu()))
                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                # Keep the extra statistic explicit so the training audit can
                # distinguish a zero inactive constraint from a finite active
                # penalty without changing the legacy loss fields.
                rate_constraint_losses.append(float(rate_constraint_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                clip_fractions.append(float((torch.abs(ratio - 1.0) > self.config.clip_epsilon).float().mean().detach().cpu()))
                gradient_norms.append(float(gradient_norm.detach().cpu()))
                actor_gradient_norms.append(actor_gradient_norm)
                critic_gradient_norms.append(critic_gradient_norm)
                actor_clip_scales.append(actor_clip_scale)
                critic_clip_scales.append(critic_clip_scale)
                if self.config.target_kl is not None and batch_kl > float(self.config.target_kl):
                    stop_early = True
                    break
            if stop_early:
                break
        dual_update_by_uav: dict[str, float] = {}
        if self.config.measurement_rate_dual_enabled:
            for uav_id in self.policy.uav_ids:
                violations = rate_violation_by_uav.get(uav_id, [])
                mean_violation = float(np.mean(violations)) if violations else 0.0
                previous = float(self.measurement_rate_dual_by_uav.get(uav_id, 0.0))
                updated = float(
                    np.clip(
                        previous + float(self.config.measurement_rate_dual_step_size) * mean_violation,
                        0.0,
                        float(self.config.measurement_rate_dual_max),
                    )
                )
                self.measurement_rate_dual_by_uav[uav_id] = updated
                dual_update_by_uav[uav_id] = updated - previous
        communication_dual_update_by_uav: dict[str, float] = {}
        if self.config.measurement_communication_constraint_enabled:
            for uav_id in self.policy.uav_ids:
                violations = communication_violation_by_uav.get(uav_id, [])
                mean_violation = float(np.mean(violations)) if violations else 0.0
                previous = float(self.measurement_communication_dual_by_uav.get(uav_id, 0.0))
                updated = float(np.clip(
                    previous + float(self.config.measurement_communication_dual_step_size) * mean_violation,
                    0.0, float(self.config.measurement_communication_dual_max),
                ))
                self.measurement_communication_dual_by_uav[uav_id] = updated
                communication_dual_update_by_uav[uav_id] = updated - previous
        self.rollout.clear()
        targets = self.config.measurement_rate_target_by_uav or {}

        def _mean_stat(values: Sequence[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        per_uav_rate_stats: dict[str, float] = {}
        for uav_id in sorted(targets):
            per_uav_rate_stats[f"measurement_rate_target_{uav_id}"] = float(targets[uav_id])
            per_uav_rate_stats[f"measurement_rate_expected_{uav_id}"] = _mean_stat(
                rate_expected_by_uav.get(uav_id, [])
            )
            per_uav_rate_stats[f"measurement_rate_upper_residual_{uav_id}"] = _mean_stat(
                rate_upper_residual_by_uav.get(uav_id, [])
            )
            per_uav_rate_stats[f"measurement_rate_lower_residual_{uav_id}"] = _mean_stat(
                rate_lower_residual_by_uav.get(uav_id, [])
            )
            per_uav_rate_stats[f"measurement_rate_raw_loss_{uav_id}"] = _mean_stat(
                rate_raw_loss_by_uav.get(uav_id, [])
            )
            per_uav_rate_stats[f"measurement_rate_active_fraction_{uav_id}"] = _mean_stat(
                rate_active_by_uav.get(uav_id, [])
            )
            per_uav_rate_stats[f"measurement_rate_violation_{uav_id}"] = _mean_stat(
                rate_violation_by_uav.get(uav_id, [])
            )
            per_uav_rate_stats[f"measurement_rate_dual_{uav_id}"] = float(
                self.measurement_rate_dual_by_uav.get(uav_id, 0.0)
            )
            per_uav_rate_stats[f"measurement_rate_dual_update_{uav_id}"] = float(
                dual_update_by_uav.get(uav_id, 0.0)
            )

        expected_values = [
            value
            for values in rate_expected_by_uav.values()
            for value in values
        ]
        upper_values = [
            value
            for values in rate_upper_residual_by_uav.values()
            for value in values
        ]
        lower_values = [
            value
            for values in rate_lower_residual_by_uav.values()
            for value in values
        ]
        active_values = [
            value
            for values in rate_active_by_uav.values()
            for value in values
        ]
        raw_constraint_loss = float(np.mean(rate_constraint_losses)) if rate_constraint_losses else 0.0
        communication_targets = self.config.measurement_communication_cost_target_by_uav or {}
        per_uav_communication_stats: dict[str, float] = {}
        for uav_id in sorted(communication_targets):
            per_uav_communication_stats[f"measurement_communication_cost_target_{uav_id}"] = float(communication_targets[uav_id])
            per_uav_communication_stats[f"measurement_communication_cost_expected_{uav_id}"] = _mean_stat(communication_expected_by_uav.get(uav_id, []))
            per_uav_communication_stats[f"measurement_communication_cost_violation_{uav_id}"] = _mean_stat(communication_violation_by_uav.get(uav_id, []))
            per_uav_communication_stats[f"measurement_communication_dual_{uav_id}"] = float(self.measurement_communication_dual_by_uav.get(uav_id, 0.0))
            per_uav_communication_stats[f"measurement_communication_dual_update_{uav_id}"] = float(communication_dual_update_by_uav.get(uav_id, 0.0))
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "approximate_kl": float(np.mean(approximate_kls)) if approximate_kls else 0.0,
            "approximate_kl_max": float(max(approximate_kls)) if approximate_kls else 0.0,
            **{f"minibatch_kl_{index:04d}": float(value) for index, value in enumerate(approximate_kls)},
            "gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
            "actor_gradient_norm": float(np.mean(actor_gradient_norms)) if actor_gradient_norms else 0.0,
            "critic_gradient_norm": float(np.mean(critic_gradient_norms)) if critic_gradient_norms else 0.0,
            "actor_clip_scale": float(np.mean(actor_clip_scales)) if actor_clip_scales else 1.0,
            "critic_clip_scale": float(np.mean(critic_clip_scales)) if critic_clip_scales else 1.0,
            "gradient_clip_scope_actor_critic": float(self.config.gradient_clip_scope == "actor_critic"),
            "value_clip_enabled": float(bool(self.config.value_clip_enabled)),
            "value_clip_epsilon": float(self.config.value_clip_epsilon),
            "target_kl": float(self.config.target_kl) if self.config.target_kl is not None else 0.0,
            "target_kl_early_stop": float(bool(stop_early)),
            "bootstrap_time_limits": float(bool(self.config.bootstrap_time_limits)),
            "measurement_rate_constraint_loss": (
                raw_constraint_loss
            ),
            "measurement_rate_constraint_weighted_loss": float(
                self.config.measurement_rate_constraint_coef * raw_constraint_loss
            ),
            "measurement_rate_dual_loss": (
                float(np.mean(rate_dual_losses)) if rate_dual_losses else 0.0
            ),
            "measurement_rate_dual_enabled": float(bool(self.config.measurement_rate_dual_enabled)),
            "measurement_rate_constraint_enabled": float(
                bool(self.config.measurement_rate_constraint_enabled)
            ),
            "measurement_rate_expected_mean": _mean_stat(expected_values),
            "measurement_rate_upper_residual_mean": _mean_stat(upper_values),
            "measurement_rate_lower_residual_mean": _mean_stat(lower_values),
            "measurement_rate_constraint_active_fraction": _mean_stat(active_values),
            "measurement_proxy_guidance_enabled": float(
                bool(self.config.measurement_proxy_guidance_enabled)
            ),
            "measurement_persistent_proxy_guidance_enabled": float(
                bool(self.config.measurement_persistent_proxy_guidance_enabled)
            ),
            "measurement_persistent_proxy_features_enabled": float(
                bool(self.config.use_measurement_persistent_proxy_features)
            ),
            "measurement_persistent_proxy_horizon": float(
                self.config.measurement_persistent_proxy_horizon
            ),
            "measurement_proxy_teacher_loss": (
                float(np.mean(proxy_teacher_losses)) if proxy_teacher_losses else 0.0
            ),
            "measurement_proxy_teacher_weighted_loss": float(
                self.config.measurement_proxy_teacher_coef
                * (float(np.mean(proxy_teacher_losses)) if proxy_teacher_losses else 0.0)
            ),
            "measurement_proxy_teacher_valid_fraction": (
                float(np.mean(proxy_teacher_valid_values))
                if proxy_teacher_valid_values
                else 0.0
            ),
            "measurement_proxy_teacher_agreement": (
                float(np.mean(proxy_teacher_agreements))
                if proxy_teacher_agreements
                else 0.0
            ),
            "measurement_communication_constraint_enabled": float(bool(self.config.measurement_communication_constraint_enabled)),
            "measurement_communication_dual_loss": float(np.mean(communication_dual_losses)) if communication_dual_losses else 0.0,
            "measurement_communication_cost_expected_mean": _mean_stat([value for values in communication_expected_by_uav.values() for value in values]),
            "measurement_communication_cost_violation_mean": _mean_stat([value for values in communication_violation_by_uav.values() for value in values]),
            "samples": float(len(records)),
            "optimizer_steps": float(len(losses)),
            **({f"measurement_head_{key}_gradient_norm": _mean_stat(value) for key, value in component_gradients.items()}
               if self.config.auxiliary_gradient_diagnostics else {}),
            "epochs": float(self.config.policy_epochs),
            **per_uav_rate_stats,
            **per_uav_communication_stats,
        }


__all__ = [
    "MAPPOConfig",
    "MAPPOPolicy",
    "MAPPOTrainer",
    "PolicyDecision",
    "PolicySample",
    "RolloutTransition",
    "central_state_to_vector",
    "central_state_vector_dim",
]
