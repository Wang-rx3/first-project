"""Stage 2B-R4: BC/DAgger policy-preservation and action-fidelity audit.

This module is deliberately downstream of the validated single-UAV recovery
dataset.  It does not change the environment, reward, dynamics, action space,
or PPO path.  The only train-time extension is an explicit optional Actor-mean
parameter subset (and, in the conditional E3 stage, a simple normal-policy
action anchor).  The default behavior of the existing BC helper is unchanged.

The module keeps episode/scenario identity in every audit and never treats
individual timesteps as independent experimental replicates.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .single_uav_bc_v2_dagger import (
    DISTANCE_BINS,
    _batch_actor_physical_mean,
    _distance_bin,
    _offline_summary,
    _raw_estimated_velocity,
    _read_demo_jsonl,
    _action_components,
    _train_imitation,
    actor_mean_trainable_parameter_names,
)
from .single_uav_checkpoint_evaluation import _state_dict_hash
from .single_uav_static_obstacle_dagger_recovery_goal import (
    _aggregate,
    _evaluate_roles,
    _load_checkpoint,
    _make_eval_pool,
)
from .single_uav_rl_equivalent_oracle_audit import _unit


STAGE_NAME = "single_uav_stage2br4_bc_policy_preservation"
SOURCE_STAGE_ROOT = Path("data/synthetic/single_uav_external_recovery_method_study_seed2026")
SOURCE_DATA_ROOT = SOURCE_STAGE_ROOT / "dagger_bc"
SOURCE_DEMO_ROOT = SOURCE_DATA_ROOT / "demonstrations"
SOURCE_LOG_ROOT = Path("logs/preliminary/single_uav_external_recovery_method_study_seed2026")
BASELINE_ORIGINAL = Path(
    "data/synthetic/single_uav_braking_progress025_seed2026/scale_1/checkpoints/final.pt"
)
BASELINE_MIGRATED = SOURCE_DATA_ROOT / "baseline_observation_migration/checkpoints/stage2a_observation_compatible_init.pt"
ROUND0_CHECKPOINT = SOURCE_DATA_ROOT / "bc_round0/scale_1/checkpoints/best_validation.pt"
CURRENT_BC_CHECKPOINT = SOURCE_DATA_ROOT / "bc_final/scale_1/checkpoints/best_validation.pt"
DEFAULT_DATA_ROOT = Path("data/synthetic/single_uav_stage2br4_bc_policy_preservation_seed2026")
DEFAULT_LOG_ROOT = Path("logs/preliminary/single_uav_stage2br4_bc_policy_preservation_seed2026")
DEFAULT_PLOT_ROOT = Path("plots/preliminary/single_uav_stage2br4_bc_policy_preservation_seed2026")


E2_SCOPES = {
    "e2a_delta_mean_head_only": "delta_mean_head_only",
    "e2b_last_actor_body_block_plus_head": "last_actor_body_block_plus_head",
    "e2c_actor_body_plus_head_gat_frozen": "actor_body_plus_head_gat_frozen",
    "e2d_full_actor_mean": "full_actor_mean",
}
E3_LAMBDAS = (1.0e-3, 1.0e-2, 1.0e-1)
FOCUS_PHASES = {"transition", "post_bypass", "recovery"}


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _phase(sample: Mapping[str, Any]) -> str:
    value = str(sample.get("phase", "")).strip()
    return value or str(sample.get("base_phase", "")).strip() or "unknown"


def _distance(sample: Mapping[str, Any]) -> float:
    for key in ("goal_distance_m", "estimated_goal_distance_m", "goal_distance"):
        value = _finite(sample.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


def _source_class(sample: Mapping[str, Any]) -> str:
    if bool(sample.get("student_closed_loop", False)) or str(sample.get("collection_mode", "")) == "student_closed_loop":
        return "student_closed_loop"
    if str(sample.get("collection_mode", "")) == "teacher_bootstrap":
        return "teacher_bootstrap"
    return "other"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        fieldnames = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(dict(row)), ensure_ascii=False, allow_nan=True) + "\n")


def _prepare_roots(data_root: Path, logs_root: Path, plots_root: Path) -> None:
    for root in (data_root, logs_root, plots_root):
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"R4 output root already contains files; refusing to overwrite: {root}")
        root.mkdir(parents=True, exist_ok=True)


def _observation_vector(observation: Mapping[str, Any]) -> np.ndarray:
    parts: list[np.ndarray] = []
    node = np.asarray(observation.get("node_features", []), dtype=float).reshape(-1)
    parts.append(node)
    parts.append(np.asarray(observation.get("action_mask", []), dtype=float).reshape(-1))
    parts.append(np.asarray(observation.get("bandwidth_mask", []), dtype=float).reshape(-1))
    parts.append(np.asarray([_finite(observation.get("remaining_time_fraction"), 0.0)], dtype=float))
    return np.concatenate(parts) if parts else np.zeros(1, dtype=float)


def _predict_actions(trainer: Any, samples: Sequence[Mapping[str, Any]], chunk_size: int = 256) -> np.ndarray:
    import torch

    if not samples:
        return np.zeros((0, 3), dtype=float)
    trainer.policy.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), int(chunk_size)):
            batch = samples[start : start + int(chunk_size)]
            chunks.append(
                _batch_actor_physical_mean(trainer, [row["observation"] for row in batch])
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
    return np.concatenate(chunks, axis=0)


def _components(trainer: Any, sample: Mapping[str, Any], action: Sequence[float]) -> dict[str, float]:
    action_array = np.asarray(action, dtype=float).reshape(3)
    goal_direction = _unit(np.asarray(sample.get("goal_direction", (0.0, 0.0, 0.0)), dtype=float))
    velocity = _raw_estimated_velocity(trainer, sample["observation"])
    velocity_direction = _unit(velocity)
    radial = float(np.dot(action_array, goal_direction))
    tangent = action_array - radial * goal_direction
    braking = max(0.0, -float(np.dot(action_array, velocity_direction)))
    return {
        "action_norm_mps": float(np.linalg.norm(action_array)),
        "radial_action_projection_mps": radial,
        "tangential_action_mps": float(np.linalg.norm(tangent)),
        "braking_magnitude_mps": braking,
    }


def _load_roles(paths: Mapping[str, Path], root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    trainers: dict[str, Any] = {}
    configs: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for role, path in paths.items():
        trainer, config, payload, identity = _load_checkpoint(path.resolve(), root / role)
        trainers[role] = trainer
        configs[role] = config
        payloads[role] = payload
        identities[role] = identity
    return trainers, configs, payloads, identities


def _external_study_report(path: Path) -> None:
    text = """# External BC / DAgger policy-preservation study

本报告只记录公开实现对本项目 R4 设计的参考，不替换当前 Actor、MAPPO 或 DAgger 架构。

## HumanCompatibleAI/imitation 的机制

官方 DAgger 实现明确将过程拆成重复的 round：learner/robot 参与 rollout，expert 在 learner 实际访问的 observation 上提供 action label；新数据追加到历史数据集，然后再对累计 demonstrations 做 BC 更新。其 trainer 还保留 round checkpoint 与各轮 demonstrations，便于恢复和审计。

- 源码：<https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/algorithms/dagger.py>
- DAgger 文档：<https://github.com/HumanCompatibleAI/imitation/blob/master/docs/algorithms/dagger.rst>
- BC 源码：<https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/algorithms/bc.py>
- BC tutorial：<https://github.com/HumanCompatibleAI/imitation/blob/master/docs/tutorials/1_train_bc.ipynb>

## 机制 → 当前问题 → 是否适用 → 最小可测试形式 → 风险

| 外部机制 | 当前问题 | 是否适用 | 最小可测试形式 | 风险 |
|---|---|---|---|---|
| learner-visited DAgger labels | 当前已有大量 `student_closed_loop`，不是简单缺数据 | 适用，但不能只增加样本量 | Audit B 统计来源/距离/phase，再做单轮分层采样 | 继续聚合会扩大冲突标签和共享参数漂移 |
| aggregated replay | final BC 修改整个 Actor mean 后 normal forgetting | 适用 | E1 只做 distance×phase 重采样，保持网络与 loss | 过度过采样稀有状态可能重复相同轨迹 |
| round-separated BC | round0 与 final 混合后难区分哪轮改变了行为 | 适用 | E0 复现，E2 只改 trainable parameter range | 训练轮次/数据集不一致会混淆归因 |
| validation holdout | 需要判断是否只是训练集拟合 | 适用 | 保持独立 scenario/pair 的 validation，不把 step 当样本 | validation 若按 step 统计会伪增样本量 |
| restricted fine-tuning | GAT、body、head 同时漂移 | 直接适用 | E2-A/B/C/D 同数据、同 loss、同训练预算 | head-only 可能没有足够表达能力 |
| behavior anchor / replay constraint | 要保住 baseline normal navigation | 适用 | E3 仅在 normal observations 加 action MSE anchor，扫描 3 个 λ | anchor 太强会压制 recovery correction |
| RNN / hidden memory | Teacher 可能有 `_release_latched` 历史状态 | 当前不直接适用 | 先做 Audit D 最近邻 label consistency | 未确认 observation 不充分前引入 RNN 会改变问题定义 |

## 对本项目的采用边界

本阶段只采用 DAgger 的数据组织思想、采样审计和最小行为锚定实验；不安装或替换为外部 imitation 库，不修改 reward、dynamics、action space、safety semantics 或 PPO。所有结果以 episode/scenario 为重复单位，训练/验证轨迹严格分离。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _action_audit(
    samples: Sequence[Mapping[str, Any]],
    trainers: Mapping[str, Any],
    logs_root: Path,
) -> dict[str, Any]:
    predictions = {role: _predict_actions(trainer, samples) for role, trainer in trainers.items()}
    raw_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        teacher_action = np.asarray(sample["teacher_action_delta_v_mps"], dtype=float).reshape(3)
        teacher_parts = _components(next(iter(trainers.values())), sample, teacher_action)
        base = {
            "sample_index": index,
            "split": sample.get("split", ""),
            "episode_id": sample.get("episode_id", ""),
            "scenario_hash": sample.get("scenario_hash", ""),
            "step_index": sample.get("step_index", ""),
            "kind": sample.get("kind", ""),
            "distance_bin": _distance_bin(_distance(sample)),
            "distance_m": _distance(sample),
            "phase": _phase(sample),
            "base_phase": sample.get("base_phase", ""),
            "source_class": _source_class(sample),
            "teacher_action_norm_mps": teacher_parts["action_norm_mps"],
            "teacher_radial_action_projection_mps": teacher_parts["radial_action_projection_mps"],
            "teacher_tangential_action_mps": teacher_parts["tangential_action_mps"],
            "teacher_braking_magnitude_mps": teacher_parts["braking_magnitude_mps"],
        }
        for role, trainer in trainers.items():
            predicted = predictions[role][index]
            parts = _components(trainer, sample, predicted)
            raw_rows.append(
                {
                    **base,
                    "actor_role": role,
                    "actor_action_norm_mps": parts["action_norm_mps"],
                    "actor_radial_action_projection_mps": parts["radial_action_projection_mps"],
                    "actor_tangential_action_mps": parts["tangential_action_mps"],
                    "actor_braking_magnitude_mps": parts["braking_magnitude_mps"],
                    "actor_teacher_action_norm_ratio": parts["action_norm_mps"] / max(teacher_parts["action_norm_mps"], 1.0e-8),
                    "radial_absolute_error_mps": abs(parts["radial_action_projection_mps"] - teacher_parts["radial_action_projection_mps"]),
                    "radial_relative_error": abs(parts["radial_action_projection_mps"] - teacher_parts["radial_action_projection_mps"])
                    / max(abs(teacher_parts["radial_action_projection_mps"]), 0.1),
                    "tangential_absolute_error_mps": abs(parts["tangential_action_mps"] - teacher_parts["tangential_action_mps"]),
                    "braking_absolute_error_mps": abs(parts["braking_magnitude_mps"] - teacher_parts["braking_magnitude_mps"]),
                    "action_mse": float(np.mean((predicted - teacher_action) ** 2)),
                }
            )
    _write_csv(logs_root / "action_audit_step.csv", raw_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for role in trainers:
        for distance_bin in DISTANCE_BINS:
            for phase in sorted({str(row["phase"]) for row in raw_rows}):
                selected = [
                    row
                    for row in raw_rows
                    if row["actor_role"] == role and row["distance_bin"] == distance_bin and row["phase"] == phase
                ]
                if not selected:
                    continue
                aggregate_rows.append(
                    {
                        "actor_role": role,
                        "distance_bin": distance_bin,
                        "phase": phase,
                        "samples": len(selected),
                        "unique_episodes": len({str(row["episode_id"]) for row in selected}),
                        "teacher_action_norm_mps": _mean(row["teacher_action_norm_mps"] for row in selected),
                        "actor_action_norm_mps": _mean(row["actor_action_norm_mps"] for row in selected),
                        "actor_teacher_action_norm_ratio": _mean(row["actor_teacher_action_norm_ratio"] for row in selected),
                        "teacher_radial_action_projection_mps": _mean(row["teacher_radial_action_projection_mps"] for row in selected),
                        "actor_radial_action_projection_mps": _mean(row["actor_radial_action_projection_mps"] for row in selected),
                        "teacher_tangential_action_mps": _mean(row["teacher_tangential_action_mps"] for row in selected),
                        "actor_tangential_action_mps": _mean(row["actor_tangential_action_mps"] for row in selected),
                        "teacher_braking_magnitude_mps": _mean(row["teacher_braking_magnitude_mps"] for row in selected),
                        "actor_braking_magnitude_mps": _mean(row["actor_braking_magnitude_mps"] for row in selected),
                        "radial_absolute_error_mps": _mean(row["radial_absolute_error_mps"] for row in selected),
                        "radial_relative_error": _mean(row["radial_relative_error"] for row in selected),
                        "tangential_absolute_error_mps": _mean(row["tangential_absolute_error_mps"] for row in selected),
                        "braking_absolute_error_mps": _mean(row["braking_absolute_error_mps"] for row in selected),
                        "action_mse": _mean(row["action_mse"] for row in selected),
                    }
                )
    _write_csv(logs_root / "action_shrinkage_by_distance_phase.csv", aggregate_rows)
    overall: dict[str, Any] = {}
    for role in trainers:
        selected = [row for row in raw_rows if row["actor_role"] == role]
        overall[role] = {
            "samples": len(selected),
            "mean_action_norm_ratio": _mean(row["actor_teacher_action_norm_ratio"] for row in selected),
            "mean_radial_absolute_error_mps": _mean(row["radial_absolute_error_mps"] for row in selected),
            "mean_tangential_absolute_error_mps": _mean(row["tangential_absolute_error_mps"] for row in selected),
            "mean_braking_absolute_error_mps": _mean(row["braking_absolute_error_mps"] for row in selected),
            "mean_action_mse": _mean(row["action_mse"] for row in selected),
        }
    report_lines = [
        "# Action shrinkage audit",
        "",
        "同一批 42D、冻结 normalizer 的 Actor observation 被分别输入 Teacher label、baseline Actor、round0 BC 和 final BC。统计单位是 episode 内的 step 记录；推断只在 episode/scenario 层解释，未把 step 当作独立重复实验。",
        "",
        "## Overall",
        "",
        "| actor | samples | action-norm ratio | radial error (m/s) | tangential error (m/s) | braking error (m/s) | action MSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role, values in overall.items():
        report_lines.append(
            f"| {role} | {values['samples']} | {_fmt(values['mean_action_norm_ratio'])} | {_fmt(values['mean_radial_absolute_error_mps'])} | {_fmt(values['mean_tangential_absolute_error_mps'])} | {_fmt(values['mean_braking_absolute_error_mps'])} | {_fmt(values['mean_action_mse'])} |"
        )
    near = [row for row in aggregate_rows if row["distance_bin"] in {"5-3m", "3-1m", "<=1m"}]
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"近目标（5–3m、3–1m、<=1m）单元的记录数为 {_unique_count(near, 'samples')}（按行记录汇总），其余距离单元仍需结合 dataset distribution audit 解读。",
            "- loss averaging：由 action MSE 训练目标导致的平均效应只能解释平均拟合，不能解释某一个 phase 的系统性幅值塌缩；本文件保留逐 bin/phase 的误差用于区分。",
            "- sample imbalance：若近目标单元样本明显少于 >8m，E1 的唯一变量将是 sampling；是否为主因由 E1 与 E0 的 fidelity/任务指标决定。",
            "- conflicting labels：由 Audit D 的最近邻 observation/action 距离检验；若 observation 很近而 label 差异很大，feed-forward 回归会产生均值化。",
            "- shared-parameter drift：由 Audit C 的参数组 restore sensitivity 检验；若恢复某组 baseline 权重显著降低 normal action drift，则该组是 forgetting 的功能来源。",
            "- normalization、tanh saturation 和 phase mixing：本轮不修改它们；报告分别给出归一化合同、动作是否接近 component limit，以及 phase 交叉统计，避免将总 MSE 当成单一原因。",
        ]
    )
    report_lines.append("")
    report_lines.append("具体的 5–3m、3–1m、<=1m 数值请见 `action_shrinkage_by_distance_phase.csv`；不使用跨 episode 的伪重复显著性检验。")
    (logs_root / "action_shrinkage_audit.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"overall": overall, "raw_rows": len(raw_rows), "aggregate_rows": aggregate_rows}


def _dataset_distribution(samples_by_split: Mapping[str, Sequence[Mapping[str, Any]]], logs_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split, samples in samples_by_split.items():
        grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for sample in samples:
            grouped[(_distance_bin(_distance(sample)), _phase(sample), _source_class(sample))].append(sample)
        for (distance_bin, phase, source), selected in sorted(grouped.items()):
            rows.append(
                {
                    "dataset_view": "balanced_jsonl",
                    "split": split,
                    "distance_bin": distance_bin,
                    "phase": phase,
                    "source_class": source,
                    "samples": len(selected),
                    "unique_episodes": len({str(row.get("episode_id", "")) for row in selected}),
                    "unique_scenarios": len({str(row.get("scenario_hash", "")) for row in selected}),
                }
            )
    raw_rows = _read_csv(SOURCE_LOG_ROOT / "dagger_bc/demonstration_step_audit.csv")
    raw_grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        split = str(row.get("split", ""))
        if split not in {"train", "validation"}:
            continue
        distance = _finite(row.get("goal_distance_m", row.get("estimated_goal_distance_m")))
        phase = str(row.get("phase", "")) or str(row.get("base_phase", "")) or "unknown"
        source = "student_closed_loop" if str(row.get("student_closed_loop", "")).lower() in {"true", "1"} else str(row.get("collection_mode", "other"))
        raw_grouped[(split, _distance_bin(distance), phase, source)].append(row)
    for (split, distance_bin, phase, source), selected in sorted(raw_grouped.items()):
        rows.append(
            {
                "dataset_view": "raw_step_audit",
                "split": split,
                "distance_bin": distance_bin,
                "phase": phase,
                "source_class": source,
                "samples": len(selected),
                "unique_episodes": len({str(row.get("episode_id", "")) for row in selected}),
                "unique_scenarios": len({str(row.get("scenario_hash", "")) for row in selected}),
            }
        )
    _write_csv(logs_root / "dataset_distribution_audit.csv", rows)

    # The heatmap is the actual balanced JSONL train view used by current BC.
    train = samples_by_split.get("train", [])
    phase_order = ["normal", "approach", "decision", "bypass", "transition", "post_bypass", "recovery", "goal_approach", "braking", "boundary_onset"]
    phase_order.extend(sorted({_phase(row) for row in train} - set(phase_order)))
    matrix = np.zeros((len(phase_order), len(DISTANCE_BINS)), dtype=int)
    for row in train:
        matrix[phase_order.index(_phase(row)), DISTANCE_BINS.index(_distance_bin(_distance(row)))] += 1
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10.5, max(4.5, 0.38 * len(phase_order) + 1.8)), constrained_layout=True)
        image = axis.imshow(matrix, cmap="viridis", aspect="auto")
        axis.set_xticks(np.arange(len(DISTANCE_BINS)), DISTANCE_BINS, rotation=25, ha="right")
        axis.set_yticks(np.arange(len(phase_order)), phase_order)
        axis.set_xlabel("distance bin")
        axis.set_ylabel("phase")
        axis.set_title("Balanced BC train distribution: distance × phase")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() * 0.45 else "black", fontsize=8)
        figure.colorbar(image, ax=axis, label="samples")
        figure.savefig(logs_root.parent.parent / "dummy.png") if False else None
        plt.close(figure)
        # Caller places plots in the proper root; this temporary marker is not used.
    except Exception as exc:  # pragma: no cover - plotting availability is environment-dependent
        heatmap_error = str(exc)
    else:
        heatmap_error = ""
    return {
        "rows": rows,
        "train_matrix": matrix.tolist(),
        "phase_order": phase_order,
        "heatmap_error": heatmap_error,
    }


def _save_distribution_heatmap(distribution: Mapping[str, Any], plots_root: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(distribution["train_matrix"], dtype=int)
    phases = list(distribution["phase_order"])
    figure, axis = plt.subplots(figsize=(10.5, max(4.5, 0.38 * len(phases) + 1.8)), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(DISTANCE_BINS)), DISTANCE_BINS, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(phases)), phases)
    axis.set_xlabel("distance bin")
    axis.set_ylabel("phase")
    axis.set_title("Balanced BC train distribution: distance × phase")
    maximum = int(matrix.max()) if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="white" if maximum and matrix[i, j] > maximum * 0.45 else "black", fontsize=8)
    figure.colorbar(image, ax=axis, label="samples")
    path = plots_root / "dataset_distribution_heatmap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return str(path.resolve())


def _distribution_report(distribution: Mapping[str, Any], path: Path) -> None:
    rows = list(distribution["rows"])
    lines = [
        "# Dataset distribution audit",
        "",
        "报告同时保留当前实际 BC JSONL（balanced_jsonl）与历史完整 step audit（raw_step_audit）。所有 count 仍按记录展示，但 unique episode/scenario 同时给出，避免将 step 当成独立实验样本。",
        "",
        "## Balanced JSONL train distance × phase",
        "",
        "| phase | >8m | 8-5m | 5-3m | 3-1m | <=1m |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    train = [row for row in rows if row["dataset_view"] == "balanced_jsonl" and row["split"] == "train"]
    for phase in distribution["phase_order"]:
        values = [next((row["samples"] for row in train if row["phase"] == phase and row["distance_bin"] == item), 0) for item in DISTANCE_BINS]
        lines.append(f"| {phase} | " + " | ".join(str(value) for value in values) + " |")
    lines.extend(["", "## Source proportions", ""])
    for split in ("train", "validation"):
        selected = [row for row in rows if row["dataset_view"] == "balanced_jsonl" and row["split"] == split]
        totals = Counter()
        for row in selected:
            totals[row["source_class"]] += int(row["samples"])
        total = sum(totals.values())
        lines.append(f"- {split}: " + ", ".join(f"{key}={value} ({value / total:.3f})" for key, value in sorted(totals.items())) if total else f"- {split}: empty")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "如果 >8m 占据绝大多数、而 5–3m/3–1m/<=1m 以及 transition/recovery/goal_approach/braking 很少，则总 MSE 会偏向远距离动作；E1 将在同一批既有轨迹上只修改 sampling 进行检验。若 student_closed_loop 已占主要来源，则‘缺 learner-visited data’不能作为解释。",
            "",
            "详细 raw/balanced 统计见 `dataset_distribution_audit.csv`；图见 `dataset_distribution_heatmap.png`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parameter_group(name: str) -> str:
    if name.startswith("gat."):
        return "GAT"
    if name.startswith("actor_body.0") or name.startswith("actor_body.2"):
        return "actor_body_early"
    if name.startswith("actor_body.4"):
        return "actor_body_late"
    if name.startswith("delta_mean_head."):
        return "delta_mean_head"
    return "other"


def _normal_forgetting_audit(
    bank: Sequence[Mapping[str, Any]],
    trainers: Mapping[str, Any],
    logs_root: Path,
) -> dict[str, Any]:
    predictions = {role: _predict_actions(trainer, bank) for role, trainer in trainers.items()}
    baseline = predictions["baseline_actor"]
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(bank):
        base_parts = _components(trainers["baseline_actor"], sample, baseline[index])
        row_base = {
            "sample_index": index,
            "episode_id": sample.get("episode_id", ""),
            "scenario_hash": sample.get("scenario_hash", ""),
            "step_index": sample.get("step_index", ""),
            "distance_bin": _distance_bin(_distance(sample)),
            "phase": _phase(sample),
        }
        for role, action_rows in predictions.items():
            parts = _components(trainers[role], sample, action_rows[index])
            rows.append(
                {
                    **row_base,
                    "policy": role,
                    "action_l2_drift_mps": float(np.linalg.norm(action_rows[index] - baseline[index])),
                    "radial_drift_mps": abs(parts["radial_action_projection_mps"] - base_parts["radial_action_projection_mps"]),
                    "tangential_drift_mps": abs(parts["tangential_action_mps"] - base_parts["tangential_action_mps"]),
                    "braking_drift_mps": abs(parts["braking_magnitude_mps"] - base_parts["braking_magnitude_mps"]),
                }
            )
    _write_csv(logs_root / "normal_action_drift.csv", rows)

    baseline_state = {key: value.detach().cpu().clone() for key, value in trainers["baseline_actor"].policy.state_dict().items()}
    final_trainer = trainers["final_bc"]
    final_state = {key: value.detach().cpu().clone() for key, value in final_trainer.policy.state_dict().items()}
    group_keys: dict[str, list[str]] = defaultdict(list)
    for name in final_state:
        group = _parameter_group(name)
        if group != "other" and name in baseline_state and tuple(final_state[name].shape) == tuple(baseline_state[name].shape):
            group_keys[group].append(name)
    sensitivity: list[dict[str, Any]] = []
    full_drift = _mean(row["action_l2_drift_mps"] for row in rows if row["policy"] == "final_bc")
    for group in ("GAT", "actor_body_early", "actor_body_late", "delta_mean_head"):
        names = group_keys.get(group, [])
        diff_norm = math.sqrt(sum(float(torch_norm(final_state[name] - baseline_state[name]) ** 2) for name in names)) if names else 0.0
        base_norm = math.sqrt(sum(float(torch_norm(baseline_state[name]) ** 2) for name in names)) if names else 0.0
        restored = dict(final_state)
        for name in names:
            restored[name] = baseline_state[name].clone()
        final_trainer.policy.load_state_dict(restored, strict=True)
        restored_predictions = _predict_actions(final_trainer, bank)
        restored_drift = float(np.mean(np.linalg.norm(restored_predictions - baseline, axis=1))) if len(bank) else 0.0
        final_trainer.policy.load_state_dict(final_state, strict=True)
        sensitivity.append(
            {
                "parameter_group": group,
                "parameter_count": int(sum(final_state[name].numel() for name in names)),
                "parameter_l2_diff": diff_norm,
                "baseline_parameter_l2": base_norm,
                "relative_parameter_drift": diff_norm / max(base_norm, 1.0e-8),
                "full_final_action_drift_mps": full_drift,
                "restore_group_action_drift_mps": restored_drift,
                "drift_reduction_when_restored_mps": full_drift - restored_drift,
            }
        )
    _write_csv(logs_root / "normal_forgetting_parameter_sensitivity.csv", sensitivity)
    overall: dict[str, Any] = {}
    for role in trainers:
        selected = [row for row in rows if row["policy"] == role]
        overall[role] = {
            "mean_action_l2_drift_mps": _mean(row["action_l2_drift_mps"] for row in selected),
            "mean_radial_drift_mps": _mean(row["radial_drift_mps"] for row in selected),
            "mean_tangential_drift_mps": _mean(row["tangential_drift_mps"] for row in selected),
            "mean_braking_drift_mps": _mean(row["braking_drift_mps"] for row in selected),
        }
    lines = [
        "# Normal-navigation forgetting audit",
        "",
        "observation bank 来自既有 baseline normal_far_distance rollout；同一批输入比较 baseline Actor、round0 BC 和 final BC。参数组敏感性采用‘将 final 某组恢复为 baseline、其余保持 final’的功能性 leave-group-restore 对照。",
        "",
        "## Action drift",
        "",
        "| policy | L2 drift | radial drift | tangential drift | braking drift |",
        "|---|---:|---:|---:|---:|",
    ]
    for role, value in overall.items():
        lines.append(f"| {role} | {_fmt(value['mean_action_l2_drift_mps'])} | {_fmt(value['mean_radial_drift_mps'])} | {_fmt(value['mean_tangential_drift_mps'])} | {_fmt(value['mean_braking_drift_mps'])} |")
    lines.extend(["", "## Parameter-group restore sensitivity", "", "| group | parameter L2 diff | relative drift | final action drift | restore action drift | reduction |", "|---|---:|---:|---:|---:|---:|"])
    for row in sensitivity:
        lines.append(f"| {row['parameter_group']} | {_fmt(row['parameter_l2_diff'])} | {_fmt(row['relative_parameter_drift'])} | {_fmt(row['full_final_action_drift_mps'])} | {_fmt(row['restore_group_action_drift_mps'])} | {_fmt(row['drift_reduction_when_restored_mps'])} |")
    lines.extend(["", "参数组只能按上述 restore sensitivity 解释为功能相关证据，不能仅凭参数范数宣称因果。完整逐 observation 结果见 `normal_action_drift.csv`。"])
    (logs_root / "normal_forgetting_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"overall": overall, "parameter_sensitivity": sensitivity, "rows": len(rows)}


def torch_norm(value: Any) -> float:
    return float(np.linalg.norm(value.detach().cpu().numpy().reshape(-1)))


def _release_lookup() -> tuple[dict[tuple[str, int], Any], str | None]:
    path = SOURCE_LOG_ROOT / "dagger_bc/demonstration_step_audit.csv"
    rows = _read_csv(path)
    if not rows:
        return {}, None
    fields = list(rows[0])
    field = next((item for item in fields if "release" in item.lower() and "latch" in item.lower()), None)
    if field is None:
        return {}, None
    lookup: dict[tuple[str, int], Any] = {}
    for row in rows:
        try:
            step = int(float(row.get("step", row.get("step_index", 0))))
        except (TypeError, ValueError):
            continue
        lookup[(str(row.get("episode_id", "")), step)] = row.get(field)
    return lookup, field


def _teacher_label_conflict_audit(samples: Sequence[Mapping[str, Any]], logs_root: Path) -> dict[str, Any]:
    selected = [sample for sample in samples if _phase(sample) in FOCUS_PHASES and str(sample.get("kind", "")) == "static_obstacle"]
    if len(selected) > 800:
        selected = [selected[index] for index in np.linspace(0, len(selected) - 1, 800, dtype=int)]
    lookup, release_field = _release_lookup()
    if len(selected) < 2:
        _write_csv(logs_root / "teacher_label_conflict_audit.csv", [])
        (logs_root / "teacher_label_conflict_report.md").write_text("# Teacher label conflict audit\n\n可用于最近邻分析的 transition/post_bypass/recovery static records 少于 2。\n", encoding="utf-8")
        return {"pairs": 0, "conflicts": 0, "release_field": release_field}
    matrix = np.stack([_observation_vector(sample["observation"]) for sample in selected])
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1.0e-8] = 1.0
    normalized = (matrix - mean) / scale
    labels = np.stack([np.asarray(sample["teacher_action_delta_v_mps"], dtype=float).reshape(3) for sample in selected])
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(selected):
        distances = np.linalg.norm(normalized - normalized[index], axis=1)
        for other_index, other in enumerate(selected):
            if str(other.get("episode_id", "")) == str(sample.get("episode_id", "")):
                distances[other_index] = float("inf")
        nearest = int(np.argmin(distances))
        obs_distance = float(distances[nearest])
        action_distance = float(np.linalg.norm(labels[index] - labels[nearest]))
        key = (str(sample.get("episode_id", "")), int(sample.get("step_index", 0) or 0))
        other_key = (str(selected[nearest].get("episode_id", "")), int(selected[nearest].get("step_index", 0) or 0))
        release = sample.get("release_latched", lookup.get(key, "unavailable"))
        other_release = selected[nearest].get("release_latched", lookup.get(other_key, "unavailable"))
        rows.append(
            {
                "episode_id": sample.get("episode_id", ""),
                "step_index": sample.get("step_index", ""),
                "phase": _phase(sample),
                "nearest_episode_id": selected[nearest].get("episode_id", ""),
                "nearest_step_index": selected[nearest].get("step_index", ""),
                "nearest_phase": _phase(selected[nearest]),
                "observation_distance_standardized_l2": obs_distance,
                "teacher_action_distance_mps": action_distance,
                "teacher_action_norm_mps": float(np.linalg.norm(labels[index])),
                "nearest_teacher_action_norm_mps": float(np.linalg.norm(labels[nearest])),
                "release_latched": release,
                "nearest_release_latched": other_release,
                "goal_distance_m": _distance(sample),
                "nearest_goal_distance_m": _distance(selected[nearest]),
                "is_label_conflict": int(obs_distance <= 1.0 and action_distance >= 0.25),
            }
        )
    rows.sort(key=lambda row: (int(row["is_label_conflict"]), -float(row["teacher_action_distance_mps"]), float(row["observation_distance_standardized_l2"])), reverse=True)
    _write_csv(logs_root / "teacher_label_conflict_audit.csv", rows)
    conflicts = [row for row in rows if int(row["is_label_conflict"])]
    report = [
        "# Teacher hidden-state / latch label-conflict audit",
        "",
        f"分析对象为 static obstacle 的 transition/post_bypass/recovery records，最近邻强制来自不同 episode。观测距离是在当前 Actor-visible normalized observation 上的 standardized L2；固定冲突判据为 observation distance <= 1.0 且 teacher action distance >= 0.25 m/s。",
        "",
        f"- candidate records={len(selected)}，nearest-neighbor pairs={len(rows)}。",
        f"- label conflicts={len(conflicts)}，rate={len(conflicts) / len(rows) if rows else 0.0:.3f}。",
        f"- release latch field={'available: ' + release_field if release_field else 'not present in historical JSONL/step audit'}。",
        "",
        "## Interpretation",
        "",
        "若冲突率高，说明仅用现有 feed-forward observation 可能无法唯一确定 Teacher 的 recovery action，应先使 Teacher label Markov-consistent；本阶段不直接加入 RNN。若冲突率低，则 hidden latch 不是当前 action shrinkage 的主要证据。",
        "",
        "详表见 `teacher_label_conflict_audit.csv`。",
    ]
    (logs_root / "teacher_label_conflict_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"pairs": len(rows), "conflicts": len(conflicts), "conflict_rate": len(conflicts) / len(rows) if rows else 0.0, "release_field": release_field}


def _mean(values: Iterable[Any]) -> float:
    numeric = [_finite(value) for value in values]
    numeric = [value for value in numeric if math.isfinite(value)]
    return float(np.mean(numeric)) if numeric else float("nan")


def _unique_count(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    # For aggregate rows, `samples` is not an identity.  This helper is used
    # only for display and therefore returns a numeric sum for that field.
    return int(sum(int(row.get(field, 0) or 0) for row in rows))


def _fmt(value: Any) -> str:
    number = _finite(value)
    return "nan" if not math.isfinite(number) else f"{number:.6g}"


def _write_demo_root(root: Path, train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    demos = root / "demonstrations"
    _write_jsonl(demos / "train.jsonl", train)
    _write_jsonl(demos / "validation.jsonl", validation)
    _write_json(demos / "manifest.json", _jsonable({
        "dataset_id": STAGE_NAME,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_episodes": len({(row.get("scenario_hash"), row.get("episode_index")) for row in train}),
        "validation_episodes": len({(row.get("scenario_hash"), row.get("episode_index")) for row in validation}),
        **dict(metadata),
    }))


def _stratified_resample(samples: Sequence[Mapping[str, Any]], count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[(_distance_bin(_distance(sample)), _phase(sample))].append(sample)
    distance_target = {">8m": 0.20, "8-5m": 0.20, "5-3m": 0.25, "3-1m": 0.20, "<=1m": 0.15}
    weights: dict[tuple[str, str], float] = {}
    for key, values in groups.items():
        weights[key] = distance_target.get(key[0], 0.1) / math.sqrt(max(len(values), 1))
    total_weight = sum(weights.values()) or 1.0
    desired = {key: float(count) * value / total_weight for key, value in weights.items()}
    quotas = {key: int(math.floor(value)) for key, value in desired.items()}
    remaining = int(count - sum(quotas.values()))
    for key in sorted(desired, key=lambda item: (-(desired[item] - quotas[item]), item))[:remaining]:
        quotas[key] += 1
    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        values = groups[key]
        quota = quotas.get(key, 0)
        if quota <= 0:
            continue
        replace = quota > len(values)
        indices = rng.choice(len(values), size=quota, replace=replace)
        selected.extend(copy.deepcopy(dict(values[int(index)])) for index in np.asarray(indices).reshape(-1))
    rng.shuffle(selected)
    return selected, {
        "sampling": "distance_phase_stratified_weighted_resample",
        "seed": int(seed),
        "requested_samples": int(count),
        "actual_samples": len(selected),
        "cell_quotas": {f"{key[0]}|{key[1]}": int(value) for key, value in sorted(quotas.items())},
        "cell_source_counts": {f"{key[0]}|{key[1]}": len(value) for key, value in sorted(groups.items())},
    }


def _scope_names(policy: Any, scope: str) -> tuple[str, ...]:
    all_names = tuple(actor_mean_trainable_parameter_names(policy))
    if scope == "delta_mean_head_only":
        selected = tuple(name for name in all_names if name.startswith("delta_mean_head."))
    elif scope == "last_actor_body_block_plus_head":
        selected = tuple(name for name in all_names if name.startswith("actor_body.4") or name.startswith("delta_mean_head."))
    elif scope == "actor_body_plus_head_gat_frozen":
        selected = tuple(name for name in all_names if name.startswith("actor_body.") or name.startswith("delta_mean_head."))
    elif scope == "full_actor_mean":
        selected = all_names
    else:
        raise ValueError(f"unknown R4 trainable scope: {scope}")
    if not selected:
        raise RuntimeError(f"scope {scope} selected no Actor mean parameters")
    return selected


def _train_candidate(
    *,
    label: str,
    scope_names: Sequence[str] | None,
    initial_checkpoint: Path,
    demonstrations_dir: Path,
    data_root: Path,
    logs_root: Path,
    plots_root: Path,
    anchor_lambda: float = 0.0,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(20262702)
    return _train_imitation(
        initial_checkpoint,
        demonstrations_dir,
        output_dir=data_root / label,
        logs_dir=logs_root / label,
        plots_dir=plots_root / label,
        max_epochs=45,
        patience=8,
        batch_size=32,
        learning_rate=1.0e-4,
        seed=20262702,
        stage=f"stage2br4_{label}",
        force=False,
        progress=True,
        trainable_parameter_names=scope_names,
        anchor_lambda=anchor_lambda,
    )


def _build_anchor_dataset(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    baseline_trainer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[list[dict[str, Any]]] = []
    for collection in (train, validation):
        predictions = _predict_actions(baseline_trainer, collection)
        enriched: list[dict[str, Any]] = []
        for sample, action in zip(collection, predictions):
            row = copy.deepcopy(dict(sample))
            row["baseline_actor_action_delta_v_mps"] = action.tolist()
            row["anchor_apply"] = str(sample.get("kind", "")) == "normal_far_distance"
            enriched.append(row)
        output.append(enriched)
    return output[0], output[1]


def _action_fidelity_summary(
    samples: Sequence[Mapping[str, Any]],
    trainers: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, trainer in trainers.items():
        predictions = _predict_actions(trainer, samples)
        for distance_bin in DISTANCE_BINS:
            selected = [index for index, sample in enumerate(samples) if _distance_bin(_distance(sample)) == distance_bin]
            if not selected:
                continue
            ratios: list[float] = []
            radial: list[float] = []
            tangential: list[float] = []
            braking: list[float] = []
            for index in selected:
                teacher = np.asarray(samples[index]["teacher_action_delta_v_mps"], dtype=float)
                t = _components(trainer, samples[index], teacher)
                p = _components(trainer, samples[index], predictions[index])
                ratios.append(p["action_norm_mps"] / max(t["action_norm_mps"], 1.0e-8))
                radial.append(abs(p["radial_action_projection_mps"] - t["radial_action_projection_mps"]))
                tangential.append(abs(p["tangential_action_mps"] - t["tangential_action_mps"]))
                braking.append(abs(p["braking_magnitude_mps"] - t["braking_magnitude_mps"]))
            rows.append(
                {
                    "policy": role,
                    "distance_bin": distance_bin,
                    "samples": len(selected),
                    "mean_action_norm_ratio": _mean(ratios),
                    "mean_radial_error_mps": _mean(radial),
                    "mean_tangential_error_mps": _mean(tangential),
                    "mean_braking_error_mps": _mean(braking),
                }
            )
    return rows


def _write_ablation_summary(
    evaluation: Mapping[str, Any],
    fidelity_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    aggregates = evaluation.get("aggregates", {})
    for split, split_values in aggregates.items():
        for role, metrics in split_values.items():
            fidelity = [row for row in fidelity_rows if row["policy"] == role]
            results.append(
                {
                    "split": split,
                    "policy": role,
                    "episodes": metrics.get("episodes", 0),
                    "entered_8m": metrics.get("entered_8m", 0),
                    "entered_8m_rate": metrics.get("entered_8m_rate", 0.0),
                    "entered_3m": metrics.get("entered_3m", 0),
                    "entered_1m": metrics.get("entered_1m", 0),
                    "stable_goal": metrics.get("stable_goal", 0),
                    "passed_obstacle": metrics.get("passed_obstacle", 0),
                    "post_bypass_recovery": metrics.get("post_bypass_recovery", 0),
                    "mean_minimum_goal_distance_m": metrics.get("mean_minimum_goal_distance_m"),
                    "mean_speed_at_8m_entry_mps": metrics.get("mean_speed_at_8m_entry_mps"),
                    "mean_speed_at_3m_entry_mps": metrics.get("mean_speed_at_3m_entry_mps"),
                    "mean_speed_at_1m_entry_mps": metrics.get("mean_speed_at_1m_entry_mps"),
                    "mean_maximum_consecutive_goal_hold_steps": metrics.get("mean_maximum_consecutive_goal_hold_steps"),
                    "hold_steps_ge_1": metrics.get("hold_steps_ge_1", 0),
                    "hold_steps_ge_3": metrics.get("hold_steps_ge_3", 0),
                    "hold_steps_ge_5": metrics.get("hold_steps_ge_5", 0),
                    "mean_action_norm_ratio": _mean(row.get("mean_action_norm_ratio") for row in fidelity),
                    "mean_radial_action_error_mps": _mean(row.get("mean_radial_error_mps") for row in fidelity),
                    "mean_tangential_action_error_mps": _mean(row.get("mean_tangential_error_mps") for row in fidelity),
                    "mean_braking_action_error_mps": _mean(row.get("mean_braking_error_mps") for row in fidelity),
                }
            )
    _write_csv(path.parent / "ablation_results.csv", results)
    control = next((row for row in results if row["split"] == "normal_far_distance" and row["policy"] == "current_bc_actor"), {})
    candidates = [
        row
        for row in results
        if row["split"] == "normal_far_distance" and row["policy"] in {"e1_sampling", "e2a_delta_mean_head_only", "e2b_last_actor_body_block_plus_head", "e2c_actor_body_plus_head_gat_frozen", "e2d_full_actor_mean"}
    ]
    static_rows = {row["policy"]: row for row in results if row["split"] == "static_holdout"}
    selected: list[str] = []
    for row in candidates:
        static = static_rows.get(row["policy"], {})
        if (
            int(row.get("entered_8m", 0) or 0) >= 19
            and int(static.get("passed_obstacle", 0) or 0) >= int(static_rows.get("current_bc_actor", {}).get("passed_obstacle", 0) or 0)
            and int(static.get("post_bypass_recovery", 0) or 0) >= int(static_rows.get("current_bc_actor", {}).get("post_bypass_recovery", 0) or 0)
            and int(static.get("entered_1m", 0) or 0) > int(static_rows.get("current_bc_actor", {}).get("entered_1m", 0) or 0)
            and int(static.get("stable_goal", 0) or 0) > int(static_rows.get("current_bc_actor", {}).get("stable_goal", 0) or 0)
        ):
            selected.append(str(row["policy"]))
    summary = {
        "normal_gate_candidates": selected,
        "current_control": control,
        "e3_allowed": bool(selected),
        "reason": "at least one E1/E2 candidate satisfies normal entered_8m >= 19 and static preservation/improvement" if selected else "no E1/E2 candidate satisfies all core gates; stop before E3/PPO",
    }
    lines = [
        "# Ablation summary",
        "",
        "E0 is the control replication; E1 changes only distance×phase sampling; E2 changes only the Actor mean trainable range. E3 is not eligible unless an E1/E2 candidate satisfies the explicit normal and static gates.",
        "",
        "| split | policy | episodes | 8m | 3m | 1m | stable | min distance | mean max hold | hold>=1 | hold>=3 | hold>=5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(f"| {row['split']} | {row['policy']} | {row['episodes']} | {row['entered_8m']} | {row['entered_3m']} | {row['entered_1m']} | {row['stable_goal']} | {_fmt(row['mean_minimum_goal_distance_m'])} | {_fmt(row['mean_maximum_consecutive_goal_hold_steps'])} | {row['hold_steps_ge_1']} | {row['hold_steps_ge_3']} | {row['hold_steps_ge_5']} |")
    lines.extend(["", f"- E3 allowed: {summary['e3_allowed']}", f"- candidates: {', '.join(selected) if selected else 'none'}", f"- decision: {summary['reason']}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results, summary


def _final_report(
    path: Path,
    *,
    audit: Mapping[str, Any],
    distribution: Mapping[str, Any],
    forgetting: Mapping[str, Any],
    conflict: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None,
    ablation_decision: Mapping[str, Any],
    e0_status: Mapping[str, Any],
) -> None:
    normal = (evaluation or {}).get("aggregates", {}).get("normal_far_distance", {})
    static = (evaluation or {}).get("aggregates", {}).get("static_holdout", {})
    lines = [
        "# Stage 2B-R4 final report",
        "",
        "本阶段未进入 PPO；没有修改 reward、dynamics、action space、safety/boundary projection、stable criterion 或 PPO 超参数。",
        "",
        "## 1. Execution status",
        "",
        f"- E0 status: {e0_status.get('status')}；reproduction_passed={e0_status.get('reproduction_passed')}。",
        f"- E1/E2 completed: {bool(evaluation)}。",
        f"- E3 allowed: {ablation_decision.get('e3_allowed')}；PPO allowed: false。",
        "",
        "## 2. Required answers",
        "",
        "1. **瓶颈是否已转移到 Actor imitation？** 当前证据若同时显示 Teacher 通过、BC action fidelity 下降、normal action drift 非零，则主要瓶颈在 Actor imitation/policy preservation，而不是重新设计 Teacher；最终以本报告指标和历史 Teacher 结果共同判断。",
        f"2. **BC action shrinkage 主因：** overall audit={json.dumps(audit.get('overall', {}), ensure_ascii=False)}；应优先检查 5–3m/3–1m 单元的 ratio 与 radial error，而不是总 MSE。",
        f"3. **distance/phase imbalance：** dataset distribution 已输出，train matrix={json.dumps(distribution.get('train_matrix', []), ensure_ascii=False)}；E1 是唯一改变 sampling 的检验。",
        f"4. **normal forgetting 参数组：** restore sensitivity={json.dumps(forgetting.get('parameter_sensitivity', []), ensure_ascii=False)}；按最大 action-drift reduction 排名，不按参数量猜测。",
        f"5. **Teacher hidden latch label conflict：** pairs={conflict.get('pairs', 0)}, conflicts={conflict.get('conflicts', 0)}, rate={_fmt(conflict.get('conflict_rate', 0.0))}, release_field={conflict.get('release_field') or 'unavailable'}。未确认 observation 不足前不引入 RNN。",
        f"6. **restricted fine-tuning 是否更好：** 由 E2 与 E0 的统一 normal/static 表决定；当前 candidate gate={ablation_decision.get('normal_gate_candidates', [])}。",
        "7. **baseline anchor 是否保住 normal 20/20：** E3 只有在 E1/E2 候选通过 normal entered_8m>=19 且 static preservation/improvement 后才允许运行；本阶段未擅自进入 PPO。",
        "8. **最小有效机制组合：** 若 candidate gate 为空，当前没有证据支持任何 E1/E2 组合，停止在 E2；若非空，候选需再经过 E3 anchor trade-off。",
        "9. **static entered_1m/stable 是否非零：** 请以 evaluation 表中 static_holdout 的各角色行读取；没有把训练集结果冒充 holdout 结果。",
        "10. **是否进入 PPO + BC anchor：** 否；只有 normal gate、static gate、action fidelity gate 同时通过后才可单独规划，且本轮不自动执行。",
        "",
        "## 3. Evaluation roles",
        "",
    ]
    if evaluation:
        lines.extend(["| split | role | 8m | 3m | 1m | stable | min distance | mean max hold |", "|---|---|---:|---:|---:|---:|---:|---:|"])
        for split, roles in evaluation.get("aggregates", {}).items():
            for role, metrics in roles.items():
                lines.append(f"| {split} | {role} | {metrics.get('entered_8m', 0)}/{metrics.get('episodes', 0)} | {metrics.get('entered_3m', 0)}/{metrics.get('episodes', 0)} | {metrics.get('entered_1m', 0)}/{metrics.get('episodes', 0)} | {metrics.get('stable_goal', 0)}/{metrics.get('episodes', 0)} | {_fmt(metrics.get('mean_minimum_goal_distance_m'))} | {_fmt(metrics.get('mean_maximum_consecutive_goal_hold_steps'))} |")
    else:
        lines.append("E0 未通过或尚未产生统一 evaluation，因此没有继续报告 E1/E2 性能表。")
    lines.extend(
        [
            "",
            "## 4. Evidence and limitations",
            "",
            "- step-level CSV 只用于 action/distribution/nearest-neighbor 诊断；性能门槛按 episode 汇总。",
            "- release latch 若历史数据没有字段，会明确标记 unavailable，不能据此虚构 hidden-state 证据。",
            "- collision/clearance/safety 机制未在本阶段改变；R4 只研究 imitation/policy preservation。",
            "- 详细文件：`external_bc_preservation_study.md`、`action_shrinkage_audit.md`、`dataset_distribution_audit.md`、`normal_forgetting_audit.md`、`teacher_label_conflict_report.md`、`ablation_summary.md`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage2br4(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    progress: bool = True,
) -> dict[str, Any]:
    """Run audits, E0, E1 and E2; conditionally permit later stages."""

    data_root, logs_root, plots_root = Path(data_root), Path(logs_root), Path(plots_root)
    _prepare_roots(data_root, logs_root, plots_root)
    required = (BASELINE_ORIGINAL, BASELINE_MIGRATED, ROUND0_CHECKPOINT, CURRENT_BC_CHECKPOINT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("R4 required checkpoint(s) missing: " + ", ".join(missing))
    train = _read_demo_jsonl(SOURCE_DEMO_ROOT / "train.jsonl")
    validation = _read_demo_jsonl(SOURCE_DEMO_ROOT / "validation.jsonl")
    if not train or not validation:
        raise ValueError("R4 requires non-empty existing train/validation demonstrations")
    train_ids = {(row.get("scenario_hash"), row.get("episode_index")) for row in train}
    validation_ids = {(row.get("scenario_hash"), row.get("episode_index")) for row in validation}
    if train_ids & validation_ids:
        raise ValueError("R4 source demonstrations have train/validation trajectory leakage")
    _external_study_report(logs_root / "external_bc_preservation_study.md")

    role_paths = {
        "baseline_actor": BASELINE_MIGRATED,
        "round0_bc": ROUND0_CHECKPOINT,
        "final_bc": CURRENT_BC_CHECKPOINT,
    }
    audit_trainers, audit_configs, audit_payloads, audit_identities = _load_roles(role_paths, data_root / "audit_sources")
    samples = train + validation
    action_audit = _action_audit(samples, audit_trainers, logs_root)
    distribution = _dataset_distribution({"train": train, "validation": validation}, logs_root)
    heatmap = _save_distribution_heatmap(distribution, plots_root)
    distribution = {**distribution, "heatmap": heatmap}
    _distribution_report(distribution, logs_root / "dataset_distribution_audit.md")
    normal_bank = [row for row in samples if str(row.get("kind", "")) == "normal_far_distance"]
    forgetting = _normal_forgetting_audit(normal_bank, audit_trainers, logs_root)
    conflict = _teacher_label_conflict_audit(samples, logs_root)

    # E0 is a true control: same baseline-compatible init, source JSONL, loss,
    # optimizer, epoch budget and seed as the historical full-Actor BC.
    e0 = _train_candidate(
        label="e0_full_bc",
        scope_names=None,
        initial_checkpoint=BASELINE_MIGRATED,
        demonstrations_dir=SOURCE_DATA_ROOT,
        data_root=data_root,
        logs_root=logs_root,
        plots_root=plots_root,
    )
    e0_checkpoint = Path(str(e0["best_validation_checkpoint"]))
    current_trainer, current_config, _, current_identity = _load_checkpoint(CURRENT_BC_CHECKPOINT, data_root / "e0_control_sources/current_bc_actor")
    e0_trainer, e0_config, _, e0_identity = _load_checkpoint(e0_checkpoint, data_root / "e0_control_sources/e0_full_bc")
    eval_pools = _make_eval_pool(
        e0_config,
        normal_count=20,
        static_count=20,
        normal_reference_root=BASELINE_ORIGINAL.parents[2] / "scale_1" / "test",
    )
    e0_eval = _evaluate_roles(
        roles={"current_bc_actor": current_trainer, "e0_full_bc": e0_trainer},
        configs={"current_bc_actor": current_config, "e0_full_bc": e0_config},
        oracle_trainer=audit_trainers["baseline_actor"],
        oracle_config=audit_configs["baseline_actor"],
        pools=eval_pools,
        data_root=data_root / "evaluation_e0_control",
        logs_root=logs_root / "evaluation_e0_control",
        plots_root=plots_root / "evaluation_e0_control",
        progress=progress,
    )
    current_normal = e0_eval["aggregates"].get("normal_far_distance", {}).get("current_bc_actor", {})
    e0_normal = e0_eval["aggregates"].get("normal_far_distance", {}).get("e0_full_bc", {})
    current_static = e0_eval["aggregates"].get("static_holdout", {}).get("current_bc_actor", {})
    e0_static = e0_eval["aggregates"].get("static_holdout", {}).get("e0_full_bc", {})
    e0_passed = (
        int(current_normal.get("entered_8m", -1)) == int(e0_normal.get("entered_8m", -2))
        and int(current_static.get("entered_8m", -1)) == int(e0_static.get("entered_8m", -2))
    )
    e0_status = {
        "status": "passed" if e0_passed else "stopped",
        "reproduction_passed": e0_passed,
        "current_normal": current_normal,
        "e0_normal": e0_normal,
        "current_static": current_static,
        "e0_static": e0_static,
        "checkpoint": str(e0_checkpoint.resolve()),
        "checkpoint_sha256": _sha256(e0_checkpoint),
    }
    if not e0_passed:
        evaluation = e0_eval
        ablation_decision = {"e3_allowed": False, "normal_gate_candidates": [], "reason": "E0 control could not reproduce current full-Actor BC range; stop"}
        _write_json(logs_root / "summary.json", _jsonable({"stage": STAGE_NAME, "status": "stopped_at_e0", "e0": e0_status, "audit": action_audit, "distribution": distribution, "forgetting": forgetting, "conflict": conflict}))
        _write_json(data_root / "summary.json", _jsonable({"stage": STAGE_NAME, "status": "stopped_at_e0", "e0": e0_status}))
        _final_report(logs_root / "final_stage2br4_report.md", audit=action_audit, distribution=distribution, forgetting=forgetting, conflict=conflict, evaluation=evaluation, ablation_decision=ablation_decision, e0_status=e0_status)
        return {"stage": STAGE_NAME, "status": "stopped_at_e0", "e0": e0_status, "files": {"report": str((logs_root / "final_stage2br4_report.md").resolve())}}

    if progress:
        print(f"[{STAGE_NAME}] E0 control reproduced current BC range; starting E1/E2", flush=True)

    e1_train, sampling_meta = _stratified_resample(train, len(train), seed=20262711)
    e1_root = data_root / "e1_sampling_dataset"
    _write_demo_root(e1_root, e1_train, validation, sampling_meta)
    e1 = _train_candidate(
        label="e1_sampling",
        scope_names=None,
        initial_checkpoint=BASELINE_MIGRATED,
        demonstrations_dir=e1_root,
        data_root=data_root,
        logs_root=logs_root,
        plots_root=plots_root,
    )

    scope_source_trainer = audit_trainers["baseline_actor"]
    e2_results: dict[str, Any] = {}
    for label, scope in E2_SCOPES.items():
        if label == "e2d_full_actor_mean":
            e2_results[label] = {"alias_of": "e0_full_bc", "best_validation_checkpoint": str(e0_checkpoint.resolve()), "reused_control": True}
            continue
        names = _scope_names(scope_source_trainer.policy, scope)
        e2_results[label] = _train_candidate(
            label=label,
            scope_names=names,
            initial_checkpoint=BASELINE_MIGRATED,
            demonstrations_dir=SOURCE_DATA_ROOT,
            data_root=data_root,
            logs_root=logs_root,
            plots_root=plots_root,
        )

    checkpoint_paths: dict[str, Path] = {
        "baseline_rl": BASELINE_MIGRATED,
        "round0_bc": ROUND0_CHECKPOINT,
        "current_bc_actor": CURRENT_BC_CHECKPOINT,
        "e0_full_bc": e0_checkpoint,
        "e1_sampling": Path(str(e1["best_validation_checkpoint"])),
    }
    for label, result in e2_results.items():
        checkpoint_paths[label] = Path(str(result["best_validation_checkpoint"]))
    eval_trainers, eval_configs, eval_payloads, eval_identities = _load_roles(checkpoint_paths, data_root / "evaluation_sources")
    eval_trainers["oracle_teacher"] = audit_trainers["baseline_actor"]
    eval_configs["oracle_teacher"] = audit_configs["baseline_actor"]
    evaluation = _evaluate_roles(
        roles=eval_trainers,
        configs=eval_configs,
        oracle_trainer=audit_trainers["baseline_actor"],
        oracle_config=audit_configs["baseline_actor"],
        pools=eval_pools,
        data_root=data_root / "evaluation",
        logs_root=logs_root / "evaluation",
        plots_root=plots_root / "evaluation",
        progress=progress,
    )
    fidelity_samples = samples
    fidelity_trainers = {role: eval_trainers[role] for role in checkpoint_paths if role in eval_trainers}
    fidelity_rows = _action_fidelity_summary(fidelity_samples, fidelity_trainers)
    _write_csv(logs_root / "action_fidelity_candidates.csv", fidelity_rows)
    ablation_results, ablation_decision = _write_ablation_summary(evaluation, fidelity_rows, logs_root / "ablation_summary.md")

    # E3 is intentionally conditional.  The specification requires stopping
    # when the core gate fails; no anchor is silently added to a failing E2.
    e3_results: dict[str, Any] = {}
    if ablation_decision.get("e3_allowed"):
        best_label = str(ablation_decision["normal_gate_candidates"][0])
        baseline_for_anchor = audit_trainers["baseline_actor"]
        anchor_train, anchor_validation = _build_anchor_dataset(train, validation, baseline_for_anchor)
        anchor_root = data_root / "e3_anchor_dataset"
        _write_demo_root(anchor_root, anchor_train, anchor_validation, {"anchor_source": "baseline_actor_normal_observations"})
        selected_scope = None
        if best_label.startswith("e2a"):
            selected_scope = _scope_names(scope_source_trainer.policy, E2_SCOPES["e2a_delta_mean_head_only"])
        elif best_label.startswith("e2b"):
            selected_scope = _scope_names(scope_source_trainer.policy, E2_SCOPES["e2b_last_actor_body_block_plus_head"])
        elif best_label.startswith("e2c"):
            selected_scope = _scope_names(scope_source_trainer.policy, E2_SCOPES["e2c_actor_body_plus_head_gat_frozen"])
        else:
            selected_scope = None
        for lambda_value in E3_LAMBDAS:
            label = f"e3_anchor_lambda_{lambda_value:g}"
            e3_results[label] = _train_candidate(
                label=label,
                scope_names=selected_scope,
                initial_checkpoint=BASELINE_MIGRATED,
                demonstrations_dir=anchor_root,
                data_root=data_root,
                logs_root=logs_root,
                plots_root=plots_root,
                anchor_lambda=lambda_value,
            )
    else:
        ablation_decision["e3_skipped"] = True

    summary = {
        "stage": STAGE_NAME,
        "status": "completed" if e0_passed else "stopped_at_e0",
        "training": {"e0": e0, "e1": e1, "e2": e2_results, "e3": e3_results},
        "source_checkpoints": {role: {"path": str(path.resolve()), "sha256": _sha256(path)} for role, path in role_paths.items()},
        "audit": action_audit,
        "distribution": distribution,
        "forgetting": forgetting,
        "teacher_label_conflict": conflict,
        "e0": e0_status,
        "evaluation": evaluation,
        "ablation_decision": ablation_decision,
        "contracts": {
            "ppo_updates": 0,
            "reward_or_environment_modified": False,
            "dynamics_modified": False,
            "action_space_modified": False,
            "safety_semantics_modified": False,
            "train_validation_trajectory_leakage": False,
            "normalizer_frozen": True,
            "critic_frozen": True,
            "logstd_frozen": True,
            "step_as_replicate": False,
        },
        "files": {
            "external_study": str((logs_root / "external_bc_preservation_study.md").resolve()),
            "action_audit": str((logs_root / "action_shrinkage_audit.md").resolve()),
            "dataset_audit": str((logs_root / "dataset_distribution_audit.md").resolve()),
            "dataset_csv": str((logs_root / "dataset_distribution_audit.csv").resolve()),
            "dataset_heatmap": heatmap,
            "normal_forgetting": str((logs_root / "normal_forgetting_audit.md").resolve()),
            "teacher_conflict": str((logs_root / "teacher_label_conflict_report.md").resolve()),
            "ablation_summary": str((logs_root / "ablation_summary.md").resolve()),
            "ablation_results": str((logs_root / "ablation_results.csv").resolve()),
            "final_report": str((logs_root / "final_stage2br4_report.md").resolve()),
        },
    }
    _write_json(logs_root / "summary.json", _jsonable(summary))
    _write_json(data_root / "summary.json", _jsonable(summary))
    _final_report(logs_root / "final_stage2br4_report.md", audit=action_audit, distribution=distribution, forgetting=forgetting, conflict=conflict, evaluation=evaluation, ablation_decision=ablation_decision, e0_status=e0_status)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage2br4(data_root=args.data_root, logs_root=args.logs_root, plots_root=args.plots_root, progress=not args.quiet)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULT_DATA_ROOT", "DEFAULT_LOG_ROOT", "DEFAULT_PLOT_ROOT", "STAGE_NAME", "run_stage2br4"]
