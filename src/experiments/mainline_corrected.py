"""Fail-closed, versioned seed42 correction protocol. Historical artifacts are read-only."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .cooperative_event_logging import accepted_counts, collect_window_events

HISTORICAL_ROOT = Path("data/synthetic/mainline_baseline_comparison_dev_seed42")


def compare_episode(source: Path, replay: Path, tolerance: float = 1e-8) -> list[dict[str, Any]]:
    """All saved action/trajectory/reward columns, not just a summary average."""
    from .baseline import _read_csv
    discrepancies: list[dict[str, Any]] = []

    def equal(a: Any, b: Any) -> bool:
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if a == b:
            return True
        try:
            x, y = float(a), float(b)
            return bool(np.isfinite(x) and np.isfinite(y) and abs(x - y) <= tolerance)
        except (ValueError, TypeError):
            return False

    for filename in ("actions.csv", "ground_truth.csv", "estimates.csv", "rewards.csv"):
        old, new = _read_csv(source / filename), _read_csv(replay / filename)
        if len(old) != len(new) or not old:
            discrepancies.append({"file": filename, "field": "row_count", "old": len(old), "new": len(new)})
            continue
        for index, (a, b) in enumerate(zip(old, new)):
            for column, x in a.items():
                if column == "policy_source":  # replay provenance, not an action
                    continue
                if column not in b:
                    discrepancies.append({"file": filename, "row": index, "field": column, "reason": "missing"})
                    continue
                y = b[column]
                if x.startswith(("{", "[")):
                    x, y = json.loads(x), json.loads(y)
                if not equal(x, y):
                    discrepancies.append({"file": filename, "row": index, "field": column, "old": x, "new": y})
            if len(discrepancies) >= 20:
                break
    return discrepancies


def replay_audit(config: Any) -> dict[str, Any]:
    """Regenerate decisions using frozen final actors or the historical rules.

    No trainer.update/optimizer.step calls; no test directory is enumerated.
    A mismatch stops restoration and prevents launching training experiments.
    """
    from .mainline_baseline_comparison import (
        METHODS, FIXED_CONTROL_METHODS, COMMUNICATION_BRANCH, _baseline_scale,
        _load_scenario_episodes, _formal_from_scale, _checkpoint_trainer, _rule_action,
        _append_event_rows, _write_table,
    )
    from .baseline import _read_json, _sha256, _reference_formal_config
    from .formal import _make_environment, _episode_metrics
    from .measurement_rate_controlled_mappo import _fit_measurement_budgets
    from .pilot import _seed_everything, _EpisodeTraceWriter, _assert_observation_finite
    from ..synthetic.episode_generator import _write_json

    root, logs = config.resolved_output_dir, config.resolved_logs_dir
    for output in (root, logs):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"corrected-v2 never overwrites existing output: {output}")
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    baseline = _baseline_scale(HISTORICAL_ROOT)
    budgets = _fit_measurement_budgets(baseline, uav_count=3)
    proxy_path = HISTORICAL_ROOT / "seed_42" / COMMUNICATION_BRANCH / "scale_3/calibration/measurement_proxy_model.json"
    proxy = _read_json(proxy_path)
    paths: set[Path] = {proxy_path}
    for method in METHODS:
        scale = (HISTORICAL_ROOT / "seed_42/rules" / method if method in FIXED_CONTROL_METHODS
                 else HISTORICAL_ROOT / "seed_42" / method / "scale_3")
        paths.add(scale / "manifest.json")
        # Explicit split allow-list: never recurse into the experiment root/test.
        for episode in _load_scenario_episodes(scale):
            paths.update(p for p in (scale / "validation" / episode["episode_id"]).iterdir() if p.is_file())
        if method not in FIXED_CONTROL_METHODS:
            paths.add(scale / "checkpoints/final.pt")
            paths.add(scale / "normalization_stats.json")
            paths.update(p for p in (scale / "calibration").iterdir() if p.is_file())
    hashes = {str(path): _sha256(path) for path in sorted(paths)}
    records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    failure: Any = None
    try:
        for method in METHODS:
            rule = method in FIXED_CONTROL_METHODS
            source = (HISTORICAL_ROOT / "seed_42/rules" / method if rule else
                      HISTORICAL_ROOT / "seed_42" / method / "scale_3")
            output = root / "seed_42" / method / "scale_3"
            formal = (_formal_from_scale(baseline, output, progress=False, budget_enabled=True, budgets=budgets)
                      if rule else _reference_formal_config(_read_json(source / "manifest.json"), output, force=False, progress=False))
            if rule:
                formal = replace(formal, mappo_variant="baseline_mappo", encoder_type="gat",
                                 measurement_quality_logit_penalty=0.0)
            episodes = _load_scenario_episodes(source)
            if len(episodes) != 20:
                raise ValueError(f"expected 20 historical validation episodes for {method}")
            trainer = None if rule else _checkpoint_trainer(source, formal, _make_environment(formal, 42, metric_scales=formal.metric_scales))
            for index, episode in enumerate(episodes):
                episode_id, seed = str(episode["episode_id"]), int(episode["seed"])
                if config.progress:
                    print(
                        f"[{config.protocol} replay] {method} {index+1}/20 {episode_id}",
                        flush=True,
                    )
                original = source / "validation" / episode_id
                scenario = _read_json(original / "scenario.json")
                _seed_everything(seed)
                env = _make_environment(formal, seed, metric_scales=formal.metric_scales)
                obs, info = env.reset(seed=seed, scenario_config=scenario["scenario_generation"])
                writer = _EpisodeTraceWriter(
                    episode_root=output / "validation" / episode_id, episode_id=episode_id,
                    scenario_id=scenario["scenario_id"], split="validation", seed=seed,
                    phase="validation", environment=env, normalization_fitted=True,
                    diagnostics_path=output / "diagnostics/attention_maps.jsonl", policy_source="frozen_replay")
                rng = np.random.default_rng(seed * 1009 + sum(map(ord, method)))
                cursors = dict.fromkeys(env.uav_ids, 0)
                events: list[dict[str, Any]] = []
                for step in range(formal.max_steps):
                    _assert_observation_finite(obs)
                    if rule:
                        actions, _ = _rule_action(method, obs, env, budgets=budgets, cursors=cursors, rng=rng, proxy_model=proxy)
                    else:
                        actions = trainer.act(obs, info["central_state"], deterministic=True).actions
                    before = accepted_counts(env)
                    next_obs, rewards, terminated, truncated, next_info = env.step(actions)
                    events.extend(collect_window_events(env, before, run_root=output, episode_id=episode_id, split="validation", step=step))
                    writer.record_step(step=step, observations_before=obs, observations_after=next_obs,
                                       actions=actions, rewards=rewards, info=next_info, graph_traces=None,
                                       policy_source="frozen_replay", action_start=step*env.config.motion.dt_s,
                                       timestamp=(step+1)*env.config.motion.dt_s, terminated=bool(terminated), truncated=bool(truncated))
                    obs, info = next_obs, next_info
                    if terminated or truncated:
                        break
                summary = writer.finalize()
                _append_event_rows(output, "validation", episode_id, events)
                discrepancies = compare_episode(original, output / "validation" / episode_id)
                records.append({"method": method, "episode_id": episode_id, "equivalent": not discrepancies,
                                "accepted_events": sum(bool(e["accepted"]) for e in events),
                                "counter_total": sum(accepted_counts(env).values()), "differences": discrepancies})
                if discrepancies:
                    failure = {"cause": "historical_replay_mismatch", "method": method, "episode_id": episode_id, "differences": discrepancies}
                    break
                row = _episode_metrics(output / "validation" / episode_id, {**summary, "seed": seed})
                if not row["accepted_update_count"]:
                    for field in ("beneficial_fraction", "harmful_fraction", "neutral_fraction"):
                        row[field] = "NA"
                metrics.append({"method": method, **row})
            if failure:
                break
    except Exception as exc:
        failure = {"cause": "replay_execution_error", "exception": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        unchanged = all(_sha256(Path(path)) == digest for path, digest in hashes.items())
        report = {"protocol": config.protocol, "stage": "replay-audit", "seed": 42,
                  "audit_ok": bool(not failure and unchanged and len(records) == 120 and all(r["equivalent"] for r in records)),
                  "historical_files_unchanged": unchanged, "completed_episodes": len(records),
                  "expected_episodes": 120, "failure": failure,
                  "test_accesses": 0, "test_access_policy": "explicit validation paths only",
                  "parameter_updates": 0, "source_hashes": hashes, "episodes": records}
        _write_json(logs / "audit_report.json", report)
        _write_table(logs / "restored_validation_metrics.csv", metrics)
    return {"audit_ok": report["audit_ok"], "audit_report": str(logs / "audit_report.json"), "failure": failure}


def run_corrected_comparison(config: Any) -> dict[str, Any]:
    if config.force:
        raise ValueError(
            "--force is prohibited for corrected-v2/stabilized-v3/stabilized-v4/stabilized-v4-corrected/core-framework-v1/core-framework-v2/core-framework-v3/core-framework-v4/core-framework-v4-delta-scale/core-framework-v5/core-framework-v6/core-framework-v7/core-framework-v8/core-framework-v9-action-selection; choose a new output directory"
        )
    if config.stage == "replay-audit":
        return replay_audit(config)
    if config.stage in {"smoke", "development"}:
        from .mainline_corrected_stages import run_stage
        return run_stage(config)
    if config.stage == "plot-only":
        from .mainline_corrected_stages import summarize
        from .mainline_compact_figures import generate_compact_figures
        from .baseline import _read_json
        manifest = _read_json(config.resolved_output_dir / "comparison_manifest.json")
        paths = manifest["method_paths"]
        active_methods = tuple(manifest.get("active_methods", paths.keys()))
        rows, isolation = summarize(
            paths,
            int(manifest["counts"][2]),
            methods=active_methods,
        )
        figure_profile = (
            "compact-proposed-v4"
            if config.protocol in {"core-framework-v4", "core-framework-v4-delta-scale", "core-framework-v5", "core-framework-v6", "core-framework-v7", "core-framework-v8", "core-framework-v9-action-selection"}
            else "compact-proposed-v3"
            if config.protocol == "core-framework-v3"
            else config.figure_profile
        )
        figures = generate_compact_figures(
            paths,
            rows,
            config.resolved_logs_dir,
            config.resolved_plots_dir,
            arm=config.experiment_arm,
            figure_profile=figure_profile,
        )
        return {"audit_ok": True, "figures": figures, "source_manifest_unmodified": True, "isolation": isolation}
    raise ValueError(config.stage)
