from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from src.experiments.cooperative_event_logging import accepted_counts, collect_window_events, require_new_episode
from src.experiments.mainline_baseline_comparison import _append_event_rows, MainlineBaselineComparisonConfig
from src.experiments.mainline_corrected import compare_episode


def event(source="uav_01", accepted=True):
    return {"target_uav_id": "uav_00", "source_uav_id": source, "accepted": accepted,
            "measurement_time_s": .2, "message_send_time_s": 0.0}


def test_reset_window_sequence_does_not_drop_events(tmp_path):
    env = SimpleNamespace(ekfs={"uav_00": SimpleNamespace(valid_neighbor_count=0)}, relative_range_update_events=[])
    recorded = []
    for step, events in enumerate([[], [event()], [], [event(), event("uav_02")], [event()]]):
        before = accepted_counts(env)
        env.relative_range_update_events = events
        env.ekfs["uav_00"].valid_neighbor_count += sum(e["accepted"] for e in events)
        recorded += collect_window_events(env, before, run_root=tmp_path, episode_id="e", split="validation", step=step)
    assert len(recorded) == 4
    assert len({e["event_id"] for e in recorded}) == 4
    assert all(e["run_id"] and e["episode_id"] == "e" for e in recorded)


def test_missing_duplicate_and_wrong_target_fail_closed(tmp_path):
    env = SimpleNamespace(ekfs={"uav_00": SimpleNamespace(valid_neighbor_count=1)}, relative_range_update_events=[])
    kwargs = dict(run_root=tmp_path, episode_id="e", split="validation", step=0)
    with pytest.raises(ValueError, match="counter mismatch"):
        collect_window_events(env, {"uav_00": 0}, **kwargs)
    env.relative_range_update_events = [event(), event()]
    with pytest.raises(ValueError, match="duplicate"):
        collect_window_events(env, {"uav_00": 0}, **kwargs)
    env.relative_range_update_events = [{**event(), "target_uav_id": "unknown"}]
    with pytest.raises(ValueError, match="unknown"):
        collect_window_events(env, {"uav_00": 0}, **kwargs)


def test_duplicate_episode_append_preserves_old_bytes(tmp_path):
    _append_event_rows(tmp_path, "validation", "e", [event()])
    path = tmp_path / "diagnostics/validation_cooperative_updates.jsonl"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        _append_event_rows(tmp_path, "validation", "e", [event()])
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        require_new_episode(path.parent)


def test_comparison_checks_nested_actions_and_trajectory_not_only_mean(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    for file in ("actions.csv", "ground_truth.csv", "estimates.csv", "rewards.csv"):
        for folder in (old, new):
            (folder/file).write_text('x,action_raw\n1,"{""delta_v"": [0, 1]}"\n', encoding="utf-8")
    assert compare_episode(old, new) == []
    (new/"actions.csv").write_text('x,action_raw\n1,"{""delta_v"": [0, 2]}"\n', encoding="utf-8")
    assert compare_episode(old, new)[0]["field"] == "action_raw"


def test_corrected_protocol_is_versioned_and_legacy_defaults_unchanged():
    config = MainlineBaselineComparisonConfig(protocol="corrected-v2", stage="replay-audit")
    assert "v2_seed42" in str(config.resolved_output_dir)
    assert config.resolved_logs_dir != MainlineBaselineComparisonConfig().resolved_logs_dir
    with pytest.raises(ValueError):
        MainlineBaselineComparisonConfig(stage="replay-audit")


def test_indexed_noise_does_not_depend_on_other_calls():
    import numpy as np
    from src.simulation.random_streams import indexed_rng
    expected=indexed_rng(42,9,"vo","uav_00").normal(size=10)
    for i in range(30):
        indexed_rng(42,i,"channel","uav_00","uav_01").normal(size=20)
    np.testing.assert_array_equal(expected,indexed_rng(42,9,"vo","uav_00").normal(size=10))
    assert not np.array_equal(expected,indexed_rng(42,9,"imu","uav_00").normal(size=10))


def test_new_mlp_keeps_edge_identity_and_no_attention():
    import copy
    import numpy as np
    import torch
    from src.learning.mappo import MAPPOConfig, MAPPOPolicy
    from src.simulation.environment import MultiUAVJointEnv, SimulationConfig
    from src.experiments.pilot import _assert_graph_trace
    env=MultiUAVJointEnv(SimulationConfig(max_steps=2))
    obs,_=env.reset(seed=42)
    local=obs["uav_00"]
    policy=MAPPOPolicy(MAPPOConfig(encoder_type="mlp_slots_v2",gat_hidden_dim=8,gat_output_dim=8),uav_ids=env.uav_ids)
    inputs=[]
    hook=policy.mlp_flat_encoder.register_forward_pre_hook(lambda module,args: inputs.append(args[0].detach().clone()))
    _,_,trace=policy.act_with_trace(local,deterministic=True)
    _assert_graph_trace(trace)
    changed=copy.deepcopy(local)
    changed["candidate_edge_features"]={"uav_01":np.ones(11),"uav_02":np.zeros(11)}
    policy.act(changed,deterministic=True)
    changed["candidate_edge_features"]={"uav_02":np.ones(11),"uav_01":np.zeros(11)}
    policy.act(changed,deterministic=True)
    assert not torch.equal(inputs[-1],inputs[-2])
    assert trace["attention_maps"]==[]
    hook.remove()


def test_teacher_cost_and_gradient_boundary():
    import numpy as np
    import torch
    from src.learning.mappo import MAPPOConfig, MAPPOPolicy
    from src.learning.measurement_cost import candidate_communication_cost
    from src.simulation.environment import MultiUAVJointEnv, SimulationConfig
    env=MultiUAVJointEnv(SimulationConfig(max_steps=2)); obs,_=env.reset(seed=42)
    local=obs["uav_00"]
    local.update(expected_information_gain_norm=[0,.8,.3],predicted_communication_cost_norm=[1,.2,.8],proxy_valid=[0,1,1],uav_id="uav_00")
    policy=MAPPOPolicy(MAPPOConfig(use_measurement_proxy_features=True,measurement_proxy_guidance_enabled=True,
        measurement_proxy_full_packet_cost=True,measurement_proxy_target_relative_cost=True,
        measurement_communication_constraint_enabled=True,measurement_communication_cost_target_by_uav={u:.5 for u in env.uav_ids}),uav_ids=env.uav_ids)
    risk=torch.tensor([0.,.2,1.]); assert torch.allclose(policy._teacher_candidate_cost(risk),candidate_communication_cost(risk,.5))
    for loss in (policy.measurement_proxy_teacher_loss(local,uav_id="uav_00")[0], policy.expected_measurement_communication_cost(local)):
        policy.zero_grad(); loss.backward()
        assert sum(float(p.grad.abs().sum()) for p in policy.measurement_head.parameters() if p.grad is not None)>0
        assert all(p.grad is None for name,p in policy.named_parameters() if not name.startswith("measurement_head."))


def test_multi_episode_terminal_gae_does_not_cross_boundaries():
    from src.learning.mappo import MAPPOTrainer
    from src.simulation.environment import MultiUAVJointEnv, SimulationConfig
    env=MultiUAVJointEnv(SimulationConfig(max_steps=1)); trainer=MAPPOTrainer.from_environment(env,seed=42,policy_epochs=1)
    obs,info=env.reset(seed=42)
    decision=trainer.act(obs,info["central_state"],deterministic=True)
    trainer.add_transition(decision,{u:1.0 for u in env.uav_ids},info["central_state"],done=True)
    trainer.add_transition(decision,{u:100.0 for u in env.uav_ids},info["central_state"],done=True)
    records=trainer._training_records()
    assert all(abs(r["return"]-1.)<1e-6 for r in records[:3])
    assert all(abs(r["return"]-100.)<1e-6 for r in records[3:])


def test_rollout_tail_and_monitor_preserve_training(tmp_path):
    import torch
    from dataclasses import replace
    from src.experiments.formal import FormalExperimentConfig, run_preliminary
    config=FormalExperimentConfig(output_dir=tmp_path/"off",seed=42,max_steps=1,warmup_episodes=1,train_episodes=5,
        validation_episodes=1,test_episodes=1,policy_epochs=1,gat_hidden_dim=8,gat_output_dim=8,actor_hidden_dim=8,critic_hidden_dim=8,
        rollout_episodes_per_update=4,random_stream_mode="indexed-v1",branch_invariant_initialization=True,progress=False)
    run_preliminary(config)
    run_preliminary(replace(config,output_dir=tmp_path/"on",development_monitor_enabled=True))
    def payload(name):return torch.load(tmp_path/name/"scale_3/checkpoints/final.pt",weights_only=False)
    assert all(torch.equal(v,payload("on")["policy_state_dict"][k]) for k,v in payload("off")["policy_state_dict"].items())
    updates=[json.loads(s) for s in (tmp_path/"on/scale_3/diagnostics/training_metrics.jsonl").read_text().splitlines()]
    assert [u["rollout_episode_count"] for u in updates]==[4,1]
    assert [u["completed_training_episodes"] for u in updates]==[4,5]
