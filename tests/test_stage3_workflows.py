"""Pure workflow boundaries; never run calibration, PPO, or environments."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments import three_uav_measurement_selection as workflow
from src.learning.normalization import FeatureNormalizer, ObservationNormalizer
from src.analysis.three_uav_fixed_navigation_probes import synthetic_observations
from src.control.fixed_navigation import fixed_navigation_command


def test_actual_config_matches_frozen_navigation_sensor_and_dimensions():
    spec = workflow.read_json(workflow.CONFIG)
    sim = workflow.effective_simulation(spec)
    cfg = workflow.policy_config(spec)
    assert cfg.critic_input_dim == 439 and cfg.communication_learning_scope == 'measurement_only'
    assert spec['range_variance_m2'] == sim.sensor.uwb_los_std_m**2
    invalid = copy.deepcopy(spec)
    invalid['range_variance_m2'] = .01
    with pytest.raises(ValueError, match='variance'):
        workflow.effective_simulation(invalid)
    invalid = copy.deepcopy(spec)
    invalid['simulation']['sensor']['uwb_los_std_m'] = 0
    with pytest.raises(ValueError):
        workflow.effective_simulation(invalid)


def test_exclusive_output_refuses_any_existing_root_and_unsafe_ids(tmp_path):
    paths = workflow.RunOutput('test_run', root=tmp_path)
    with pytest.raises(FileExistsError):
        workflow.RunOutput('test_run', root=tmp_path)
    with pytest.raises(ValueError):
        workflow.RunOutput('../escape', root=tmp_path)
    (tmp_path/'plots/preliminary'/workflow.STEM/'occupied').mkdir()
    with pytest.raises(FileExistsError):
        workflow.RunOutput('occupied', root=tmp_path)
    assert not (tmp_path/'data/synthetic'/workflow.STEM/'occupied').exists()


@pytest.mark.parametrize('field', ['schema', 'communication_learning_scope', 'critic_dim', 'config_hash',
    'normalization_hash', 'calibration_sha256', 'scene_hash', 'sources', 'effective_sensor_config'])
def test_checkpoint_metadata_rejects_wrong_mode_or_provenance_before_loading(field, monkeypatch):
    expected = {'schema': workflow.SCHEMA, 'communication_learning_scope': 'measurement_only', 'critic_dim': 439,
        'config_hash': 'c', 'normalization_hash': 'n', 'calibration_sha256': 'a', 'scene_hash': 's',
        'sources': {'module': 'sha'}, 'effective_sensor_config': {'known': True}}
    workflow.validate_checkpoint_metadata({'contract': expected}, expected)
    bad = copy.deepcopy(expected)
    bad[field] = 'wrong'
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Deserialization reached'))
    with pytest.raises(ValueError, match='Checkpoint contract'):
        workflow.validate_checkpoint_metadata({'contract': bad}, expected)
    with pytest.raises(ValueError):
        workflow.validate_checkpoint_metadata({'motion_action_mode': 'joint'}, expected)


def test_load_policy_checks_sidecar_before_torch_load_or_model_creation(monkeypatch):
    monkeypatch.setattr(workflow, 'read_json', lambda *a: {'contract': {'schema': 'old_joint'}})
    monkeypatch.setattr(workflow, 'checkpoint_contract', lambda *a: {'schema': workflow.SCHEMA})
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('torch.load reached'))
    monkeypatch.setattr(workflow, 'MAPPOPolicy', lambda *a, **k: pytest.fail('model creation reached'))
    with pytest.raises(ValueError, match='Checkpoint contract'):
        workflow.load_policy('old_model', {}, None, {}, '')


def calibration_fixture(spec):
    observations = synthetic_observations()
    # Simulated numerical matrices only; no env data, no training or validation fit.
    actor = ObservationNormalizer().fit([observations])
    critic = FeatureNormalizer().fit(np.stack([np.zeros(439), np.ones(439)]))
    return {'schema': workflow.SCHEMA, 'ready': True, 'fit_split': 'train',
        'fit_scene_ids': [s['scene_id'] for s in workflow.contract().make_scene_splits()['train']],
        'config_hash': workflow.content_hash(spec), 'scene_hash': workflow.content_hash(workflow.contract().make_scene_splits()),
        'sources': workflow.source_manifest(), 'effective_sensor_config': workflow.asdict(workflow.effective_simulation(spec).sensor),
        'actor_normalizer': actor.state_dict(), 'critic_normalizer': critic.state_dict(), 'credit_scale': .1}


def test_calibration_is_frozen_and_refuses_validation_provenance(monkeypatch):
    spec = workflow.read_json(workflow.CONFIG)
    cfg = workflow.policy_config(spec)
    saved = calibration_fixture(spec)
    original_read = workflow.read_json
    monkeypatch.setattr(workflow, 'read_json', lambda path: saved if Path(path).name == 'calibration.json' else original_read(path))
    original_hash = workflow.file_hash
    monkeypatch.setattr(workflow, 'file_hash', lambda path: 'calibration_sha' if Path(path).name == 'calibration.json' else original_hash(path))
    _, actor, critic, _ = workflow.load_calibration('synthetic_bank', spec, cfg)
    with pytest.raises(RuntimeError, match='frozen'):
        actor.fit([])
    with pytest.raises(RuntimeError, match='frozen'):
        critic.fit(np.zeros((1, 439)))
    saved['fit_split'] = 'validation'
    with pytest.raises(ValueError, match='TRAIN'):
        workflow.load_calibration('synthetic_bank', spec, cfg)


def test_failed_calibration_never_instantiates_trainer_or_loads_checkpoint(monkeypatch, tmp_path):
    spec = workflow.read_json(workflow.CONFIG)
    saved = calibration_fixture(spec)
    saved['ready'] = False
    original_read = workflow.read_json
    monkeypatch.setattr(workflow, 'read_json', lambda path: saved if Path(path).name == 'calibration.json' else original_read(path))
    monkeypatch.setattr(workflow, 'MAPPOTrainer', lambda *a, **k: pytest.fail('Trainer created'))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Weights loaded'))
    with pytest.raises(ValueError, match='ready=false'):
        workflow.training_workflow(workflow.RunOutput('negative', root=tmp_path), spec,
            workflow.effective_simulation(spec), workflow.policy_config(spec), {}, 'synthetic_bank')


def test_short_precheck_hard_forbids_training_loading_and_neural_execution(monkeypatch, tmp_path):
    spec = workflow.read_json(workflow.CONFIG)
    counters = {'joint_environment_steps': 0, 'ppo_updates': 0, 'optimizer_steps': 0,
        'rollout_additions': 0, 'checkpoint_loads': 0, 'normalization_fits': 0, 'neural_actions_executed': 0}
    calls = []
    def fake_episode(*args, **kwargs):
        with pytest.raises(RuntimeError, match='Forbidden'):
            workflow.MAPPOTrainer.__init__(object())
        with pytest.raises(RuntimeError, match='Forbidden'):
            torch.load('old.pt')
        with pytest.raises(RuntimeError, match='Forbidden'):
            workflow.MAPPOPolicy.act(object(), {})
        with pytest.raises(RuntimeError, match='Forbidden'):
            torch.optim.Adam.step(object())
        counters['joint_environment_steps'] += 8
        calls.append(True)
        return {'accepted_updates': 1, 'actual_measurement_overhead_packets': 1}
    monkeypatch.setattr(workflow, 'episode', fake_episode)
    result = workflow.precheck_workflow(workflow.RunOutput('guard', root=tmp_path), spec,
        workflow.effective_simulation(spec), workflow.policy_config(spec), counters)
    assert result['passed'] and len(calls) == 3
    assert result['training_readiness'] == 'not_evaluated' and counters['optimizer_steps'] == 0


def test_native_trainer_decision_on_saved_real_graph_is_measurement_only(monkeypatch):
    """Saved Stage-2 data are only an interface fixture, never fitted or added."""
    path = workflow.ROOT/'data/synthetic/three_uav_fixed_navigation_v2_seed2026_20261011/saved_observation_probes.json'
    frame = next(f for f in workflow.read_json(path) if f['episode_id'] == 'three_nav_dev_normal' and f['step'] == 25)
    raw = copy.deepcopy(frame['observations'])
    for uid in workflow.UIDS:
        raw[uid].update(measurement_budget_remaining=30, measurement_budget_remaining_ratio=1.,
            measurement_update_eligibility=np.array([other != uid for other in workflow.UIDS]))
    raw = workflow.contract().admissible_observations(raw)
    central = workflow.central_accounting({'central_state': frame['central_state']}, raw)
    cfg = workflow.policy_config(workflow.read_json(workflow.CONFIG))
    trainer = workflow.MAPPOTrainer(cfg, uav_ids=workflow.UIDS, seed=2026116200)
    monkeypatch.setattr(trainer.optimizer, 'step', lambda *a, **k: pytest.fail('Optimizer update attempted'))
    monkeypatch.setattr(trainer, 'add_transition', lambda *a, **k: pytest.fail('Rollout attempted'))
    original_transform = trainer.normalizer.transform
    def poison_nn_input(values):
        result = original_transform(values)
        for local in result.values():
            local['estimate'] = np.ones(15)*1e6
            local['goal'] = np.ones(3)*-1e6
            local['node_features'] = np.ones_like(local['node_features'])*999
        return result
    trainer.normalizer.transform = poison_nn_input
    decision = trainer.act(raw, central, deterministic=True)
    assert decision.central_vector.shape == (439,)
    for uid in workflow.UIDS:
        expected = fixed_navigation_command(raw[uid]['estimate'], raw[uid]['goal'])['command_delta_v_mps']
        np.testing.assert_array_equal(decision.actions[uid]['delta_v'], expected)
        sample = decision.samples[uid]
        assert sample.delta_v.size == sample.bandwidth.size == sample.link_selection.size == 0
        assert not any(key in decision.observations[uid] for key in ('central_state', 'truth_error', 'true_position'))
        with torch.no_grad():
            lp, entropy = trainer.policy.evaluate_actions(decision.observations[uid], sample)
        assert float(lp) == pytest.approx(sample.old_log_prob, abs=2e-6) and torch.isfinite(entropy)
    assert not trainer.rollout and not trainer.optimizer.state
