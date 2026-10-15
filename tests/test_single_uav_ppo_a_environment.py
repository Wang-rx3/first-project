import copy
from dataclasses import replace

import pytest

from src.experiments.single_uav_ppo_baseline import StrictSingleUAVPPOEnv, SingleUAVPPOConfig
from src.experiments.single_uav_ppo_a_environment import create_foundation_environment
from src.experiments.single_uav_ppo_a_scenarios import build_a_foundation_scenarios


def test_normal_marker_requires_zero_physical_obstacles():
    scenarios = build_a_foundation_scenarios()
    env = create_foundation_environment(SingleUAVPPOConfig())
    scene = copy.deepcopy(scenarios['validation_single'][0]['scenario'])
    scene['foundation_task_kind'] = 'normal'
    with pytest.raises(ValueError, match='physical obstacle'):
        env.reset(seed=2026, scenario_config=scene)


def test_empty_unmarked_and_physical_tasks_delegate(monkeypatch):
    env = create_foundation_environment(SingleUAVPPOConfig())
    scenarios = build_a_foundation_scenarios()
    for scene in (scenarios['train_pool'][0], scenarios['validation_single'][0]):
        config = copy.deepcopy(scene['scenario'])
        config.pop('foundation_task_kind', None)
        env.reset(seed=2026, scenario_config=config)
        assert not env._normal_enabled
        result = ({}, {}, False, False, {'marker': object()})
        monkeypatch.setattr(StrictSingleUAVPPOEnv, 'step', lambda self, action: result)
        actual = env.step({})
        assert actual[4] is result[4]
        assert actual == result


def test_truth_hold_reset_bonus_clip_and_terminal_guard(monkeypatch):
    env = create_foundation_environment(SingleUAVPPOConfig())
    scene = build_a_foundation_scenarios()['normal_train'][0]['scenario']
    env.reset(seed=2026, scenario_config=scene)
    values = iter([(0.5, 0.4), (0.5, 0.4), (1.1, 0.4)] + [(0.5, 0.4)] * 5)
    def fake_parent(self, action):
        distance, speed = next(values)
        return {}, {}, False, False, {
            'strict_goal_distance_m_actual': distance, 'strict_speed_mps_actual': speed,
            'strict_physical_pass_complete': False, 'learning_rewards': {'uav_00': -0.2},
            'learning_reward_components': {'learning_reward_preclip_by_uav': {'uav_00': -0.2}},
        }
    monkeypatch.setattr(StrictSingleUAVPPOEnv, 'step', fake_parent)
    for expected in (1, 2, 0, 1, 2, 3, 4):
        _, _, terminated, _, info = env.step({})
        assert info['strict_hold_steps'] == expected
        assert not terminated and info['strict_success_bonus'] == 0
    _, _, terminated, truncated, info = env.step({})
    assert terminated and not truncated and info['strict_success']
    assert info['strict_hold_steps'] == 5
    assert info['strict_success_bonus'] == 5.0
    assert info['learning_rewards']['uav_00'] == 1.0
    assert info['learning_reward_components']['learning_reward_preclip_by_uav']['uav_00'] == 4.8
    assert not info['strict_physical_pass_complete'] and not info['strict_physical_pass_applicable']
    with pytest.raises(RuntimeError, match='reset required'):
        env.step({})


def test_collision_prevents_normal_success(monkeypatch):
    env = create_foundation_environment(SingleUAVPPOConfig())
    env.reset(seed=2026, scenario_config=build_a_foundation_scenarios()['normal_train'][0]['scenario'])
    env._strict_collision_permanent_failure = True
    monkeypatch.setattr(StrictSingleUAVPPOEnv, 'step', lambda self, action: ({}, {}, True, False, {
        'strict_goal_distance_m_actual': 0., 'strict_speed_mps_actual': 0.,
        'learning_reward_components': {}, 'strict_physical_pass_complete': False,
    }))
    for _ in range(6):
        *_, info = env.step({})
        assert not info['strict_success'] and info['strict_hold_steps'] == 0
