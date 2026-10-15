"""Opt-in normal-task termination repair, with unchanged physical-task path."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..simulation.obstacle_model import is_physical_obstacle
from .single_uav_ppo_baseline import StrictSingleUAVPPOEnv, build_single_uav_simulation_config


class FoundationSingleUAVPPOEnv(StrictSingleUAVPPOEnv):
    def __init__(self, config, baseline):
        self._normal_enabled = False
        self._normal_hold = 0
        self._normal_success = False
        self._normal_bonus_awarded = False
        super().__init__(config, baseline)

    def reset(self, *, seed=None, scenario_config=None):
        self._normal_enabled = False
        self._normal_hold = 0
        self._normal_success = False
        self._normal_bonus_awarded = False
        obs, info = super().reset(seed=seed, scenario_config=scenario_config)
        declared = (scenario_config or {}).get('foundation_task_kind') == 'normal'
        has_physical = any(is_physical_obstacle(obstacle) for obstacle in self.obstacles)
        if declared and has_physical:
            raise ValueError('Declared normal task contains a physical obstacle')
        self._normal_enabled = bool(declared and not has_physical)
        info.update(self.strict_status())
        return obs, info

    def strict_status(self):
        status = super().strict_status()
        if not self._normal_enabled:
            return status
        status.update(strict_hold_steps=self._normal_hold, strict_success=self._normal_success,
                      strict_physical_pass_applicable=False, foundation_task_kind='normal')
        # Physical-pass status retains the parent's false value; N/A is never a bypass.
        return status

    def step(self, action):
        if self._normal_enabled and self._normal_success:
            raise RuntimeError('Cannot step after normal-task success; reset required')
        obs, rewards, terminated, truncated, info = super().step(action)
        if not self._normal_enabled:
            return obs, rewards, terminated, truncated, info
        qualifies = bool(not self._strict_collision_permanent_failure
                         and info['strict_goal_distance_m_actual'] <= 1.0
                         and info['strict_speed_mps_actual'] <= 0.5)
        self._normal_hold = self._normal_hold + 1 if qualifies else 0
        self._normal_success = bool(self._normal_hold >= 5 and qualifies)
        triggered = self._normal_success and not self._normal_bonus_awarded
        bonus = self.baseline.strict_success_bonus if triggered else 0.0
        if triggered:
            components = dict(info['learning_reward_components'])
            preclip = dict(components['learning_reward_preclip_by_uav'])
            updated = {uid: float(value) + bonus for uid, value in preclip.items()}
            clip = self.config.reward.learning_reward_clip
            info['learning_rewards'] = {uid: float(np.clip(value, -clip, clip)) for uid, value in updated.items()}
            components.update(learning_reward_preclip_by_uav=updated,
                              learning_reward_total_by_uav=dict(info['learning_rewards']),
                              learning_reward_total=float(np.mean(list(info['learning_rewards'].values()))),
                              single_uav_learning_reward_preclip=float(np.mean(list(updated.values()))),
                              single_uav_learning_reward=float(np.mean(list(info['learning_rewards'].values()))),
                              single_uav_goal_bonus=bonus, strict_success_bonus=bonus)
            info['learning_reward_components'] = components
            self._normal_bonus_awarded = True
        if self._normal_success:
            terminated, truncated = True, False
            info.update(termination_reason='strict_normal_goal_reached', task_complete=True)
        info.update(single_uav_goal_condition=qualifies, single_uav_goal_reached=self._normal_success,
                    single_uav_goal_stable=self._normal_success, single_uav_stable_steps=self._normal_hold,
                    single_uav_goal_bonus_triggered=triggered, single_uav_goal_bonus=bonus,
                    strict_success_bonus=bonus, terminated=bool(terminated), truncated=bool(truncated))
        info.update(self.strict_status())
        components = dict(info['learning_reward_components'])
        components.update({key: value for key, value in info.items() if key.startswith(('strict_', 'single_uav_goal_'))
                           or key in {'single_uav_stable_steps', 'single_uav_stable_hold_steps'}})
        info['learning_reward_components'] = components
        return obs, rewards, bool(terminated), bool(truncated), info


def create_foundation_environment(base):
    return FoundationSingleUAVPPOEnv(build_single_uav_simulation_config(base), base)
