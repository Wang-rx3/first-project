"""Fixed development scenes; preserve all training RNG and frozen actor parameters."""
from __future__ import annotations
import copy
import random
import numpy as np


def run_training_monitor(config, trainer, scale_root, metric_scales, *, completed_episodes):
    import torch
    from .formal import _make_environment, _scenario_for
    from .pilot import _append_jsonl, _parameter_vector, _seed_everything
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.random.get_rng_state()
    trainer_rng = copy.deepcopy(trainer.rng.bit_generator.state)
    training = trainer.policy.training
    before = _parameter_vector(trainer).clone()
    try:
        trainer.policy.eval()
        for index in range(5):
            seed = int(config.seed) + 900_000 + index * 1009
            _seed_everything(seed)
            env = _make_environment(config, seed, metric_scales=metric_scales)
            obs, info = env.reset(seed=seed, scenario_config=_scenario_for(config, seed, split="train", episode_index=index))
            rewards_history, squared_errors = [], []
            for step in range(config.max_steps):
                with torch.no_grad():
                    decision = trainer.act(obs, info["central_state"], deterministic=True)
                obs, rewards, terminated, truncated, info = env.step(decision.actions)
                rewards_history.append(float(np.mean(list(rewards.values()))))
                squared_errors.extend(float(np.sum((env.ekfs[u].position - env.states[u].position_m)**2)) for u in env.uav_ids)
                if terminated or truncated:
                    break
            _append_jsonl(scale_root / "diagnostics/development_monitor.jsonl", {
                "completed_training_episodes": completed_episodes, "monitor_scene": index,
                "scene_seed": seed, "split": "development_monitor", "training_seed": config.seed,
                "mean_reward": float(np.mean(rewards_history)), "cumulative_return": float(sum(rewards_history)),
                "position_rmse_norm_m": float(np.sqrt(np.mean(squared_errors))), "windows": len(rewards_history),
                "parameter_updates": 0, "used_for_checkpoint_selection": False,
            })
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng)
        torch.random.set_rng_state(torch_rng); trainer.rng.bit_generator.state = trainer_rng
        trainer.policy.train(training)
    if not torch.equal(before, _parameter_vector(trainer)):
        raise RuntimeError("development monitor changed policy parameters")
