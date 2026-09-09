"""Tests for the learning-side GAT, normalization, and MAPPO interfaces."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from src.learning.normalization import FeatureNormalizer, MetricNormalizer, ObservationNormalizer


class NormalizationTests(unittest.TestCase):
    def test_metric_normalizer_uses_training_p95_and_serializes(self) -> None:
        normalizer = MetricNormalizer().fit(
            [
                {"uncertainty": 1.0, "packet_loss": 0.1, "aoi": 2.0, "delay": 0.2},
                {"uncertainty": 3.0, "packet_loss": 0.3, "aoi": 4.0, "delay": 0.4},
            ]
        )
        transformed = normalizer.transform(
            {"uncertainty": 3.0, "packet_loss": 0.3, "aoi": 4.0, "delay": 0.4}
        )
        self.assertTrue(all(0.0 <= value <= 1.0 for value in transformed.values()))
        restored = MetricNormalizer.from_state_dict(normalizer.state_dict())
        self.assertEqual(restored.state_dict(), normalizer.state_dict())

    def test_feature_normalizer_round_trip_and_missing_values(self) -> None:
        normalizer = FeatureNormalizer(clip=None).fit(
            np.asarray([[0.0, 10.0], [2.0, 14.0]], dtype=float)
        )
        transformed = normalizer.transform([[0.0, 10.0], [2.0, 14.0]])
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-6)

        missing = normalizer.transform(
            [[-1.0, -1.0]],
            valid_mask=[[False, False]],
            preserve_invalid=True,
            missing_value=-1.0,
        )
        np.testing.assert_allclose(missing, [[-1.0, -1.0]])

        restored = FeatureNormalizer.from_state_dict(normalizer.state_dict())
        np.testing.assert_allclose(restored.transform([[1.0, 12.0]]), normalizer.transform([[1.0, 12.0]]))

    def test_observation_normalizer_preserves_non_feature_fields(self) -> None:
        observation = {
            "uav_00": {
                "node_features": np.asarray([[1.0, 2.0]], dtype=float),
                "candidate_edge_features": {
                    "uav_01": np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -80.0, 10.0, 0.2, 0.1, 0.3, 2.0])
                },
                "action_mask": np.asarray([True, False]),
                "metadata": "kept",
            }
        }
        normalizer = ObservationNormalizer().fit([observation])
        transformed = normalizer.transform(observation)
        self.assertEqual(transformed["uav_00"]["metadata"], "kept")
        np.testing.assert_array_equal(
            transformed["uav_00"]["action_mask"], observation["uav_00"]["action_mask"]
        )
        # Missing candidate geometry/LOS remains an explicit sentinel instead
        # of being treated as a physical distance or LOS value.
        np.testing.assert_array_equal(
            transformed["uav_00"]["candidate_edge_features"]["uav_01"][:5],
            np.full(5, -1.0, dtype=np.float32),
        )


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for GAT/MAPPO tests")
class LearningNetworkTests(unittest.TestCase):
    def test_edge_gat_forward_backward_and_attention_normalization(self) -> None:
        import torch

        from src.learning.gat import EdgeGATNetwork

        torch.manual_seed(3)
        network = EdgeGATNetwork(
            node_dim=4,
            edge_dim=3,
            hidden_dim=5,
            output_dim=6,
            heads=2,
            layers=2,
        )
        node_features = torch.randn(3, 4)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_features = torch.randn(2, 3)
        output = network(node_features, edge_index, edge_features)
        self.assertEqual(tuple(output.shape), (3, 6))
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())

        for layer in network.layers:
            recorded = layer.last_attention
            self.assertIsNotNone(recorded)
            _indices, alpha, _logits = recorded
            for target in range(3):
                positions = torch.nonzero(_indices[1] == target, as_tuple=False).flatten()
                self.assertTrue(torch.allclose(alpha[positions].sum(dim=0), torch.ones(2), atol=1e-5))

    def test_mappo_collects_and_updates_environment_rollout(self) -> None:
        from src.learning.mappo import MAPPOTrainer
        from src.simulation.environment import MultiUAVJointEnv, SimulationConfig

        environment = MultiUAVJointEnv(SimulationConfig(max_steps=3, gnss_mode="denied"))
        trainer = MAPPOTrainer.from_environment(
            environment,
            seed=5,
            gat_hidden_dim=8,
            gat_output_dim=8,
            gat_heads=1,
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            max_obstacles=4,
            policy_epochs=1,
            minibatch_size=4,
        )
        observations, info = environment.reset(seed=12)
        decision = trainer.act(observations, info["central_state"])
        self.assertEqual(set(decision.actions), set(environment.uav_ids))
        for action in decision.actions.values():
            self.assertNotIn("relay_uav_id", action)
            self.assertLessEqual(sum(action["bandwidth"].values()), 1.0 + 1e-6)
            self.assertEqual(len(action["delta_v"]), 3)
            self.assertTrue(np.isfinite(action["delta_v"]).all())

        next_observations, rewards, terminated, truncated, next_info = environment.step(decision.actions)
        self.assertEqual(set(next_observations), set(environment.uav_ids))
        trainer.add_transition(
            decision,
            rewards,
            next_info["central_state"],
            done=bool(terminated or truncated),
        )
        stats = trainer.update()
        self.assertGreaterEqual(stats["samples"], float(len(environment.uav_ids)))
        self.assertTrue(np.isfinite(list(stats.values())).all())

    def test_seeded_paired_policy_keeps_shared_modules_identical(self) -> None:
        from src.learning.mappo import MAPPOConfig, MAPPOPolicy

        common = {
            "uav_count": 3,
            "node_feature_dim": 37,
            "edge_feature_dim": 11,
            "gat_hidden_dim": 8,
            "gat_output_dim": 8,
            "gat_heads": 1,
            "gat_layers": 1,
            "actor_hidden_dim": 16,
            "critic_hidden_dim": 16,
            "parameter_initialization_seed": 2026,
        }
        baseline = MAPPOPolicy(
            MAPPOConfig(**common, use_measurement_quality_features=False)
        )
        quality = MAPPOPolicy(
            MAPPOConfig(**common, use_measurement_quality_features=True)
        )
        baseline_hashes = baseline.initial_parameter_initialization_hashes()
        quality_hashes = quality.initial_parameter_initialization_hashes()
        for module in (
            "gat",
            "actor_body",
            "bandwidth_head",
            "delta_mean_head",
            "delta_logstd",
            "critic",
        ):
            self.assertEqual(baseline_hashes[module], quality_hashes[module], module)
        self.assertNotEqual(
            baseline_hashes["measurement_head"], quality_hashes["measurement_head"]
        )


if __name__ == "__main__":
    unittest.main()
