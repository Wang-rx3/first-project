"""Pure tests for stable frozen-runtime normalizer identity snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.experiments.single_uav_goal_recovery_correction_evaluation import (
    _canonical_hash,
    _runtime_identity_jsonable,
)
from src.experiments.single_uav_lateral_exit_braking_assistance import (
    FROZEN_OFF_BASELINE_FINGERPRINT,
    preregistration_hash,
)
from src.learning.normalization import FeatureNormalizer, ObservationNormalizer


PROJECT = Path(__file__).resolve().parents[1]


def nested_normalizer() -> ObservationNormalizer:
    normalizer = ObservationNormalizer(eps=1e-6, clip=10.0)
    normalizer.normalizers["node_features"] = FeatureNormalizer(
        eps=1e-6,
        clip=10.0,
        mean=np.asarray([1.0, 2.0], dtype=np.float64),
        scale=np.asarray([0.5, 0.25], dtype=np.float64),
        count=np.asarray([7, 9], dtype=np.int64),
    )
    return normalizer


def snapshot_hash(value: object) -> str:
    return _canonical_hash(_runtime_identity_jsonable(value))


class RuntimeIdentitySerializerTests(unittest.TestCase):
    def test_nested_feature_normalizer_is_json_safe(self) -> None:
        snapshot = _runtime_identity_jsonable(nested_normalizer())
        encoded = json.dumps(snapshot, sort_keys=True, allow_nan=False)
        self.assertIn("FeatureNormalizer", encoded)
        self.assertEqual(len(_canonical_hash(snapshot)), 64)

    def test_equal_nested_normalizer_state_has_equal_hash(self) -> None:
        self.assertEqual(
            snapshot_hash(nested_normalizer()),
            snapshot_hash(nested_normalizer()),
        )

    def test_nested_statistics_change_changes_hash(self) -> None:
        baseline = nested_normalizer()
        changed = nested_normalizer()
        changed.normalizers["node_features"].mean[0] += 0.125
        self.assertNotEqual(snapshot_hash(baseline), snapshot_hash(changed))

    def test_ndarray_snapshot_is_stable_and_explicit(self) -> None:
        value = np.asarray([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
        first = _runtime_identity_jsonable(value)
        second = _runtime_identity_jsonable(value.copy())
        self.assertEqual(first, second)
        self.assertEqual(first["__ndarray__"]["dtype"], value.dtype.str)
        self.assertEqual(first["__ndarray__"]["shape"], [4])
        json.dumps(first, sort_keys=True, allow_nan=False)

    def test_numpy_scalars_are_stable_python_values(self) -> None:
        self.assertEqual(_runtime_identity_jsonable(np.int64(7)), 7)
        self.assertEqual(_runtime_identity_jsonable(np.float32(1.5)), 1.5)
        self.assertIsInstance(_runtime_identity_jsonable(np.float64(2.5)), float)
        self.assertEqual(_runtime_identity_jsonable(np.bool_(True)), True)

    def test_nested_dict_list_tuple_is_stable(self) -> None:
        value = {"outer": [1, {"inner": (2.0, np.int32(3))}]}
        snapshot = _runtime_identity_jsonable(value)
        self.assertEqual(snapshot, _runtime_identity_jsonable(copy.deepcopy(value)))
        self.assertIn('"__sequence_type__": "tuple"', json.dumps(snapshot))

    def test_self_reference_fails_closed(self) -> None:
        class SelfReferential:
            pass

        value = SelfReferential()
        value.self = value
        with self.assertRaisesRegex(ValueError, "^RUNTIME_IDENTITY_OBJECT_CYCLE$"):
            _runtime_identity_jsonable(value)

    def test_callable_object_field_is_excluded(self) -> None:
        class StateWithCallable:
            def __init__(self) -> None:
                self.count = 3
                self.callback = lambda: None

        snapshot = _runtime_identity_jsonable(StateWithCallable())
        self.assertEqual(snapshot["state"], {"count": 3})
        self.assertNotIn("callback", snapshot["state"])
        with self.assertRaisesRegex(TypeError, "callable runtime identity state"):
            _runtime_identity_jsonable(lambda: None)

    def test_serializer_does_not_modify_original_object(self) -> None:
        value = nested_normalizer()
        before = copy.deepcopy(value.state_dict())
        _runtime_identity_jsonable(value)
        after = value.state_dict()
        self.assertEqual(before["eps"], after["eps"])
        self.assertEqual(before["clip"], after["clip"])
        self.assertEqual(before["normalizers"], after["normalizers"])

    def test_assistance_helper_source_hash_is_unchanged(self) -> None:
        source = (PROJECT / "src/control/lateral_exit_braking_assistance.py").read_bytes()
        self.assertEqual(
            hashlib.sha256(source).hexdigest(),
            "19a729bed2cd301f216f94f770c19e9802a8c38dcba2b6ca5e5bac914fc3c383",
        )

    def test_preregistration_hash_is_unchanged(self) -> None:
        self.assertEqual(
            preregistration_hash(),
            "a93837621a04c9fa7895881155a90878b664eeac318fbcb0cf7dded5afedd2c2",
        )

    def test_off_baseline_fingerprint_constant_is_unchanged(self) -> None:
        self.assertEqual(
            FROZEN_OFF_BASELINE_FINGERPRINT,
            "d0d6ec4891d314923c01be839cc3def1d071462cbccff1971a0f0e22ebcfd07b",
        )


if __name__ == "__main__":
    unittest.main()
