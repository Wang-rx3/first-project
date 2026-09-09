"""Training-only feature normalization for actor and critic inputs.

The simulator keeps physical values in their native units for logging and
evaluation.  This module provides a separate, serializable preprocessing
stage so that statistics are fitted on the training split and reused without
data leakage during validation, testing, or deployment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


def _as_feature_matrix(values: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    original_shape = tuple(array.shape)
    if array.ndim == 1:
        return array.reshape(1, -1), original_shape
    return array.reshape(-1, array.shape[-1]), original_shape


def _broadcast_valid_mask(mask: Any | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    candidate = np.asarray(mask, dtype=bool)
    try:
        return np.broadcast_to(candidate, shape).copy()
    except ValueError as exc:
        raise ValueError(f"valid_mask shape {candidate.shape} cannot broadcast to {shape}") from exc


@dataclass
class FeatureNormalizer:
    """Per-feature standardizer with explicit missing-value handling."""

    eps: float = 1e-6
    clip: float | None = 10.0
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    count: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.scale is not None and self.count is not None

    @property
    def feature_dim(self) -> int | None:
        return None if self.mean is None else int(self.mean.shape[0])

    def fit(self, values: Any, valid_mask: Any | None = None) -> "FeatureNormalizer":
        matrix, _ = _as_feature_matrix(values)
        mask = _broadcast_valid_mask(valid_mask, matrix.shape)
        mask &= np.isfinite(matrix)
        count = mask.sum(axis=0).astype(np.int64)
        safe_count = np.maximum(count, 1)
        safe_values = np.where(mask, matrix, 0.0)
        mean = safe_values.sum(axis=0) / safe_count
        centered = np.where(mask, matrix - mean, 0.0)
        variance = (centered * centered).sum(axis=0) / safe_count
        scale = np.sqrt(np.maximum(variance, self.eps * self.eps))
        mean[count == 0] = 0.0
        scale[count == 0] = 1.0
        self.mean = mean.astype(np.float64)
        self.scale = scale.astype(np.float64)
        self.count = count
        return self

    def _require_fitted(self, feature_dim: int) -> None:
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before transform")
        if self.feature_dim != feature_dim:
            raise ValueError(
                f"feature dimension mismatch: fitted={self.feature_dim}, received={feature_dim}"
            )

    def transform(
        self,
        values: Any,
        valid_mask: Any | None = None,
        *,
        preserve_invalid: bool = False,
        missing_value: float = -1.0,
    ) -> np.ndarray:
        matrix, original_shape = _as_feature_matrix(values)
        if not self.fitted:
            return np.asarray(values, dtype=np.float64).copy()
        self._require_fitted(matrix.shape[-1])
        mask = _broadcast_valid_mask(valid_mask, matrix.shape)
        mask &= np.isfinite(matrix)
        normalized = (matrix - self.mean) / self.scale
        if self.clip is not None:
            normalized = np.clip(normalized, -float(self.clip), float(self.clip))
        normalized[~mask] = float(missing_value) if preserve_invalid else 0.0
        return normalized.reshape(original_shape).astype(np.float32)

    def transform_with_mask(
        self,
        values: Any,
        valid_mask: Any | None = None,
        *,
        preserve_invalid: bool = False,
        missing_value: float = -1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix, original_shape = _as_feature_matrix(values)
        mask = _broadcast_valid_mask(valid_mask, matrix.shape)
        mask &= np.isfinite(matrix)
        return (
            self.transform(
                values,
                mask,
                preserve_invalid=preserve_invalid,
                missing_value=missing_value,
            ),
            mask.reshape(original_shape).astype(bool),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "eps": float(self.eps),
            "clip": None if self.clip is None else float(self.clip),
            "mean": None if self.mean is None else self.mean.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
            "count": None if self.count is None else self.count.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FeatureNormalizer":
        normalizer = cls(eps=float(state.get("eps", 1e-6)), clip=state.get("clip", 10.0))
        if state.get("mean") is not None:
            normalizer.mean = np.asarray(state["mean"], dtype=np.float64)
            normalizer.scale = np.asarray(state["scale"], dtype=np.float64)
            normalizer.count = np.asarray(state["count"], dtype=np.int64)
        return normalizer


def _edge_valid_mask(values: np.ndarray) -> np.ndarray:
    """Ignore missing geometry/LOS sentinels in the current 11-D edge vector."""

    matrix, _ = _as_feature_matrix(values)
    mask = np.ones_like(matrix, dtype=bool)
    if matrix.shape[-1] >= 4:
        geometry_valid = matrix[:, 3] >= 0.0
        mask[:, :4] = geometry_valid[:, None]
    if matrix.shape[-1] >= 5:
        mask[:, 4] = matrix[:, 4] >= 0.0
    return mask.reshape(np.asarray(values).shape)


class ObservationNormalizer:
    """Normalize the actor-visible observation dictionaries.

    The class intentionally handles only feature arrays.  Discrete IDs,
    action masks, link dictionaries, and physical-unit logging fields remain
    unchanged.  A fitted instance can be serialized through ``state_dict``.
    """

    FEATURE_KEYS = (
        "node_features",
        "effective_edge_features",
        "candidate_edge_features",
        "candidate_incoming_edge_features",
        "potential_edge_features",
    )

    def __init__(self, *, eps: float = 1e-6, clip: float | None = 10.0) -> None:
        self.eps = float(eps)
        self.clip = clip
        self.normalizers: dict[str, FeatureNormalizer] = {}

    @property
    def fitted(self) -> bool:
        return bool(self.normalizers) and all(item.fitted for item in self.normalizers.values())

    @staticmethod
    def _local_observations(observations: Any) -> list[Mapping[str, Any]]:
        if isinstance(observations, Mapping) and "node_features" in observations:
            return [observations]
        if isinstance(observations, Mapping):
            return [value for value in observations.values() if isinstance(value, Mapping)]
        return [value for value in observations if isinstance(value, Mapping)]

    @staticmethod
    def _values_for_key(local: Mapping[str, Any], key: str) -> list[np.ndarray]:
        value = local.get(key)
        if value is None:
            return []
        if isinstance(value, Mapping):
            return [np.asarray(item, dtype=np.float64) for item in value.values() if np.asarray(item).size]
        array = np.asarray(value, dtype=np.float64)
        return [] if array.size == 0 else [array]

    def fit(self, observations: Iterable[Any]) -> "ObservationNormalizer":
        collected: dict[str, list[np.ndarray]] = {key: [] for key in self.FEATURE_KEYS}
        # Accept both ``fit([obs_t, obs_{t+1}, ...])`` and one complete
        # multi-agent observation dictionary.  The latter is convenient when
        # fitting statistics from the first collected training episode.
        if isinstance(observations, Mapping):
            observation_iterable: Iterable[Any] = (observations,)
        else:
            observation_iterable = observations
        for observation in observation_iterable:
            for local in self._local_observations(observation):
                for key in self.FEATURE_KEYS:
                    collected[key].extend(self._values_for_key(local, key))
        self.normalizers = {}
        for key, values in collected.items():
            if not values:
                continue
            matrix = np.concatenate([_as_feature_matrix(value)[0] for value in values], axis=0)
            valid = None
            if "edge_features" in key:
                valid = _edge_valid_mask(matrix)
            self.normalizers[key] = FeatureNormalizer(eps=self.eps, clip=self.clip).fit(matrix, valid)
        return self

    def _transform_array(self, key: str, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        normalizer = self.normalizers.get(key)
        if normalizer is None:
            return array.copy().astype(np.float32)
        valid = _edge_valid_mask(array) if "edge_features" in key else None
        return normalizer.transform(array, valid, preserve_invalid=True, missing_value=-1.0)

    def transform(self, observations: Any) -> Any:
        if isinstance(observations, Mapping) and "node_features" in observations:
            return self._transform_local(observations)
        if isinstance(observations, Mapping):
            return {key: self._transform_local(value) for key, value in observations.items()}
        return [self._transform_local(value) for value in observations]

    def _transform_local(self, local: Mapping[str, Any]) -> dict[str, Any]:
        output = copy.deepcopy(dict(local))
        for key in self.FEATURE_KEYS:
            if key not in output:
                continue
            value = output[key]
            if isinstance(value, Mapping):
                output[key] = {
                    item_key: self._transform_array(key, item_value)
                    for item_key, item_value in value.items()
                }
            else:
                output[key] = self._transform_array(key, value)
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "eps": self.eps,
            "clip": self.clip,
            "normalizers": {key: value.state_dict() for key, value in self.normalizers.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ObservationNormalizer":
        normalizer = cls(eps=float(state.get("eps", 1e-6)), clip=state.get("clip", 10.0))
        normalizer.normalizers = {
            key: FeatureNormalizer.from_state_dict(value)
            for key, value in dict(state.get("normalizers", {})).items()
        }
        return normalizer


@dataclass
class MetricNormalizer:
    """Training-only p95 scales for scalar reward metrics.

    Unlike :class:`FeatureNormalizer`, this class does not center metrics. It
    stores a positive 95th-percentile scale and maps each metric to
    ``clip(value / scale, 0, 1)``. A fitted instance must be reused unchanged
    for validation and test episodes.
    """

    quantile: float = 0.95
    eps: float = 1e-9
    scales: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not 0.0 < float(self.quantile) <= 1.0:
            raise ValueError("quantile must be in (0, 1]")

    @property
    def fitted(self) -> bool:
        return bool(self.scales)

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> "MetricNormalizer":
        aliases = {
            "uncertainty": "uncertainty",
            "loss": "packet_loss",
            "packet_loss": "packet_loss",
            "aoi": "aoi",
            "delay": "delay",
            "connectivity": "connectivity",
        }
        values: dict[str, list[float]] = {key: [] for key in aliases}
        for row in rows:
            for output_key, input_key in aliases.items():
                try:
                    value = float(row[input_key])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value) and value >= 0.0:
                    values[output_key].append(value)
        self.scales = {
            key: max(float(self.eps), float(np.quantile(np.asarray(items), self.quantile)))
            if items
            else 1.0
            for key, items in values.items()
        }
        return self

    def transform(self, values: Mapping[str, Any]) -> dict[str, float]:
        if not self.fitted:
            raise RuntimeError("MetricNormalizer must be fitted before transform")
        assert self.scales is not None
        source_keys = {"loss": "packet_loss"}
        return {
            key: float(
                np.clip(
                    float(values.get(source_keys.get(key, key), 0.0)) / max(self.eps, scale),
                    0.0,
                    1.0,
                )
            )
            for key, scale in self.scales.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "quantile": float(self.quantile),
            "eps": float(self.eps),
            "scales": None if self.scales is None else {key: float(value) for key, value in self.scales.items()},
            "fit_policy": "training_only_p95",
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "MetricNormalizer":
        normalizer = cls(
            quantile=float(state.get("quantile", 0.95)),
            eps=float(state.get("eps", 1e-9)),
        )
        if state.get("scales") is not None:
            normalizer.scales = {str(key): float(value) for key, value in dict(state["scales"]).items()}
        return normalizer


__all__ = ["FeatureNormalizer", "ObservationNormalizer", "MetricNormalizer"]
