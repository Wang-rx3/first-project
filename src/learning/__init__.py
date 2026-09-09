"""Learning components for the pure-Python multi-UAV environment.

Normalization is usable with the NumPy-only environment.  The GAT and MAPPO
exports are loaded when the PyTorch dependency is installed; keeping this
small fallback importable lets schema, environment, and normalization tests
run before the learning dependency is provisioned.
"""

from .normalization import FeatureNormalizer, MetricNormalizer, ObservationNormalizer
from .visual_odometry_quality import (
    covariance_for_quality,
    covariance_scale_for_quality,
    fit_visual_odometry_quality_model,
)

__all__ = [
    "FeatureNormalizer",
    "MetricNormalizer",
    "ObservationNormalizer",
    "covariance_for_quality",
    "covariance_scale_for_quality",
    "fit_visual_odometry_quality_model",
]

try:
    from .gat import EdgeGATLayer, EdgeGATNetwork
    from .mappo import (
        MAPPOConfig,
        MAPPOTrainer,
        MAPPOPolicy,
        PolicyDecision,
        PolicySample,
        central_state_to_vector,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "EdgeGATLayer",
            "EdgeGATNetwork",
            "MAPPOConfig",
            "MAPPOPolicy",
            "MAPPOTrainer",
            "PolicyDecision",
            "PolicySample",
            "central_state_to_vector",
        ]
    )
