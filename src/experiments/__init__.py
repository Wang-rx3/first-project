"""Reproducible experiment drivers for the research environment."""

from .formal import (
    FormalExperimentConfig,
    default_ekf_parameters,
    load_selected_ekf_config,
    run_preliminary,
    summarize_preliminary,
    validate_preliminary,
)
from .pilot import PilotConfig, run_pilot
from .plotting import plot_main_figures
from .scenario_screening import ScenarioScreeningConfig, run_scenario_screening
from .quality_feature_sensitivity_audit import (
    QualityFeatureSensitivityAuditConfig,
    run_quality_feature_sensitivity_audit,
)
from .safety_mechanism_audit import (
    SafetyMechanismAuditConfig,
    run_safety_mechanism_audit,
)
from .safety_timing_counterfactual import (
    SafetyTimingCounterfactualConfig,
    run_safety_timing_counterfactual,
)
from .measurement_policy_stability_audit import (
    MeasurementPolicyStabilityAuditConfig,
    run_measurement_policy_stability_audit,
)
from .measurement_rate_controlled_mappo import (
    MeasurementRateControlledMAPPOConfig,
    run_measurement_rate_controlled_mappo,
)
from .measurement_rate_constraint_audit import (
    MeasurementRateConstraintAuditConfig,
    run_measurement_rate_constraint_audit,
)
from .measurement_rate_sampling_audit import (
    MeasurementRateSamplingAuditConfig,
    run_measurement_rate_sampling_audit,
)
from .measurement_rate_constraint_mechanism_audit import (
    MeasurementRateConstraintMechanismAuditConfig,
    run_measurement_rate_constraint_mechanism_audit,
)
from .adaptive_budgeted_quality_aware_mappo import (
    AdaptiveMeasurementRateMAPPOConfig,
    run_adaptive_budgeted_quality_aware_mappo,
)
from .proxy_guided_adaptive_budgeted_quality_aware_mappo import (
    ProxyGuidedAdaptiveMAPPOConfig,
    ProxyTeacherAuditConfig,
    run_proxy_guided_adaptive_budgeted_quality_aware_mappo,
    run_proxy_teacher_audit,
)
from .proxy_update_quality_audit import (
    ProxyUpdateQualityAuditConfig,
    run_proxy_guided_update_quality_audit,
)
from .communication_constrained_proxy_mappo import (
    CommunicationConstrainedProxyMAPPOConfig,
    MeasurementCommunicationCostAuditConfig,
    run_communication_constrained_proxy_mappo,
    run_measurement_communication_cost_audit,
)
from .persistent_proxy_guided_quality_mappo import (
    PersistentProxyAuditConfig,
    PersistentProxyMAPPOConfig,
    run_persistent_proxy_audit,
    run_persistent_proxy_guided_quality_mappo,
)
from .persistent_proxy_mechanism_audit import (
    PersistentProxyMechanismAuditConfig,
    run_persistent_proxy_mechanism_audit,
)
from .ekf_sensor_interaction_audit import (
    EKFSensorInteractionAuditConfig,
    run_ekf_sensor_interaction_audit,
)
from .mainline_baseline_comparison import (
    MainlineBaselineComparisonConfig,
    run_mainline_baseline_comparison,
)
from .vo_interaction_audit import run as run_vo_interaction_audit

__all__ = [
    "FormalExperimentConfig",
    "default_ekf_parameters",
    "load_selected_ekf_config",
    "PilotConfig",
    "run_pilot",
    "run_preliminary",
    "summarize_preliminary",
    "validate_preliminary",
    "plot_main_figures",
    "ScenarioScreeningConfig",
    "run_scenario_screening",
    "QualityFeatureSensitivityAuditConfig",
    "run_quality_feature_sensitivity_audit",
    "SafetyMechanismAuditConfig",
    "run_safety_mechanism_audit",
    "SafetyTimingCounterfactualConfig",
    "run_safety_timing_counterfactual",
    "MeasurementPolicyStabilityAuditConfig",
    "run_measurement_policy_stability_audit",
    "MeasurementRateControlledMAPPOConfig",
    "run_measurement_rate_controlled_mappo",
    "MeasurementRateConstraintAuditConfig",
    "run_measurement_rate_constraint_audit",
    "MeasurementRateSamplingAuditConfig",
    "run_measurement_rate_sampling_audit",
    "MeasurementRateConstraintMechanismAuditConfig",
    "run_measurement_rate_constraint_mechanism_audit",
    "AdaptiveMeasurementRateMAPPOConfig",
    "run_adaptive_budgeted_quality_aware_mappo",
    "ProxyGuidedAdaptiveMAPPOConfig",
    "ProxyTeacherAuditConfig",
    "run_proxy_guided_adaptive_budgeted_quality_aware_mappo",
    "run_proxy_teacher_audit",
    "ProxyUpdateQualityAuditConfig",
    "run_proxy_guided_update_quality_audit",
    "CommunicationConstrainedProxyMAPPOConfig",
    "MeasurementCommunicationCostAuditConfig",
    "run_communication_constrained_proxy_mappo",
    "run_measurement_communication_cost_audit",
    "PersistentProxyAuditConfig",
    "PersistentProxyMAPPOConfig",
    "run_persistent_proxy_audit",
    "run_persistent_proxy_guided_quality_mappo",
    "PersistentProxyMechanismAuditConfig",
    "run_persistent_proxy_mechanism_audit",
    "EKFSensorInteractionAuditConfig",
    "run_ekf_sensor_interaction_audit",
    "MainlineBaselineComparisonConfig",
    "run_mainline_baseline_comparison",
    "run_vo_interaction_audit",
]
