"""Synthetic benchmark generation utilities."""

from .channel_model import ChannelConfig, LinkMetrics, compute_link_metrics
from .generator import SyntheticConfig, generate_benchmark

__all__ = [
    "ChannelConfig",
    "LinkMetrics",
    "compute_link_metrics",
    "SyntheticConfig",
    "generate_benchmark",
]
