"""Calibratable statistical channel model for the synthetic UAV benchmark.

The model intentionally stops at an effective link abstraction.  It does not
simulate OFDM symbols, MIMO processing, beamforming, or a concrete MAC
protocol.  All powers are represented in dBm, RSSI/SNR in dB, bandwidth in Hz,
and rates in bit/s.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelConfig:
    """Parameters for the log-distance link-budget model.

    The defaults are reproducible Python-simulation baseline values.  Formal
    studies document them and vary them through controlled sensitivity,
    repeated-seed, and ablation experiments; no external dataset is required.
    """

    carrier_frequency_hz: float = 2.4e9
    reference_distance_m: float = 1.0
    tx_power_dbm: float = 10.0
    total_bandwidth_hz: float = 20_000_000.0
    path_loss_exponent_los: float = 2.0
    path_loss_exponent_nlos: float = 3.2
    nlos_extra_loss_db: float = 16.0
    shadowing_std_los_db: float = 2.0
    shadowing_std_nlos_db: float = 4.0
    rssi_measurement_std_db: float = 1.2
    noise_density_dbm_hz: float = -174.0
    noise_figure_db: float = 7.0
    phy_efficiency: float = 0.8
    packet_size_bytes: int = 512
    propagation_speed_mps: float = 299_792_458.0
    max_packet_loss: float = 0.95
    max_delay_s: float = 1.0
    min_rssi_dbm: float = -100.0
    min_snr_db: float = 0.0
    aoi_cap_s: float = 60.0
    # Logistic packet-loss coefficients, controlled by the Python experiment
    # configuration and sensitivity protocol.
    loss_beta0: float = 4.5
    loss_beta1: float = 0.3
    loss_beta2: float = 1.0
    loss_beta3: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            ("carrier_frequency_hz", self.carrier_frequency_hz),
            ("reference_distance_m", self.reference_distance_m),
            ("total_bandwidth_hz", self.total_bandwidth_hz),
            ("path_loss_exponent_los", self.path_loss_exponent_los),
            ("path_loss_exponent_nlos", self.path_loss_exponent_nlos),
            ("propagation_speed_mps", self.propagation_speed_mps),
            ("max_delay_s", self.max_delay_s),
            ("aoi_cap_s", self.aoi_cap_s),
        )
        for name, value in positive:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.phy_efficiency <= 1.0:
            raise ValueError("phy_efficiency must be in (0, 1]")
        if self.packet_size_bytes <= 0:
            raise ValueError("packet_size_bytes must be positive")
        if not 0.0 <= self.max_packet_loss <= 1.0:
            raise ValueError("max_packet_loss must be in [0, 1]")
        if self.shadowing_std_los_db < 0 or self.shadowing_std_nlos_db < 0:
            raise ValueError("shadowing standard deviations must be non-negative")
        if self.rssi_measurement_std_db < 0:
            raise ValueError("rssi_measurement_std_db must be non-negative")


@dataclass(frozen=True)
class LinkMetrics:
    """One directed link evaluation at one simulation timestamp."""

    distance_m: float
    los: int
    tx_power_dbm: float
    rx_power_dbm: float
    noise_power_dbm: float
    rssi_dbm: float
    snr_db: float
    channel_bandwidth_hz: float
    rate_bps: float
    packet_loss: float
    raw_delay_s: float
    delay_s: float
    aoi_s: float
    queue_length_packets: float
    candidate_edge: int
    effective_edge: int
    packet_success: int
    link_state: str
    drop_reason: str

    @property
    def bandwidth_kbps(self) -> float:
        """Backward-compatible alias for the effective rate in kbit/s."""

        return self.rate_bps / 1000.0


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _free_space_path_loss_db(config: ChannelConfig) -> float:
    return 20.0 * math.log10(
        4.0
        * math.pi
        * config.reference_distance_m
        * config.carrier_frequency_hz
        / config.propagation_speed_mps
    )


def _logistic_packet_loss(
    snr_db: float,
    los: bool,
    queue_length_packets: float,
    config: ChannelConfig,
) -> float:
    exponent = (
        config.loss_beta0
        - config.loss_beta1 * snr_db
        + config.loss_beta2 * (0.0 if los else 1.0)
        + config.loss_beta3 * max(0.0, queue_length_packets)
    )
    # Numerically stable enough for the calibrated ranges used here and
    # explicitly clipped to make the CSV contract robust for extreme inputs.
    if exponent >= 0.0:
        loss = 1.0 / (1.0 + math.exp(-exponent))
    else:
        exp_value = math.exp(exponent)
        loss = exp_value / (1.0 + exp_value)
    return _clip(loss, 0.0, 1.0)


def compute_link_metrics(
    *,
    distance_m: float,
    los: bool,
    bandwidth_ratio: float,
    queue_length_packets: float,
    previous_aoi_s: float,
    dt_s: float,
    rng: random.Random,
    config: ChannelConfig | None = None,
    max_comm_range_m: float | None = None,
    tx_power_dbm: float | None = None,
    aoi_reset_on_success: bool = True,
) -> LinkMetrics:
    """Compute physical/statistical metrics and sample one packet event.

    ``bandwidth_ratio`` is a per-candidate-link share of the sender's total
    bandwidth.  A policy that allocates a vector of shares can call this
    function once per directed edge.  The function keeps the packet event
    explicit: a failed packet does not silently trigger retransmission.
    ``aoi_reset_on_success`` is kept configurable because the standalone
    channel fixture reports a successful packet immediately, whereas the
    action-driven environment resets AoI only when the delayed packet reaches
    the receiver buffer.
    """

    model = config or ChannelConfig()
    if distance_m < 0:
        raise ValueError("distance_m must be non-negative")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    ratio = _clip(float(bandwidth_ratio), 0.0, 1.0)
    queue = max(0.0, float(queue_length_packets))
    distance = max(float(distance_m), model.reference_distance_m)
    bandwidth_hz = model.total_bandwidth_hz * ratio
    tx_power = model.tx_power_dbm if tx_power_dbm is None else float(tx_power_dbm)
    los_flag = bool(los)
    max_range = max_comm_range_m
    if max_range is not None and max_range <= 0:
        raise ValueError("max_comm_range_m must be positive when provided")

    path_loss_exponent = (
        model.path_loss_exponent_los
        if los_flag
        else model.path_loss_exponent_nlos
    )
    path_loss_db = _free_space_path_loss_db(model) + 10.0 * path_loss_exponent * math.log10(
        distance / model.reference_distance_m
    )
    shadowing_std = (
        model.shadowing_std_los_db
        if los_flag
        else model.shadowing_std_nlos_db
    )
    shadowing_db = rng.gauss(0.0, shadowing_std)
    obstacle_loss_db = 0.0 if los_flag else model.nlos_extra_loss_db
    rx_power_dbm = tx_power - path_loss_db - shadowing_db - obstacle_loss_db

    noise_bandwidth_hz = max(bandwidth_hz, 1.0)
    noise_power_dbm = (
        model.noise_density_dbm_hz
        + 10.0 * math.log10(noise_bandwidth_hz)
        + model.noise_figure_db
    )
    rssi_dbm = rx_power_dbm + rng.gauss(0.0, model.rssi_measurement_std_db)
    snr_db = rx_power_dbm - noise_power_dbm

    if bandwidth_hz > 0.0:
        linear_snr = 10.0 ** (snr_db / 10.0)
        rate_bps = model.phy_efficiency * bandwidth_hz * math.log2(1.0 + linear_snr)
    else:
        rate_bps = 0.0
    rate_bps = max(0.0, rate_bps)
    packet_loss = (
        1.0
        if bandwidth_hz <= 0.0
        else _logistic_packet_loss(snr_db, los_flag, queue, model)
    )

    out_of_range = max_range is not None and distance_m > max_range
    if out_of_range:
        packet_loss = 1.0
    if rate_bps > 0.0 and not out_of_range:
        serialization_s = model.packet_size_bytes * 8.0 / rate_bps
        queue_s = queue * serialization_s
        raw_delay_s = distance_m / model.propagation_speed_mps + serialization_s + queue_s
    else:
        raw_delay_s = model.max_delay_s
    raw_delay_s = max(0.0, raw_delay_s)
    delay_s = min(model.max_delay_s, raw_delay_s)

    candidate = int(
        not out_of_range
        and bandwidth_hz > 0.0
        and rssi_dbm >= model.min_rssi_dbm
        and snr_db >= model.min_snr_db
        and packet_loss <= model.max_packet_loss
        and raw_delay_s <= model.max_delay_s
    )
    packet_success = int(candidate and rng.random() >= packet_loss)
    effective = packet_success
    previous_aoi = max(0.0, float(previous_aoi_s))
    if effective and aoi_reset_on_success:
        aoi_s = delay_s
        drop_reason = ""
    elif effective:
        aoi_s = min(model.aoi_cap_s, previous_aoi + dt_s)
        drop_reason = ""
    else:
        aoi_s = min(model.aoi_cap_s, previous_aoi + dt_s)
        if out_of_range:
            drop_reason = "out_of_range"
        elif bandwidth_hz <= 0.0:
            drop_reason = "no_bandwidth"
        elif not candidate:
            drop_reason = "link_threshold"
        else:
            drop_reason = "random_loss" if los_flag else "nlos"

    if out_of_range:
        link_state = "out_of_range"
    elif bandwidth_hz <= 0.0:
        link_state = "no_bandwidth"
    elif candidate and effective:
        link_state = "connected"
    elif candidate:
        link_state = "unstable"
    else:
        link_state = "unavailable"

    return LinkMetrics(
        distance_m=float(distance_m),
        los=int(los_flag),
        tx_power_dbm=tx_power,
        rx_power_dbm=rx_power_dbm,
        noise_power_dbm=noise_power_dbm,
        rssi_dbm=rssi_dbm,
        snr_db=snr_db,
        channel_bandwidth_hz=bandwidth_hz,
        rate_bps=rate_bps,
        packet_loss=packet_loss,
        raw_delay_s=raw_delay_s,
        delay_s=delay_s,
        aoi_s=aoi_s,
        queue_length_packets=queue,
        candidate_edge=candidate,
        effective_edge=effective,
        packet_success=packet_success,
        link_state=link_state,
        drop_reason=drop_reason,
    )


__all__ = ["ChannelConfig", "LinkMetrics", "compute_link_metrics"]
