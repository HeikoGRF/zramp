"""Low-cost V2V MAC, airtime, and co-channel interference model.

The path powers come from the exact per-frame Sionna trace. Candidate model
transfers are assigned to a small number of reusable time-frequency resources.
Every transfer is bidirectional: providers transmit in the first directional
airtime phase and receivers transmit in the second. Links sharing a resource
interfere through their already-traced directed path powers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TransferProposal:
    receiver: int
    provider: int
    priority: float
    provider_to_receiver_bytes: int
    receiver_to_provider_bytes: int


@dataclass(frozen=True)
class TransferMetrics:
    forward_sinr_db: float
    reverse_sinr_db: float
    forward_capacity_bytes: float
    reverse_capacity_bytes: float


@dataclass(frozen=True)
class ScheduledTransfer:
    proposal: TransferProposal
    resource: int
    metrics: TransferMetrics


@dataclass(frozen=True)
class RejectedTransfer:
    proposal: TransferProposal
    reason: str


@dataclass(frozen=True)
class ScheduleResult:
    accepted: tuple[ScheduledTransfer, ...]
    rejected: tuple[RejectedTransfer, ...]


def _dbm_to_mw(value_dbm: float) -> float:
    return 10.0 ** (float(value_dbm) / 10.0)


def _sinr_db(
    desired_dbm: float,
    interferer_dbm: Sequence[float],
    *,
    noise_floor_dbm: float,
) -> float:
    desired_mw = _dbm_to_mw(desired_dbm)
    denominator_mw = _dbm_to_mw(noise_floor_dbm) + sum(
        _dbm_to_mw(value) for value in interferer_dbm
    )
    return 10.0 * math.log10(max(desired_mw / denominator_mw, 1.0e-30))


def _capacity_bytes(
    sinr_db: float,
    *,
    bandwidth_hz: float,
    direction_airtime_s: float,
    efficiency: float,
    max_spectral_efficiency: float,
) -> float:
    linear = 10.0 ** (float(sinr_db) / 10.0)
    spectral_efficiency = min(
        math.log2(1.0 + linear), float(max_spectral_efficiency)
    )
    return (
        float(bandwidth_hz)
        * float(direction_airtime_s)
        * float(efficiency)
        * spectral_efficiency
        / 8.0
    )


def _resource_metrics(
    proposals: Sequence[TransferProposal],
    *,
    received_power_dbm: Mapping[tuple[int, int], float],
    missing_power_dbm: float,
    noise_floor_dbm: float,
    bandwidth_hz: float,
    direction_airtime_s: float,
    efficiency: float,
    max_spectral_efficiency: float,
) -> dict[TransferProposal, TransferMetrics]:
    result: dict[TransferProposal, TransferMetrics] = {}
    for proposal in proposals:
        forward_interference = [
            float(
                received_power_dbm.get(
                    (int(other.provider), int(proposal.receiver)),
                    missing_power_dbm,
                )
            )
            for other in proposals
            if other is not proposal
        ]
        reverse_interference = [
            float(
                received_power_dbm.get(
                    (int(other.receiver), int(proposal.provider)),
                    missing_power_dbm,
                )
            )
            for other in proposals
            if other is not proposal
        ]
        forward_sinr = _sinr_db(
            float(
                received_power_dbm.get(
                    (int(proposal.provider), int(proposal.receiver)),
                    missing_power_dbm,
                )
            ),
            forward_interference,
            noise_floor_dbm=noise_floor_dbm,
        )
        reverse_sinr = _sinr_db(
            float(
                received_power_dbm.get(
                    (int(proposal.receiver), int(proposal.provider)),
                    missing_power_dbm,
                )
            ),
            reverse_interference,
            noise_floor_dbm=noise_floor_dbm,
        )
        result[proposal] = TransferMetrics(
            forward_sinr_db=float(forward_sinr),
            reverse_sinr_db=float(reverse_sinr),
            forward_capacity_bytes=_capacity_bytes(
                forward_sinr,
                bandwidth_hz=bandwidth_hz,
                direction_airtime_s=direction_airtime_s,
                efficiency=efficiency,
                max_spectral_efficiency=max_spectral_efficiency,
            ),
            reverse_capacity_bytes=_capacity_bytes(
                reverse_sinr,
                bandwidth_hz=bandwidth_hz,
                direction_airtime_s=direction_airtime_s,
                efficiency=efficiency,
                max_spectral_efficiency=max_spectral_efficiency,
            ),
        )
    return result


def _all_transfers_fit(
    metrics: Mapping[TransferProposal, TransferMetrics],
    *,
    min_sinr_db: float,
) -> bool:
    return all(
        value.forward_sinr_db >= float(min_sinr_db)
        and value.reverse_sinr_db >= float(min_sinr_db)
        and value.forward_capacity_bytes
        >= float(proposal.provider_to_receiver_bytes)
        and value.reverse_capacity_bytes
        >= float(proposal.receiver_to_provider_bytes)
        for proposal, value in metrics.items()
    )


def schedule_transfers(
    proposals: Sequence[TransferProposal],
    *,
    received_power_dbm: Mapping[tuple[int, int], float],
    noise_floor_dbm: float,
    min_sinr_db: float,
    resource_count: int,
    bandwidth_hz: float,
    direction_airtime_s: float,
    efficiency: float,
    max_spectral_efficiency: float,
    missing_power_dbm: float,
) -> ScheduleResult:
    """Greedily schedule bidirectional transfers with global half duplex."""

    if int(resource_count) <= 0:
        raise ValueError("resource_count must be positive")
    if float(bandwidth_hz) <= 0.0 or float(direction_airtime_s) <= 0.0:
        raise ValueError("bandwidth and directional airtime must be positive")
    if not 0.0 < float(efficiency) <= 1.0:
        raise ValueError("efficiency must be in (0, 1]")
    if float(max_spectral_efficiency) <= 0.0:
        raise ValueError("max_spectral_efficiency must be positive")

    resources: list[list[TransferProposal]] = [
        [] for _ in range(int(resource_count))
    ]
    used_nodes: set[int] = set()
    rejected: list[RejectedTransfer] = []
    ordered = sorted(
        proposals,
        key=lambda proposal: (
            -float(proposal.priority),
            int(proposal.receiver),
            int(proposal.provider),
        ),
    )
    for proposal in ordered:
        if int(proposal.receiver) == int(proposal.provider):
            rejected.append(RejectedTransfer(proposal, "self_link"))
            continue
        if (
            int(proposal.receiver) in used_nodes
            or int(proposal.provider) in used_nodes
        ):
            rejected.append(RejectedTransfer(proposal, "half_duplex_conflict"))
            continue

        feasible_resources: list[tuple[float, int]] = []
        for resource, existing in enumerate(resources):
            trial = [*existing, proposal]
            metrics = _resource_metrics(
                trial,
                received_power_dbm=received_power_dbm,
                missing_power_dbm=missing_power_dbm,
                noise_floor_dbm=noise_floor_dbm,
                bandwidth_hz=bandwidth_hz,
                direction_airtime_s=direction_airtime_s,
                efficiency=efficiency,
                max_spectral_efficiency=max_spectral_efficiency,
            )
            if not _all_transfers_fit(metrics, min_sinr_db=min_sinr_db):
                continue
            margins = [
                min(
                    value.forward_sinr_db - float(min_sinr_db),
                    value.reverse_sinr_db - float(min_sinr_db),
                    value.forward_capacity_bytes
                    / max(1.0, float(item.provider_to_receiver_bytes))
                    - 1.0,
                    value.reverse_capacity_bytes
                    / max(1.0, float(item.receiver_to_provider_bytes))
                    - 1.0,
                )
                for item, value in metrics.items()
            ]
            feasible_resources.append((float(min(margins)), int(resource)))
        if not feasible_resources:
            rejected.append(RejectedTransfer(proposal, "sinr_or_airtime"))
            continue
        _margin, selected_resource = max(
            feasible_resources, key=lambda row: (row[0], -row[1])
        )
        resources[selected_resource].append(proposal)
        used_nodes.add(int(proposal.receiver))
        used_nodes.add(int(proposal.provider))

    accepted: list[ScheduledTransfer] = []
    for resource, rows in enumerate(resources):
        metrics = _resource_metrics(
            rows,
            received_power_dbm=received_power_dbm,
            missing_power_dbm=missing_power_dbm,
            noise_floor_dbm=noise_floor_dbm,
            bandwidth_hz=bandwidth_hz,
            direction_airtime_s=direction_airtime_s,
            efficiency=efficiency,
            max_spectral_efficiency=max_spectral_efficiency,
        )
        accepted.extend(
            ScheduledTransfer(proposal, int(resource), metrics[proposal])
            for proposal in rows
        )
    accepted.sort(
        key=lambda row: (
            int(row.proposal.receiver),
            int(row.proposal.provider),
        )
    )
    return ScheduleResult(tuple(accepted), tuple(rejected))
