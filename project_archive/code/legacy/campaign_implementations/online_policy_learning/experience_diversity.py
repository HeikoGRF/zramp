"""Policy-free top-experience predictor sharing.

Provider ranking and one-shot aggregation both use a quantity-times-diversity
score computed from the receiver-visible dataset of the current zone visit.
The temporal descriptor deliberately uses fixed Fourier frequencies so scores
remain comparable across independently trained predictor models.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.optim as optim

import rl_reward_experiment.sim as rre_sim
from SUMO.sumo_rl import SumoT2Simulation

from .aggregation import weighted_average


EXPERIENCE_METADATA_BYTES = 4  # one IEEE-754 float32 per directed contact


def fixed_fourier_time_features(
    times: torch.Tensor,
    *,
    num_frequencies: int,
    min_period: float,
    max_period: float,
    time_unit: float = 1.0,
) -> torch.Tensor:
    """Apply the predictor's initialized Fourier transform without learning it."""

    if int(num_frequencies) < 1:
        raise ValueError("num_frequencies must be positive")
    if float(min_period) <= 0.0 or float(max_period) <= float(min_period):
        raise ValueError("Require 0 < min_period < max_period")
    if float(time_unit) <= 0.0:
        raise ValueError("time_unit must be positive")
    values = torch.as_tensor(times, dtype=torch.float32)
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    if values.ndim != 2 or int(values.shape[1]) != 1:
        raise ValueError("times must have shape [num_samples] or [num_samples, 1]")
    periods = torch.logspace(
        math.log10(float(min_period)),
        math.log10(float(max_period)),
        steps=int(num_frequencies),
        device=values.device,
        dtype=values.dtype,
    )
    omega = (2.0 * math.pi / periods).unsqueeze(0)
    scaled = values / float(time_unit)
    phase = scaled * omega
    trend = torch.log1p(torch.clamp_min(scaled, 0.0))
    return torch.cat((trend, torch.sin(phase), torch.cos(phase)), dim=-1)


def build_experience_descriptors(
    predictor_inputs: torch.Tensor,
    *,
    num_time_frequencies: int = 8,
    min_time_period: float = 2.0,
    max_time_period: float = 1000.0,
    time_unit: float = 1.0,
) -> torch.Tensor:
    """Build ``[normalized tx, normalized rx, fixed time features]`` rows."""

    inputs = torch.as_tensor(predictor_inputs, dtype=torch.float32)
    if inputs.ndim != 2 or int(inputs.shape[1]) not in {4, 5}:
        raise ValueError("predictor_inputs must have shape [num_samples, 4 or 5]")
    coordinates = inputs[:, :4]
    if int(inputs.shape[1]) == 4:
        return coordinates
    time_features = fixed_fourier_time_features(
        inputs[:, 4:5],
        num_frequencies=int(num_time_frequencies),
        min_period=float(min_time_period),
        max_period=float(max_time_period),
        time_unit=float(time_unit),
    )
    return torch.cat((coordinates, time_features), dim=-1)


def experience_score(
    descriptors: torch.Tensor,
    max_diversity_samples: int = 512,
    *,
    generator: torch.Generator | None = None,
) -> float:
    """Return sample count times mean pairwise descriptor distance."""

    rows = torch.as_tensor(descriptors, dtype=torch.float32)
    if rows.ndim != 2:
        raise ValueError("descriptors must have shape [num_samples, descriptor_dim]")
    if int(max_diversity_samples) < 2:
        raise ValueError("max_diversity_samples must be at least two")
    num_samples = int(rows.shape[0])
    if num_samples < 2:
        return float(num_samples)
    if num_samples > int(max_diversity_samples):
        indices = torch.randperm(
            num_samples,
            device=rows.device,
            generator=generator,
        )[: int(max_diversity_samples)]
        diversity_samples = rows[indices]
    else:
        diversity_samples = rows
    diversity = torch.pdist(diversity_samples, p=2).mean()
    return float((float(num_samples) * diversity).item())


def select_highest_experience(
    provider_ids: Sequence[int],
    provider_experiences: Sequence[float],
    budget: int,
) -> list[int]:
    """Select exactly min(B, feasible providers), with stable ID tie-breaking."""

    ids = [int(value) for value in provider_ids]
    scores = [float(value) for value in provider_experiences]
    if len(ids) != len(scores):
        raise ValueError("provider IDs and experiences must have the same length")
    if int(budget) < 0:
        raise ValueError("budget must be non-negative")
    if any(not math.isfinite(value) or value < 0.0 for value in scores):
        raise ValueError("experience scores must be finite and non-negative")
    ranked = sorted(zip(ids, scores), key=lambda item: (-item[1], item[0]))
    return [provider_id for provider_id, _score in ranked[: int(budget)]]


def normalize_experience_weights(experiences: Sequence[float]) -> tuple[torch.Tensor, bool]:
    """Normalize exact scores; use uniform weights only for zero total mass."""

    if not experiences:
        raise ValueError("at least one experience score is required")
    values = torch.tensor(list(experiences), dtype=torch.float64)
    if not torch.isfinite(values).all() or bool(torch.any(values < 0.0)):
        raise ValueError("experience scores must be finite and non-negative")
    total = float(values.sum())
    zero_total_fallback = total <= 0.0
    if zero_total_fallback:
        values.fill_(1.0 / float(values.numel()))
    else:
        values /= total
    return values.to(dtype=torch.float32), zero_total_fallback


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu").clone()
        for name, tensor in model.state_dict().items()
    }


def _state_nbytes(state: dict[str, torch.Tensor]) -> int:
    return int(sum(int(value.numel()) * int(value.element_size()) for value in state.values()))


class ExperienceDiversitySimulation(SumoT2Simulation):
    """Pull top-B experience scores and aggregate own plus pulled predictors."""

    _LOG_FIELDS = (
        "step",
        "mode",
        "receiver_idx",
        "zone",
        "neighbor_count",
        "budget",
        "selected_count",
        "selected_provider_ids",
        "self_experience",
        "provider_experiences",
        "selected_experiences",
        "aggregation_weights",
        "zero_total_fallback",
        "descriptor_samples",
        "metadata_bytes",
        "model_bytes",
    )

    def __init__(
        self,
        *args,
        pull_budget: int,
        max_diversity_samples: int = 512,
        **kwargs,
    ) -> None:
        budget = int(pull_budget)
        if float(pull_budget) != float(budget) or budget < 0:
            raise ValueError("pull_budget must be a non-negative integer")
        if int(max_diversity_samples) < 2:
            raise ValueError("max_diversity_samples must be at least two")
        self.pull_budget = budget
        self.max_diversity_samples = int(max_diversity_samples)
        self._experience_cache: dict[int, tuple[tuple[object, ...], float]] = {}
        self._experience_log_file = None
        self._experience_log_writer: csv.DictWriter | None = None
        self._experience_log_rows = 0
        kwargs["local_policy_share"] = False
        super().__init__(*args, **kwargs)
        self.zramp_policy_mode = "experience-diversity-top-b"
        self._communication_assumptions = self._build_communication_assumptions()
        self._experience_log_path = Path(self.cfg.results_dir) / "experience_aggregation.csv"
        try:
            self._experience_log_path.unlink()
        except FileNotFoundError:
            pass

    def _init_local_policy_agents(self) -> None:
        """This baseline intentionally has no trainable decision model."""

        self.local_agents.clear()
        self._local_policy_pending_transitions.clear()
        self._local_policy_versions.clear()
        self._local_policy_initial_rngs.clear()

    def _train_rl_agents(self, step: int | None = None) -> dict[str, float]:
        del step
        return {mode: 0.0 for mode in self.agents}

    def _sample_recency_weights(self, *args, **kwargs) -> None:
        """Train predictors uniformly on the full current-zone visit buffer."""

        del args, kwargs
        return None

    def _normalized_links(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None,
    ) -> list[tuple[int, int, int]]:
        if contact_links is None:
            return [
                (int(zone), int(a), int(b))
                for zone, indices in zone_nodes.items()
                for offset, a in enumerate(sorted(indices))
                for b in sorted(indices)[offset + 1 :]
            ]
        unique: set[tuple[int, int, int]] = set()
        for zone, a, b in contact_links:
            ia, ib = sorted((int(a), int(b)))
            if ia == ib or not (0 <= ia < len(self.nodes) and 0 <= ib < len(self.nodes)):
                continue
            if int(self.nodes[ia].current_az) != int(self.nodes[ib].current_az):
                continue
            unique.add((int(zone), ia, ib))
        return sorted(unique)

    def _node_experience(self, node_idx: int) -> float:
        ns = self.nodes[int(node_idx)]
        rows = ns.current_visit_samples_x
        num_samples = len(rows)
        first_time = float(rows[0][-1]) if num_samples and len(rows[0]) >= 5 else None
        last_time = float(rows[-1][-1]) if num_samples and len(rows[-1]) >= 5 else None
        cache_key = (int(ns.current_az), num_samples, first_time, last_time)
        cached = self._experience_cache.get(int(node_idx))
        if cached is not None and cached[0] == cache_key:
            return float(cached[1])
        raw = np.asarray(rows, dtype=np.float32)
        if num_samples == 0:
            raw = np.empty((0, self._predictor_input_dim()), dtype=np.float32)
        descriptors = build_experience_descriptors(
            torch.as_tensor(raw, dtype=torch.float32),
            num_time_frequencies=int(self.cfg.predictor_time_num_frequencies),
            min_time_period=float(self.cfg.predictor_time_min_period),
            max_time_period=float(self.cfg.predictor_time_max_period),
            time_unit=float(self.cfg.predictor_time_unit),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            rre_sim._stable_torch_seed(
                self.cfg.seed,
                "experience-diversity",
                int(node_idx),
                int(ns.current_az),
                num_samples,
                first_time,
                last_time,
            )
        )
        score = experience_score(
            descriptors,
            self.max_diversity_samples,
            generator=generator,
        )
        self._experience_cache[int(node_idx)] = (cache_key, score)
        return score

    def _ensure_experience_log(self) -> csv.DictWriter:
        if self._experience_log_writer is None:
            self._experience_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._experience_log_file = open(
                self._experience_log_path, "w", newline="", encoding="utf-8"
            )
            self._experience_log_writer = csv.DictWriter(
                self._experience_log_file, fieldnames=self._LOG_FIELDS
            )
            self._experience_log_writer.writeheader()
        return self._experience_log_writer

    def _write_experience_log(self, row: dict[str, object]) -> None:
        self._ensure_experience_log().writerow(
            {field: row.get(field, "") for field in self._LOG_FIELDS}
        )
        self._experience_log_rows += 1

    def _flush_experience_log(self) -> None:
        if self._experience_log_file is not None:
            self._experience_log_file.flush()

    def _close_experience_log(self) -> None:
        if self._experience_log_file is not None:
            self._experience_log_file.flush()
            self._experience_log_file.close()
        self._experience_log_file = None
        self._experience_log_writer = None

    def _gossip_step(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Snapshot all scores/models, then independently update each receiver."""

        links = self._normalized_links(zone_nodes, contact_links)
        neighbors: dict[int, list[int]] = defaultdict(list)
        for _zone, a, b in links:
            neighbors[a].append(b)
            neighbors[b].append(a)
            self._next_enc_id += 1
        for receiver_idx in neighbors:
            neighbors[receiver_idx] = sorted(set(neighbors[receiver_idx]))
        if not neighbors:
            return

        experiences = {
            node_idx: self._node_experience(node_idx)
            for node_idx in range(len(self.nodes))
        }
        snapshots = {
            mode: {
                node_idx: _clone_state(ns.variants[mode].model)
                for node_idx, ns in enumerate(self.nodes)
            }
            for mode in self.agents
        }

        for mode in self.agents:
            for receiver_idx in sorted(neighbors):
                provider_ids = neighbors[receiver_idx]
                provider_scores = [experiences[idx] for idx in provider_ids]
                selected_ids = select_highest_experience(
                    provider_ids, provider_scores, self.pull_budget
                )
                candidate_ids = [receiver_idx, *selected_ids]
                candidate_scores = [experiences[idx] for idx in candidate_ids]
                weights, zero_total_fallback = normalize_experience_weights(candidate_scores)
                aggregated = weighted_average(
                    [snapshots[mode][idx] for idx in candidate_ids], weights
                )
                variant = self.nodes[receiver_idx].variants[mode]
                variant.model.load_state_dict(aggregated)
                variant.opt = optim.Adam(variant.model.parameters(), lr=self.cfg.local_lr)
                variant.t_wait = 0
                variant.last_rmse_available = False
                self._refresh_variant_signature(variant)

                selected_set = set(selected_ids)
                score_by_provider = dict(zip(provider_ids, provider_scores))
                metadata_bytes = len(provider_ids) * EXPERIENCE_METADATA_BYTES
                model_bytes = sum(
                    _state_nbytes(snapshots[mode][provider_idx])
                    for provider_idx in selected_ids
                )
                for provider_idx in provider_ids:
                    action = int(provider_idx in selected_set)
                    self.decision_log.append(
                        {
                            "step": int(step),
                            "mode": mode,
                            "node_i": receiver_idx,
                            "node_j": provider_idx,
                            "az": int(self.nodes[receiver_idx].current_az),
                            "action": action,
                            "experience_score": float(score_by_provider[provider_idx]),
                        }
                    )
                    self._decision_action_counts[str(mode)][action] += 1

                self._write_experience_log(
                    {
                        "step": int(step),
                        "mode": mode,
                        "receiver_idx": receiver_idx,
                        "zone": int(self.nodes[receiver_idx].current_az),
                        "neighbor_count": len(provider_ids),
                        "budget": self.pull_budget,
                        "selected_count": len(selected_ids),
                        "selected_provider_ids": ";".join(map(str, selected_ids)),
                        "self_experience": f"{experiences[receiver_idx]:.9g}",
                        "provider_experiences": ";".join(
                            f"{idx}:{experiences[idx]:.9g}" for idx in provider_ids
                        ),
                        "selected_experiences": ";".join(
                            f"{idx}:{experiences[idx]:.9g}" for idx in selected_ids
                        ),
                        "aggregation_weights": ";".join(
                            f"{idx}:{float(weight):.9g}"
                            for idx, weight in zip(candidate_ids, weights.tolist())
                        ),
                        "zero_total_fallback": int(zero_total_fallback),
                        "descriptor_samples": len(
                            self.nodes[receiver_idx].current_visit_samples_x
                        ),
                        "metadata_bytes": metadata_bytes,
                        "model_bytes": model_bytes,
                    }
                )

    def _build_communication_assumptions(self) -> dict[str, int | float | str | bool]:
        assumptions = super()._build_communication_assumptions()
        model_bytes = int(assumptions.get("B_model_bytes", 0))
        assumptions.update(
            {
                "zramp_policy_mode": "experience-diversity-top-b",
                "policy_model": False,
                "policy_params": 0,
                "B_policy_bytes": 0,
                "B_decision_meta_bytes_per_directed_decision": EXPERIENCE_METADATA_BYTES,
                "B_decision_scalar_meta_bytes_per_directed_decision": EXPERIENCE_METADATA_BYTES,
                "B_model_signature_bytes_per_directed_decision": 0,
                "B_accepted_merge_meta_bytes_per_pull": 0,
                "B_accepted_pull_bytes": model_bytes,
                "B_local_policy_pull_bytes": 0,
                "B_local_zramp_accepted_pull_bytes": model_bytes,
                "pull_budget_expected_slots": float(self.pull_budget),
                "provider_selection": "top-B quantity-times-descriptor-diversity experience",
                "predictor_aggregation": "simultaneous own-plus-selected exact experience weighting",
                "experience_definition": "m*mean_pairwise_distance([normalized_tx,normalized_rx,fixed_fourier_time])",
                "experience_diversity_max_samples": int(self.max_diversity_samples),
                "experience_zero_total_fallback": "uniform-only-when-all-candidate-scores-are-zero",
                "metadata_note": (
                    "Each feasible provider sends one float32 experience score. "
                    "The receiver downloads exactly min(B,N) predictor states and no policy."
                ),
            }
        )
        return assumptions

    def _write_partial_outputs(self, *args, **kwargs) -> None:
        super()._write_partial_outputs(*args, **kwargs)
        self._flush_experience_log()

    def run(self) -> None:
        try:
            super().run()
        finally:
            self._close_experience_log()

