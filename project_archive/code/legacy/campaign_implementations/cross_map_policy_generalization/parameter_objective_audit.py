"""Offline audit of parameter-only signals against true map improvement.

The all-map evaluation rows in this module are diagnostic labels only.  They
are never exposed to the online provider policy, aggregation rule, predictor
training, or artificial-link generator.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch

from online_policy_learning.local_validation_reward import interpolate_states
from online_policy_learning.parameter_geometry import select_geometry_aggregation
from cross_map_policy_generalization.parameter_geometry_role_simulation import (
    ParameterGeometryRoleSimulation,
)


class ParameterObjectiveAuditMixin:
    """Collect counterfactual global-map rewards for feasible contacts."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._audit_every = max(
            1, int(os.environ.get("PARAMETER_AUDIT_EVERY", "20"))
        )
        self._audit_pair_cap = max(
            1, int(os.environ.get("PARAMETER_AUDIT_PAIR_CAP", "24"))
        )
        self._audit_test_cap = max(
            16, int(os.environ.get("PARAMETER_AUDIT_TEST_CAP", "256"))
        )
        self._audit_sketch_dim = max(
            8, int(os.environ.get("PARAMETER_AUDIT_SKETCH_DIM", "256"))
        )
        self._audit_save_states = (
            os.environ.get("PARAMETER_AUDIT_SAVE_STATES", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        self._audit_states: dict[tuple[int, str, int], object] = {}
        self._audit_alpha_grid = np.asarray(
            [
                float(value)
                for value in os.environ.get(
                    "PARAMETER_AUDIT_ALPHA_GRID", "0,0.25,0.5,0.75,1"
                ).split(",")
            ],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(self._audit_alpha_grid)) or np.any(
            (self._audit_alpha_grid < 0.0)
            | (self._audit_alpha_grid > 1.0)
        ):
            raise ValueError("PARAMETER_AUDIT_ALPHA_GRID must lie in [0, 1]")
        self._audit_alpha_grid = np.unique(
            np.concatenate((self._audit_alpha_grid, np.asarray([1.0])))
        )
        if (
            int(self._route_evaluation_X.shape[0]) == 0
            and self.measurement_trace_in
        ):
            with np.load(self.measurement_trace_in, allow_pickle=False) as replay:
                if "fid_0000_z0_X" in replay.files:
                    self._route_evaluation_X = np.asarray(
                        replay["fid_0000_z0_X"], dtype=np.float32
                    ).reshape(-1, 4)
                    self._route_evaluation_y = np.maximum(
                        np.asarray(
                            replay["fid_0000_z0_y"], dtype=np.float32
                        ).reshape(-1),
                        float(self.cfg.noise_floor_dbm),
                    )
        if int(self._route_evaluation_X.shape[0]) == 0:
            raise ValueError(
                "parameter-objective audit requires a fixed evaluation set "
                "in the measurement trace"
            )
        self._audit_X, self._audit_y = self._fixed_audit_test()
        threshold = float(
            self.cfg.noise_floor_dbm + self.cfg.snr_min_db
        )
        self._audit_feasible = self._audit_y >= threshold
        self._audit_file = None
        self._audit_writer: csv.DictWriter | None = None
        self._audit_rows = 0

    def _fixed_audit_test(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a deterministic, class-stratified subset of the fixed map."""

        X = np.asarray(self._route_evaluation_X, dtype=np.float32)
        y = np.asarray(self._route_evaluation_y, dtype=np.float32).reshape(-1)
        cap = min(int(y.shape[0]), int(self._audit_test_cap))
        if cap == int(y.shape[0]):
            return X.copy(), y.copy()
        threshold = float(
            self.cfg.noise_floor_dbm + self.cfg.snr_min_db
        )
        feasible = np.flatnonzero(y >= threshold)
        unavailable = np.flatnonzero(y < threshold)
        rng = np.random.default_rng(20260727)
        feasible_cap = min(
            int(feasible.size),
            max(1, int(round(cap * feasible.size / max(1, y.size)))),
        )
        unavailable_cap = min(int(unavailable.size), cap - feasible_cap)
        if feasible_cap + unavailable_cap < cap:
            feasible_cap = min(
                int(feasible.size), cap - unavailable_cap
            )
        selected = np.concatenate(
            (
                rng.choice(feasible, size=feasible_cap, replace=False),
                rng.choice(
                    unavailable, size=unavailable_cap, replace=False
                ),
            )
        )
        rng.shuffle(selected)
        return X[selected].copy(), y[selected].copy()

    @staticmethod
    def _rmse(error: np.ndarray, mask: np.ndarray | None = None) -> float:
        values = error if mask is None else error[mask]
        if int(values.size) == 0:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))

    def _evaluate_state(
        self, state: dict[str, torch.Tensor]
    ) -> tuple[float, float, float]:
        """Evaluate a snapshot on the fixed map without mutating a vehicle."""

        self._load_model_state(self._cv_eval_model, dict(state))
        self._cv_eval_model.eval()
        features = torch.as_tensor(
            self._audit_X, dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            prediction = self._denorm_dbm(
                self._cv_eval_model(features)
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            ).astype(np.float64, copy=False)
        error = prediction - self._audit_y.astype(np.float64, copy=False)
        return (
            self._rmse(error),
            self._rmse(error, self._audit_feasible),
            self._rmse(error, ~self._audit_feasible),
        )

    def _parameter_sketch(self, vector: torch.Tensor) -> np.ndarray:
        """Deterministic CountSketch of the normalized initial-model delta."""

        values = vector.detach().to(device="cpu", dtype=torch.float64).numpy()
        output = np.zeros(self._audit_sketch_dim, dtype=np.float64)
        if int(values.size) == 0:
            return output.astype(np.float32)
        indices = np.arange(values.size, dtype=np.uint64)
        hashed = indices * np.uint64(11400714819323198485)
        buckets = np.asarray(
            hashed % np.uint64(self._audit_sketch_dim), dtype=np.int64
        )
        signs = np.where(
            ((hashed >> np.uint64(32)) & np.uint64(1)) == 0,
            1.0,
            -1.0,
        )
        np.add.at(output, buckets, values * signs)
        output /= np.sqrt(max(1, int(values.size)))
        return output.astype(np.float32)

    def _audit_fields(self) -> list[str]:
        scalar = [
            "seed",
            "step",
            "mode",
            "receiver_idx",
            "provider_idx",
            "alpha",
            "baseline_rmse",
            "candidate_rmse",
            "oracle_gain_db",
            "baseline_feasible_rmse",
            "candidate_feasible_rmse",
            "feasible_gain_db",
            "baseline_unavailable_rmse",
            "candidate_unavailable_rmse",
            "unavailable_gain_db",
            "geometry_objective",
            "geometry_baseline_objective",
            "geometry_selected_alpha",
            "geometry_gross_reward",
            "receiver_radial",
            "provider_radial",
            "receiver_training_stability",
            "provider_training_stability",
            "receiver_merge_stability",
            "provider_merge_stability",
            "receiver_maturity",
            "provider_maturity",
            "pair_distance",
            "novelty",
            "cosine",
            "cancellation_ratio",
            "trust_ratio",
            "test_rows",
            "test_feasible_rows",
            "test_unavailable_rows",
        ]
        embedding_dim = int(self.embedding_dim)
        scalar.extend(
            f"receiver_embedding_{index:03d}"
            for index in range(embedding_dim)
        )
        scalar.extend(
            f"provider_embedding_{index:03d}"
            for index in range(embedding_dim)
        )
        scalar.extend(
            f"receiver_sketch_{index:03d}"
            for index in range(self._audit_sketch_dim)
        )
        scalar.extend(
            f"provider_sketch_{index:03d}"
            for index in range(self._audit_sketch_dim)
        )
        return scalar

    def _ensure_writer(self) -> csv.DictWriter:
        if self._audit_writer is not None:
            return self._audit_writer
        path = Path(self.cfg.results_dir) / "parameter_objective_audit.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_file = path.open("w", newline="", encoding="utf-8")
        self._audit_writer = csv.DictWriter(
            self._audit_file, fieldnames=self._audit_fields()
        )
        self._audit_writer.writeheader()
        self._audit_file.flush()
        return self._audit_writer

    def _select_pairs(
        self, links: list[tuple[int, int, int]], step: int
    ) -> list[tuple[int, int]]:
        directed = sorted(
            {
                pair
                for _zone, a, b in links
                for pair in ((int(a), int(b)), (int(b), int(a)))
            }
        )
        if len(directed) <= self._audit_pair_cap:
            return directed
        rng = np.random.default_rng(
            int(self.cfg.seed) * 1_000_003 + int(step)
        )
        selected = rng.choice(
            len(directed), size=self._audit_pair_cap, replace=False
        )
        return [directed[int(index)] for index in sorted(selected)]

    def _run_parameter_audit(
        self, *, step: int, links: list[tuple[int, int, int]]
    ) -> None:
        pairs = self._select_pairs(links, step)
        if not pairs:
            return
        writer = self._ensure_writer()
        for mode in self.agents:
            state_cache = {
                node_idx: self._clone_state(
                    self.nodes[node_idx].variants[str(mode)].model
                )
                for node_idx in sorted(
                    {node for pair in pairs for node in pair}
                )
            }
            vector_cache = {
                node_idx: self._geometry_tracker(
                    node_idx, str(mode)
                ).vector(state)
                for node_idx, state in state_cache.items()
            }
            sketch_cache = {
                node_idx: self._parameter_sketch(vector)
                for node_idx, vector in vector_cache.items()
            }
            raw_state_cache = {
                node_idx: self._raw_state(
                    node_idx, str(mode), model_state=state
                )
                for node_idx, state in state_cache.items()
            }
            if self._audit_save_states:
                for node_idx, raw_state in raw_state_cache.items():
                    self._audit_states[
                        (int(step), str(mode), int(node_idx))
                    ] = raw_state.clone()
            for receiver_idx, provider_idx in pairs:
                state_a = state_cache[receiver_idx]
                state_b = state_cache[provider_idx]
                tracker_a = self._geometry_tracker(receiver_idx, str(mode))
                tracker_b = self._geometry_tracker(provider_idx, str(mode))
                geometry = select_geometry_aggregation(
                    state_a,
                    state_b,
                    tracker_a,
                    tracker_b,
                    alpha_grid=self._audit_alpha_grid,
                    radial_scale=self.geometry_radial_scale,
                    cancellation_penalty=self.geometry_cancellation_penalty,
                    trust_penalty=self.geometry_trust_penalty,
                    trust_radius=self.geometry_trust_radius,
                )
                objective_by_alpha = dict(geometry.evaluations)
                baseline = self._evaluate_state(state_a)
                receiver_state = raw_state_cache[receiver_idx]
                provider_state = raw_state_cache[provider_idx]
                receiver_agent = self.local_agents[str(mode)][receiver_idx]
                receiver_embedding = (
                    receiver_agent.policy_embedding(receiver_state)
                    .reshape(-1)
                    .numpy()
                )
                provider_embedding = (
                    receiver_agent.policy_embedding(provider_state)
                    .reshape(-1)
                    .numpy()
                )
                for alpha in self._audit_alpha_grid:
                    candidate = (
                        baseline
                        if float(alpha) == 1.0
                        else self._evaluate_state(
                            interpolate_states(state_a, state_b, float(alpha))
                        )
                    )
                    row: dict[str, object] = {
                        "seed": int(self.cfg.seed),
                        "step": int(step),
                        "mode": str(mode),
                        "receiver_idx": int(receiver_idx),
                        "provider_idx": int(provider_idx),
                        "alpha": float(alpha),
                        "baseline_rmse": baseline[0],
                        "candidate_rmse": candidate[0],
                        "oracle_gain_db": baseline[0] - candidate[0],
                        "baseline_feasible_rmse": baseline[1],
                        "candidate_feasible_rmse": candidate[1],
                        "feasible_gain_db": baseline[1] - candidate[1],
                        "baseline_unavailable_rmse": baseline[2],
                        "candidate_unavailable_rmse": candidate[2],
                        "unavailable_gain_db": baseline[2] - candidate[2],
                        "geometry_objective": objective_by_alpha[float(alpha)],
                        "geometry_baseline_objective": objective_by_alpha[1.0],
                        "geometry_selected_alpha": geometry.alpha,
                        "geometry_gross_reward": geometry.gross_reward,
                        "receiver_radial": geometry.receiver.radial_distance,
                        "provider_radial": geometry.provider.radial_distance,
                        "receiver_training_stability": (
                            geometry.receiver.training_stability
                        ),
                        "provider_training_stability": (
                            geometry.provider.training_stability
                        ),
                        "receiver_merge_stability": (
                            geometry.receiver.merge_stability
                        ),
                        "provider_merge_stability": (
                            geometry.provider.merge_stability
                        ),
                        "receiver_maturity": geometry.receiver.maturity,
                        "provider_maturity": geometry.provider.maturity,
                        "pair_distance": geometry.pair_distance,
                        "novelty": geometry.normalized_novelty,
                        "cosine": geometry.cosine,
                        "cancellation_ratio": geometry.cancellation_ratio,
                        "trust_ratio": geometry.trust_ratio,
                        "test_rows": int(self._audit_y.size),
                        "test_feasible_rows": int(
                            np.count_nonzero(self._audit_feasible)
                        ),
                        "test_unavailable_rows": int(
                            np.count_nonzero(~self._audit_feasible)
                        ),
                    }
                    row.update(
                        {
                            f"receiver_embedding_{index:03d}": float(value)
                            for index, value in enumerate(receiver_embedding)
                        }
                    )
                    row.update(
                        {
                            f"provider_embedding_{index:03d}": float(value)
                            for index, value in enumerate(provider_embedding)
                        }
                    )
                    row.update(
                        {
                            f"receiver_sketch_{index:03d}": float(value)
                            for index, value in enumerate(
                                sketch_cache[receiver_idx]
                            )
                        }
                    )
                    row.update(
                        {
                            f"provider_sketch_{index:03d}": float(value)
                            for index, value in enumerate(
                                sketch_cache[provider_idx]
                            )
                        }
                    )
                    writer.writerow(row)
                    self._audit_rows += 1
        self._audit_file.flush()
        print(
            "[PARAMETER-AUDIT] "
            f"step={step} pairs={len(pairs)} rows_total={self._audit_rows}",
            flush=True,
        )

    def _gossip_step(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> None:
        links = self._normalized_contact_links(zone_nodes, contact_links)
        if int(step) % self._audit_every == 0:
            self._run_parameter_audit(step=int(step), links=links)
        super()._gossip_step(step, zone_nodes, contact_links=links)

    def run(self) -> None:
        try:
            super().run()
        finally:
            if self._audit_save_states and self._audit_states:
                ordered = sorted(self._audit_states)
                states = [self._audit_states[key] for key in ordered]
                payload = {
                    "step": np.asarray(
                        [key[0] for key in ordered], dtype=np.int32
                    ),
                    "mode": np.asarray(
                        [key[1] for key in ordered], dtype=np.str_
                    ),
                    "node_idx": np.asarray(
                        [key[2] for key in ordered], dtype=np.int32
                    ),
                }
                trajectories = [
                    state.trajectory.detach().cpu().numpy().astype(
                        np.float32, copy=False
                    )
                    for state in states
                ]
                lengths = np.asarray(
                    [trajectory.shape[0] for trajectory in trajectories],
                    dtype=np.int32,
                )
                max_length = int(lengths.max(initial=0))
                trajectory_dim = int(trajectories[0].shape[1])
                padded = np.zeros(
                    (len(trajectories), max_length, trajectory_dim),
                    dtype=np.float32,
                )
                for index, trajectory in enumerate(trajectories):
                    padded[index, : trajectory.shape[0]] = trajectory
                payload["trajectory"] = padded
                payload["trajectory_length"] = lengths
                group_count = len(states[0].model_groups)
                for group_index in range(group_count):
                    payload[f"group_{group_index:03d}"] = np.stack(
                        [
                            state.model_groups[group_index]
                            .detach()
                            .cpu()
                            .numpy()
                            for state in states
                        ]
                    ).astype(np.float32)
                np.savez_compressed(
                    Path(self.cfg.results_dir)
                    / "parameter_objective_states.npz",
                    **payload,
                )
            if self._audit_file is not None:
                self._audit_file.flush()
                self._audit_file.close()
            self._audit_file = None
            self._audit_writer = None


class ParameterObjectiveAuditSimulation(
    ParameterObjectiveAuditMixin,
    ParameterGeometryRoleSimulation,
):
    """Tiny-map parameter-objective audit."""
