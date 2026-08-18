"""Exact decentralized simulation for the 40-vehicle MergeTestMap trace.

The class keeps real feasible observations and private validation semantics in
the established exact sequential simulator.  Artificial floor links augment
predictor training only.  Predictor and policy communication is restricted to
links that are both radio-feasible and direct-path clear in the Sionna trace.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from online_policy_learning.local_validation_reward import ValidationSubset
from online_policy_learning.online_local_validation_policy import (
    ExactSequentialBidirectionalSimulation,
)
from SUMO.artificial_link_support import (
    VehicleEvidence,
    support_distance_candidates,
    temporal_artificial_key,
    temporal_contradiction_keep_mask,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALL_LINK_TRACE = (
    ROOT
    / "cross_map_policy_generalization"
    / "role_sionna_40_seed01"
    / "role_sionna_all_links_seed01.npz"
)


class RoleExactSequentialSimulation(ExactSequentialBidirectionalSimulation):
    """One global coordinate-plus-optional-time predictor per vehicle."""

    def __init__(
        self,
        *args,
        all_link_trace: str | Path = DEFAULT_ALL_LINK_TRACE,
        artificial_min_real_samples: int = 128,
        artificial_full_weight_samples: int = 2048,
        artificial_ratio: float = 0.5,
        artificial_support_min_m: float = 7.5,
        artificial_support_high_m: float = 10.0,
        artificial_low_weight_start: float = 0.10,
        artificial_low_weight_end: float = 0.50,
        artificial_high_weight_start: float = 0.25,
        artificial_high_weight_end: float = 1.00,
        artificial_maturity_exponent: float = 1.0,
        artificial_validation: bool = True,
        artificial_candidate_pool: int = 512,
        artificial_max_new_per_vehicle_step: int = 16,
        **kwargs,
    ) -> None:
        cfg = args[0] if args else kwargs.get("cfg")
        node_count = int(getattr(cfg, "num_nodes", 0))
        if node_count <= 0:
            raise ValueError("role simulation requires a positive node count")
        minimum = int(artificial_min_real_samples)
        full = int(artificial_full_weight_samples)
        if minimum < 1 or full <= minimum:
            raise ValueError(
                "artificial maturity requires 1 <= minimum < full-weight samples"
            )
        if float(artificial_ratio) < 0.0:
            raise ValueError("artificial_ratio cannot be negative")
        if float(artificial_support_min_m) <= 0.0:
            raise ValueError("artificial support distance must be positive")
        if float(artificial_support_high_m) < float(artificial_support_min_m):
            raise ValueError("high support distance must be at least the minimum")
        if float(artificial_maturity_exponent) <= 0.0:
            raise ValueError("artificial maturity exponent must be positive")

        self.all_link_trace = Path(all_link_trace).expanduser().resolve()
        self.artificial_min_real_samples = minimum
        self.artificial_full_weight_samples = full
        self.artificial_ratio = float(artificial_ratio)
        self.artificial_support_min_m = float(artificial_support_min_m)
        self.artificial_support_high_m = float(artificial_support_high_m)
        self.artificial_low_weight_start = float(artificial_low_weight_start)
        self.artificial_low_weight_end = float(artificial_low_weight_end)
        self.artificial_high_weight_start = float(artificial_high_weight_start)
        self.artificial_high_weight_end = float(artificial_high_weight_end)
        self.artificial_maturity_exponent = float(artificial_maturity_exponent)
        self.artificial_validation = bool(artificial_validation)
        self.artificial_candidate_pool = int(artificial_candidate_pool)
        self.artificial_max_new_per_vehicle_step = int(
            artificial_max_new_per_vehicle_step
        )
        if self.artificial_candidate_pool < 0:
            raise ValueError(
                "artificial candidate pool must be nonnegative; zero uses all coordinates"
            )
        if self.artificial_max_new_per_vehicle_step < 0:
            raise ValueError("artificial per-step cap must be nonnegative")

        self._clear_contact_pairs_by_step = self._load_clear_contact_pairs(
            self.all_link_trace
        )
        self._artificial_evidence = [VehicleEvidence() for _ in range(node_count)]
        self._real_samples_received = np.zeros(node_count, dtype=np.int64)
        self._artificial_features: list[list[list[float]]] = [
            [] for _ in range(node_count)
        ]
        self._artificial_raw: list[list[list[float]]] = [
            [] for _ in range(node_count)
        ]
        self._artificial_high: list[list[bool]] = [
            [] for _ in range(node_count)
        ]
        self._artificial_steps: list[list[int]] = [
            [] for _ in range(node_count)
        ]
        # Disjoint 80/10/10 allocation: predictor fit, FedAvg optimization,
        # and pull reward. Artificial validation rows never train the model.
        self._artificial_split: list[list[int]] = [
            [] for _ in range(node_count)
        ]
        self._artificial_keys: list[set[tuple[int, int, int, int, int]]] = [
            set() for _ in range(node_count)
        ]
        self._artificial_generated = np.zeros(node_count, dtype=np.int64)
        self._artificial_removed = np.zeros(node_count, dtype=np.int64)
        self._artificial_shortfall = np.zeros(node_count, dtype=np.int64)
        self._eligible_clear_contacts = 0

        super().__init__(*args, **kwargs)
        self.aggregation_alpha_grid_size = 9
        if self._predictor_input_dim() not in {4, 5}:
            raise ValueError("role predictor must use four coordinates and optional time")
        if self.selection_mode == "policy":
            self.share_policy_every_contact = True
            self.local_policy_share = True
            self._communication_assumptions = (
                self._build_communication_assumptions()
            )
        self._route_evaluation_X = np.empty((0, 4), dtype=np.float32)
        self._route_evaluation_y = np.empty((0,), dtype=np.float32)
        if self.measurement_trace_in:
            with np.load(self.measurement_trace_in, allow_pickle=False) as replay:
                if "evaluation_route_weighted_X" in replay.files:
                    self._route_evaluation_X = np.asarray(
                        replay["evaluation_route_weighted_X"], dtype=np.float32
                    ).reshape(-1, 4)
                    self._route_evaluation_y = np.asarray(
                        replay["evaluation_route_weighted_y"], dtype=np.float32
                    ).reshape(-1)

    @staticmethod
    def _load_clear_contact_pairs(path: Path) -> dict[int, set[tuple[int, int]]]:
        with np.load(path, allow_pickle=False) as archive:
            steps = np.asarray(archive["step"], dtype=np.int32)
            tx = np.asarray(archive["tx_vehicle_index"], dtype=np.int16)
            rx = np.asarray(archive["rx_vehicle_index"], dtype=np.int16)
            feasible = np.asarray(archive["feasible"], dtype=np.bool_)
            blocked = np.asarray(
                archive["direct_path_blocked"], dtype=np.bool_
            )
        selected = np.flatnonzero(feasible & (~blocked))
        result: dict[int, set[tuple[int, int]]] = {}
        if len(selected):
            order = selected[np.argsort(steps[selected], kind="stable")]
            ordered_steps = steps[order]
            unique, starts = np.unique(ordered_steps, return_index=True)
            ends = list(starts[1:]) + [len(order)]
            for raw_step, start, end in zip(unique, starts, ends):
                indices = order[int(start) : int(end)]
                result[int(raw_step)] = {
                    tuple(sorted((int(tx[index]), int(rx[index]))))
                    for index in indices
                    if int(tx[index]) != int(rx[index])
                }
        return result

    def _contact_links_from_measurements(
        self,
        meas: list[tuple[int, int, int, float]],
    ) -> list[tuple[int, int, int]]:
        links = super()._contact_links_from_measurements(meas)
        step = int(getattr(self, "_current_sumo_step", 0))
        clear_pairs = self._clear_contact_pairs_by_step.get(step, set())
        filtered = [
            (zone, first, second)
            for zone, first, second in links
            if tuple(sorted((int(first), int(second)))) in clear_pairs
        ]
        self._eligible_clear_contacts += len(filtered)
        return filtered

    @staticmethod
    def _artificial_key(
        raw_pair: np.ndarray, step: int
    ) -> tuple[int, int, int, int, int]:
        return temporal_artificial_key(raw_pair, int(step))

    def _reset_artificial_node(self, node_idx: int) -> None:
        node = int(node_idx)
        if not 0 <= node < len(self._artificial_evidence):
            return
        self._artificial_evidence[node] = VehicleEvidence()
        self._real_samples_received[node] = 0
        self._artificial_features[node] = []
        self._artificial_raw[node] = []
        self._artificial_high[node] = []
        self._artificial_steps[node] = []
        self._artificial_split[node] = []
        self._artificial_keys[node] = set()

    def _reset_respawned_node(
        self, node_idx: int, *, generation: int | None = None
    ) -> None:
        super()._reset_respawned_node(node_idx, generation=generation)
        self._reset_artificial_node(int(node_idx))

    def _remove_artificial_contradictions(
        self,
        node_idx: int,
        real_pairs: np.ndarray,
        *,
        step: int,
    ) -> int:
        """Remove only negatives contradicted at the same raw timestamp."""

        node = int(node_idx)
        if not self._artificial_raw[node] or int(real_pairs.shape[0]) == 0:
            return 0
        raw = np.asarray(self._artificial_raw[node], dtype=np.float32)
        keep = temporal_contradiction_keep_mask(
            raw,
            np.asarray(self._artificial_steps[node], dtype=np.int64),
            real_pairs,
            step=int(step),
            minimum_distance_m=self.artificial_support_min_m,
        )
        removed = int(np.count_nonzero(~keep))
        if removed <= 0:
            return 0
        self._artificial_features[node] = [
            value for value, retain in zip(self._artificial_features[node], keep)
            if bool(retain)
        ]
        self._artificial_raw[node] = [
            value for value, retain in zip(self._artificial_raw[node], keep)
            if bool(retain)
        ]
        self._artificial_high[node] = [
            value for value, retain in zip(self._artificial_high[node], keep)
            if bool(retain)
        ]
        self._artificial_steps[node] = [
            value for value, retain in zip(self._artificial_steps[node], keep)
            if bool(retain)
        ]
        self._artificial_split[node] = [
            value for value, retain in zip(self._artificial_split[node], keep)
            if bool(retain)
        ]
        self._artificial_keys[node] = {
            self._artificial_key(
                np.asarray(value, dtype=np.float32), retained_step
            )
            for value, retained_step in zip(
                self._artificial_raw[node], self._artificial_steps[node]
            )
        }
        self._artificial_removed[node] += removed
        return removed

    def _train_predictors_from_current_measurements(
        self,
        *,
        step: int,
        measurements: list[tuple[int, int, int, float]],
    ) -> None:
        real_by_receiver: dict[int, list[np.ndarray]] = {}
        for _zone, tx_idx, rx_idx, value in measurements:
            if self._snr_from_rx_power_dbm(float(value)) < float(
                self.cfg.snr_min_db
            ):
                continue
            tx = int(tx_idx)
            rx = int(rx_idx)
            raw = np.asarray(
                [
                    self.nodes[tx].node.x,
                    self.nodes[tx].node.y,
                    self.nodes[rx].node.x,
                    self.nodes[rx].node.y,
                ],
                dtype=np.float32,
            )
            real_by_receiver.setdefault(rx, []).append(raw)

        for receiver, rows in sorted(real_by_receiver.items()):
            real = np.asarray(rows, dtype=np.float32).reshape(-1, 4)
            self._remove_artificial_contradictions(receiver, real, step=int(step))
            evidence = self._artificial_evidence[receiver]
            for raw in real:
                evidence.observe(raw)
            self._real_samples_received[receiver] += int(real.shape[0])
            received = int(self._real_samples_received[receiver])
            if received < self.artificial_min_real_samples:
                continue
            generation_maturity = float(
                np.clip(
                    (received - self.artificial_min_real_samples)
                    / float(
                        self.artificial_full_weight_samples
                        - self.artificial_min_real_samples
                    ),
                    0.0,
                    1.0,
                )
            )
            desired = int(
                math.floor(
                    self.artificial_ratio * generation_maturity * received
                )
            )
            requested = min(
                max(0, desired - len(self._artificial_raw[receiver])),
                self.artificial_max_new_per_vehicle_step,
            )
            if requested <= 0:
                continue
            candidates = support_distance_candidates(
                evidence,
                receiver_xy=np.asarray(
                    [
                        self.nodes[receiver].node.x,
                        self.nodes[receiver].node.y,
                    ],
                    dtype=np.float32,
                ),
                number=requested,
                rng=np.random.default_rng(
                    int(self.cfg.seed) * 1_000_003
                    + int(step) * 10_007
                    + int(receiver) * 97
                ),
                candidate_pool=self.artificial_candidate_pool,
                minimum_distance_m=self.artificial_support_min_m,
                high_distance_m=self.artificial_support_high_m,
                low_weight=0.0,
                high_weight=1.0,
            )
            added = 0
            for raw, tier, _distance in candidates:
                key = self._artificial_key(raw, int(step))
                if key in self._artificial_keys[receiver]:
                    continue
                feature = self._pair_model_features(
                    (float(raw[0]), float(raw[1])),
                    (float(raw[2]), float(raw[3])),
                    step=int(step),
                    zone=0,
                )
                self._artificial_features[receiver].append(feature)
                self._artificial_raw[receiver].append(raw.tolist())
                self._artificial_high[receiver].append(bool(tier >= 0.5))
                self._artificial_steps[receiver].append(int(step))
                ordinal = int(self._artificial_generated[receiver]) + added
                remainder = ordinal % 10
                self._artificial_split[receiver].append(
                    1 if remainder == 8 else (2 if remainder == 9 else 0)
                )
                self._artificial_keys[receiver].add(key)
                added += 1
            self._artificial_generated[receiver] += added
            self._artificial_shortfall[receiver] += requested - added

        super()._train_predictors_from_current_measurements(
            step=step, measurements=measurements
        )

    def _maturity(self, node_idx: int) -> float:
        received = int(self._real_samples_received[int(node_idx)])
        linear = float(
            np.clip(
                (received - self.artificial_min_real_samples)
                / float(
                    self.artificial_full_weight_samples
                    - self.artificial_min_real_samples
                ),
                0.0,
                1.0,
            )
        )
        return linear ** self.artificial_maturity_exponent

    def _artificial_weights(
        self, node_idx: int, indices: np.ndarray
    ) -> np.ndarray:
        maturity = self._maturity(int(node_idx))
        low = self.artificial_low_weight_start + maturity * (
            self.artificial_low_weight_end - self.artificial_low_weight_start
        )
        high = self.artificial_high_weight_start + maturity * (
            self.artificial_high_weight_end - self.artificial_high_weight_start
        )
        tiers = np.asarray(
            self._artificial_high[int(node_idx)], dtype=np.bool_
        )[indices]
        return np.where(tiers, high, low).astype(np.float32)

    def _artificial_partition_indices(
        self,
        node_idx: int,
        partition: int,
        *,
        real_count: int | None = None,
    ) -> np.ndarray:
        splits = np.asarray(
            self._artificial_split[int(node_idx)], dtype=np.int8
        )
        if self.artificial_validation:
            indices = np.flatnonzero(splits == int(partition))
        elif int(partition) == 0:
            indices = np.arange(len(splits), dtype=np.int64)
        else:
            indices = np.empty((0,), dtype=np.int64)
        if real_count is None or len(indices) == 0:
            return indices
        maximum = int(math.floor(self.artificial_ratio * int(real_count)))
        if maximum <= 0:
            return np.empty((0,), dtype=np.int64)
        if len(indices) <= maximum:
            return indices
        # Preserve coverage across the full private history instead of taking
        # only the earliest or latest artificial rows.
        positions = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
        return indices[positions]

    def _artificial_label_dbm(self) -> float:
        """Physical target for a provisionally unavailable link."""

        return float(self.cfg.noise_floor_dbm)

    def _augment_predictor_training_arrays(
        self,
        *,
        node_idx: int,
        train_X: np.ndarray,
        train_y: np.ndarray,
        train_steps: np.ndarray,
        sample_weights: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        real_count = int(train_X.shape[0])
        real_weights = (
            np.ones(real_count, dtype=np.float32)
            if sample_weights is None
            else np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        )
        indices = self._artificial_partition_indices(int(node_idx), 0)
        if len(indices) == 0:
            return train_X, train_y, train_steps, real_weights
        artificial_weights = self._artificial_weights(int(node_idx), indices)
        artificial_X = np.asarray(
            self._artificial_features[int(node_idx)], dtype=np.float32
        ).reshape(-1, self._predictor_input_dim())[indices]
        artificial_y = np.full(
            (len(indices), 1),
            float(self._artificial_label_dbm()),
            dtype=np.float32,
        )
        artificial_steps = np.asarray(
            self._artificial_steps[int(node_idx)], dtype=np.float32
        )[indices]
        return (
            np.concatenate((train_X, artificial_X), axis=0),
            np.concatenate((train_y, artificial_y), axis=0),
            np.concatenate((train_steps, artificial_steps), axis=0),
            np.concatenate((real_weights, artificial_weights), axis=0),
        )

    def _artificial_validation_context(
        self, subset: ValidationSubset
    ) -> tuple[int, int] | None:
        for node_idx, state in enumerate(self._zone_validation):
            if subset is state.optimization:
                return node_idx, 1
            if subset is state.reward:
                return node_idx, 2
        return None

    def _validation_arrays_with_artificial(
        self,
        subset: ValidationSubset,
        *,
        quality: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if quality <= 0.0 or not subset.features:
            return (
                np.empty((0, self._predictor_input_dim()), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        real_X, real_y = self._subset_arrays(subset)
        context = self._artificial_validation_context(subset)
        if context is None:
            return (
                real_X,
                real_y,
                np.ones((len(real_X),), dtype=np.float32),
            )
        node_idx, partition = context
        indices = self._artificial_partition_indices(
            node_idx, partition, real_count=len(real_X)
        )
        if len(indices) == 0:
            return (
                real_X,
                real_y,
                np.ones((len(real_X),), dtype=np.float32),
            )
        artificial_X = np.asarray(
            self._artificial_features[node_idx], dtype=np.float32
        ).reshape(-1, self._predictor_input_dim())[indices]
        artificial_y = np.full(
            (len(indices), 1),
            float(self._artificial_label_dbm()),
            dtype=np.float32,
        )
        return (
            np.concatenate((real_X, artificial_X), axis=0),
            np.concatenate((real_y, artificial_y), axis=0),
            np.concatenate(
                (
                    np.ones((len(real_X),), dtype=np.float32),
                    self._artificial_weights(node_idx, indices),
                ),
                axis=0,
            ),
        )

    def _prepare_validation_pair(
        self,
        subset_a: ValidationSubset,
        subset_b: ValidationSubset,
        *,
        quality_a: float,
        quality_b: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        arrays: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        counts: list[int] = []
        for subset, quality in (
            (subset_a, float(quality_a)),
            (subset_b, float(quality_b)),
        ):
            X, y_rssi, row_weights = self._validation_arrays_with_artificial(
                subset, quality=quality
            )
            arrays.append(X)
            targets.append(y_rssi)
            weights.append(row_weights)
            counts.append(int(X.shape[0]))
        if sum(counts) == 0:
            return (
                torch.empty(
                    (0, self._predictor_input_dim()),
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                counts[0],
                counts[1],
            )
        features = torch.as_tensor(
            np.concatenate(arrays, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        y_rssi = np.concatenate(targets, axis=0)
        target_loss = torch.as_tensor(
            self._rssi_to_loss_db(y_rssi).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        row_weights = torch.as_tensor(
            np.concatenate(weights, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        return features, target_loss, row_weights, counts[0], counts[1]

    def _pair_mses(
        self,
        state: dict[str, torch.Tensor],
        prepared: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, int, int
        ],
    ) -> tuple[float, float]:
        features, target_loss, weights, count_a, count_b = prepared
        if int(features.shape[0]) == 0:
            return 0.0, 0.0
        self._load_model_state(self._cv_eval_model, dict(state))
        self._cv_eval_model.eval()
        lo = float(self._loss_min_db())
        hi = float(self._loss_max_db())
        with torch.inference_mode():
            prediction = self._cv_eval_model(features).reshape(-1)
            prediction_loss = torch.clamp(
                prediction * (hi - lo) + lo,
                min=lo,
                max=hi,
            )
            squared = torch.square(prediction_loss - target_loss)

            def weighted_mse(start: int, count: int) -> float:
                if count <= 0:
                    return 0.0
                row_loss = squared[start : start + count]
                row_weights = weights[start : start + count]
                denominator = torch.sum(row_weights)
                if float(denominator.item()) <= 0.0:
                    return 0.0
                return float(
                    (torch.sum(row_loss * row_weights) / denominator).item()
                )

            return weighted_mse(0, count_a), weighted_mse(count_a, count_b)

    def _role_experiment_metadata(self) -> dict[str, object]:
        return {
            "format": "merge_test_role_exact_sequential_v1",
            "all_link_trace": str(self.all_link_trace),
            "predictors_per_vehicle": 1,
            "predictor_inputs": [
                "tx_x", "tx_y", "rx_x", "rx_y",
                *(
                    ["original_absolute_timestamp_seconds"]
                    if self._predictor_input_dim() == 5 else []
                ),
            ],
            "predictor_time_encoding": (
                "small learned scalar MLP, trained end-to-end from raw replay "
                "timestamps"
                if self._predictor_input_dim() == 5 else None
            ),
            "role_labels_visible_to_policy": False,
            "local_observations": "all privately received feasible links",
            "communication_contact_rule": (
                "bidirectionally feasible and direct_path_blocked=false"
            ),
            "policy_gossip": (
                "every eligible contact" if self.selection_mode == "policy" else "none"
            ),
            "communication_penalty_beta_db": float(
                self.communication_penalty
            ),
            "maximum_pull_tokens_per_vehicle_step": float(
                self.pull_budget / self.token_window_steps
            ),
            "policy_decision_rule": (
                "pull iff predicted (private-validation RMSE gain - beta) > 0; "
                "epsilon exploration may still probe"
                if self.policy_fixed_trigger_db == 0.0
                else "configured trigger rule"
            ),
            "fedavg_alpha_grid": np.linspace(
                0.0, 1.0, self.aggregation_alpha_grid_size
            ).tolist(),
            "artificial_validation_or_reward": self.artificial_validation,
            "artificial_split": (
                {
                    "predictor_training": 0.8,
                    "fedavg_optimization": 0.1,
                    "pull_reward": 0.1,
                }
                if self.artificial_validation
                else {
                    "predictor_training": 1.0,
                    "fedavg_optimization": 0.0,
                    "pull_reward": 0.0,
                }
            ),
            "artificial_min_real_samples": self.artificial_min_real_samples,
            "artificial_full_weight_samples": self.artificial_full_weight_samples,
            "artificial_ratio": self.artificial_ratio,
            "artificial_support_min_m": self.artificial_support_min_m,
            "artificial_support_high_m": self.artificial_support_high_m,
            "artificial_label_dbm": float(self._artificial_label_dbm()),
            "artificial_identity": (
                "quantized local position pair plus original raw simulation step"
            ),
            "artificial_contradiction_scope": (
                "real feasible evidence revokes a provisional negative only "
                "at the same raw simulation step"
            ),
            "receiver_noise_floor_dbm": float(self.cfg.noise_floor_dbm),
            "real_sample_rx_threshold_dbm": float(
                self.cfg.noise_floor_dbm + self.cfg.snr_min_db
            ),
            "artificial_low_weight_range": [
                self.artificial_low_weight_start,
                self.artificial_low_weight_end,
            ],
            "artificial_high_weight_range": [
                self.artificial_high_weight_start,
                self.artificial_high_weight_end,
            ],
            "artificial_maturity_exponent": self.artificial_maturity_exponent,
            "artificial_max_new_per_vehicle_step": (
                self.artificial_max_new_per_vehicle_step
            ),
        }

    def _compute_fidelity_row(self, step: int) -> dict[str, float | int]:
        row = super()._compute_fidelity_row(step)
        if int(self._route_evaluation_X.shape[0]) == 0:
            return row
        truth = self._route_evaluation_y.astype(np.float64, copy=False)
        reachable = truth >= float(
            self.cfg.noise_floor_dbm + self.cfg.snr_min_db
        )
        active = getattr(self, "_current_node_active", None)
        for mode in self.reward_modes:
            predictions = []
            for node_idx, node in enumerate(self.nodes):
                if active is not None and not bool(active[node_idx]):
                    continue
                prediction, _support = self._predict_variant_fidelity(
                    node, mode, self._route_evaluation_X
                )
                predictions.append(np.asarray(prediction, dtype=np.float64))
            if not predictions:
                continue
            matrix = np.stack(predictions)
            error = matrix - truth[None, :]
            row[f"{mode}_route_weighted_rmse_total"] = float(
                np.sqrt(np.mean(np.square(error)))
            )
            if np.any(reachable):
                row[f"{mode}_route_weighted_feasible_rmse_total"] = float(
                    np.sqrt(np.mean(np.square(error[:, reachable])))
                )
            unavailable = ~reachable
            if np.any(unavailable):
                row[f"{mode}_route_weighted_unavailable_rmse_total"] = float(
                    np.sqrt(np.mean(np.square(error[:, unavailable])))
                )
                excess = np.maximum(
                    matrix[:, unavailable]
                    - float(self.cfg.noise_floor_dbm + self.cfg.snr_min_db),
                    0.0,
                )
                row[
                    f"{mode}_route_weighted_unavailable_censored_rmse_total"
                ] = float(np.sqrt(np.mean(np.square(excess))))
        return row

    def run(self) -> None:
        output = Path(self.cfg.results_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "role_experiment_config.json").write_text(
            json.dumps(self._role_experiment_metadata(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        super().run()

    def _save_outputs(self) -> None:
        super()._save_outputs()
        receivers_by_step: dict[str, list[int]] = {}
        receiver_pull_counts_by_step: dict[str, dict[str, int]] = {}
        for row in self._token_decision_rows:
            if not bool(row.get("attempted", False)):
                continue
            step = str(int(row["step"]))
            receiver = int(row["receiver_idx"])
            receivers_by_step.setdefault(step, []).append(
                receiver
            )
            counts = receiver_pull_counts_by_step.setdefault(step, {})
            key = str(receiver)
            counts[key] = int(counts.get(key, 0)) + 1
        for step in receivers_by_step:
            receivers_by_step[step] = sorted(set(receivers_by_step[step]))
        pulls_by_step = {
            step: int(sum(counts.values()))
            for step, counts in receiver_pull_counts_by_step.items()
        }
        schedule_payload = {
            "format": "exact_policy_realized_pull_schedule_v2",
            "source_selection_mode": str(self.selection_mode),
            "communication_penalty_beta_db": float(
                self.communication_penalty
            ),
            "receivers_by_step": receivers_by_step,
            "receiver_pull_counts_by_step": receiver_pull_counts_by_step,
            "pulls_by_step": pulls_by_step,
            "total_pulls": int(sum(pulls_by_step.values())),
            "total_receiver_step_pulls": int(
                sum(len(rows) for rows in receivers_by_step.values())
            ),
        }
        schedule_path = (
            Path(self.cfg.results_dir) / "realized_pull_schedule.json"
        )
        schedule_path.write_text(
            json.dumps(schedule_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload = {
            **self._role_experiment_metadata(),
            "eligible_clear_contact_pair_steps": int(
                self._eligible_clear_contacts
            ),
            "real_samples_received_by_vehicle": self._real_samples_received.tolist(),
            "artificial_retained_by_vehicle": [
                len(rows) for rows in self._artificial_raw
            ],
            "artificial_generated_by_vehicle": self._artificial_generated.tolist(),
            "artificial_removed_by_vehicle": self._artificial_removed.tolist(),
            "artificial_generation_shortfall_by_vehicle": (
                self._artificial_shortfall.tolist()
            ),
        }
        output = Path(self.cfg.results_dir) / "role_experiment_summary.json"
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
