"""Cell-native support variants for the Luxembourg dissemination benchmark."""

from __future__ import annotations

from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.capsule_greedy.run_capsule_greedy import average_state_dicts
from experiments.place_wallis_benchmark.cell_grid_support import (
    CELL_GRID_SUPPORT_SEMANTICS,
    add_segments_to_grid,
    link_confidence_profiles,
    link_support_profile,
    link_support_profiles,
    sparse_grid_payload_bytes,
)
from experiments.place_wallis_benchmark import run_support_expert_bank as core
from experiments.support_acquisition_pretraining.pretrain_spatial_grid_gain import (
    PROFILE_KIND_CELL_TRAVERSAL,
)


class CellGridExpertBankSimulation(core.LearnedAcquisitionExpertBankSimulation):
    """Learned-acquisition expert bank with cell-crossing binary support."""

    checkpoint_format = "place_wallis_cell_grid_expert_bank_metrics_v1"

    def __init__(
        self,
        cfg,
        *,
        cell_confidence_mode: str = "binary",
        cell_minimum_intensity: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self._local_cell_profiles: list[np.ndarray] = []
        self._cell_profiles_by_key: dict[core.ExpertKey, np.ndarray] = {}
        self._cell_confidence_mode = str(cell_confidence_mode)
        self._cell_minimum_intensity = float(cell_minimum_intensity)
        if self._cell_minimum_intensity <= 0.0:
            raise ValueError("minimum cell intensity must be positive")
        if self._cell_confidence_mode not in {
            "binary", "path-ratio", "global-ratio"
        }:
            raise ValueError(
                f"unknown cell confidence mode {self._cell_confidence_mode!r}"
            )
        super().__init__(cfg, **kwargs)
        payload = torch.load(
            self.acquisition_bundle,
            map_location="cpu",
            weights_only=False,
        )
        semantics = str(payload.get("support_profile_semantics", ""))
        if semantics != PROFILE_KIND_CELL_TRAVERSAL:
            raise ValueError(
                "cell-grid support requires an acquisition bundle pretrained "
                f"with {PROFILE_KIND_CELL_TRAVERSAL!r}, got {semantics!r}"
            )
        self._local_cell_profiles = [
            np.zeros(
                (self._gain_grid_resolution, self._gain_grid_resolution),
                dtype=np.float32,
            )
            for _ in range(int(cfg.num_nodes))
        ]
        self._communication_assumptions.update({
            "support_representation": CELL_GRID_SUPPORT_SEMANTICS,
            "support_update": (
                "every positive link increments each traversed cell"
            ),
            "support_gate": (
                "every cell traversed by the query must reach the minimum intensity"
            ),
            "minimum_cell_intensity": self._cell_minimum_intensity,
            "prediction_confidence": self._cell_confidence_mode,
            "support_aggregation": "pointwise maximum",
            "support_intensity_use": "prediction gate, acquisition, and redundancy",
            "support_payload": (
                "exact sparse uint32 cell-index/count pairs or dense uint32 grid"
            ),
        })

    def _result_method_id(self) -> str:
        suffix = f"_{self.method_tag}" if self.method_tag else ""
        return (
            "cell_grid_expert_bank_learned_acquisition"
            f"_step{self._acquisition_checkpoint_step}{suffix}"
        )

    def _result_method_name(self) -> str:
        suffix = f", {self.method_tag}" if self.method_tag else ""
        return (
            "Cell-grid learned-acquisition expert bank "
            f"(pretrained step {self._acquisition_checkpoint_step}{suffix})"
        )

    def _reset_aux_node(
        self,
        i: int,
        *,
        old_az: int | None = None,
        new_az: int | None = None,
    ) -> None:
        super()._reset_aux_node(i, old_az=old_az, new_az=new_az)
        index = int(i)
        if index < len(self._local_cell_profiles):
            self._local_cell_profiles[index].fill(0.0)

    def _register_cell_record(self, receiver: int) -> core.ExpertKey:
        index = int(receiver)
        self._local_versions[index] += 1
        key = (
            index,
            int(self._expert_incarnations[index]),
            int(self._local_versions[index]),
        )
        profile = np.asarray(
            self._local_cell_profiles[index], dtype=np.float32
        ).copy()
        profile.setflags(write=False)
        self._cell_profiles_by_key[key] = profile
        self._expert_registry[key] = core.ExpertRecord(
            key=key,
            experience=int(self.greedy_m_samples[index]),
            capsules=(),
            model_state=core._cpu_state(self.greedy_models[index]),
        )
        return key

    def _refresh_local_expert(self, receiver: int) -> None:
        index = int(receiver)
        previous = list(self._expert_banks[index])
        previous_profile = self._grid_bank_profile(previous)
        key = self._register_cell_record(index)
        self._expert_banks[index] = self._insert_candidate(
            index, previous, key
        )
        if key in self._expert_banks[index]:
            if key not in previous:
                self._cache_grid_bank_profile(
                    self._expert_banks[index],
                    np.maximum(previous_profile, self._grid_profile_for_key(key)),
                )

    def _grid_profile_for_key(self, key: core.ExpertKey) -> np.ndarray:
        profile = self._cell_profiles_by_key.get(key)
        if profile is None:
            raise KeyError(f"missing cell-grid support for expert {key}")
        return profile.reshape(-1)

    def _prune_registry(self) -> None:
        super()._prune_registry()
        live = set(self._expert_registry)
        self._cell_profiles_by_key = {
            key: profile
            for key, profile in self._cell_profiles_by_key.items()
            if key in live
        }

    def _expert_support_profile(
        self, key: core.ExpertKey, query_m: np.ndarray
    ) -> np.ndarray:
        normalized = np.asarray(query_m, dtype=np.float64) / float(
            self.cfg.map_size
        )
        return link_support_profile(
            self._grid_profile_for_key(key),
            normalized,
            binary=True,
            minimum_intensity=self._cell_minimum_intensity,
        )

    def _expert_support_profiles(
        self, keys: list[core.ExpertKey], query_m: np.ndarray
    ) -> dict[core.ExpertKey, np.ndarray]:
        if not keys:
            return {}
        normalized = np.asarray(query_m, dtype=np.float64) / float(
            self.cfg.map_size
        )
        values = link_confidence_profiles(
            np.stack([self._grid_profile_for_key(key) for key in keys]),
            normalized,
            mode=self._cell_confidence_mode,
            minimum_intensity=self._cell_minimum_intensity,
        )
        return {key: values[index] for index, key in enumerate(keys)}

    def _expert_normalized_predictions(
        self,
        template: torch.nn.Module,
        record: core.ExpertRecord,
        xt: torch.Tensor,
        routing_profile: np.ndarray,
    ) -> np.ndarray:
        template.load_state_dict(record.model_state)
        template.eval()
        floor = float(template.floor_prior_norm)
        confidence = np.clip(
            np.asarray(routing_profile, dtype=np.float32), 0.0, 1.0
        )
        prediction = np.full(len(confidence), floor, dtype=np.float32)
        selected = np.flatnonzero(confidence > 0.0)
        if len(selected):
            with torch.no_grad():
                rows = torch.as_tensor(
                    selected, dtype=torch.long, device=xt.device
                )
                raw = (
                    template.base(xt.index_select(0, rows))
                    .detach().cpu().numpy().reshape(-1)
                )
                prediction[selected] += confidence[selected] * (
                    raw - floor
                )
        return prediction

    def _prediction_choice(
        self,
        *,
        step: int,
        receiver: int,
        keys: list[core.ExpertKey],
        routing: np.ndarray,
    ) -> np.ndarray:
        del step, receiver
        experience = np.asarray([
            max(1, int(self._expert_registry[key].experience)) for key in keys
        ], dtype=np.float64)
        score = np.where(routing > 0.0, experience[None, :], -1.0)
        return np.argmax(score, axis=1)

    def _expert_support_unit_count(self, key: core.ExpertKey) -> int:
        return int(np.count_nonzero(self._grid_profile_for_key(key) > 0.0))

    def _pulled_support_payload_values(self, key: core.ExpertKey) -> int:
        payload_bytes = sparse_grid_payload_bytes(
            self._grid_profile_for_key(key)
        )
        return int(math.ceil(payload_bytes / 4.0))

    def _train_staged_local_samples(self, step: int) -> None:
        measurements = self._staged_measurements or []
        rows_by_receiver: dict[
            int, list[tuple[list[float], float, np.ndarray]]
        ] = {}
        self._meas_per_node = {}
        for zone, tx_idx, rx_idx, value in measurements:
            tx_node = self.nodes[int(tx_idx)].node
            rx_node = self.nodes[int(rx_idx)].node
            features = self._pair_model_features(
                (tx_node.x, tx_node.y),
                (rx_node.x, rx_node.y),
                step=step,
                zone=int(zone),
            )
            segment = np.asarray(
                [[tx_node.x, tx_node.y], [rx_node.x, rx_node.y]],
                dtype=np.float64,
            ) / float(self.cfg.map_size)
            rows_by_receiver.setdefault(int(rx_idx), []).append(
                (features, float(value), segment)
            )
        active = {
            index
            for index in range(int(self.cfg.num_nodes))
            if bool(self._current_node_active[index])
        }
        receivers = sorted(
            set(rows_by_receiver)
            | (active & set(self._replay_buffers))
            | self._additional_training_receivers(active)
        )
        for receiver in receivers:
            rows = rows_by_receiver.get(receiver, [])
            if rows:
                add_segments_to_grid(
                    self._local_cell_profiles[receiver],
                    np.asarray([row[2] for row in rows], dtype=np.float64),
                )
            X = np.asarray(
                [row[0] for row in rows], dtype=np.float32
            ).reshape(-1, 4)
            y = np.asarray(
                [row[1] for row in rows], dtype=np.float32
            ).reshape(-1, 1)
            replay = self._replay_buffers.get(receiver)
            rng = np.random.default_rng(np.random.SeedSequence([
                int(self.cfg.seed), int(step), int(receiver)
            ]))
            updated = False
            full_dataset = self.training_params.full_dataset_epochs > 0
            if rows:
                self._train_array(
                    receiver,
                    X,
                    y,
                    epochs=self.training_params.new_data_epochs,
                    rng=rng,
                )
                updated = True
            if full_dataset and rows:
                if replay is None:
                    replay = core.ReplayBuffer(
                        self.training_params.replay_capacity, 4
                    )
                    self._replay_buffers[receiver] = replay
                replay.add(X, y)
                self.greedy_m_samples[receiver] += int(len(rows))
                self.greedy_n_samples[receiver] = self.greedy_m_samples[receiver]
            if replay is not None and replay.size > 0:
                if full_dataset:
                    replay_X, replay_y = replay.all_data()
                    self._train_array(
                        receiver,
                        replay_X,
                        replay_y,
                        epochs=self.training_params.full_dataset_epochs,
                        rng=rng,
                    )
                else:
                    recent_start = (
                        self.training_params.replay_batches
                        - self.training_params.recent_replay_batches
                    )
                    for batch_index in range(self.training_params.replay_batches):
                        replay_X, replay_y = replay.sample(
                            rng,
                            int(self.cfg.local_batch_size),
                            recent_window=(
                                self.training_params.recent_window
                                if batch_index >= recent_start else None
                            ),
                        )
                        self._train_array(
                            receiver, replay_X, replay_y, epochs=1, rng=rng
                        )
                updated = True
            if rows and not full_dataset:
                if replay is None:
                    replay = core.ReplayBuffer(
                        self.training_params.replay_capacity, 4
                    )
                    self._replay_buffers[receiver] = replay
                replay.add(X, y)
                self.greedy_m_samples[receiver] += int(len(rows))
                self.greedy_n_samples[receiver] = self.greedy_m_samples[receiver]
            if self._train_additional_receiver_data(step, receiver, rng):
                updated = True
            if updated:
                self._refresh_local_expert(receiver)
        self._staged_measurements = None
        self._prune_registry()

    def _save_checkpoint(self, step: int) -> None:
        super()._save_checkpoint(step)
        output = Path(self.cfg.results_dir)
        status_path = output / "checkpoint_status.json"
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        status["cell_grid_support"] = {
            "semantics": CELL_GRID_SUPPORT_SEMANTICS,
            "resolution": int(self._gain_grid_resolution),
            "gate": "all query-traversed cells reach the minimum intensity",
            "minimum_cell_intensity": self._cell_minimum_intensity,
            "prediction_confidence": self._cell_confidence_mode,
            "intensity_aggregation": "pointwise maximum",
        }
        core.atomic_json(status_path, status)
        metrics_path = output / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        metrics["support"] = status["cell_grid_support"]
        core.atomic_json(metrics_path, metrics)


class CellGridLocalOnlySimulation(core.LocalOnlySupportSimulation):
    """One persistent traversed-cell-grid predictor per vehicle, without sharing."""

    checkpoint_format = "place_wallis_cell_grid_local_only_metrics_v1"

    def __init__(
        self,
        cfg,
        *,
        cell_confidence_mode: str = "binary",
        cell_minimum_intensity: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self._gain_grid_resolution = 300
        self._local_cell_profiles: list[np.ndarray] = []
        self._cell_profiles_by_key: dict[core.ExpertKey, np.ndarray] = {}
        self._cell_confidence_mode = str(cell_confidence_mode)
        self._cell_minimum_intensity = float(cell_minimum_intensity)
        if self._cell_minimum_intensity <= 0.0:
            raise ValueError("minimum cell intensity must be positive")
        if self._cell_confidence_mode != "binary":
            raise ValueError("the local-only cell-grid baseline uses binary support")
        super().__init__(cfg, **kwargs)
        self._local_cell_profiles = [
            np.zeros(
                (self._gain_grid_resolution, self._gain_grid_resolution),
                dtype=np.float32,
            )
            for _ in range(int(cfg.num_nodes))
        ]
        self._communication_assumptions.update({
            "method": "local-only traversed-cell support baseline",
            "support_representation": CELL_GRID_SUPPORT_SEMANTICS,
            "support_update": "every positive link increments each traversed cell",
            "support_gate": (
                "every cell traversed by the query reaches the minimum intensity"
            ),
            "minimum_cell_intensity": self._cell_minimum_intensity,
            "prediction_confidence": "binary",
            "support_resolution": int(self._gain_grid_resolution),
            "support_shared": False,
            "raw_samples_shared": False,
            "model_parameters_shared": False,
        })

    def _result_method_id(self) -> str:
        suffix = f"_{self.method_tag}" if self.method_tag else ""
        return f"cell_grid_local_only{suffix}"

    def _result_method_name(self) -> str:
        suffix = f" ({self.method_tag})" if self.method_tag else ""
        return f"Local-only cell-grid support-gated MLP{suffix}"

    def _reset_aux_node(
        self,
        i: int,
        *,
        old_az: int | None = None,
        new_az: int | None = None,
    ) -> None:
        core.SupportExpertBankSimulation._reset_aux_node(
            self, i, old_az=old_az, new_az=new_az
        )
        index = int(i)
        if index < len(self._local_cell_profiles):
            self._local_cell_profiles[index].fill(0.0)

    _register_cell_record = CellGridExpertBankSimulation._register_cell_record
    _grid_profile_for_key = CellGridExpertBankSimulation._grid_profile_for_key
    _expert_support_profile = CellGridExpertBankSimulation._expert_support_profile
    _expert_support_profiles = CellGridExpertBankSimulation._expert_support_profiles
    _expert_normalized_predictions = (
        CellGridExpertBankSimulation._expert_normalized_predictions
    )
    _expert_support_unit_count = (
        CellGridExpertBankSimulation._expert_support_unit_count
    )
    _pulled_support_payload_values = (
        CellGridExpertBankSimulation._pulled_support_payload_values
    )
    _train_staged_local_samples = (
        CellGridExpertBankSimulation._train_staged_local_samples
    )

    def _refresh_local_expert(self, receiver: int) -> None:
        index = int(receiver)
        key = self._register_cell_record(index)
        self._expert_banks[index] = [key]

    def _prune_registry(self) -> None:
        core.SupportExpertBankSimulation._prune_registry(self)
        live = set(self._expert_registry)
        self._cell_profiles_by_key = {
            key: profile
            for key, profile in self._cell_profiles_by_key.items()
            if key in live
        }

    def _save_checkpoint(self, step: int) -> None:
        super()._save_checkpoint(step)
        output = Path(self.cfg.results_dir)
        support = {
            "semantics": CELL_GRID_SUPPORT_SEMANTICS,
            "resolution": int(self._gain_grid_resolution),
            "gate": "all query-traversed cells reach the minimum intensity",
            "minimum_cell_intensity": self._cell_minimum_intensity,
            "prediction_confidence": "binary",
            "intensity_aggregation": "local increments only",
        }
        status_path = output / "checkpoint_status.json"
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        status["cell_grid_support"] = support
        core.atomic_json(status_path, status)
        metrics_path = output / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        metrics["support"] = support
        metrics["method"] = {
            "id": self._result_method_id(),
            "name": self._result_method_name(),
            "model": "one persistent 4-64-64-1 cell-grid-gated MLP per vehicle",
        }
        core.atomic_json(metrics_path, metrics)


class CellGridCentralSimulation(core.CentralSupportSimulation):
    """One central predictor with the same traversed-cell support gate."""

    checkpoint_format = "place_wallis_cell_grid_central_metrics_v1"

    def __init__(
        self,
        cfg,
        *,
        cell_confidence_mode: str = "binary",
        cell_minimum_intensity: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self._gain_grid_resolution = 300
        self._cell_confidence_mode = str(cell_confidence_mode)
        self._cell_minimum_intensity = float(cell_minimum_intensity)
        if self._cell_minimum_intensity <= 0.0:
            raise ValueError("minimum cell intensity must be positive")
        if self._cell_confidence_mode != "binary":
            raise ValueError("the central cell-grid baseline uses binary support")
        self._cell_profiles_by_key: dict[core.ExpertKey, np.ndarray] = {}
        super().__init__(cfg, **kwargs)
        self._central_cell_profile = np.zeros(
            (self._gain_grid_resolution, self._gain_grid_resolution),
            dtype=np.float32,
        )
        self._communication_assumptions.update({
            "method": "central traversed-cell support baseline",
            "central_support": "all feasible measurement traversed cells",
            "support_representation": CELL_GRID_SUPPORT_SEMANTICS,
            "support_gate": "every cell traversed by the query reaches the minimum intensity",
            "minimum_cell_intensity": self._cell_minimum_intensity,
            "support_resolution": int(self._gain_grid_resolution),
        })

    def _result_method_id(self) -> str:
        suffix = f"_{self.method_tag}" if self.method_tag else ""
        return f"cell_grid_central{suffix}"

    def _result_method_name(self) -> str:
        suffix = f" ({self.method_tag})" if self.method_tag else ""
        return f"Central cell-grid support-gated MLP{suffix}"

    def _update_central_support(
        self, rows: list[tuple[list[float], float, np.ndarray]]
    ) -> None:
        segments = np.asarray([row[2] for row in rows], dtype=np.float64)
        add_segments_to_grid(
            self._central_cell_profile,
            segments / float(self.cfg.map_size),
        )

    def _refresh_central_record(self) -> None:
        self._central_version += 1
        key = (0, 0, int(self._central_version))
        profile = np.asarray(
            self._central_cell_profile, dtype=np.float32
        ).copy()
        profile.setflags(write=False)
        self._cell_profiles_by_key = {key: profile}
        self._expert_registry = {
            key: core.ExpertRecord(
                key=key,
                experience=int(self._central_experience),
                capsules=(),
                model_state=core._cpu_state(
                    self.greedy_models[self._central_model_index]
                ),
            )
        }
        self._support_profiles.clear()

    def _grid_profile_for_key(self, key: core.ExpertKey) -> np.ndarray:
        return self._cell_profiles_by_key[key].reshape(-1)

    _expert_support_profile = CellGridExpertBankSimulation._expert_support_profile
    _expert_support_profiles = CellGridExpertBankSimulation._expert_support_profiles
    _expert_normalized_predictions = (
        CellGridExpertBankSimulation._expert_normalized_predictions
    )
    _expert_support_unit_count = (
        CellGridExpertBankSimulation._expert_support_unit_count
    )
    _pulled_support_payload_values = (
        CellGridExpertBankSimulation._pulled_support_payload_values
    )

    def _save_checkpoint(self, step: int) -> None:
        super()._save_checkpoint(step)
        output = Path(self.cfg.results_dir)
        status_path = output / "checkpoint_status.json"
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        status["cell_grid_support"] = {
            "semantics": CELL_GRID_SUPPORT_SEMANTICS,
            "resolution": int(self._gain_grid_resolution),
            "gate": "all query-traversed cells reach the minimum intensity",
            "minimum_cell_intensity": self._cell_minimum_intensity,
            "prediction_confidence": "binary",
        }
        core.atomic_json(status_path, status)
        metrics_path = output / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        metrics["support"] = status["cell_grid_support"]
        metrics["method"] = {
            "id": self._result_method_id(),
            "name": self._result_method_name(),
            "model": "one map-wide 4-64-64-1 cell-grid-gated MLP",
        }
        core.atomic_json(metrics_path, metrics)


class CellGridWeightedSingleSimulation(CellGridExpertBankSimulation):
    """One model per vehicle with configurable ranked weighted pulls."""

    checkpoint_format = "place_wallis_cell_grid_weighted_single_metrics_v1"

    def __init__(
        self,
        cfg,
        *,
        weighted_pulls_per_receiver_step: int = 1,
        weighted_pull_interval_steps: int = 1,
        weighted_pull_schedule_anchor: str = "entry",
        weighted_acquisition: bool = False,
        weighted_acquisition_fixed_budget: bool = False,
        weighted_selection: str = "experience",
        **kwargs: Any,
    ) -> None:
        self._weighted_seen_models: list[set[core.ExpertKey]] = []
        self._weighted_previous_active = np.zeros(int(cfg.num_nodes), dtype=np.bool_)
        self._weighted_next_pull_step = np.zeros(int(cfg.num_nodes), dtype=np.int64)
        self._weighted_pull_count = 0
        self._weighted_pull_interval_steps = int(
            weighted_pull_interval_steps
        )
        self._weighted_pull_schedule_anchor = str(
            weighted_pull_schedule_anchor
        )
        self._weighted_pulls_per_receiver_step = int(
            weighted_pulls_per_receiver_step
        )
        self._weighted_acquisition_fixed_budget = bool(
            weighted_acquisition_fixed_budget
        )
        self._weighted_selection = str(weighted_selection)
        if self._weighted_selection not in {
            "experience",
            "grid-intensity",
            "random-grid-intensity",
        }:
            raise ValueError(
                f"unknown weighted selection: {self._weighted_selection}"
            )
        self._weighted_acquisition = bool(weighted_acquisition)
        if self._weighted_pulls_per_receiver_step < 0:
            raise ValueError(
                "weighted pulls per receiver-step must be non-negative"
            )
        if self._weighted_pull_interval_steps <= 0:
            raise ValueError(
                "weighted pull interval must be positive"
            )
        if self._weighted_pull_schedule_anchor not in {"entry", "global"}:
            raise ValueError(
                "weighted pull schedule anchor must be entry or global"
            )
        if (
            self._weighted_acquisition_fixed_budget
            and not self._weighted_acquisition
        ):
            raise ValueError(
                "fixed-budget acquisition requires acquisition ranking"
            )
        if (
            self._weighted_acquisition
            and self._weighted_selection != "experience"
        ):
            raise ValueError(
                "weighted selection applies only without acquisition ranking"
            )
        super().__init__(cfg, **kwargs)
        self._weighted_seen_models = [
            set() for _ in range(int(cfg.num_nodes))
        ]
        self._communication_assumptions.update({
            "method": "cell-grid sample-count-weighted single model",
            "acquisition_model_used": self._weighted_acquisition,
            "pull_rule": (
                "rank candidates by predicted relative grid gain, subject "
                "only to the fixed pull budget"
                if self._weighted_acquisition_fixed_budget
                else (
                "iteratively pull the highest predicted relative grid gain "
                "above kappa"
                if self._weighted_acquisition
                else (
                    "highest-sample-count unseen neighbour models, then coverage; "
                    "zero pull budget means all available models"
                )
                )
            ),
            "post_pull_rejection": False,
            "model_merge": "sample-count-weighted parameter average",
            "merged_sample_count": "maximum to avoid gossip double counting",
            "optimizer_after_merge": "Adam state reset",
            "models_per_vehicle": 1,
            "pull_interval_steps": int(self._weighted_pull_interval_steps),
            "pull_schedule_anchor": (
                "first active step after vehicle entry"
                if self._weighted_pull_schedule_anchor == "entry"
                else "global simulation steps divisible by the pull interval"
            ),
        })
        if self._weighted_selection in {
            "grid-intensity",
            "random-grid-intensity",
        }:
            self._communication_assumptions.update({
                "pull_rule": (
                    "uniform random unseen neighbour models within the fixed budget"
                    if self._weighted_selection == "random-grid-intensity"
                    else "highest advertised summed grid intensity"
                ),
                "model_merge": "summed-grid-intensity-weighted parameter average",
                "advertisement": "model key and one summed-intensity scalar",
            })

    def _result_method_id(self) -> str:
        suffix = f"_{self.method_tag}" if self.method_tag else ""
        mode = "acquisition_" if self._weighted_acquisition else ""
        return f"cell_grid_{mode}weighted_single_model{suffix}"

    def _result_method_name(self) -> str:
        suffix = f", {self.method_tag}" if self.method_tag else ""
        mode = "acquisition-driven " if self._weighted_acquisition else ""
        return (
            f"Cell-grid {mode}weighted single model "
            f"({suffix.lstrip(', ')})"
        )

    def _refresh_local_expert(self, receiver: int) -> None:
        index = int(receiver)
        key = self._register_cell_record(index)
        self._expert_banks[index] = [key]

    def _reset_aux_node(
        self,
        i: int,
        *,
        old_az: int | None = None,
        new_az: int | None = None,
    ) -> None:
        super()._reset_aux_node(i, old_az=old_az, new_az=new_az)
        index = int(i)
        if index < len(self._weighted_previous_active):
            self._weighted_previous_active[index] = False
            self._weighted_next_pull_step[index] = int(
                getattr(self, "_current_sumo_step", 0)
            )
        if index < len(self._weighted_seen_models):
            self._weighted_seen_models[index].clear()

    def _advertisement_scalar_values_for_key(
        self, key: core.ExpertKey
    ) -> int:
        if self._quantized_patch_advertisements:
            payload_bytes = (
                self._patch_count * self._patch_codebook_groups + 4
            )
            return int((payload_bytes + 3) // 4)
        if not self._sparse_patch_advertisements:
            return int(self._advertisement_latent_dim)
        encoding = self._advertisement_for_key(key).encoding
        codes = encoding[:-1].reshape(
            self._patch_count, self._patch_latent_channels
        )
        active = int(np.count_nonzero(np.any(codes != 0.0, axis=1)))
        return int(active * (self._patch_latent_channels + 1) + 1)

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        del zone_nodes
        step = int(getattr(self, "_current_sumo_step", 0))
        self._restore_logs()
        active = np.asarray(self._current_node_active, dtype=np.bool_)
        entered = active & ~self._weighted_previous_active
        if self._weighted_pull_schedule_anchor == "global":
            due_mask = active & (
                step % self._weighted_pull_interval_steps == 0
            )
        else:
            self._weighted_next_pull_step[entered] = step
            due_mask = active & (step >= self._weighted_next_pull_step)
        due_receivers = set(np.flatnonzero(due_mask).tolist())
        if self._weighted_pull_schedule_anchor == "entry":
            self._weighted_next_pull_step[due_mask] = (
                step + self._weighted_pull_interval_steps
            )
        self._weighted_previous_active = active.copy()
        links = sorted({
            (int(zone), min(int(a), int(b)), max(int(a), int(b)))
            for zone, a, b in (contact_links or [])
            if int(a) != int(b)
        })
        neighbours: dict[int, list[int]] = {}
        for _zone, left, right in links:
            neighbours.setdefault(left, []).append(right)
            neighbours.setdefault(right, []).append(left)
        pre_banks = {
            index: list(self._expert_banks[index]) for index in neighbours
        }
        if self._weighted_acquisition:
            self._ensure_grid_advertisements([
                key
                for bank in pre_banks.values()
                for key in bank
            ])

        coverage_cache: dict[core.ExpertKey, int] = {}

        def coverage(key: core.ExpertKey) -> int:
            value = coverage_cache.get(key)
            if value is None:
                value = int(np.count_nonzero(
                    self._grid_profile_for_key(key) > 0.0
                ))
                coverage_cache[key] = value
            return value

        intensity_cache: dict[core.ExpertKey, float] = {}

        def intensity(key: core.ExpertKey) -> float:
            value = intensity_cache.get(key)
            if value is None:
                value = float(np.sum(
                    self._grid_profile_for_key(key), dtype=np.float64
                ))
                intensity_cache[key] = value
            return value

        next_banks: dict[int, list[core.ExpertKey]] = {}
        pulls = manifest_records = support_values = parameter_values = 0
        advertisement_values = 0
        for receiver in sorted(neighbours):
            current_keys = [
                key for key in pre_banks[receiver]
                if key in self._expert_registry
            ]
            current = current_keys[-1] if current_keys else None
            if receiver not in due_receivers:
                next_banks[receiver] = current_keys[-1:] if current_keys else []
                continue
            cached_ads = self._receiver_advertisement_cache[receiver]
            offers: list[core.ExpertKey] = []
            for sender in sorted(neighbours[receiver]):
                sender_keys = [
                    key for key in pre_banks.get(sender, [])
                    if key in self._expert_registry
                ]
                manifest_records += len(sender_keys)
                for key in sender_keys:
                    if (
                        (
                            self._weighted_acquisition
                            or self._weighted_selection in {
                                "grid-intensity",
                                "random-grid-intensity",
                            }
                        )
                        and key not in cached_ads
                    ):
                        advertisement_values += (
                            self._advertisement_scalar_values_for_key(key)
                            if self._weighted_acquisition else 1
                        )
                        cached_ads.add(key)
                    if (
                        key != current
                        and key not in self._weighted_seen_models[receiver]
                    ):
                        offers.append(key)
            offers = list(dict.fromkeys(offers))
            if not offers:
                next_banks[receiver] = current_keys[-1:] if current_keys else []
                continue
            if self._weighted_acquisition:
                remaining = list(offers)
                working = current_keys[-1:] if current_keys else []
                selected: list[core.ExpertKey] = []
                while remaining:
                    if (
                        self._weighted_acquisition_fixed_budget
                        and self._weighted_pulls_per_receiver_step > 0
                        and len(selected)
                        >= self._weighted_pulls_per_receiver_step
                    ):
                        break
                    if self._weighted_acquisition_fixed_budget:
                        candidate = self._union_provider_candidate(
                            working,
                            remaining,
                            minimum_relative_gain=float("-inf"),
                        )
                        if candidate is None and remaining:
                            candidate = max(
                                remaining,
                                key=lambda key: (
                                    int(self._expert_registry[key].experience),
                                    key,
                                ),
                            )
                    else:
                        candidate = self._provider_candidate(
                            working, remaining
                        )
                    if candidate is None:
                        break
                    selected.append(candidate)
                    working.append(candidate)
                    remaining = [
                        key for key in remaining if key != candidate
                    ]
            elif self._weighted_selection == "random-grid-intensity":
                budget = self._weighted_pulls_per_receiver_step
                count = len(offers) if budget == 0 else min(budget, len(offers))
                rng = np.random.default_rng(np.random.SeedSequence([
                    int(self.cfg.seed),
                    int(step),
                    int(receiver),
                    0x524E4457,
                ]))
                positions = rng.choice(len(offers), size=count, replace=False)
                selected = [offers[int(position)] for position in positions]
            else:
                ranked = sorted(
                    offers,
                    key=lambda key: (
                        intensity(key)
                        if self._weighted_selection == "grid-intensity"
                        else float(self._expert_registry[key].experience),
                        int(self._expert_registry[key].experience),
                        coverage(key),
                        key,
                    ),
                    reverse=True,
                )
                budget = self._weighted_pulls_per_receiver_step
                selected = ranked if budget == 0 else ranked[:budget]
            if not selected:
                next_banks[receiver] = current_keys[-1:] if current_keys else []
                continue
            records = [self._expert_registry[key] for key in selected]
            states = [record.model_state for record in records]
            profiles = [self._grid_profile_for_key(key) for key in selected]
            experiences = [int(record.experience) for record in records]
            if self._weighted_selection in {
                "grid-intensity",
                "random-grid-intensity",
            }:
                weights = [
                    max(1, int(np.sum(profile, dtype=np.float64)))
                    for profile in profiles
                ]
            else:
                weights = [max(1, value) for value in experiences]
            if current is not None:
                current_record = self._expert_registry[current]
                current_profile = self._grid_profile_for_key(current)
                states.insert(0, current_record.model_state)
                profiles.insert(0, current_profile)
                experiences.insert(0, int(current_record.experience))
                weights.insert(
                    0,
                    max(1, int(np.sum(current_profile, dtype=np.float64)))
                    if self._weighted_selection in {
                        "grid-intensity",
                        "random-grid-intensity",
                    }
                    else max(1, int(current_record.experience)),
                )
            merged_state = (
                copy.deepcopy(states[0])
                if len(states) == 1
                else average_state_dicts(states, weights)
            )
            merged_profile = np.maximum.reduce(profiles).reshape(
                self._gain_grid_resolution, self._gain_grid_resolution
            ).copy()
            merged_experience = max(experiences)
            self.greedy_models[receiver].load_state_dict(merged_state)
            self.greedy_opts[receiver].state.clear()
            self._local_cell_profiles[receiver] = merged_profile
            self.greedy_m_samples[receiver] = merged_experience
            self.greedy_n_samples[receiver] = merged_experience
            key = self._register_cell_record(receiver)
            next_banks[receiver] = [key]
            self._weighted_seen_models[receiver].update(selected)
            pulls += len(selected)
            support_values += sum(
                self._pulled_support_payload_values(key) for key in selected
            )
            parameter_values += sum(
                sum(int(value.numel()) for value in record.model_state.values())
                for record in records
            )
        for receiver, bank in next_banks.items():
            self._expert_banks[receiver] = bank
        self._weighted_pull_count += pulls
        self._model_transfers += pulls
        self._manifest_records += manifest_records
        if self._weighted_acquisition:
            self._learned_pull_attempts += int(pulls)
            self._learned_pull_accepts += int(pulls)
            self._advertisement_scalar_values += int(advertisement_values)
            self._pulled_support_scalar_values += int(support_values)
            self._pulled_model_parameter_values += int(parameter_values)

        self._network_step_stats.update({
            "weighted_single_model_pulls": int(pulls),
            "expert_bank_model_messages": int(pulls),
            "expert_bank_manifest_records": int(manifest_records),
            "expert_bank_advertisement_scalar_values": int(advertisement_values),
            "expert_bank_advertisement_bytes": int(4 * advertisement_values),
            "capsule_scalar_values_sent": int(support_values),
            "capsule_payload_bytes": int(4 * support_values),
            "model_parameter_values_pulled": int(parameter_values),
            "model_payload_bytes": int(4 * parameter_values),
        })
        self._train_staged_local_samples(step)
        return int(pulls)

    def _save_checkpoint(self, step: int) -> None:
        super()._save_checkpoint(step)
        output = Path(self.cfg.results_dir)
        status_path = output / "checkpoint_status.json"
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        if not self._weighted_acquisition:
            status.pop("learned_acquisition", None)
        status["bank_capacity"] = 1
        status["weighted_single_model"] = {
            "pulls": int(self._weighted_pull_count),
            "selection": (
                "pretrained predicted relative grid coverage/intensity gain"
                if self._weighted_acquisition
                else "advertised sample count then covered-cell count"
            ),
            "pulls_per_receiver_per_step": (
                "until no predicted gain exceeds kappa"
                if self._weighted_acquisition
                else ("all available"
                    if self._weighted_pulls_per_receiver_step == 0
                    else int(self._weighted_pulls_per_receiver_step))
            ),
            "post_pull_rejection": False,
            "merge": "sample-count-weighted parameter average",
            "optimizer_reset": True,
            "pull_interval_steps": int(self._weighted_pull_interval_steps),
            "pull_schedule_anchor": (
                "first active step after vehicle entry"
                if self._weighted_pull_schedule_anchor == "entry"
                else "global simulation steps divisible by the pull interval"
            ),
        }
        if self._weighted_selection in {
            "grid-intensity",
            "random-grid-intensity",
        }:
            status["weighted_single_model"].update({
                "selection": (
                    "uniform random unseen neighbour model"
                    if self._weighted_selection == "random-grid-intensity"
                    else "advertised summed grid intensity"
                ),
                "merge": "summed-grid-intensity-weighted parameter average",
                "advertisement": "one summed-intensity scalar per model version",
            })
        if self._weighted_acquisition_fixed_budget:
            status["weighted_single_model"].update({
                "acquisition_fixed_budget": True,
                "pulls_per_receiver_per_step": (
                    "all available" if self._weighted_pulls_per_receiver_step == 0
                    else int(self._weighted_pulls_per_receiver_step)
                ),
            })
        if self._weighted_acquisition:
            status["learned_acquisition"].update({
                "rejected_pulls": 0,
                "post_pull_gain_rejections": 0,
                "post_pull_gain_threshold": None,
                "post_pull_gain_measure": None,
                "sampled_gain_validation": False,
            })
            status["post_pull_validation"] = "none"
        core.atomic_json(status_path, status)
        metrics_path = output / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if not self._weighted_acquisition:
            metrics.pop("learned_acquisition", None)
        metrics["weighted_single_model"] = status["weighted_single_model"]
        metrics["expert_bank"].update({
            "capacity_including_local": 1,
            "selection": (
                "frozen pretrained relative grid-gain acquisition"
                if self._weighted_acquisition
                else "advertised sample count then covered-cell count"
            ),
            "advertisement": (
                "cached product-quantized grid encoding, model key and experience"
                if self._weighted_acquisition
                else "model key, sample count, and covered-cell count"
            ),
            "post_pull_grid_validation": "none",
            "post_pull_gain_measure": None,
            "post_pull_gain_threshold": None,
            "sampled_gain_validation": False,
            "post_pull_exact_validation": False,
            "iterative_pull_until_no_positive_candidate": self._weighted_acquisition,
            "pull_budget_per_receiver_step": (
                "predicted relative gain above kappa"
                if self._weighted_acquisition
                else ("all available"
                    if self._weighted_pulls_per_receiver_step == 0
                    else int(self._weighted_pulls_per_receiver_step))
            ),
        })
        if self._weighted_selection in {
            "grid-intensity",
            "random-grid-intensity",
        }:
            metrics["expert_bank"].update({
                "selection": (
                    "uniform random unseen neighbour model"
                    if self._weighted_selection == "random-grid-intensity"
                    else "advertised summed grid intensity"
                ),
                "advertisement": "model key and one summed-intensity scalar",
                "model_merge": "summed-grid-intensity-weighted parameter average",
            })
        if self._weighted_acquisition_fixed_budget:
            metrics["expert_bank"].update({
                "iterative_pull_until_no_positive_candidate": False,
                "pull_budget_per_receiver_step": (
                    "all available" if self._weighted_pulls_per_receiver_step == 0
                    else int(self._weighted_pulls_per_receiver_step)
                ),
            })
        if self._weighted_acquisition:
            metrics["learned_acquisition"].update({
                "rejected_pulls": 0,
                "post_pull_gain_rejections": 0,
                "post_pull_exact_validation": False,
                "post_pull_gain_threshold": None,
                "post_pull_gain_measure": None,
                "sampled_gain_validation": False,
                "decision": (
                    "highest predicted relative gain under fixed pull budget"
                    if self._weighted_acquisition_fixed_budget
                    else "pre-pull predicted relative gain above kappa"
                ),
            })
        core.atomic_json(metrics_path, metrics)
