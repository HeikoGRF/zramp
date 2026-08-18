"""Online decentralized simulation with bounded predictor expert banks.

Each physical vehicle trains only its own local specialist from measurements it
received. Predictor pulls copy frozen specialists between bounded per-AZ banks;
they never copy measurement rows and never average predictor parameters. At
inference, the expert with the strongest calibrated spatial support is used;
outside the union of learned supports the conservative max-loss prior is used.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from .local_validation_reward import FLOAT32_BYTES, PullResult
from .online_local_validation_policy import (
    ExactPrivateState,
    SampleSharingExactSequentialSimulation,
    _TrainingExample,
    exact_model_groups,
)
from .expert_bank import (
    AcquisitionReward,
    CellValidationCertificate,
    ExpertBank,
    PredictorExpert,
    SupportProfile,
    ValidationCertificate,
    bilateral_zone_reward,
    pareto_safe_acquisition,
    support_overlap,
)
from .expert_bank_policy import (
    DecentralizedLinUCB,
    bank_support_vector,
    expert_pull_features,
    profile_vector,
)


@dataclass(frozen=True)
class _AcquisitionTrial:
    bank: ExpertBank
    reward: AcquisitionReward
    objective_evaluations: int


class ExpertBankSampleSharingSimulation(
    SampleSharingExactSequentialSimulation
):
    """Persistent per-vehicle/per-AZ expert banks with local policy learning."""

    policy_transfer_rule = (
        "versioned-frozen-single-expert-and-policy-sample-gossip"
    )

    def __init__(
        self,
        *args,
        expert_bank_capacity: int = 4,
        expert_bank_temperature: float = 0.15,
        expert_bank_support_threshold: float = 0.05,
        expert_bank_support_grid_points: int = 7,
        expert_bank_support_validation_recall: float = 0.9,
        expert_bank_min_local_samples: int = 10,
        expert_bank_diversity_weight_db: float = 0.25,
        expert_bank_external_utility_weight: float = 0.5,
        expert_bank_gate_biases: tuple[float, ...] = (-1.0, 0.0, 1.0),
        expert_bank_certificate_grid_points: int = 3,
        expert_bank_certificate_min_samples: int = 8,
        expert_bank_certificate_epoch_steps: int = 60,
        expert_bank_bandit_exploration: float = 0.35,
        **kwargs,
    ) -> None:
        if not args:
            raise TypeError("simulation configuration is required")
        cfg = args[0]
        feature_dim = 5 if bool(
            getattr(cfg, "predictor_include_time", False)
        ) else 4
        self.expert_bank_capacity = int(expert_bank_capacity)
        self.expert_bank_temperature = float(expert_bank_temperature)
        self.expert_bank_support_threshold = float(
            expert_bank_support_threshold
        )
        self.expert_bank_support_grid_points = int(
            expert_bank_support_grid_points
        )
        self.expert_bank_support_validation_recall = float(
            expert_bank_support_validation_recall
        )
        self.expert_bank_min_local_samples = int(
            expert_bank_min_local_samples
        )
        self.expert_bank_diversity_weight_db = float(
            expert_bank_diversity_weight_db
        )
        self.expert_bank_external_utility_weight = float(
            expert_bank_external_utility_weight
        )
        self.expert_bank_gate_biases = tuple(
            float(value) for value in expert_bank_gate_biases
        )
        self.expert_bank_certificate_grid_points = int(
            expert_bank_certificate_grid_points
        )
        self.expert_bank_certificate_min_samples = int(
            expert_bank_certificate_min_samples
        )
        self.expert_bank_certificate_epoch_steps = int(
            expert_bank_certificate_epoch_steps
        )
        self.expert_bank_bandit_exploration = float(
            expert_bank_bandit_exploration
        )
        if self.expert_bank_capacity <= 0:
            raise ValueError("expert bank capacity must be positive")
        if self.expert_bank_min_local_samples <= 0:
            raise ValueError("minimum local expert samples must be positive")
        if self.expert_bank_support_grid_points < 2:
            raise ValueError("expert support grid needs at least two points")
        if not 0.0 < self.expert_bank_support_validation_recall <= 1.0:
            raise ValueError("expert support recall must be in (0, 1]")
        if not self.expert_bank_gate_biases:
            raise ValueError("at least one expert gate bias is required")
        if self.expert_bank_certificate_grid_points < 2:
            raise ValueError("certificate grid needs at least two points")
        if self.expert_bank_certificate_min_samples <= 0:
            raise ValueError("certificate minimum samples must be positive")
        if self.expert_bank_certificate_epoch_steps <= 0:
            raise ValueError("certificate epoch steps must be positive")
        if self.expert_bank_bandit_exploration < 0.0:
            raise ValueError("bandit exploration cannot be negative")
        kwargs["policy_support_dim"] = 4 * feature_dim + 2
        super().__init__(*args, **kwargs)
        if not self.symmetric_pulls:
            raise ValueError("expert-bank exchanges require symmetric pulls")
        if self.policy_reward_metric != "rmse-gain":
            raise ValueError("expert-bank policy rewards must use RMSE gain")
        self._expert_banks: list[dict[int, dict[str, ExpertBank]]] = [
            {} for _ in self.nodes
        ]
        self._local_expert_versions: Counter[str] = Counter()
        self._advertised_experts: dict[
            tuple[int, str, int, int], PredictorExpert
        ] = {}
        self._advertisement_cache_step = -1
        self._selected_experts: dict[
            tuple[int, str, int, int, int], PredictorExpert
        ] = {}
        self._selected_features: dict[
            tuple[int, str, int, int, int], np.ndarray
        ] = {}
        self._pull_bandits: dict[
            tuple[int, int, str], DecentralizedLinUCB
        ] = {}
        self._receiver_policy_observations: dict[
            tuple[int, str, int, int], tuple[ExactPrivateState, torch.Tensor]
        ] = {}
        self.zramp_policy_mode = (
            "decentralized-versioned-single-expert-bank-sample-gossip"
        )
        self._communication_assumptions = self._build_communication_assumptions()

    # -------------------------------------------------------------- bank state

    def _new_bank(self) -> ExpertBank:
        return ExpertBank(
            capacity=self.expert_bank_capacity,
            temperature=self.expert_bank_temperature,
            support_threshold=self.expert_bank_support_threshold,
            prior_normalized_loss=1.0,
            routing="hard-certified",
        )

    def _bank(
        self, node_idx: int, zone: int, mode: str, *, create: bool = True
    ) -> ExpertBank:
        node = int(node_idx)
        az = int(zone)
        if not hasattr(self, "_expert_banks"):
            return self._new_bank()
        zones = self._expert_banks[node]
        if not create and az not in zones:
            return self._new_bank()
        modes = zones.setdefault(az, {})
        if not create and str(mode) not in modes:
            return self._new_bank()
        return modes.setdefault(str(mode), self._new_bank())

    def _bank_snapshot(self, bank: ExpertBank) -> ExpertBank:
        """Freeze the membership advertised at the beginning of one step."""

        return ExpertBank(
            list(bank.experts),
            capacity=bank.capacity,
            temperature=bank.temperature,
            logit_bias_by_lineage=dict(bank.logit_bias_by_lineage),
            support_threshold=bank.support_threshold,
            prior_normalized_loss=bank.prior_normalized_loss,
            uncertainty_floor_db=bank.uncertainty_floor_db,
            routing=bank.routing,
        )

    def _lineage(self, node_idx: int, zone: int, mode: str) -> str:
        generations = getattr(self, "_node_generations", [])
        generation = (
            int(generations[int(node_idx)])
            if int(node_idx) < len(generations)
            else 0
        )
        return (
            f"seed{int(self.cfg.seed)}:vehicle{int(node_idx)}:"
            f"generation{generation}:az{int(zone)}:{str(mode)}:local"
        )

    @staticmethod
    def _subset_arrays(subset, feature_dim: int) -> tuple[np.ndarray, np.ndarray]:
        features = (
            np.asarray(subset.features, dtype=np.float32)
            if subset.features
            else np.empty((0, int(feature_dim)), dtype=np.float32)
        )
        targets = np.asarray(subset.targets, dtype=np.float32).reshape(-1)
        return features, targets

    def _reward_validation(
        self, node_idx: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        state = self._zone_validation[int(node_idx)]
        features, targets = self._subset_arrays(
            state.reward, self._predictor_input_dim()
        )
        return features, targets, float(state.reward.quality)

    def _private_validation_for_expert(
        self, node_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        state = self._zone_validation[int(node_idx)]
        rows = [
            self._subset_arrays(state.optimization, self._predictor_input_dim()),
            self._subset_arrays(state.reward, self._predictor_input_dim()),
        ]
        features = [x for x, _y in rows if int(x.shape[0]) > 0]
        targets = [y for x, y in rows if int(x.shape[0]) > 0]
        if not features:
            return (
                np.empty((0, self._predictor_input_dim()), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        return np.concatenate(features), np.concatenate(targets)

    def _model_rmse_dbm(
        self, model: torch.nn.Module, features: np.ndarray, target_dbm: np.ndarray
    ) -> float:
        if int(features.shape[0]) == 0:
            return float(self.cfg.rssi_max_dbm - self.cfg.rssi_min_dbm)
        model.eval()
        with torch.inference_mode():
            values = model(
                torch.as_tensor(features, dtype=torch.float32, device=self.device)
            ).detach().cpu().numpy().reshape(-1)
        prediction = self._denorm_dbm(values)
        return float(
            np.sqrt(
                np.mean(
                    np.square(
                        prediction.astype(np.float64)
                        - np.asarray(target_dbm, dtype=np.float64).reshape(-1)
                    )
                )
            )
        )


    def _attach_cell_certificates(
        self,
        expert: PredictorExpert,
        features: np.ndarray,
        target_dbm: np.ndarray,
        *,
        validator_id: str,
    ) -> None:
        if int(features.shape[0]) == 0:
            return
        normalized = ExpertBank._model_prediction(expert.model, features)
        prediction_dbm = (
            float(self.cfg.rssi_max_dbm)
            - (
                float(self.cfg.rssi_max_dbm)
                - float(self.cfg.rssi_min_dbm)
            )
            * np.clip(normalized, 0.0, 1.0)
        ).astype(np.float64)
        target = np.asarray(target_dbm, dtype=np.float64).reshape(-1)
        error_limit = float(self.cfg.rssi_max_dbm - self.cfg.rssi_min_dbm)
        squared_error = np.square(
            np.clip(prediction_dbm - target, -error_limit, error_limit)
        )
        prior_error = np.square(
            np.clip(
                float(self.cfg.rssi_min_dbm) - target,
                -error_limit,
                error_limit,
            )
        )
        grid = int(
            getattr(self, "expert_bank_certificate_grid_points", 3)
        )
        cells = SupportProfile.coarse_cell_indices(
            features, grid_points=grid
        )
        epoch = int(getattr(self, "_current_sumo_step", 0)) // int(
            getattr(self, "expert_bank_certificate_epoch_steps", 60)
        )

        def attach(cell_index: int, mask: np.ndarray) -> None:
            count = int(np.count_nonzero(mask))
            if count <= 0:
                return
            expert.add_cell_certificate(
                CellValidationCertificate(
                    expert_hash=expert.content_hash,
                    validator_id=str(validator_id),
                    epoch=epoch,
                    cell_index=int(cell_index),
                    grid_points=grid,
                    sample_count=count,
                    squared_error_sum_db2=float(np.sum(squared_error[mask])),
                    prior_squared_error_sum_db2=float(
                        np.sum(prior_error[mask])
                    ),
                )
            )

        attach(-1, np.ones(len(target), dtype=bool))
        for cell in np.unique(cells):
            mask = cells == int(cell)
            if int(np.count_nonzero(mask)) >= int(
                getattr(self, "expert_bank_certificate_min_samples", 8)
            ):
                attach(int(cell), mask)

    def _sync_local_expert(self, node_idx: int, mode: str) -> None:
        node = int(node_idx)
        ns = self.nodes[node]
        zone = int(getattr(ns, "current_az", -1))
        if zone < 0:
            return
        features = np.asarray(ns.current_visit_samples_x, dtype=np.float32)
        if features.ndim != 2 or int(features.shape[0]) < self.expert_bank_min_local_samples:
            return
        validation_x, validation_y = self._private_validation_for_expert(node)
        if int(validation_x.shape[0]) == 0:
            return
        lineage = self._lineage(node, zone, mode)
        bank = self._bank(node, zone, mode)
        model = copy.deepcopy(ns.variants[str(mode)].model)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._local_expert_versions[lineage] += 1
        version = int(self._local_expert_versions[lineage])
        support = SupportProfile.fit(
            features,
            time_scale=float(
                getattr(self.cfg, "predictor_learned_time_scale", 1800.0)
            ),
            spatial_grid_points=self.expert_bank_support_grid_points,
            calibration_features=validation_x,
            target_recall=self.expert_bank_support_validation_recall,
        )
        base_state = super()._raw_state(
            node, str(mode), model_state=model.state_dict()
        )
        candidate_state = ExactPrivateState(
            model_groups=base_state.model_groups,
            trajectory=base_state.trajectory,
            support=profile_vector(support),
        )
        source_embedding = (
            self.local_agents[str(mode)][node].policy_embedding(
                candidate_state
            )
            if self.selection_mode != "random"
            else None
        )
        expert = next(
            (row for row in bank.experts if row.lineage_id == lineage), None
        )
        if expert is None:
            expert = PredictorExpert(
                lineage_id=lineage,
                model=model,
                support=support,
                own_validation_rmse_db=self._model_rmse_dbm(
                    model, validation_x, validation_y
                ),
                experience=int(features.shape[0]),
                version=version,
                policy_embedding=source_embedding,
                locally_owned=True,
            )
            bank.append(expert, allow_probation=len(bank.experts) >= bank.capacity)
        else:
            expert.model = model
            expert.support = support
            expert.own_validation_rmse_db = self._model_rmse_dbm(
                model, validation_x, validation_y
            )
            expert.experience = int(features.shape[0])
            expert.version = version
            expert.policy_embedding = (
                None
                if source_embedding is None
                else source_embedding.detach().cpu().clone()
            )
            expert.refresh_content_hash()
        self._attach_cell_certificates(
            expert,
            validation_x,
            validation_y,
            validator_id=f"vehicle{node}:az{zone}:source",
        )
        if len(bank.experts) > bank.capacity:
            bank.resolve_probation(
                validation_x,
                validation_y,
                diversity_weight_db=self.expert_bank_diversity_weight_db,
                external_utility_weight=self.expert_bank_external_utility_weight,
                rssi_min_dbm=float(self.cfg.rssi_min_dbm),
                rssi_max_dbm=float(self.cfg.rssi_max_dbm),
            )

    def _train_local(self, ns, X, y_dbm, **kwargs) -> None:
        super()._train_local(ns, X, y_dbm, **kwargs)
        node_idx = self.node_idx(ns)
        if node_idx >= 0:
            for mode in self.agents:
                self._sync_local_expert(node_idx, str(mode))

    def _reset_respawned_node(
        self, node_idx: int, *, generation: int | None = None
    ) -> None:
        super()._reset_respawned_node(node_idx, generation=generation)
        if hasattr(self, "_expert_banks"):
            self._expert_banks[int(node_idx)] = {}

    def _advertisement_key(
        self, node_idx: int, zone: int, mode: str
    ) -> tuple[int, str, int, int]:
        return (
            int(getattr(self, "_current_sumo_step", 0)),
            str(mode),
            int(node_idx),
            int(zone),
        )

    def _advertised_expert(
        self, node_idx: int, zone: int, mode: str
    ) -> PredictorExpert | None:
        key = self._advertisement_key(node_idx, zone, mode)
        if int(key[0]) != int(self._advertisement_cache_step):
            self._advertised_experts.clear()
            self._selected_experts.clear()
            self._selected_features.clear()
            self._receiver_policy_observations.clear()
            self._advertisement_cache_step = int(key[0])
        cached = self._advertised_experts.get(key)
        if cached is not None:
            return cached
        bank = self._bank(int(node_idx), int(zone), str(mode), create=False)
        if not bank.experts:
            return None
        ordered = sorted(
            bank.experts,
            key=lambda row: (row.lineage_id, int(row.version)),
        )
        step = int(key[0])
        position = (
            step + 31 * int(node_idx) + 7 * int(zone)
        ) % len(ordered)
        snapshot = ordered[position].transferred_copy()
        self._advertised_experts[key] = snapshot
        return snapshot

    def _candidate_state(
        self, expert: PredictorExpert, *, trajectory_width: int
    ) -> ExactPrivateState:
        return ExactPrivateState(
            model_groups=exact_model_groups(expert.model.state_dict()),
            trajectory=torch.empty(
                (0, int(trajectory_width)), dtype=torch.float32
            ),
            support=profile_vector(expert.support),
        )

    # -------------------------------------------------------------- policy I/O

    def _raw_state(self, node_idx: int, mode: str, **kwargs) -> ExactPrivateState:
        state = super()._raw_state(node_idx, mode, **kwargs)
        zone = int(getattr(self.nodes[int(node_idx)], "current_az", -1))
        bank = self._bank(int(node_idx), zone, str(mode), create=False)
        return ExactPrivateState(
            model_groups=state.model_groups,
            trajectory=state.trajectory,
            support=bank_support_vector(
                bank, feature_dim=self._predictor_input_dim()
            ),
        )

    def _make_peer_view(self, ns_j, mode: str):
        view = super()._make_peer_view(ns_j, mode)
        node_idx = self.node_idx(ns_j)
        zone = int(getattr(ns_j, "current_az", -1))
        bank = self._bank(node_idx, zone, str(mode), create=False)
        view._expert_bank_snapshot = self._bank_snapshot(bank)  # type: ignore[attr-defined]
        view._expert_manifest = bank.manifest(  # type: ignore[attr-defined]
            grid_points=self.expert_bank_certificate_grid_points
        )
        view._advertised_expert = self._advertised_expert(  # type: ignore[attr-defined]
            node_idx, zone, str(mode)
        )
        if self.selection_mode != "random":
            receiver_state = self._raw_state(node_idx, str(mode))
            receiver_embedding = self.local_agents[str(mode)][
                node_idx
            ].policy_embedding(receiver_state)
            self._receiver_policy_observations[
                self._advertisement_key(node_idx, zone, str(mode))
            ] = (
                receiver_state.clone(),
                receiver_embedding.detach().cpu().clone(),
            )
        return view

    def _provider_policy_observation(
        self, node_idx: int, mode: str, provider_view: object
    ) -> tuple[ExactPrivateState, torch.Tensor]:
        candidate = getattr(provider_view, "_advertised_expert", None)
        if not isinstance(candidate, PredictorExpert):
            return super()._provider_policy_observation(
                node_idx, mode, provider_view
            )
        embedding = candidate.policy_embedding
        if embedding is None:
            raise RuntimeError(
                "policy candidate is missing its immutable source embedding"
            )
        receiver_state = self._raw_state(int(node_idx), str(mode))
        state = self._candidate_state(
            candidate,
            trajectory_width=int(receiver_state.trajectory.shape[1]),
        )
        return state, embedding.detach().cpu().clone()


    def _selection_key(
        self,
        receiver_idx: int,
        provider_idx: int,
        zone: int,
        mode: str,
    ) -> tuple[int, str, int, int, int]:
        return (
            int(getattr(self, "_current_sumo_step", 0)),
            str(mode),
            int(receiver_idx),
            int(provider_idx),
            int(zone),
        )

    def _bandit(
        self, node_idx: int, zone: int, mode: str
    ) -> DecentralizedLinUCB:
        key = (int(node_idx), int(zone), str(mode))
        if key not in self._pull_bandits:
            self._pull_bandits[key] = DecentralizedLinUCB(
                exploration=self.expert_bank_bandit_exploration,
                reward_scale_db=30.0,
                sample_capacity=max(512, int(self.policy_sample_capacity)),
            )
        return self._pull_bandits[key]

    def _select_provider_expert(
        self,
        *,
        receiver_idx: int,
        provider_idx: int,
        zone: int,
        mode: str,
        provider_bank: ExpertBank,
    ) -> PredictorExpert | None:
        key = self._selection_key(
            receiver_idx, provider_idx, zone, mode
        )
        cached = self._selected_experts.get(key)
        if cached is not None:
            return cached
        receiver_bank = self._bank(
            int(receiver_idx), int(zone), str(mode), create=False
        )
        candidates = receiver_bank.usable_provider_experts(provider_bank)
        if not candidates:
            return None
        if self.selection_mode == "random":
            position = (
                int(self.cfg.seed) * 1_000_003
                + int(key[0]) * 10_007
                + int(receiver_idx) * 1_009
                + int(provider_idx) * 101
                + int(zone) * 17
            ) % len(candidates)
            selected = candidates[position]
            features = expert_pull_features(
                receiver_bank,
                selected,
                grid_points=self.expert_bank_certificate_grid_points,
            )
        else:
            bandit = self._bandit(receiver_idx, zone, mode)
            ranked: list[tuple[float, float, str, PredictorExpert, np.ndarray]] = []
            for candidate in candidates:
                features = expert_pull_features(
                    receiver_bank,
                    candidate,
                    grid_points=self.expert_bank_certificate_grid_points,
                )
                certified_gain = receiver_bank.certified_marginal_gain_db(
                    candidate,
                    grid_points=self.expert_bank_certificate_grid_points,
                )
                score = float(certified_gain + bandit.score(features))
                ranked.append(
                    (
                        score,
                        float(certified_gain),
                        candidate.content_hash,
                        candidate,
                        features,
                    )
                )
            _score, _gain, _hash, selected, features = max(
                ranked, key=lambda row: (row[0], row[1], row[2])
            )
        snapshot = selected.transferred_copy()
        self._selected_experts[key] = snapshot
        self._selected_features[key] = np.asarray(
            features, dtype=np.float64
        ).copy()
        return snapshot

    def _policy_candidate_score(
        self,
        *,
        receiver_idx: int,
        provider_idx: int,
        mode: str,
        provider_view: object,
        agent,
        receiver_embedding: torch.Tensor,
        provider_embedding: torch.Tensor,
    ) -> float:
        del agent, receiver_embedding, provider_embedding
        zone = int(getattr(self.nodes[int(receiver_idx)], "current_az", -1))
        provider_bank = getattr(
            provider_view, "_expert_bank_snapshot", None
        )
        if not isinstance(provider_bank, ExpertBank):
            return -300.0
        candidate = self._select_provider_expert(
            receiver_idx=int(receiver_idx),
            provider_idx=int(provider_idx),
            zone=zone,
            mode=str(mode),
            provider_bank=provider_bank,
        )
        if candidate is None:
            return -300.0
        features = self._selected_features[
            self._selection_key(
                receiver_idx, provider_idx, zone, mode
            )
        ]
        return float(
            self._bank(receiver_idx, zone, mode, create=False)
            .certified_marginal_gain_db(
                candidate,
                grid_points=self.expert_bank_certificate_grid_points,
            )
            + self._bandit(receiver_idx, zone, mode).score(features)
        )

    # ---------------------------------------------------------- private CV pull

    def _trial_acquisition(
        self,
        receiver_bank: ExpertBank,
        provider_expert: PredictorExpert,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        provider_x: np.ndarray,
        provider_y: np.ndarray,
        receiver_quality: float,
        provider_quality: float,
        *,
        validator_id: str,
    ) -> _AcquisitionTrial | None:
        receiver_before = receiver_bank.rmse_db(
            receiver_x,
            receiver_y,
            rssi_min_dbm=float(self.cfg.rssi_min_dbm),
            rssi_max_dbm=float(self.cfg.rssi_max_dbm),
        )
        provider_before = receiver_bank.rmse_db(
            provider_x,
            provider_y,
            rssi_min_dbm=float(self.cfg.rssi_min_dbm),
            rssi_max_dbm=float(self.cfg.rssi_max_dbm),
        )
        trials: list[_AcquisitionTrial] = []
        evaluations = 0
        current = receiver_bank.expert_for_lineage(provider_expert.lineage_id)
        usable = bool(
            current is None
            or (
                not current.locally_owned
                and int(provider_expert.version) > int(current.version)
            )
        )
        if usable:
            gate_biases = (
                (0.0,)
                if receiver_bank.routing in {
                    "hard-max-support",
                    "hard-certified",
                }
                else self.expert_bank_gate_biases
            )
            for bias in gate_biases:
                evaluations += 1
                biases = dict(receiver_bank.logit_bias_by_lineage)
                biases[provider_expert.lineage_id] = float(bias)
                bank = ExpertBank(
                    list(receiver_bank.experts),
                    capacity=receiver_bank.capacity,
                    temperature=receiver_bank.temperature,
                    logit_bias_by_lineage=biases,
                    support_threshold=receiver_bank.support_threshold,
                    prior_normalized_loss=receiver_bank.prior_normalized_loss,
                    uncertainty_floor_db=receiver_bank.uncertainty_floor_db,
                    routing=receiver_bank.routing,
                )
                bank.install(
                    provider_expert.transferred_copy(),
                    allow_probation=len(bank.experts) >= bank.capacity,
                )
                if len(bank.experts) > bank.capacity:
                    bank.resolve_probation(
                        receiver_x,
                        receiver_y,
                        diversity_weight_db=self.expert_bank_diversity_weight_db,
                        external_utility_weight=self.expert_bank_external_utility_weight,
                        rssi_min_dbm=float(self.cfg.rssi_min_dbm),
                        rssi_max_dbm=float(self.cfg.rssi_max_dbm),
                    )
                receiver_after = bank.rmse_db(
                    receiver_x,
                    receiver_y,
                    rssi_min_dbm=float(self.cfg.rssi_min_dbm),
                    rssi_max_dbm=float(self.cfg.rssi_max_dbm),
                )
                provider_after = bank.rmse_db(
                    provider_x,
                    provider_y,
                    rssi_min_dbm=float(self.cfg.rssi_min_dbm),
                    rssi_max_dbm=float(self.cfg.rssi_max_dbm),
                )
                overlap = (
                    float(
                        np.mean(
                            [
                                support_overlap(
                                    row.support, provider_expert.support
                                )
                                for row in receiver_bank.experts
                                if row.lineage_id
                                != provider_expert.lineage_id
                            ]
                        )
                    )
                    if any(
                        row.lineage_id != provider_expert.lineage_id
                        for row in receiver_bank.experts
                    )
                    else 0.0
                )
                joint, receiver_gain, provider_gain = bilateral_zone_reward(
                    receiver_before,
                    receiver_after,
                    receiver_quality,
                    provider_before,
                    provider_after,
                    provider_quality,
                    overlap=overlap,
                )
                reward = AcquisitionReward(
                    joint_gain_db=float(joint),
                    receiver_gain_db=float(receiver_gain),
                    provider_gain_db=float(provider_gain),
                    receiver_before_rmse_db=float(receiver_before),
                    receiver_after_rmse_db=float(receiver_after),
                    provider_before_rmse_db=float(provider_before),
                    provider_after_rmse_db=float(provider_after),
                    duplicate_lineage=False,
                )
                trials.append(
                    _AcquisitionTrial(
                        bank=bank,
                        reward=reward,
                        objective_evaluations=evaluations,
                    )
                )
        if not trials:
            reward = AcquisitionReward(
                joint_gain_db=0.0,
                receiver_gain_db=0.0,
                provider_gain_db=0.0,
                receiver_before_rmse_db=float(receiver_before),
                receiver_after_rmse_db=float(receiver_before),
                provider_before_rmse_db=float(provider_before),
                provider_after_rmse_db=float(provider_before),
                duplicate_lineage=True,
            )
            return _AcquisitionTrial(
                bank=self._bank_snapshot(receiver_bank),
                reward=reward,
                objective_evaluations=0,
            )
        selected = max(
            trials,
            key=lambda row: (
                row.reward.joint_gain_db,
                row.reward.receiver_gain_db,
                tuple(row.bank.lineages),
            ),
        )
        pareto_safe = pareto_safe_acquisition(selected.reward)
        if pareto_safe:
            installed = selected.bank.expert_for_lineage(
                provider_expert.lineage_id
            )
            if installed is not None:
                installed.add_certificate(
                    ValidationCertificate(
                        validator_id=str(validator_id),
                        marginal_gain_db=float(selected.reward.joint_gain_db),
                        coverage_quality=float(provider_quality),
                    )
                )
                self._attach_cell_certificates(
                    installed,
                    provider_x,
                    provider_y,
                    validator_id=str(validator_id),
                )
        return _AcquisitionTrial(
            bank=selected.bank,
            reward=selected.reward,
            objective_evaluations=int(evaluations),
        )

    @staticmethod
    def _expert_model_bytes(expert: PredictorExpert, state_nbytes, clone_state) -> int:
        return int(
            state_nbytes(clone_state(expert.model))
            + expert.support.wire_nbytes
            + expert.certificate_wire_nbytes
            + len(expert.lineage_id.encode("utf-8"))
            + 16
        )

    def _execute_validation_pull(
        self,
        *,
        step: int,
        mode: str,
        receiver,
        provider,
        zone: int,
        provider_view,
        diagnostic: bool = False,
    ) -> PullResult:
        receiver_idx = self.node_idx(receiver)
        provider_idx = self.node_idx(provider)
        receiver_bank = self._bank(receiver_idx, zone, mode)
        provider_bank = (
            provider_view._expert_bank_snapshot  # type: ignore[attr-defined]
            if provider_view is not None
            and hasattr(provider_view, "_expert_bank_snapshot")
            else self._bank_snapshot(self._bank(provider_idx, zone, mode))
        )
        provider_candidate = self._select_provider_expert(
            receiver_idx=receiver_idx,
            provider_idx=provider_idx,
            zone=zone,
            mode=mode,
            provider_bank=provider_bank,
        )
        receiver_candidate = self._select_provider_expert(
            receiver_idx=provider_idx,
            provider_idx=receiver_idx,
            zone=zone,
            mode=mode,
            provider_bank=self._bank_snapshot(receiver_bank),
        )
        receiver_x, receiver_y, qa = self._reward_validation(receiver_idx)
        provider_x, provider_y, qb = self._reward_validation(provider_idx)
        metadata_bytes = int(
            self._metadata_for(provider_idx, mode).wire_nbytes
            + receiver_bank.manifest_wire_nbytes
            + provider_bank.manifest_wire_nbytes
        )
        if not isinstance(provider_candidate, PredictorExpert) or not isinstance(
            receiver_candidate, PredictorExpert
        ):
            result = PullResult(
                valid=False,
                reason="empty_advertised_expert",
                model_messages=0,
            )
            return result
        model_bytes = self._expert_model_bytes(
            receiver_candidate, self._state_nbytes, self._clone_state
        ) + self._expert_model_bytes(
            provider_candidate, self._state_nbytes, self._clone_state
        )
        write_pull_log = (
            (lambda *_args, **_kwargs: None)
            if diagnostic
            else self._write_pull_log
        )
        if not diagnostic:
            self._cv_last_provider_pull_step[
                (str(mode), receiver_idx, provider_idx, int(zone))
            ] = int(step)
            self._cv_last_provider_pull_step[
                (str(mode), provider_idx, receiver_idx, int(zone))
            ] = int(step)
            self._cv_step_pulls[mode] += 1
            self._cv_step_model_messages[mode] += 2
            self._cv_step_model_bytes[mode] += int(model_bytes)
        if qa <= 0.0 or qb <= 0.0:
            result = PullResult(
                valid=False,
                reason=(
                    "zero_receiver_reward_quality"
                    if qa <= 0.0
                    else "zero_provider_reward_quality"
                ),
                model_messages=2,
            )
            write_pull_log(
                step, mode, receiver_idx, provider_idx, zone, result,
                0.0, 0.0, qa, qb, metadata_bytes, model_bytes,
            )
            return result
        forward = self._trial_acquisition(
            receiver_bank,
            provider_candidate,
            receiver_x,
            receiver_y,
            provider_x,
            provider_y,
            qa,
            qb,
            validator_id=f"vehicle{provider_idx}:az{zone}",
        )
        reverse = self._trial_acquisition(
            provider_bank,
            receiver_candidate,
            provider_x,
            provider_y,
            receiver_x,
            receiver_y,
            qb,
            qa,
            validator_id=f"vehicle{receiver_idx}:az{zone}",
        )
        if forward is None or reverse is None:
            result = PullResult(
                valid=False,
                reason="empty_expert_bank",
                model_messages=2,
            )
            write_pull_log(
                step, mode, receiver_idx, provider_idx, zone, result,
                0.0, 0.0, qa, qb, metadata_bytes, model_bytes,
            )
            return result
        receiver_adopted = pareto_safe_acquisition(forward.reward)
        provider_adopted = pareto_safe_acquisition(reverse.reward)
        if not diagnostic and self.selection_mode == "policy":
            forward_key = self._selection_key(
                receiver_idx, provider_idx, zone, mode
            )
            reverse_key = self._selection_key(
                provider_idx, receiver_idx, zone, mode
            )
            if forward_key in self._selected_features:
                self._bandit(receiver_idx, zone, mode).update(
                    self._selected_features[forward_key],
                    forward.reward.joint_gain_db,
                    sample_id=(
                        f"{step}:{mode}:{receiver_idx}:{provider_idx}:"
                        f"{provider_candidate.content_hash}"
                    ),
                )
            if reverse_key in self._selected_features:
                self._bandit(provider_idx, zone, mode).update(
                    self._selected_features[reverse_key],
                    reverse.reward.joint_gain_db,
                    sample_id=(
                        f"{step}:{mode}:{provider_idx}:{receiver_idx}:"
                        f"{receiver_candidate.content_hash}"
                    ),
                )
        if not diagnostic:
            if receiver_adopted:
                self._expert_banks[receiver_idx][int(zone)][str(mode)] = forward.bank
            if provider_adopted:
                self._expert_banks[provider_idx][int(zone)][str(mode)] = reverse.bank
            if receiver_adopted or provider_adopted:
                if not hasattr(self, "_cv_receiver_aggregations"):
                    self._cv_receiver_aggregations = Counter()
                if receiver_adopted:
                    self._cv_receiver_aggregations[(str(mode), receiver_idx)] += 1
                if provider_adopted:
                    self._cv_receiver_aggregations[(str(mode), provider_idx)] += 1
        joint_reward = 0.5 * (
            forward.reward.joint_gain_db + reverse.reward.joint_gain_db
        )
        evaluations = int(
            forward.objective_evaluations + reverse.objective_evaluations
        )
        scalar_loss_messages = 2 * evaluations + 4
        scalar_control_messages = evaluations + 2
        scalar_messages = scalar_loss_messages + scalar_control_messages
        if not diagnostic:
            self._cv_step_scalar_loss_messages[mode] += scalar_loss_messages
            self._cv_step_scalar_control_messages[mode] += scalar_control_messages
            self._cv_step_scalar_messages[mode] += scalar_messages
            self._cv_step_scalar_bytes[mode] += scalar_messages * FLOAT32_BYTES
            self._cv_step_valid_pulls[mode] += 1
        result = PullResult(
            valid=True,
            reason=(
                "adopted_pair"
                if receiver_adopted and provider_adopted
                else "adopted_receiver"
                if receiver_adopted
                else "adopted_provider"
                if provider_adopted
                else "retained_pair"
            ),
            objective_evaluations=evaluations,
            before_loss=float(forward.reward.receiver_before_rmse_db**2),
            after_loss=float(forward.reward.receiver_after_rmse_db**2),
            reward=float(forward.reward.joint_gain_db),
            adopted=bool(receiver_adopted or provider_adopted),
            joint_reward=float(joint_reward),
            receiver_before_loss=float(forward.reward.receiver_before_rmse_db**2),
            receiver_after_loss=float(forward.reward.receiver_after_rmse_db**2),
            provider_before_loss=float(reverse.reward.receiver_before_rmse_db**2),
            provider_after_loss=float(reverse.reward.receiver_after_rmse_db**2),
            receiver_reward=float(forward.reward.joint_gain_db),
            provider_reward=float(reverse.reward.joint_gain_db),
            receiver_adopted=bool(receiver_adopted),
            provider_adopted=bool(provider_adopted),
            model_messages=2,
            scalar_loss_messages=scalar_loss_messages,
            scalar_control_messages=scalar_control_messages,
            scalar_messages=scalar_messages,
        )
        write_pull_log(
            step, mode, receiver_idx, provider_idx, zone, result,
            0.0, 0.0, qa, qb, metadata_bytes, model_bytes,
        )
        return result

    def _policy_result_gain(self, result: PullResult) -> float | None:
        """The initiating policy is labeled by its directional acquisition."""

        return (
            None
            if not result.valid or result.receiver_reward is None
            else float(result.receiver_reward)
        )

    @staticmethod
    def _policy_training_gains(
        result: PullResult,
    ) -> tuple[float | None, float | None]:
        if not result.valid:
            return None, None
        return result.receiver_reward, result.provider_reward

    def _exchange_training_examples(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        provider_idx: int,
        receiver_state: ExactPrivateState,
        provider_state: ExactPrivateState,
        receiver_embedding: torch.Tensor,
        provider_embedding: torch.Tensor,
        receiver_gain: float | None,
        provider_gain: float | None,
        propensity: float,
    ) -> list[tuple[int, _TrainingExample]]:
        """Label each direction with the exact frozen expert it advertised."""

        rows: list[tuple[int, _TrainingExample]] = []
        zone = int(getattr(self.nodes[int(receiver_idx)], "current_az", -1))
        forward_candidate = self._selected_experts.get(
            self._selection_key(
                receiver_idx, provider_idx, zone, mode
            )
        )
        if (
            forward_candidate is not None
            and forward_candidate.policy_embedding is not None
        ):
            provider_state = self._candidate_state(
                forward_candidate,
                trajectory_width=int(receiver_state.trajectory.shape[1]),
            )
            provider_embedding = (
                forward_candidate.policy_embedding.detach().cpu().clone()
            )
        if receiver_gain is not None:
            rows.append(
                (
                    int(receiver_idx),
                    _TrainingExample(
                        sample_id=self._policy_sample_id(
                            step=step,
                            mode=mode,
                            receiver_idx=receiver_idx,
                            provider_idx=provider_idx,
                        ),
                        provider_idx=int(provider_idx),
                        receiver_state=receiver_state.clone(),
                        receiver_embedding=receiver_embedding.detach().cpu().clone(),
                        provider_state=provider_state.clone(),
                        provider_embedding=provider_embedding.detach().cpu().clone(),
                        target_gain=float(receiver_gain),
                        propensity=float(propensity),
                    ),
                )
            )
        if self.symmetric_pulls and provider_gain is not None:
            provider_zone = int(
                getattr(self.nodes[int(provider_idx)], "current_az", -1)
            )
            observation = self._receiver_policy_observations.get(
                self._advertisement_key(
                    provider_idx, provider_zone, str(mode)
                )
            )
            receiver_candidate = self._selected_experts.get(
                self._selection_key(
                    provider_idx,
                    receiver_idx,
                    provider_zone,
                    str(mode),
                )
            )
            if observation is None or receiver_candidate is None:
                raise RuntimeError(
                    "missing pre-pull observation for symmetric expert label"
                )
            reverse_receiver_state, reverse_receiver_embedding = observation
            reverse_candidate_embedding = receiver_candidate.policy_embedding
            if reverse_candidate_embedding is None:
                raise RuntimeError(
                    "reverse candidate is missing its source embedding"
                )
            reverse_candidate_state = self._candidate_state(
                receiver_candidate,
                trajectory_width=int(
                    reverse_receiver_state.trajectory.shape[1]
                ),
            )
            rows.append(
                (
                    int(provider_idx),
                    _TrainingExample(
                        sample_id=self._policy_sample_id(
                            step=step,
                            mode=mode,
                            receiver_idx=provider_idx,
                            provider_idx=receiver_idx,
                        ),
                        provider_idx=int(receiver_idx),
                        receiver_state=reverse_receiver_state.clone(),
                        receiver_embedding=(
                            reverse_receiver_embedding.detach().cpu().clone()
                        ),
                        provider_state=reverse_candidate_state,
                        provider_embedding=(
                            reverse_candidate_embedding.detach().cpu().clone()
                        ),
                        target_gain=float(provider_gain),
                        propensity=float(propensity),
                    ),
                )
            )
        return rows

    # ------------------------------------------------------- operational output

    def _operational_prediction(
        self, ns, mode: str, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        node_idx = self.node_idx(ns)
        zone = int(getattr(ns, "current_az", -1))
        bank = self._bank(node_idx, zone, str(mode), create=False)
        prediction = bank.predict(X)
        return prediction.normalized_loss, prediction.supported

    def _predict_variant_fidelity(self, ns, mode_id, X):
        normalized, supported = self._operational_prediction(ns, mode_id, X)
        return self._denorm_dbm(normalized), supported.astype(np.float32)

    def predict_loss_db(self, ns, mode: str, X: np.ndarray) -> np.ndarray:
        normalized, _supported = self._operational_prediction(ns, mode, X)
        return self._denorm_loss_db(normalized)

    def predict_dbm(self, ns, mode: str, X: np.ndarray) -> np.ndarray:
        normalized, _supported = self._operational_prediction(ns, mode, X)
        return self._denorm_dbm(normalized)

    def eval_mse(self, ns, mode: str, X: np.ndarray, y_dbm: np.ndarray) -> float:
        if int(X.shape[0]) == 0:
            return 0.0
        prediction = self.predict_loss_db(ns, mode, X)
        target = self._rssi_to_loss_db(y_dbm).reshape(-1)
        return float(np.mean(np.square(prediction.reshape(-1) - target)))

    def _compute_fidelity_row(self, step: int):
        row = super()._compute_fidelity_row(step)
        active_mask = getattr(self, "_current_node_active", None)
        for mode in self.agents:
            banks: list[ExpertBank] = []
            for node_idx, ns in enumerate(self.nodes):
                if active_mask is not None and not bool(active_mask[node_idx]):
                    continue
                zone = int(getattr(ns, "current_az", -1))
                if zone < 0:
                    continue
                banks.append(
                    self._bank(node_idx, zone, str(mode), create=False)
                )
            sizes = [len(bank.experts) for bank in banks]
            received = [
                sum(not expert.locally_owned for expert in bank.experts)
                for bank in banks
            ]
            frozen_violations = sum(
                any(
                    parameter.requires_grad
                    for parameter in expert.model.parameters()
                )
                for bank in banks
                for expert in bank.experts
                if not expert.locally_owned
            )
            row[f"{mode}_expert_bank_active_count"] = int(len(banks))
            row[f"{mode}_expert_bank_mean_size"] = (
                float(np.mean(sizes)) if sizes else 0.0
            )
            row[f"{mode}_expert_bank_max_size"] = int(max(sizes, default=0))
            row[f"{mode}_expert_bank_mean_received"] = (
                float(np.mean(received)) if received else 0.0
            )
            row[f"{mode}_expert_bank_frozen_violations"] = int(
                frozen_violations
            )
            row[f"{mode}_expert_bank_unique_lineages"] = int(
                len(
                    {
                        expert.lineage_id
                        for bank in banks
                        for expert in bank.experts
                    }
                )
            )
            row[f"{mode}_expert_bank_mean_version"] = (
                float(
                    np.mean(
                        [
                            expert.version
                            for bank in banks
                            for expert in bank.experts
                        ]
                    )
                )
                if any(bank.experts for bank in banks)
                else 0.0
            )
        return row

    # ---------------------------------------------------------- audit metadata

    def _build_communication_assumptions(self):
        assumptions = super()._build_communication_assumptions()
        state = getattr(self, "template_state", {})
        one_model = self._state_nbytes(state) if state else 0
        feature_dim = 5 if bool(
            getattr(getattr(self, "cfg", None), "predictor_include_time", False)
        ) else 4
        one_support = 4 * (
            4 * feature_dim
            + 2
            + self.expert_bank_support_grid_points**4
            + 2
        )
        one_certificate = 16 + 64 + 6 * 4
        certificate_budget = 128 * one_certificate
        manifest_entry = (
            16
            + 64
            + 5 * 4
            + 4 * self.expert_bank_certificate_grid_points**4
        )
        one_manifest = self.expert_bank_capacity * manifest_entry
        one_expert_bytes = (
            int(one_model)
            + int(one_support)
            + int(certificate_budget)
            + 64
        )
        one_direction_bytes = one_expert_bytes + one_manifest
        assumptions.update(
            {
                "zramp_policy_mode": getattr(
                    self,
                    "zramp_policy_mode",
                    "decentralized-versioned-single-expert-bank-sample-gossip",
                ),
                "predictor_parameter_aggregation": False,
                "prediction_measurement_sharing": False,
                "received_experts_trainable": False,
                "expert_bank_capacity": int(self.expert_bank_capacity),
                "expert_bank_temperature": float(self.expert_bank_temperature),
                "expert_bank_support_threshold": float(
                    self.expert_bank_support_threshold
                ),
                "expert_bank_support_grid_points": int(
                    self.expert_bank_support_grid_points
                ),
                "expert_bank_support_validation_recall": float(
                    self.expert_bank_support_validation_recall
                ),
                "expert_bank_certificate_grid_points": int(
                    self.expert_bank_certificate_grid_points
                ),
                "expert_bank_certificate_min_samples": int(
                    self.expert_bank_certificate_min_samples
                ),
                "expert_bank_certificate_epoch_steps": int(
                    self.expert_bank_certificate_epoch_steps
                ),
                "measurement_interval_seconds": float(
                    getattr(
                        getattr(self, "cfg", None),
                        "predictor_time_step_duration",
                        1.0,
                    )
                ),
                "expert_bank_certificate_epoch_seconds": float(
                    self.expert_bank_certificate_epoch_steps
                    * getattr(
                        getattr(self, "cfg", None),
                        "predictor_time_step_duration",
                        1.0,
                    )
                ),
                "token_window_seconds": float(
                    getattr(self, "token_window_steps", 1)
                    * getattr(
                        getattr(self, "cfg", None),
                        "predictor_time_step_duration",
                        1.0,
                    )
                ),
                "expert_bank_bandit_exploration": float(
                    self.expert_bank_bandit_exploration
                ),
                "expert_bank_inference": (
                    "hard-lowest-certified-risk-within-spatial-support-"
                    "or-max-loss-prior"
                ),
                "expert_retention": (
                    "local-private-validation-plus-support-diversity-plus-"
                    "external-scalar-certificates"
                ),
                "policy_input": (
                    "receiver-bank-manifest-plus-exact-missing-capsule-"
                    "certified-gain-support-coverage-version-and-experience"
                ),
                "policy_reward": (
                    "directional-exact-expert-two-private-distribution-"
                    "RMSE-gain-scalars"
                ),
                "expert_advertisement": (
                    "complete-content-addressed-manifest-then-one-requested-"
                    "missing-or-newer-versioned-expert"
                ),
                "expert_refresh": (
                    "newer-same-lineage-snapshot-replaces-older-without-"
                    "using-an-extra-bank-slot"
                ),
                "B_model_bytes": int(one_direction_bytes),
                "B_expert_manifest_bytes": int(one_manifest),
                "B_expert_certificate_budget_bytes": int(certificate_budget),
                "B_accepted_pull_bytes": int(
                    2 * one_direction_bytes + 64
                ),
            }
        )
        return assumptions
