#!/usr/bin/env python3
"""Equal-weight all-neighbour MLP sharing on the Place Wallis trace."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from archive_paths import activate as activate_shared_runtime  # noqa: E402

activate_shared_runtime()

from experiments.place_wallis_benchmark.training_utils import (  # noqa: E402
    ReplayBuffer,
    TrainingParams,
)
from experiments.place_wallis_benchmark.tail_metrics import (  # noqa: E402
    make_tail_evaluation_steps,
    temporal_metric_summary,
)
from rl_reward_experiment.config import build_config_from_env  # noqa: E402
from SUMO.sumo_rl import SumoT2Simulation  # noqa: E402

DATA_ROOT = Path(
    "/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/"
    "place_wallis_300m_30min_opaque_buildings_no_vehicle_blockers"
)
DEFAULT_TRACE = (
    DATA_ROOT
    / "rssi/place_wallis_vehicles_0745_0815_1s_opaque_"
    "no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz"
)
DEFAULT_TESTSET = (
    DATA_ROOT
    / "testset/place_wallis_street_pairs_10000_opaque_"
    "no_vehicle_blockers_static_floor100.npz"
)
DEFAULT_NET = (
    ROOT
    / "SUMO/luxembourg_real_city/place_wallis/map/sionna/"
    "place_wallis_300m_radio_bounds.net.xml"
)
DEFAULT_RESULTS = (
    ROOT / "artifacts/place_wallis_benchmark/methods/equal_greedy_eval50_tail10x25"
)


def equal_average(
    states: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Return the unweighted arithmetic mean of compatible model states."""
    if not states:
        raise ValueError("cannot average an empty model list")
    result: dict[str, torch.Tensor] = {}
    weight = 1.0 / float(len(states))
    for name, first in states[0].items():
        if not (first.is_floating_point() or first.is_complex()):
            result[name] = first.detach().clone()
            continue
        value = torch.zeros_like(first)
        for state in states:
            value.add_(state[name], alpha=weight)
        result[name] = value
    return result


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class EqualGreedySimulation(SumoT2Simulation):
    checkpoint_format = "place_wallis_equal_greedy_checkpoint_v1"

    def __init__(
        self,
        cfg,
        *,
        testset: Path,
        reception_floor_dbm: float,
        training_params: TrainingParams,
        tail_evaluation_steps: tuple[int, ...],
        method_tag: str = "",
        resume: Path | None = None,
        **kwargs: Any,
    ) -> None:
        self._testset_path = Path(testset).resolve()
        self.reception_floor_dbm = float(reception_floor_dbm)
        self.training_params = training_params
        self.tail_evaluation_steps = tuple(
            int(step) for step in tail_evaluation_steps
        )
        if not self.tail_evaluation_steps:
            raise ValueError("tail evaluation schedule cannot be empty")
        self.method_tag = str(method_tag).strip().replace(" ", "_")
        if any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in self.method_tag
        ):
            raise ValueError(
                "method_tag must contain only letters, digits, _ or -"
            )
        self._replay_buffers: dict[int, ReplayBuffer] = {}
        self._staged_measurements: list[tuple[int, int, int, float]] | None = None
        self._resume_step = -1
        self._resume_payload: dict[str, Any] | None = None
        self._resume_logs_restored = True
        super().__init__(cfg, aux_baselines="greedy", **kwargs)
        self._communication_assumptions.update(
            {
                "method": "synchronous equal-weight all-neighbour averaging",
                "model_average_members": "self plus every feasible neighbour",
                "model_average_weights": "equal",
                "raw_samples_shared": False,
                "support_shared": False,
                "experience_shared": False,
                "round_order": "synchronous aggregate, then local train",
                "optimizer_reset": "after every external model average",
                "local_training": asdict(training_params),
                "vehicle_identity": "persistent while inactive and on re-entry",
                "reception_floor_dbm": self.reception_floor_dbm,
                "definitive_evaluation_steps": list(
                    self.tail_evaluation_steps
                ),
            }
        )
        if resume is not None:
            self._load_checkpoint(Path(resume))

    def _reset_aux_node(
        self,
        i: int,
        *,
        old_az: int | None = None,
        new_az: int | None = None,
    ) -> None:
        super()._reset_aux_node(i, old_az=old_az, new_az=new_az)
        self._replay_buffers.pop(int(i), None)

    def _load_measurement_trace(self, path: Path) -> dict[str, object]:
        replay = super()._load_measurement_trace(path)
        metadata = replay["meta"]
        assert isinstance(metadata, dict)
        if "permanently assigned" not in str(
            metadata.get("identity_semantics", "")
        ):
            raise ValueError("trace does not declare persistent vehicle identities")
        states = replay["node_states"]
        active = replay["node_active"]
        assert isinstance(states, np.ndarray)
        assert isinstance(active, np.ndarray)
        replay["node_active"] = active & (states[:, :, 2] >= 0.0)
        measurements = replay["measurements"]
        assert isinstance(measurements, dict)
        replay["measurements"] = {
            int(step): rows[rows[:, 4] >= self.reception_floor_dbm]
            for step, rows in measurements.items()
        }
        return replay

    def _build_fidelity_grid(self, n_pairs: int | None = None, zones=None) -> None:
        del zones
        if int(n_pairs or 0) <= 0:
            self.fidelity_grid = {}
            return
        with np.load(self._testset_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["meta_json"].item()))
            if not bool(metadata.get("buildings_opaque", False)):
                raise ValueError("test set does not use opaque buildings")
            if bool(metadata.get("dynamic_vehicle_blockers", True)):
                raise ValueError("test set uses vehicle blockers")
            if float(metadata["rssi_floor_dbm"]) != self.reception_floor_dbm:
                raise ValueError("test-set and experiment floors differ")
            X = np.asarray(archive["X"], dtype=np.float32)
            raw_y = np.asarray(archive["rssi_dbm"], dtype=np.float32).reshape(
                -1, 1
            )
        limit = min(int(n_pairs or len(X)), int(len(X)))
        # Exact-floor targets are censored non-feasible links.
        self._fidelity_feasible = (
            raw_y[:limit].reshape(-1) > self.reception_floor_dbm
        )
        self.fidelity_grid = {
            0: (
                X[:limit],
                np.maximum(raw_y[:limit], self.reception_floor_dbm),
            )
        }

    def _gossip_step(self, *args, **kwargs) -> None:
        return None

    def _train_predictors_from_current_measurements(
        self,
        *,
        step: int,
        measurements: list[tuple[int, int, int, float]],
    ) -> None:
        self._staged_measurements = (
            None if int(step) <= self._resume_step else list(measurements)
        )

    def _restore_logs(self) -> None:
        if self._resume_logs_restored or self._resume_payload is None:
            return
        self.sharing_rows = list(self._resume_payload.get("sharing_rows", []))
        self.local_policy_rows = list(
            self._resume_payload.get("local_policy_rows", [])
        )
        self.fidelity_history = list(
            self._resume_payload.get("fidelity_history", [])
        )
        self._resume_logs_restored = True

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        del zone_nodes
        step = int(getattr(self, "_current_sumo_step", 0))
        if step <= self._resume_step:
            self.sharing_rows.clear()
            self.local_policy_rows.clear()
            return 0
        self._restore_logs()
        links = sorted(
            {
                (int(zone), min(int(a), int(b)), max(int(a), int(b)))
                for zone, a, b in (contact_links or [])
                if int(a) != int(b)
            }
        )
        neighbours: dict[int, list[int]] = {}
        for _zone, left, right in links:
            neighbours.setdefault(left, []).append(right)
            neighbours.setdefault(right, []).append(left)
        participants = sorted(neighbours)
        pre_states = {
            index: {
                name: value.detach().clone()
                for name, value in self.greedy_models[index].state_dict().items()
            }
            for index in participants
        }
        next_states = {
            receiver: equal_average(
                [
                    pre_states[sender]
                    for sender in [receiver, *sorted(neighbours[receiver])]
                ]
            )
            for receiver in participants
        }
        for receiver, state in next_states.items():
            self.greedy_models[receiver].load_state_dict(state)
            self.greedy_opts[receiver].state.clear()
        parameter_count = (
            sum(tensor.numel() for tensor in next(iter(pre_states.values())).values())
            if pre_states
            else 0
        )
        transfers = 2 * len(links)
        self._network_step_stats.update(
            {
                "synchronous_greedy_receivers": int(len(participants)),
                "equal_weight_model_transfers": int(transfers),
                "model_payload_bytes": int(4 * parameter_count * transfers),
            }
        )
        self._train_local(step)
        return int(transfers)

    def _train_local(self, step: int) -> None:
        measurements = self._staged_measurements or []
        rows_by_receiver: dict[int, list[tuple[list[float], float]]] = {}
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
            rows_by_receiver.setdefault(int(rx_idx), []).append(
                (features, float(value))
            )
        active = {
            index
            for index in range(int(self.cfg.num_nodes))
            if bool(self._current_node_active[index])
        }
        receivers = sorted(
            set(rows_by_receiver) | (active & set(self._replay_buffers))
        )
        for receiver in receivers:
            rows = rows_by_receiver.get(receiver, [])
            X = np.asarray([row[0] for row in rows], dtype=np.float32).reshape(
                -1, 4
            )
            y = np.asarray([row[1] for row in rows], dtype=np.float32).reshape(
                -1, 1
            )
            replay = self._replay_buffers.get(receiver)
            rng = np.random.default_rng(
                np.random.SeedSequence([self.cfg.seed, step, receiver])
            )
            if rows:
                self._train_array(
                    receiver,
                    X,
                    y,
                    epochs=self.training_params.new_data_epochs,
                    rng=rng,
                )
            full_dataset_training = (
                self.training_params.full_dataset_epochs > 0
            )
            if full_dataset_training and rows:
                if replay is None:
                    replay = ReplayBuffer(
                        self.training_params.replay_capacity, X.shape[1]
                    )
                    self._replay_buffers[receiver] = replay
                replay.add(X, y)
                self.greedy_m_samples[receiver] += len(rows)
                self.greedy_n_samples[receiver] = self.greedy_m_samples[receiver]
            if replay is not None and replay.size > 0:
                if full_dataset_training:
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
                    for batch_index in range(
                        self.training_params.replay_batches
                    ):
                        replay_X, replay_y = replay.sample(
                            rng,
                            int(self.cfg.local_batch_size),
                            recent_window=(
                                self.training_params.recent_window
                                if batch_index >= recent_start
                                else None
                            ),
                        )
                        self._train_array(
                            receiver, replay_X, replay_y, epochs=1, rng=rng
                        )
            if rows and not full_dataset_training:
                if replay is None:
                    replay = ReplayBuffer(
                        self.training_params.replay_capacity, X.shape[1]
                    )
                    self._replay_buffers[receiver] = replay
                replay.add(X, y)
                # Local experience is never transmitted or merged.
                self.greedy_m_samples[receiver] += len(rows)
                self.greedy_n_samples[receiver] = self.greedy_m_samples[receiver]
        self._staged_measurements = None

    def _train_array(
        self,
        receiver: int,
        X: np.ndarray,
        y_dbm: np.ndarray,
        *,
        epochs: int,
        rng: np.random.Generator,
    ) -> None:
        model = self.greedy_models[receiver]
        optimizer = self.greedy_opts[receiver]
        y = self._normalize_target_from_rssi(y_dbm)
        batch_size = int(self.cfg.local_batch_size)
        model.train()
        for _epoch in range(int(epochs)):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                indices = order[start : start + batch_size]
                bx = torch.as_tensor(
                    X[indices], dtype=torch.float32, device=self.aux_device
                )
                by = torch.as_tensor(
                    y[indices], dtype=torch.float32, device=self.aux_device
                )
                optimizer.zero_grad(set_to_none=True)
                loss = self._weighted_regression_loss(model(bx), by, None)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.training_params.gradient_clip_norm,
                )
                optimizer.step()

    def _evaluate_fidelity_now(
        self, step: int, *, n_pairs: int, is_final: int
    ) -> dict[str, float | int]:
        if int(step) <= self._resume_step and self._resume_payload is not None:
            for row in reversed(self._resume_payload.get("fidelity_history", [])):
                if int(row.get("step", -1)) == int(step):
                    return dict(row)
            return {"step": int(step)}
        self._build_fidelity_grid(n_pairs=n_pairs)
        X, y = self.fidelity_grid[0]
        truth = y.reshape(-1)
        feasible = self._fidelity_feasible
        active = [
            i
            for i in range(int(self.cfg.num_nodes))
            if bool(self._current_node_active[i])
            and int(self.greedy_m_samples[i]) > 0
        ]
        sums = {"all": 0.0, "feasible": 0.0, "infeasible": 0.0}
        predicted = {"all": 0, "feasible": 0, "infeasible": 0}
        model_rmse: list[float] = []
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.aux_device)
        for index in active:
            model = self.greedy_models[index]
            model.eval()
            with torch.no_grad():
                prediction = self._denorm_dbm(
                    model(xt).detach().cpu().numpy().reshape(-1)
                )
            error_sq = np.square(prediction - truth)
            positive = prediction > self.reception_floor_dbm
            sums["all"] += float(error_sq.sum())
            sums["feasible"] += float(error_sq[feasible].sum())
            sums["infeasible"] += float(error_sq[~feasible].sum())
            predicted["all"] += int(positive.sum())
            predicted["feasible"] += int(positive[feasible].sum())
            predicted["infeasible"] += int(positive[~feasible].sum())
            model_rmse.append(float(np.sqrt(error_sq.mean())))
        all_count = len(active) * len(truth)
        feasible_count = len(active) * int(feasible.sum())
        infeasible_count = len(active) * int((~feasible).sum())

        def rmse(key: str, count: int) -> float:
            return (
                float(math.sqrt(sums[key] / count))
                if count
                else float("nan")
            )

        def ratio(key: str, count: int) -> float:
            return float(predicted[key] / count) if count else float("nan")

        row: dict[str, float | int] = {
            "step": int(step),
            "eval_n_pairs_per_zone": int(len(X)),
            "eval_is_final": int(is_final),
            "greedy_total": rmse("all", all_count),
            "greedy_mean_model_rmse": (
                float(np.mean(model_rmse)) if model_rmse else float("nan")
            ),
            "greedy_feasible_rmse": rmse("feasible", feasible_count),
            "greedy_infeasible_rmse": rmse("infeasible", infeasible_count),
            "greedy_active_experienced_models": int(len(active)),
            "greedy_predicted_feasible_fraction": ratio("all", all_count),
            "greedy_feasible_recall": ratio("feasible", feasible_count),
            "greedy_non_feasible_false_positive_rate": ratio(
                "infeasible", infeasible_count
            ),
            "greedy_mean_local_experience": (
                float(np.mean([self.greedy_m_samples[i] for i in active]))
                if active
                else 0.0
            ),
            "greedy_max_local_experience": (
                int(max(self.greedy_m_samples[i] for i in active))
                if active
                else 0
            ),
        }
        self.fidelity_history.append(row)
        return row

    @staticmethod
    def _cpu_tree(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {
                key: EqualGreedySimulation._cpu_tree(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [EqualGreedySimulation._cpu_tree(item) for item in value]
        return value

    def _save_checkpoint(self, step: int) -> None:
        experienced = [
            i for i, count in enumerate(self.greedy_m_samples) if int(count) > 0
        ]
        output = Path(self.cfg.results_dir)
        output.mkdir(parents=True, exist_ok=True)
        latest = self.fidelity_history[-1] if self.fidelity_history else {}
        temporal = temporal_metric_summary(
            self.fidelity_history,
            evaluation_steps=self.tail_evaluation_steps,
            metric_keys=(
                "greedy_total",
                "greedy_feasible_rmse",
                "greedy_infeasible_rmse",
                "greedy_predicted_feasible_fraction",
                "greedy_non_feasible_false_positive_rate",
            ),
        )
        means = temporal["mean"]
        deviations = temporal["standard_deviation"]
        tail_complete = bool(temporal["complete"])
        status = (
            "complete"
            if step >= int(self.cfg.sim_steps) and tail_complete
            else "running"
        )
        method_suffix = f"_{self.method_tag}" if self.method_tag else ""
        method_label = f" ({self.method_tag})" if self.method_tag else ""
        atomic_json(
            output / "checkpoint_status.json",
            {
                "format": self.checkpoint_format,
                "step": int(step),
                "checkpoint_kind": "metrics-only",
                "path": None,
                "experienced_models": len(experienced),
                "latest_fidelity": latest,
                "temporal_evaluation": temporal,
            },
        )
        atomic_json(
            output / "metrics.json",
            {
                "schema": "place_wallis_benchmark_result_v1",
                "status": status,
                "method": {
                    "id": f"equal_greedy{method_suffix}",
                    "name": f"Ungated equal-weight greedy sharing{method_label}",
                    "model": "4-64-64-1 MLP",
                },
                "checkpoint": {
                    "step": int(step),
                    "final_step": int(self.cfg.sim_steps),
                },
                "metrics_db": {
                    "overall_rmse": means["greedy_total"],
                    "feasible_rmse": means["greedy_feasible_rmse"],
                    "non_feasible_rmse": means["greedy_infeasible_rmse"],
                },
                "metrics_db_standard_deviation": {
                    "overall_rmse": deviations["greedy_total"],
                    "feasible_rmse": deviations["greedy_feasible_rmse"],
                    "non_feasible_rmse": deviations[
                        "greedy_infeasible_rmse"
                    ],
                },
                "evaluation": {
                    "noise_floor_dbm": self.reception_floor_dbm,
                    "true_feasible_rule": "rssi_dbm > -100",
                    "predicted_feasible_fraction": means[
                        "greedy_predicted_feasible_fraction"
                    ],
                    "non_feasible_false_positive_rate": means[
                        "greedy_non_feasible_false_positive_rate"
                    ],
                },
                "temporal_evaluation": temporal,
                "training": asdict(self.training_params),
                "latest_fidelity": latest,
                "testset": {
                    "path": str(self._testset_path),
                    "samples": int(
                        latest.get("eval_n_pairs_per_zone", 0)
                    ),
                },
            },
        )
        if latest:
            print(
                f"[EQUAL-GREEDY] metrics={step} "
                f"overall={float(latest['greedy_total']):.4f}dB "
                f"feasible={float(latest['greedy_feasible_rmse']):.4f}dB "
                f"non-feasible={float(latest['greedy_infeasible_rmse']):.4f}dB "
                f"false-positive={100 * float(latest['greedy_non_feasible_false_positive_rate']):.1f}%",
                flush=True,
            )

    def _load_checkpoint(self, path: Path) -> None:
        payload = torch.load(
            path.resolve(), map_location=self.aux_device, weights_only=False
        )
        if payload.get("format") != self.checkpoint_format:
            raise ValueError("unsupported checkpoint format")
        if payload.get("training_params") != asdict(self.training_params):
            raise ValueError("checkpoint training parameters differ")
        if tuple(payload.get("tail_evaluation_steps", ())) != (
            self.tail_evaluation_steps
        ):
            raise ValueError("checkpoint tail evaluation schedule differs")
        if str(Path(payload["trace"]).resolve()) != str(
            Path(self.measurement_trace_in).resolve()
        ):
            raise ValueError("checkpoint trace differs")
        self._resume_step = int(payload["step"])
        experience = [int(value) for value in payload["local_experience"]]
        self.greedy_m_samples = list(experience)
        self.greedy_n_samples = list(experience)
        for index, state in payload["models"].items():
            self.greedy_models[int(index)].load_state_dict(state)
        for index, state in payload["optimizers"].items():
            self.greedy_opts[int(index)].load_state_dict(state)
        self._replay_buffers = {
            int(index): ReplayBuffer.from_state_dict(state)
            for index, state in payload["replay_buffers"].items()
        }
        self._resume_payload = payload
        self._resume_logs_restored = False
        print(
            f"[EQUAL-GREEDY] resumed step={self._resume_step} from {path}",
            flush=True,
        )

    def _write_partial_outputs(self, **kwargs) -> None:
        step = int(kwargs.get("step", 0))
        if step <= self._resume_step and self._resume_payload is not None:
            return
        super()._write_partial_outputs(**kwargs)
        if step > 0 and (
            step % max(1, int(self.flush_every)) == 0
            or step >= int(self.cfg.sim_steps)
        ):
            self._save_checkpoint(step)


def validate_dataset(trace: Path, testset: Path) -> dict[str, int | float]:
    with np.load(trace, allow_pickle=False) as archive:
        trace_meta = json.loads(str(archive["meta_json"].item()))
        if not bool(trace_meta.get("buildings_opaque", False)):
            raise ValueError("trace does not use opaque buildings")
        if bool(trace_meta.get("dynamic_vehicle_blockers", True)):
            raise ValueError("trace uses vehicle blockers")
        if not np.all(np.asarray(archive["node_generations"]) == 0):
            raise ValueError("trace reuses physical vehicle slots")
        minimum = trace_meta.get("measurement_filter", {}).get(
            "minimum_rssi_dbm"
        )
        if float(minimum) != -100.0:
            raise ValueError("trace is not filtered at -100 dBm")
        trace_bounds = trace_meta.get("zone_layout", {}).get(
            "bounds_local_xy_m"
        )
        if trace_bounds is None:
            raise ValueError("trace metadata does not define local map bounds")
    with np.load(testset, allow_pickle=False) as archive:
        test_meta = json.loads(str(archive["meta_json"].item()))
        if float(test_meta["rssi_floor_dbm"]) != -100.0:
            raise ValueError("test set is not censored at -100 dBm")
        test_count = int(archive["X"].shape[0])
        test_bounds = test_meta.get("region_bounds_local_xy_m")
        if test_bounds is None:
            raise ValueError("test-set metadata does not define local map bounds")

    def square_map_size(bounds: object, source: str) -> float:
        values = np.asarray(bounds, dtype=np.float64).reshape(-1)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError(f"{source} map bounds must contain four finite values")
        xmin, ymin, xmax, ymax = (float(value) for value in values)
        width, height = xmax - xmin, ymax - ymin
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{source} map bounds must have positive extent")
        if not math.isclose(width, height, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(f"{source} map bounds must define a square zone")
        if not math.isclose(xmin, 0.0, abs_tol=1.0e-6) or not math.isclose(
            ymin, 0.0, abs_tol=1.0e-6
        ):
            raise ValueError(f"{source} map bounds must use a local zero origin")
        return float(width)

    trace_map_size = square_map_size(trace_bounds, "trace")
    test_map_size = square_map_size(test_bounds, "test-set")
    if not math.isclose(
        trace_map_size, test_map_size, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError("trace and test-set map sizes differ")
    return {
        "num_nodes": int(trace_meta["num_nodes"]),
        "num_zones": int(trace_meta["num_zones"]),
        "sim_steps": int(trace_meta["sim_steps"]),
        "tx_power_dbm": float(trace_meta["tx_power_dbm"]),
        "rssi_max_dbm": float(trace_meta["rssi_max_dbm"]),
        "test_count": test_count,
        "map_size": trace_map_size,
    }


def self_test() -> None:
    states = [
        {"x": torch.tensor([1.0, 3.0])},
        {"x": torch.tensor([3.0, 5.0])},
        {"x": torch.tensor([5.0, 7.0])},
    ]
    assert torch.allclose(
        equal_average(states)["x"], torch.tensor([3.0, 5.0])
    )
    replay = ReplayBuffer(3)
    replay.add(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        np.arange(4, dtype=np.float32).reshape(-1, 1),
    )
    assert ReplayBuffer.from_state_dict(replay.state_dict()).size == 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--method-tag", default="")
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--local-lr", type=float, default=5.0e-4)
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--new-data-epochs", type=int, default=2)
    parser.add_argument("--replay-batches", type=int, default=8)
    parser.add_argument("--recent-replay-batches", type=int, default=4)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--full-dataset-epochs", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--tail-eval-count", type=int, default=10)
    parser.add_argument("--tail-eval-stride", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("equal greedy self-test passed")
        return 0
    metadata = validate_dataset(args.trace.resolve(), args.testset.resolve())
    sim_steps = (
        int(metadata["sim_steps"])
        if args.sim_steps is None
        else int(args.sim_steps)
    )
    if sim_steps > int(metadata["sim_steps"]):
        raise ValueError("requested steps exceed trace")
    tail_evaluation_steps = make_tail_evaluation_steps(
        sim_steps,
        count=int(args.tail_eval_count),
        stride=int(args.tail_eval_stride),
    )
    results_dir = args.results_dir.resolve()
    resume = args.resume
    automatic = results_dir / "checkpoint_latest.pt"
    if resume is None and args.resume_if_exists and automatic.exists():
        resume = automatic
    training = TrainingParams(
        replay_capacity=int(args.replay_capacity),
        new_data_epochs=int(args.new_data_epochs),
        replay_batches=int(args.replay_batches),
        recent_replay_batches=int(args.recent_replay_batches),
        recent_window=int(args.recent_window),
        full_dataset_epochs=int(args.full_dataset_epochs),
        gradient_clip_norm=float(args.gradient_clip_norm),
    )
    checkpoint_every = max(1, int(args.checkpoint_every))
    cfg = build_config_from_env(
        seed=int(args.seed),
        num_nodes=int(metadata["num_nodes"]),
        num_zones=int(metadata["num_zones"]),
        sim_steps=sim_steps,
        map_size=300.0,
        active_modes=(),
        results_dir=str(results_dir),
        tx_power_dbm=float(metadata["tx_power_dbm"]),
        rssi_min_dbm=-100.0,
        rssi_max_dbm=float(metadata["rssi_max_dbm"]),
        noise_floor_dbm=-100.0,
        snr_min_db=0.0,
        model_transfer_snr_min_db=0.0,
        rssi_model="tiny",
        predictor_prior="none",
        predictor_include_time=False,
        local_lr=float(args.local_lr),
        local_batch_size=int(args.local_batch_size),
        local_epochs=1,
        merge_strategy="average",
        fidelity_grid_per_zone=int(metadata["test_count"]),
        fidelity_eval_every=checkpoint_every,
        final_fidelity_grid_per_zone=int(metadata["test_count"]),
        fidelity_final_steps=tail_evaluation_steps,
        fidelity_log_every=0,
        verbose=not bool(args.quiet),
        spike_recovery_enabled=False,
    )
    simulation = EqualGreedySimulation(
        cfg,
        sumo_config=str(args.net.resolve()),
        sumo_net=str(args.net.resolve()),
        measurement_trace_in=str(args.trace.resolve()),
        testset=args.testset.resolve(),
        reception_floor_dbm=-100.0,
        training_params=training,
        tail_evaluation_steps=tail_evaluation_steps,
        method_tag=str(args.method_tag),
        resume=resume,
        progress_every=int(args.progress_every),
        log_rmse_every=0,
        flush_every=checkpoint_every,
        random_od_routing=False,
        local_policy_share=False,
    )
    simulation.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
