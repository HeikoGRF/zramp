#!/usr/bin/env python3
"""
SUMO mobility + RL reward experiment semantics.

Runs parallel RL heads: multiple β for ``--reward-t`` (compound ids ``tT_bβ``), plus plain
future-window variants ``t1``…``t7`` via ``--also-windows``, optional oracle ``v4``. Local-only,
greedy-share, and central-per-zone models share one SUMO rollout.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
import zlib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROUTE_REASSIGN_EDGE_BUFFER = 4
INVALID_SUMO_POSITION = -1.0e9
DECISION_LOG_FIELDS = [
    "step",
    "enc_id",
    "node_i",
    "node_j",
    "az",
    "dist",
    "mode",
    "action",
    "merge_weight",
    "predicted_gain",
    "gain_threshold",
    "exploratory",
    "reward",
    "deferred",
]


def _valid_sumo_position(x: float, y: float) -> bool:
    return float(x) > INVALID_SUMO_POSITION and float(y) > INVALID_SUMO_POSITION

from model import TinyRSSIPredictor, make_rssi_predictor
from rl_reward_experiment.config import (
    WINDOW_T_BY_MODE,
    WINDOW_T_VALUES,
    build_config_from_env,
    parse_modes,
)
from rl_reward_experiment.mobility import (
    collides_with_walls,
    group_pairs_by_tx,
    zone_bounds,
    zone_of,
)
import rl_reward_experiment.sim as rre_sim
from rl_reward_experiment.node_state import bound_raw_samples, saturate_n_samples
from rl_reward_experiment.rl_agent import DQNAgent
from rl_reward_experiment.sim import Simulation
from SUMO.sumo_sionna_map import SumoNetSionnaMap, read_net_bounds, sionna_variant_for_net


class _TraceReplayMap:
    """Minimal map facade used when replaying cached measurements."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.walls: list = []
        self.dynamic_obstacles: list = []
        self.dynamic_schedule = None
        self._last_dynamic_active: tuple[str, ...] = ()

    def build(self):
        return None

    def cleanup(self) -> None:
        return None


class _TraceReplayRayTracer:
    """Dummy tracer; replay mode injects cached grids before this is used."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def step_measurements(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("Trace replay should read cached step measurements")

    def measure_pairs(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("Trace replay should read cached fidelity grids")


class _PackedModelState:
    """One contiguous CPU tensor plus a reusable state-dict layout spec."""

    __slots__ = ("flat", "spec")

    def __init__(self, flat: torch.Tensor, spec: tuple[tuple[str, tuple[int, ...], torch.dtype, int], ...]):
        self.flat = flat
        self.spec = spec

    def to_state_dict(self, *, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape, dtype, numel in self.spec:
            view = self.flat.narrow(0, offset, int(numel)).view(shape)
            if view.dtype != dtype:
                view = view.to(dtype=dtype)
            if device is not None:
                view = view.to(device)
            out[name] = view
            offset += int(numel)
        return out


class SumoT2Simulation(Simulation):
    def __init__(
        self,
        cfg,
        *,
        sumo_config: str,
        sumo_net: str,
        dynamic_map: str | None = None,
        warmup_limit_steps: int = 5000,
        skip_aux_baselines: bool = False,
        aux_baselines: str | None = None,
        progress_every: int = 0,
        log_rmse_every: int = 0,
        flush_every: int = 0,
        max_wall_seconds: float | None = None,
        random_od_routing: bool = False,
        route_min_zone_distance: int = 1,
        route_max_zone_distance: int | None = 1,
        open_boundary_routing: bool = False,
        open_boundary_probability: float = 0.45,
        open_boundary_margin: float = 0.12,
        open_boundary_exit_margin: float = 0.035,
        open_boundary_respawn_buffer: int = 2,
        jam_reroute_wait_seconds: float = 25.0,
        intersection_control: bool = False,
        intersection_wait_seconds: float = 12.0,
        intersection_release_steps: int = 8,
        intersection_stop_distance: float = 24.0,
        zone_model_memory: bool = False,
        local_policy_share: bool = True,
        local_policy_initial_pull: str = "byte-match",
        local_policy_initial_pull_prob: float | None = None,
        local_policy_updates_per_batch: int = 1,
        central_accumulate_samples: bool = False,
        mobility_trace_in: str | None = None,
        measurement_trace_in: str | None = None,
        measurement_trace_out: str | None = None,
        trace_record_only: bool = False,
    ):
        self.sumo_config = str(Path(sumo_config).resolve())
        self.sumo_net = str(Path(sumo_net).resolve())
        self.dynamic_map = str(Path(dynamic_map).resolve()) if dynamic_map else None
        self.mobility_trace_in = (
            str(Path(mobility_trace_in).resolve()) if mobility_trace_in else None
        )
        self.measurement_trace_in = (
            str(Path(measurement_trace_in).resolve()) if measurement_trace_in else None
        )
        self.measurement_trace_out = (
            str(Path(measurement_trace_out).resolve()) if measurement_trace_out else None
        )
        self.trace_record_only = bool(trace_record_only)
        self._trace_replay = (
            self._load_measurement_trace(Path(self.measurement_trace_in))
            if self.measurement_trace_in
            else None
        )
        self._mobility_trace = (
            self._load_mobility_trace(Path(self.mobility_trace_in))
            if self.mobility_trace_in
            else None
        )
        if self._trace_replay is not None and self._mobility_trace is not None:
            raise ValueError("--measurement-trace-in and --mobility-trace-in are mutually exclusive")
        if self._trace_replay is not None:
            trace_meta = self._trace_replay.get("meta", {})
            trace_steps = int(trace_meta.get("sim_steps", 0)) if isinstance(trace_meta, dict) else 0
            if trace_steps < int(cfg.sim_steps):
                raise ValueError(
                    f"Measurement trace {self.measurement_trace_in} has sim_steps={trace_steps}, "
                    f"but this run requested sim_steps={int(cfg.sim_steps)}"
                )
        if self._mobility_trace is not None:
            positions = self._mobility_trace.get("positions")
            if not isinstance(positions, np.ndarray):
                raise ValueError(f"Mobility trace {self.mobility_trace_in} did not load positions")
            if positions.shape[0] < int(cfg.sim_steps) + 1:
                raise ValueError(
                    f"Mobility trace {self.mobility_trace_in} has {positions.shape[0]} stored steps, "
                    f"but this run needs {int(cfg.sim_steps) + 1}"
                )
            if positions.shape[1] < int(cfg.num_nodes):
                raise ValueError(
                    f"Mobility trace {self.mobility_trace_in} has {positions.shape[1]} vehicles, "
                    f"but this run requested {int(cfg.num_nodes)} nodes"
                )
        self._trace_record_path = (
            Path(self.measurement_trace_out) if self.measurement_trace_out else None
        )
        self._trace_node_state_by_step: dict[int, np.ndarray] = {}
        self._trace_node_generation_by_step: dict[int, np.ndarray] = {}
        self._current_node_active = np.ones((int(cfg.num_nodes),), dtype=np.bool_)
        self._node_generations = [0 for _ in range(int(cfg.num_nodes))]
        self._replacement_pending_nodes: set[int] = set()
        self._trace_synced_by_step: dict[int, int] = {}
        self._trace_measurement_rows: list[np.ndarray] = []
        self._trace_fidelity_events: list[dict[str, object]] = []
        self._trace_fidelity_read_index = 0
        self._trace_fidelity_step = 0
        self._trace_dynamic_by_step: dict[int, str] = {}
        self._trace_refresh_zones_by_step: dict[int, list[int]] = {}
        self._warmup_limit_steps = int(warmup_limit_steps)
        self.central_accumulate_samples = bool(central_accumulate_samples)
        if aux_baselines is None:
            aux_baselines = "none" if skip_aux_baselines else "all"
        requested_aux = {
            part.strip().lower()
            for part in str(aux_baselines).replace("+", ",").split(",")
            if part.strip()
        }
        if "all" in requested_aux:
            requested_aux = {"iso", "greedy", "central"}
        if "none" in requested_aux:
            requested_aux = set()
        unknown_aux = requested_aux.difference({"iso", "greedy", "central"})
        if unknown_aux:
            raise ValueError(f"Unknown auxiliary baselines: {sorted(unknown_aux)}")
        self.aux_baselines = requested_aux
        self.skip_aux_baselines = not self.aux_baselines
        self.progress_every = max(0, int(progress_every))
        self.log_rmse_every = max(0, int(log_rmse_every))
        self.flush_every = max(0, int(flush_every))
        self.max_wall_seconds = (
            None if max_wall_seconds is None else max(0.0, float(max_wall_seconds))
        )
        self.random_od_routing = bool(random_od_routing)
        self.route_min_zone_distance = max(1, int(route_min_zone_distance))
        self.route_max_zone_distance = (
            None if route_max_zone_distance is None else max(1, int(route_max_zone_distance))
        )
        self.open_boundary_routing = bool(open_boundary_routing)
        self.open_boundary_probability = max(0.0, min(1.0, float(open_boundary_probability)))
        self.open_boundary_margin = max(0.01, min(0.45, float(open_boundary_margin)))
        self.open_boundary_exit_margin = max(0.005, min(0.12, float(open_boundary_exit_margin)))
        self.open_boundary_respawn_buffer = max(0, int(open_boundary_respawn_buffer))
        self.jam_reroute_wait_seconds = max(0.0, float(jam_reroute_wait_seconds))
        self.intersection_control = bool(intersection_control)
        self.intersection_wait_seconds = max(0.0, float(intersection_wait_seconds))
        self.intersection_release_steps = max(1, int(intersection_release_steps))
        self.intersection_stop_distance = max(5.0, float(intersection_stop_distance))
        self.zone_model_memory = bool(zone_model_memory)
        self.zramp_policy_mode = "local"
        self.local_policy_share = bool(local_policy_share)
        self.local_policy_initial_pull = str(local_policy_initial_pull).strip().lower().replace("_", "-")
        if self.local_policy_initial_pull not in {"greedy", "byte-match", "fixed"}:
            raise ValueError("local_policy_initial_pull must be greedy, byte-match, or fixed")
        self.local_policy_initial_pull_prob_override = (
            None
            if local_policy_initial_pull_prob is None
            else max(0.0, min(1.0, float(local_policy_initial_pull_prob)))
        )
        self.local_policy_updates_per_batch = max(1, int(local_policy_updates_per_batch))
        self.local_policy_initial_pull_probability = 1.0
        self._last_local_policy_train_updates_this_step = 0
        self.local_agents: dict[str, list[DQNAgent]] = {}
        self._local_policy_pending_transitions: dict[str, list[int]] = {}
        self._local_policy_versions: dict[str, list[int]] = {}
        self._local_policy_train_updates: Counter[str] = Counter()
        self._local_policy_pull_updates: Counter[str] = Counter()
        self._local_policy_initial_accepts: Counter[str] = Counter()
        self._local_policy_initial_decisions: Counter[str] = Counter()
        self._local_policy_initial_rngs: dict[str, list[random.Random]] = {}
        self._last_local_policy_queued_transitions = 0
        self._last_local_policy_pull_updates = 0
        self._comm_cumulative_bytes: Counter[str] = Counter()
        self._communication_assumptions: dict[str, int | float | str | bool] = {}
        self._decision_log_stream_path = Path(cfg.results_dir) / "decisions.csv"
        self._decision_log_stream_file = None
        self._decision_log_stream_writer = None
        self._decision_log_stream_count = 0
        self._decision_action_counts: dict[str, Counter[int]] = defaultdict(Counter)
        self._packed_state_specs: dict[tuple[tuple[str, tuple[int, ...], str, int], ...], tuple[tuple[str, tuple[int, ...], torch.dtype, int], ...]] = {}
        self.local_policy_rows: list[dict[str, int | float]] = []
        self.sharing_rows: list[dict[str, int | float]] = []
        self._veh_ids: list[str] = []
        self._sumo_bbox: tuple[float, float, float, float] | None = None
        self._sumo_open = False
        self._route_rng: random.Random | None = None
        self._routing_edges_by_zone: dict[int, list[str]] = {}
        self._routing_edge_zone: dict[str, int] = {}
        self._routing_edge_shape: dict[str, list[tuple[float, float]]] = {}
        self._routing_edge_lane: dict[str, str] = {}
        self._routing_edge_midpoint: dict[str, tuple[float, float]] = {}
        self._routing_zone_capacity: dict[int, float] = {}
        self._routing_boundary_edges_by_side: dict[str, list[str]] = {}
        self._routing_junction_in_edges: dict[str, list[str]] = {}
        self._junction_release_state: dict[str, tuple[str, int]] = {}
        self._junction_held_vehicle_ids: set[str] = set()
        self._route_target_edge: dict[str, str] = {}
        self._route_target_zone: dict[str, int] = {}
        self._route_target_zone_count: Counter[int] = Counter()
        self._route_zone_load: Counter[int] = Counter()
        self._route_zone_load_by_vehicle: dict[str, Counter[int]] = {}
        self._route_exit_side: dict[str, str] = {}
        self._route_changes = 0
        self._respawn_seq = 0
        self._missing_tracked_vehicles: set[str] = set()
        self._missing_tracked_vehicle_since: dict[str, int] = {}
        # Keep all RL/local-training semantics from rl_reward_experiment,
        # but swap scene builder to one derived from the SUMO network.
        _sionna_variant = sionna_variant_for_net(self.sumo_net)
        orig_map_cls = rre_sim.Complex100mMap
        orig_ray_cls = rre_sim.RayTracer
        if self._trace_replay is not None:
            rre_sim.Complex100mMap = _TraceReplayMap
            rre_sim.RayTracer = _TraceReplayRayTracer
        else:
            rre_sim.Complex100mMap = partial(
                SumoNetSionnaMap,
                net_path=self.sumo_net,
                sionna_variant=_sionna_variant,
                dynamic_schedule_path=self.dynamic_map,
            )
        try:
            super().__init__(cfg)
        finally:
            rre_sim.Complex100mMap = orig_map_cls
            rre_sim.RayTracer = orig_ray_cls
        try:
            self._decision_log_stream_path.unlink()
        except FileNotFoundError:
            pass
        if self._trace_replay is not None and self._explicit_fidelity_schedule_enabled():
            self._trace_fidelity_read_index = 0
            self.fidelity_grid = {}
        self._init_local_policy_agents()
        self._communication_assumptions = self._build_communication_assumptions()
        self.local_policy_initial_pull_probability = self._resolve_local_initial_pull_probability()
        # Additional baselines can be selected independently.  This keeps the
        # t2_b1-vs-iso run cheap while preserving the older "all" comparison.
        self.aux_device = self.device
        self.iso_models: list[nn.Module] = []
        self.iso_opts: list[optim.Optimizer] = []
        self.iso_m_samples: list[int] = []
        self.iso_n_samples: list[int] = []
        self.greedy_models: list[nn.Module] = []
        self.greedy_opts: list[optim.Optimizer] = []
        self.greedy_m_samples: list[int] = []
        self.greedy_n_samples: list[int] = []
        self.central_models: dict[int, nn.Module] = {}
        self.central_opts: dict[int, optim.Optimizer] = {}
        self._central_accumulated_x: dict[int, list[list[float]]] = defaultdict(list)
        self._central_accumulated_y: dict[int, list[float]] = defaultdict(list)
        self._aux_template_state = {
            k: v.detach().clone() for k, v in self.template_state.items()
        }
        if self.aux_baselines:
            for _ in range(cfg.num_nodes):
                if "iso" in self.aux_baselines:
                    m_iso = self._make_predictor().to(self.aux_device)
                    m_iso.load_state_dict(self._aux_template_state)
                    self.iso_models.append(m_iso)
                    self.iso_opts.append(optim.Adam(m_iso.parameters(), lr=cfg.local_lr))
                    self.iso_m_samples.append(0)
                    self.iso_n_samples.append(0)

                if "greedy" in self.aux_baselines:
                    m_g = self._make_predictor().to(self.aux_device)
                    m_g.load_state_dict(self._aux_template_state)
                    self.greedy_models.append(m_g)
                    self.greedy_opts.append(optim.Adam(m_g.parameters(), lr=cfg.local_lr))
                    self.greedy_m_samples.append(0)
                    self.greedy_n_samples.append(0)
            if "central" in self.aux_baselines:
                for az in range(cfg.num_zones):
                    m_c = self._make_predictor().to(self.aux_device)
                    m_c.load_state_dict(self._aux_template_state)
                    self.central_models[az] = m_c
                    self.central_opts[az] = optim.Adam(m_c.parameters(), lr=cfg.local_lr)
        self._communication_assumptions.update(
            {
                "fidelity_primary_metric": (
                    "pooled-rmse-over-every-individual-model-pair-prediction"
                ),
                "fidelity_mean_model_metric": (
                    "arithmetic-mean-of-independently-computed-model-rmses"
                ),
                "central_training_samples": (
                    "all accumulated feasible directed-link measurements per zone"
                    if self.central_accumulate_samples
                    else "current-step feasible directed-link measurements per zone"
                ),
                "central_accumulate_samples": bool(
                    self.central_accumulate_samples
                ),
            }
        )
        self._node_zone_memory: list[dict[int, dict[str, object]]] = [
            {} for _ in range(cfg.num_nodes)
        ]
        self._aux_zone_memory: list[dict[int, dict[str, object]]] = [
            {} for _ in range(cfg.num_nodes)
        ]
        self._route_rng = random.Random(int(cfg.seed) + 8_271_913)
        if self.random_od_routing:
            self._build_routing_edge_index()
        if self._mobility_trace is not None:
            synced = self._apply_mobility_trace_node_state(0, reset_on_zone_change=True)
            if self._trace_record_path is not None:
                self._record_trace_node_state(0, synced=synced)
        elif self._trace_replay is None:
            self._attach_sumo_nodes()
            if self._trace_record_path is not None:
                self._record_trace_node_state(0, synced=self.cfg.num_nodes)
        else:
            self._apply_trace_node_state(0, reset_on_zone_change=True)

    def _explicit_fidelity_schedule_enabled(self) -> bool:
        final_steps = tuple(int(s) for s in getattr(self.cfg, "fidelity_final_steps", ()) or ())
        return int(getattr(self.cfg, "fidelity_eval_every", 0) or 0) > 0 or bool(final_steps)

    def _fidelity_schedule_spec(self, step: int) -> tuple[int, int] | None:
        step_i = int(step)
        final_steps = {
            int(s)
            for s in (getattr(self.cfg, "fidelity_final_steps", ()) or ())
            if 1 <= int(s) <= int(self.cfg.sim_steps)
        }
        if step_i in final_steps:
            return int(self.cfg.final_fidelity_grid_per_zone), 1
        every = int(getattr(self.cfg, "fidelity_eval_every", 0) or 0)
        if every > 0 and step_i % every == 0:
            return int(self.cfg.fidelity_grid_per_zone), 0
        return None

    def _evaluate_fidelity_now(self, step: int, *, n_pairs: int, is_final: int) -> dict[str, float | int]:
        self._trace_fidelity_step = int(step)
        self._build_fidelity_grid(n_pairs=int(n_pairs))
        self._trace_refresh_zones_by_step[int(step)] = list(range(int(self.cfg.num_zones)))
        row = self._compute_fidelity_row(int(step))
        row = self._compute_aux_fidelity(row)
        row["eval_n_pairs_per_zone"] = int(n_pairs)
        row["eval_is_final"] = int(is_final)
        self.fidelity_history.append(row)
        return row

    def _latest_fidelity_at_step(self, step: int) -> dict[str, float | int] | None:
        step_i = int(step)
        for row in reversed(self.fidelity_history):
            if int(row.get("step", -1)) == step_i:
                return dict(row)
        return None

    def _load_measurement_trace(self, path: Path) -> dict[str, object]:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"].item()))
            trace_format = str(meta.get("format", ""))
            if trace_format not in {"sumo_rssi_trace_v2", "sumo_rssi_trace_v3"}:
                raise ValueError(
                    f"Measurement trace {path} does not preserve vehicle replacement "
                    "generations; regenerate it with the v2/v3 trace writer"
                )
            node_states = np.asarray(data["node_states"], dtype=np.float32)
            if "node_generations" not in data.files:
                raise ValueError("v2 measurement trace is missing node_generations")
            node_generations = np.asarray(data["node_generations"], dtype=np.int32)
            if node_generations.shape != node_states.shape[:2]:
                raise ValueError("node_generations shape differs from node_states")
            node_active = (
                np.asarray(data["node_active"], dtype=np.bool_)
                if "node_active" in data.files
                else np.ones(node_states.shape[:2], dtype=np.bool_)
            )
            if node_active.shape != node_states.shape[:2]:
                raise ValueError("node_active shape differs from node_states")
            synced_raw = data["synced"] if "synced" in data.files else np.zeros((node_states.shape[0],), dtype=np.int32)
            meas_raw = data["measurements"] if "measurements" in data.files else np.zeros((0, 5), dtype=np.float32)
            synced = np.asarray(synced_raw, dtype=np.int32)
            measurements = np.asarray(meas_raw, dtype=np.float32)

            meas_by_step: dict[int, np.ndarray] = {}
            if measurements.size:
                steps = measurements[:, 0].astype(np.int32)
                order = np.argsort(steps, kind="stable")
                sorted_steps = steps[order]
                sorted_meas = measurements[order]
                unique, starts = np.unique(sorted_steps, return_index=True)
                ends = list(starts[1:]) + [len(sorted_meas)]
                for raw_step, start, end in zip(unique, starts, ends):
                    meas_by_step[int(raw_step)] = sorted_meas[int(start):int(end)]

            fidelity_events: list[dict[str, object]] = []
            for idx, event_meta in enumerate(meta.get("fidelity_events", [])):
                zones = [int(z) for z in event_meta.get("zones", [])]
                grids: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                for az in zones:
                    x_key = f"fid_{idx:04d}_z{az}_X"
                    y_key = f"fid_{idx:04d}_z{az}_y"
                    grids[int(az)] = (
                        np.asarray(data[x_key], dtype=np.float32),
                        np.asarray(data[y_key], dtype=np.float32).reshape(-1, 1),
                    )
                fidelity_events.append({**event_meta, "zones": zones, "grids": grids})

        return {
            "path": str(path),
            "meta": meta,
            "node_states": node_states,
            "node_generations": node_generations,
            "node_active": node_active,
            "synced": synced,
            "measurements": meas_by_step,
            "fidelity_events": fidelity_events,
            "dynamic_by_step": {
                int(k): str(v) for k, v in dict(meta.get("dynamic_by_step", {})).items()
            },
            "refresh_zones_by_step": {
                int(k): [int(z) for z in v]
                for k, v in dict(meta.get("refresh_zones_by_step", {})).items()
            },
        }

    def _load_mobility_trace(self, path: Path) -> dict[str, object]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        traces = data.get("traces", {})
        if not isinstance(traces, dict) or not traces:
            raise ValueError(f"Mobility trace {path} contains no traces")
        raw_vehicle_ids = data.get("vehicle_ids")
        if isinstance(raw_vehicle_ids, list) and raw_vehicle_ids:
            vehicle_ids = [str(v) for v in raw_vehicle_ids]
        else:
            vehicle_ids = sorted(str(v) for v in traces.keys())

        arrays: list[np.ndarray] = []
        used_ids: list[str] = []
        for vid in vehicle_ids:
            points = traces.get(vid)
            if points is None:
                continue
            cleaned: list[list[float]] = []
            last_xy: list[float] | None = None
            for item in points:
                if item is None:
                    if last_xy is None:
                        cleaned.append([0.0, 0.0])
                    else:
                        cleaned.append([float(last_xy[0]), float(last_xy[1])])
                    continue
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    raise ValueError(f"Mobility trace {path} has invalid point for vehicle {vid}: {item!r}")
                last_xy = [float(item[0]), float(item[1])]
                cleaned.append(last_xy)
            arr = np.asarray(cleaned, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(f"Mobility trace {path} has invalid points for vehicle {vid}")
            arrays.append(arr[:, :2])
            used_ids.append(vid)
        if not arrays:
            raise ValueError(f"Mobility trace {path} has no usable vehicle trajectories")
        min_len = min(int(arr.shape[0]) for arr in arrays)
        if min_len <= 0:
            raise ValueError(f"Mobility trace {path} has empty vehicle trajectories")
        positions = np.stack([arr[:min_len] for arr in arrays], axis=1).astype(np.float32, copy=False)
        raw_active = data.get("active_traces", {})
        active_columns: list[np.ndarray] = []
        for vid in used_ids:
            values = raw_active.get(vid) if isinstance(raw_active, dict) else None
            if values is None:
                active_columns.append(np.ones((min_len,), dtype=np.bool_))
                continue
            active_column = np.asarray(values, dtype=np.bool_).reshape(-1)
            if active_column.shape[0] < min_len:
                raise ValueError(
                    f"Mobility trace {path} has only {active_column.shape[0]} active flags "
                    f"for vehicle {vid}, expected at least {min_len}"
                )
            active_columns.append(active_column[:min_len])
        node_active = np.stack(active_columns, axis=1)
        if (
            bool(data.get("open_boundary_routing", False))
            and str(data.get("format", ""))
            not in {"sumo_mobility_trace_v3", "sumo_mobility_trace_v4"}
        ):
            raise ValueError(
                f"Mobility trace {path} does not preserve vehicle replacement "
                "events; regenerate it with the v3 mobility exporter"
            )
        node_generations = np.zeros((min_len, len(arrays)), dtype=np.int32)
        for event in data.get("respawn_events", []):
            node_idx = int(event["node_idx"])
            first_step = int(event["first_step"])
            if 0 <= node_idx < len(arrays) and first_step < min_len:
                node_generations[max(0, first_step):, node_idx] += 1
        return {
            "path": str(path),
            "meta": {
                "seed": int(data.get("seed", 0)),
                "max_step": int(data.get("max_step", min_len - 1)),
                "open_boundary_routing": bool(data.get("open_boundary_routing", False)),
            },
            "vehicle_ids": used_ids,
            "positions": positions,
            "node_generations": node_generations,
            "node_active": node_active,
        }

    def _record_trace_node_state(self, step: int, *, synced: int) -> None:
        if self._trace_record_path is None:
            return
        rows = np.zeros((int(self.cfg.num_nodes), 3), dtype=np.float32)
        for i, ns in enumerate(self.nodes):
            rows[i, 0] = float(ns.node.x)
            rows[i, 1] = float(ns.node.y)
            rows[i, 2] = float(ns.current_az)
        self._trace_node_state_by_step[int(step)] = rows
        self._trace_node_generation_by_step[int(step)] = np.asarray(
            self._node_generations, dtype=np.int32
        ).copy()
        self._trace_synced_by_step[int(step)] = int(synced)

    def _record_trace_measurements(self, step: int, meas: list[tuple[int, int, int, float]]) -> None:
        if self._trace_record_path is None:
            return
        if not meas:
            return
        arr = np.empty((len(meas), 5), dtype=np.float32)
        arr[:, 0] = float(step)
        for idx, (az, tx_idx, rx_idx, val) in enumerate(meas):
            arr[idx, 1] = float(az)
            arr[idx, 2] = float(tx_idx)
            arr[idx, 3] = float(rx_idx)
            arr[idx, 4] = float(val)
        self._trace_measurement_rows.append(arr)

    def _record_trace_fidelity_event(self, *, n_pairs: int, zones: list[int]) -> None:
        if self._trace_record_path is None:
            return
        if self._explicit_fidelity_schedule_enabled() and int(self._trace_fidelity_step) <= 0:
            return
        grids: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for az in zones:
            X, y = self.fidelity_grid[int(az)]
            grids[int(az)] = (
                np.asarray(X, dtype=np.float32).copy(),
                np.asarray(y, dtype=np.float32).reshape(-1, 1).copy(),
            )
        self._trace_fidelity_events.append(
            {
                "step": int(self._trace_fidelity_step),
                "n_pairs": int(n_pairs),
                "zones": [int(z) for z in zones],
                "grids": grids,
            }
        )

    def _save_measurement_trace(self, *, last_step: int, reason: str) -> None:
        if self._trace_record_path is None:
            return
        path = self._trace_record_path
        path.parent.mkdir(parents=True, exist_ok=True)
        node_states = np.zeros((int(self.cfg.sim_steps) + 1, int(self.cfg.num_nodes), 3), dtype=np.float32)
        synced = np.zeros((int(self.cfg.sim_steps) + 1,), dtype=np.int32)
        node_generations = np.zeros(
            (int(self.cfg.sim_steps) + 1, int(self.cfg.num_nodes)),
            dtype=np.int32,
        )
        for step in range(int(self.cfg.sim_steps) + 1):
            rows = self._trace_node_state_by_step.get(step)
            if rows is not None:
                node_states[step] = rows
            elif step > 0:
                node_states[step] = node_states[step - 1]
            generation_row = self._trace_node_generation_by_step.get(step)
            if generation_row is not None:
                node_generations[step] = generation_row
            elif step > 0:
                node_generations[step] = node_generations[step - 1]
            synced[step] = int(self._trace_synced_by_step.get(step, self.cfg.num_nodes))
        measurements = (
            np.concatenate(self._trace_measurement_rows, axis=0)
            if self._trace_measurement_rows
            else np.zeros((0, 5), dtype=np.float32)
        )
        fidelity_meta = []
        arrays: dict[str, np.ndarray] = {
            "node_states": node_states,
            "node_generations": node_generations,
            "synced": synced,
            "measurements": measurements,
        }
        for idx, event in enumerate(self._trace_fidelity_events):
            zones = [int(z) for z in event["zones"]]  # type: ignore[index]
            fidelity_meta.append(
                {
                    "step": int(event["step"]),
                    "n_pairs": int(event["n_pairs"]),
                    "zones": zones,
                }
            )
            grids = event["grids"]  # type: ignore[index]
            assert isinstance(grids, dict)
            for az in zones:
                X, y = grids[int(az)]
                arrays[f"fid_{idx:04d}_z{az}_X"] = np.asarray(X, dtype=np.float32)
                arrays[f"fid_{idx:04d}_z{az}_y"] = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        meta = {
            "format": "sumo_rssi_trace_v2",
            "seed": int(self.cfg.seed),
            "sim_steps": int(self.cfg.sim_steps),
            "num_nodes": int(self.cfg.num_nodes),
            "num_zones": int(self.cfg.num_zones),
            "map_size": float(self.cfg.map_size),
            "num_rays": int(self.cfg.num_rays),
            "max_depth": int(self.cfg.max_depth),
            "trace_tx_batch_size": int(self.cfg.trace_tx_batch_size),
            "freq_hz": float(self.cfg.freq_hz),
            "tx_power_dbm": float(self.cfg.tx_power_dbm),
            "rssi_min_dbm": float(self.cfg.rssi_min_dbm),
            "rssi_max_dbm": float(self.cfg.rssi_max_dbm),
            "sumo_config": str(self.sumo_config),
            "sumo_net": str(self.sumo_net),
            "dynamic_map": str(self.dynamic_map or ""),
            "mobility_trace": str(self.mobility_trace_in or ""),
            "last_step": int(last_step),
            "reason": str(reason),
            "replacement_semantics": "complete-cold-start-before-first-new-vehicle-frame",
            "replacement_events": int(np.sum(np.diff(node_generations, axis=0) > 0)),
            "fidelity_events": fidelity_meta,
            "dynamic_by_step": {str(k): v for k, v in self._trace_dynamic_by_step.items()},
            "refresh_zones_by_step": {
                str(k): [int(z) for z in v]
                for k, v in self._trace_refresh_zones_by_step.items()
            },
        }
        arrays["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True))
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
        try:
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        print(
            f"[SUMO-RRE] wrote measurement trace {path} "
            f"({measurements.shape[0]} link samples, {len(fidelity_meta)} fidelity events)",
            flush=True,
        )

    def _apply_trace_node_state(self, step: int, *, reset_on_zone_change: bool) -> int:
        assert self._trace_replay is not None
        node_states = self._trace_replay["node_states"]
        assert isinstance(node_states, np.ndarray)
        if int(step) >= node_states.shape[0]:
            raise RuntimeError(f"Trace has no node state for step {int(step)}")
        rows = node_states[int(step)]
        generation_rows = self._trace_replay["node_generations"]
        assert isinstance(generation_rows, np.ndarray)
        node_count = int(
            getattr(getattr(self, "cfg", None), "num_nodes", len(self.nodes))
        )
        active_rows = self._trace_replay.get("node_active")
        if active_rows is None:
            self._current_node_active = np.ones((node_count,), dtype=np.bool_)
        else:
            assert isinstance(active_rows, np.ndarray)
            self._current_node_active = active_rows[
                int(step), :node_count
            ].copy()
        for i, ns in enumerate(self.nodes):
            generation = int(generation_rows[int(step), i])
            replaced = generation != int(self._node_generations[i])
            if replaced:
                self._reset_respawned_node(i, generation=generation)
            ns.node.x = float(rows[i, 0])
            ns.node.y = float(rows[i, 1])
            if not bool(self._current_node_active[i]):
                continue
            new_az = int(round(float(rows[i, 2])))
            if replaced:
                ns.current_az = int(new_az)
                self._replacement_pending_nodes.discard(i)
            elif new_az != int(ns.current_az) and reset_on_zone_change:
                old_az = int(ns.current_az)
                self._reset_node_for_zone_change(ns, new_az)
                self._reset_aux_node(i, old_az=old_az, new_az=int(new_az))
            else:
                ns.current_az = int(new_az)
            self._update_visited(ns)
        synced = self._trace_replay.get("synced")
        if isinstance(synced, np.ndarray) and int(step) < synced.shape[0]:
            return int(synced[int(step)])
        return int(np.count_nonzero(self._current_node_active))

    def _apply_mobility_trace_node_state(self, step: int, *, reset_on_zone_change: bool) -> int:
        assert self._mobility_trace is not None
        positions = self._mobility_trace["positions"]
        generations = self._mobility_trace["node_generations"]
        active_rows = self._mobility_trace["node_active"]
        assert isinstance(positions, np.ndarray)
        assert isinstance(generations, np.ndarray)
        assert isinstance(active_rows, np.ndarray)
        if int(step) >= positions.shape[0]:
            raise RuntimeError(f"Mobility trace has no node state for step {int(step)}")
        rows = positions[int(step)]
        self._current_node_active = active_rows[int(step), : int(self.cfg.num_nodes)].copy()
        for i, ns in enumerate(self.nodes):
            generation = int(generations[int(step), i])
            replaced = generation != int(self._node_generations[i])
            if replaced:
                self._reset_respawned_node(i, generation=generation)
            ns.node.x = float(rows[i, 0])
            ns.node.y = float(rows[i, 1])
            if not bool(self._current_node_active[i]):
                continue
            new_az = int(
                zone_of(
                    ns.node.x, ns.node.y, self.cfg.map_size, self.cfg.num_zones
                )
            )
            if replaced:
                ns.current_az = int(new_az)
                self._replacement_pending_nodes.discard(i)
            elif new_az != int(ns.current_az) and reset_on_zone_change:
                old_az = int(ns.current_az)
                self._reset_node_for_zone_change(ns, new_az)
                self._reset_aux_node(i, old_az=old_az, new_az=int(new_az))
            else:
                ns.current_az = int(new_az)
            self._update_visited(ns)
        return int(np.count_nonzero(self._current_node_active))

    def _trace_measurements_for_step(self, step: int) -> list[tuple[int, int, int, float]]:
        assert self._trace_replay is not None
        by_step = self._trace_replay["measurements"]
        assert isinstance(by_step, dict)
        rows = by_step.get(int(step))
        if rows is None or len(rows) == 0:
            return []
        return [
            (int(round(float(r[1]))), int(round(float(r[2]))), int(round(float(r[3]))), float(r[4]))
            for r in rows
        ]

    def _load_next_trace_fidelity_grid(
        self,
        *,
        n_pairs: int,
        zones: set[int] | list[int] | tuple[int, ...] | None,
    ) -> None:
        assert self._trace_replay is not None
        events = self._trace_replay["fidelity_events"]
        assert isinstance(events, list)
        if self._trace_fidelity_read_index >= len(events):
            raise RuntimeError("Measurement trace has no remaining fidelity grid event")
        event = events[self._trace_fidelity_read_index]
        self._trace_fidelity_read_index += 1
        event_zones = [int(z) for z in event.get("zones", [])]
        event_step = int(event.get("step", -1))
        requested_step = int(self._trace_fidelity_step)
        if requested_step > 0 and event_step != requested_step:
            raise RuntimeError(
                "Measurement trace fidelity event step mismatch: "
                f"requested step {requested_step}, next event is step {event_step}"
            )
        requested = (
            set(range(int(self.cfg.num_zones)))
            if zones is None
            else {int(z) for z in zones}
        )
        if not requested.issubset(set(event_zones)):
            raise RuntimeError(
                f"Trace fidelity event zones {event_zones} do not cover requested {sorted(requested)}"
            )
        grids = event.get("grids", {})
        assert isinstance(grids, dict)
        for az in event_zones:
            if az not in requested:
                continue
            X, y = grids[int(az)]
            self.fidelity_grid[int(az)] = (
                self._adapt_predictor_features(
                    np.asarray(X, dtype=np.float32),
                    step=int(event.get("step", 0)),
                    zone=int(az),
                ),
                np.asarray(y, dtype=np.float32).reshape(-1, 1),
            )
        if self.cfg.verbose:
            print(
                f"[SUMO-RRE] Loaded trace fidelity event "
                f"{self._trace_fidelity_read_index}/{len(events)} "
                f"step={int(event.get('step', -1))} zones={sorted(requested)} n_pairs={int(n_pairs)}",
                flush=True,
            )

    def _closed_route_edges_from_sumo_config(self) -> set[str]:
        closed: set[str] = set()
        cfg_path = Path(self.sumo_config)
        try:
            root = ET.parse(cfg_path).getroot()
        except (OSError, ET.ParseError):
            return closed
        add_files: list[Path] = []
        for elem in root.findall(".//additional-files"):
            raw = elem.get("value", "")
            for part in raw.split(","):
                item = part.strip()
                if not item:
                    continue
                p = Path(item)
                add_files.append(p if p.is_absolute() else cfg_path.parent / p)
        for path in add_files:
            if not path.is_file() or not path.name.endswith(".xml"):
                continue
            try:
                add_root = ET.parse(path).getroot()
            except ET.ParseError:
                continue
            for closing in add_root.findall(".//closingReroute"):
                edge_id = closing.get("id")
                if edge_id:
                    closed.add(edge_id)
        return closed

    def _build_routing_edge_index(self) -> None:
        bounds = read_net_bounds(self.sumo_net)
        root = ET.parse(self.sumo_net).getroot()
        closed_targets = self._closed_route_edges_from_sumo_config()
        edge_zone: dict[str, int] = {}
        edge_shape: dict[str, list[tuple[float, float]]] = {}
        edge_lane: dict[str, str] = {}
        edge_midpoint: dict[str, tuple[float, float]] = {}
        by_zone: dict[int, list[str]] = defaultdict(list)
        zone_capacity: dict[int, float] = defaultdict(float)
        boundary_edges: dict[str, list[str]] = defaultdict(list)
        incoming_by_node: dict[str, list[str]] = defaultdict(list)
        outgoing_by_node: dict[str, list[str]] = defaultdict(list)
        for edge in root.findall("edge"):
            edge_id = edge.get("id", "")
            if not edge_id or edge_id.startswith(":") or edge.get("function") == "internal":
                continue
            if edge_id in closed_targets:
                continue
            lane = edge.find("lane")
            if lane is None:
                continue
            shape_raw = lane.get("shape")
            if not shape_raw:
                continue
            pts: list[tuple[float, float]] = []
            for item in shape_raw.split():
                x, y = item.split(",")[:2]
                pts.append((float(x), float(y)))
            if len(pts) < 2:
                continue
            mx = sum(p[0] for p in pts) / float(len(pts))
            my = sum(p[1] for p in pts) / float(len(pts))
            nx = (mx - bounds.x0) / max(1e-9, bounds.width)
            ny = (my - bounds.y0) / max(1e-9, bounds.height)
            zx = max(0.0, min(self.cfg.map_size, nx * self.cfg.map_size))
            zy = max(0.0, min(self.cfg.map_size, ny * self.cfg.map_size))
            az = zone_of(zx, zy, self.cfg.map_size, self.cfg.num_zones)
            edge_zone[edge_id] = int(az)
            edge_shape[edge_id] = pts
            edge_lane[edge_id] = lane.get("id") or f"{edge_id}_0"
            edge_midpoint[edge_id] = (float(zx), float(zy))
            by_zone[int(az)].append(edge_id)
            from_node = edge.get("from") or ""
            to_node = edge.get("to") or ""
            if from_node:
                outgoing_by_node[from_node].append(edge_id)
            if to_node:
                incoming_by_node[to_node].append(edge_id)
            try:
                zone_capacity[int(az)] += float(lane.get("length") or 0.0)
            except ValueError:
                zone_capacity[int(az)] += 0.0
            dists = {
                "left": float(zx),
                "right": float(self.cfg.map_size) - float(zx),
                "bottom": float(zy),
                "top": float(self.cfg.map_size) - float(zy),
            }
            side, dist = min(dists.items(), key=lambda item: item[1])
            if dist <= float(self.cfg.map_size) * self.open_boundary_margin:
                boundary_edges[side].append(edge_id)
        self._routing_edge_zone = edge_zone
        self._routing_edge_shape = edge_shape
        self._routing_edge_lane = edge_lane
        self._routing_edge_midpoint = edge_midpoint
        self._routing_edges_by_zone = dict(by_zone)
        self._routing_zone_capacity = dict(zone_capacity)
        self._routing_boundary_edges_by_side = {k: list(v) for k, v in boundary_edges.items()}
        self._routing_junction_in_edges = {
            node: sorted(edges)
            for node, edges in incoming_by_node.items()
            if len(edges) >= 3 and len(outgoing_by_node.get(node, ())) >= 3
        }
    @staticmethod
    def _lane_allows_passenger(edge: ET.Element, lane: ET.Element) -> bool:
        allow_raw = lane.get("allow") or edge.get("allow") or ""
        disallow_raw = lane.get("disallow") or edge.get("disallow") or ""
        allow = {part.strip() for part in allow_raw.split() if part.strip()}
        disallow = {part.strip() for part in disallow_raw.split() if part.strip()}
        if "all" in disallow or "passenger" in disallow or "private" in disallow:
            return False
        if allow:
            return bool({"all", "passenger", "private"}.intersection(allow))
        return True

    @staticmethod
    def _net_to_map_xy(
        x: float,
        y: float,
        bounds,
        map_size: float,
    ) -> tuple[float, float]:
        nx = (float(x) - float(bounds.x0)) / max(1e-9, float(bounds.width))
        ny = (float(y) - float(bounds.y0)) / max(1e-9, float(bounds.height))
        return (
            max(0.0, min(float(map_size), nx * float(map_size))),
            max(0.0, min(float(map_size), ny * float(map_size))),
        )

    def _build_street_fidelity_points(self) -> dict[int, list[tuple[float, float]]]:
        cached = getattr(self, "_street_fidelity_points_by_zone", None)
        if cached is not None:
            return cached

        cfg = self.cfg
        bounds = read_net_bounds(self.sumo_net)
        root = ET.parse(self.sumo_net).getroot()
        closed_targets = self._closed_route_edges_from_sumo_config()
        by_zone: dict[int, list[tuple[float, float]]] = {
            int(az): [] for az in range(int(cfg.num_zones))
        }
        seen: dict[int, set[tuple[float, float]]] = {
            int(az): set() for az in range(int(cfg.num_zones))
        }
        spacing_m = 2.0

        for edge in root.findall("edge"):
            edge_id = edge.get("id", "")
            if not edge_id or edge_id.startswith(":") or edge.get("function") == "internal":
                continue
            if edge_id in closed_targets:
                continue
            for lane in edge.findall("lane"):
                if not self._lane_allows_passenger(edge, lane):
                    continue
                shape_raw = lane.get("shape")
                if not shape_raw:
                    continue
                pts: list[tuple[float, float]] = []
                for item in shape_raw.split():
                    parts = item.split(",")
                    if len(parts) < 2:
                        continue
                    pts.append((float(parts[0]), float(parts[1])))
                if len(pts) < 2:
                    continue

                for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                    mx0, my0 = self._net_to_map_xy(x0, y0, bounds, cfg.map_size)
                    mx1, my1 = self._net_to_map_xy(x1, y1, bounds, cfg.map_size)
                    seg_len = math.hypot(mx1 - mx0, my1 - my0)
                    n_steps = max(1, int(math.ceil(seg_len / spacing_m)))
                    for k in range(n_steps + 1):
                        t = float(k) / float(n_steps)
                        x = mx0 + t * (mx1 - mx0)
                        y = my0 + t * (my1 - my0)
                        if not (
                            cfg.xy_margin <= x <= cfg.map_size - cfg.xy_margin
                            and cfg.xy_margin <= y <= cfg.map_size - cfg.xy_margin
                        ):
                            continue
                        az = int(zone_of(x, y, cfg.map_size, cfg.num_zones))
                        key = (round(float(x), 3), round(float(y), 3))
                        if key in seen[az]:
                            continue
                        seen[az].add(key)
                        by_zone[az].append((float(x), float(y)))

        empty_zones = [az for az, pts in by_zone.items() if not pts]
        if empty_zones:
            raise RuntimeError(
                "street-only fidelity sampling found no drivable lane points "
                f"in zones {empty_zones}; check SUMO net coverage and zone layout"
            )
        self._street_fidelity_points_by_zone = by_zone
        return by_zone

    def _fidelity_sampling_walls(self):
        walls = list(self.walls)
        dynamic_schedule = getattr(self.map_engine, "dynamic_schedule", None)
        active_ids = set(getattr(self.map_engine, "_last_dynamic_active", ()) or ())
        if dynamic_schedule is not None and active_ids:
            for obs in dynamic_schedule.obstacles:
                if obs.scene_id in active_ids:
                    walls.append(obs.as_wall())
        return walls

    def _sample_street_oracle_pairs(
        self,
        zone: int,
        walls,
        *,
        n_tx: int,
        n_pairs: int,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        street_points = self._build_street_fidelity_points()
        candidates = [
            p for p in street_points[int(zone)] if not collides_with_walls(p[0], p[1], walls)
        ]
        if not candidates:
            raise RuntimeError(
                "street-only fidelity sampling has no active drivable points "
                f"in zone {int(zone)} after obstacle filtering"
            )

        pts = np.asarray(candidates, dtype=np.float64)
        rng = self._rng_grid
        rx_per_tx = max(1, math.ceil(int(n_pairs) / max(1, int(n_tx))))
        tx_count = max(1, int(n_tx))
        tx_idx = rng.choice(
            len(pts),
            size=tx_count,
            replace=len(pts) < tx_count,
        )

        pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for raw_txi in tx_idx:
            if len(pairs) >= int(n_pairs):
                break
            txi = int(raw_txi)
            tx = (float(pts[txi, 0]), float(pts[txi, 1]))
            rx_idx = rng.choice(
                len(pts),
                size=rx_per_tx,
                replace=len(pts) < rx_per_tx,
            )
            for raw_rxi in rx_idx:
                if len(pairs) >= int(n_pairs):
                    break
                rxi = int(raw_rxi)
                if len(pts) > 1 and rxi == txi:
                    rxi = (rxi + 1 + int(rng.integers(0, len(pts) - 1))) % len(pts)
                rx = (float(pts[rxi, 0]), float(pts[rxi, 1]))
                pairs.append((tx, rx))
        return pairs

    def _build_fidelity_grid(
        self,
        n_pairs: int | None = None,
        zones: set[int] | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """Ray-trace held-out street-only pairs per zone for fidelity reporting."""
        cfg = self.cfg
        n_pairs = int(cfg.fidelity_grid_per_zone if n_pairs is None else n_pairs)
        zone_ids = list(range(cfg.num_zones) if zones is None else sorted({int(z) for z in zones}))
        if self._trace_replay is not None:
            if n_pairs <= 0:
                # A replay smoke test may intentionally disable fidelity even
                # when the source trace contains later held-out checkpoints.
                # Do not consume a checkpoint merely because finalization calls
                # this helper at the last simulated step.
                self.fidelity_grid = {}
                return
            if (
                self._explicit_fidelity_schedule_enabled()
                and int(self._trace_fidelity_step) <= 0
            ):
                self.fidelity_grid = {}
                return
            events = self._trace_replay.get("fidelity_events", [])
            self._load_next_trace_fidelity_grid(n_pairs=n_pairs, zones=zones)
            return
        fidelity_walls = self._fidelity_sampling_walls()
        for az in zone_ids:
            pairs = self._sample_street_oracle_pairs(
                int(az),
                fidelity_walls,
                n_tx=cfg.fidelity_grid_n_tx,
                n_pairs=n_pairs,
            )
            groups = group_pairs_by_tx(pairs)
            rssi_groups = self.tracer.measure_pairs(groups)
            X = []
            y = []
            for (tx, rxs), rssi_list in zip(groups, rssi_groups):
                for rx, val in zip(rxs, rssi_list):
                    X.append(
                        self._pair_model_features(
                            tx,
                            rx,
                            step=self._trace_fidelity_step,
                            zone=int(az),
                        )
                    )
                    y.append(val)
            self.fidelity_grid[int(az)] = (
                np.asarray(X, dtype=np.float32),
                np.asarray(y, dtype=np.float32).reshape(-1, 1),
            )
            if cfg.verbose:
                print(f"[SUMO-RRE] Street fidelity grid zone {int(az)}: {len(X)} pairs")
        self._record_trace_fidelity_event(
            n_pairs=n_pairs,
            zones=[int(z) for z in zone_ids],
        )

    def _zone_distance(self, a: int, b: int) -> int:
        side = int(round(float(self.cfg.num_zones) ** 0.5))
        if side * side != int(self.cfg.num_zones):
            return 0 if a == b else 1
        ax, ay = int(a) % side, int(a) // side
        bx, by = int(b) % side, int(b) // side
        return abs(ax - bx) + abs(ay - by)

    def _routing_zone_for_edge(self, edge_id: str) -> int | None:
        return self._routing_edge_zone.get(edge_id)

    @staticmethod
    def _edge_from_lane_id(lane_id: str) -> str:
        return str(lane_id).rsplit("_", 1)[0]

    def _lane_can_reach_edge(self, lane_id: str, edge_id: str) -> bool:
        if not lane_id or not edge_id:
            return False
        try:
            links = traci.lane.getLinks(lane_id, extended=True)
        except traci.TraCIException:
            return True
        for link in links:
            if not link:
                continue
            target_lane = str(link[0])
            if target_lane and self._edge_from_lane_id(target_lane) == edge_id:
                return True
        return False

    def _candidate_destination_zones(
        self,
        current_zone: int | None,
        *,
        ignore_min_distance: bool = False,
    ) -> list[int]:
        zones = sorted(z for z, edges in self._routing_edges_by_zone.items() if edges)
        if current_zone is None:
            return zones
        candidates: list[int] = []
        for z in zones:
            if z == current_zone:
                continue
            dist = self._zone_distance(current_zone, z)
            if not ignore_min_distance and dist < self.route_min_zone_distance:
                continue
            if self.route_max_zone_distance is not None and dist > self.route_max_zone_distance:
                continue
            candidates.append(z)
        if candidates:
            return candidates
        if self.route_max_zone_distance is not None:
            close = [
                z
                for z in zones
                if z != current_zone
                and self._zone_distance(current_zone, z) <= self.route_max_zone_distance
            ]
            if close:
                return close
        return [z for z in zones if z != current_zone] or zones

    def _zone_expected_count(self, zone: int) -> float:
        total_capacity = sum(max(1.0, float(v)) for v in self._routing_zone_capacity.values())
        if total_capacity <= 0.0:
            return max(1.0, float(self.cfg.num_nodes) / max(1, len(self._routing_edges_by_zone)))
        zone_capacity = max(1.0, float(self._routing_zone_capacity.get(int(zone), 1.0)))
        return max(1.0, float(self.cfg.num_nodes) * zone_capacity / total_capacity)

    def _zone_density(self, zone: int, live_zone_count: Counter[int]) -> float:
        return float(live_zone_count[int(zone)]) / self._zone_expected_count(int(zone))

    def _choose_destination_zone(self, zones: list[int]) -> int:
        assert self._route_rng is not None
        return self._route_rng.choice(list(zones))

    def _central_route_zones(self) -> set[int]:
        side = int(round(float(self.cfg.num_zones) ** 0.5))
        if side * side != int(self.cfg.num_zones) or side <= 2:
            return set()
        if side % 2 == 1:
            mid = side // 2
            return {mid + side * mid}
        lo = side // 2 - 1
        hi = side // 2
        return {x + side * y for x in (lo, hi) for y in (lo, hi)}

    def _route_score(
        self,
        edges: list[str],
        dest_zone: int,
        live_zone_count: Counter[int] | None = None,
        edge_pressure: Counter[str] | None = None,
    ) -> float:
        zone_counts = Counter(
            self._routing_edge_zone[e] for e in edges if e in self._routing_edge_zone
        )
        live_zone_count = live_zone_count or Counter()
        edge_pressure = edge_pressure or Counter()
        edge_jam = sum(float(edge_pressure.get(edge_id, 0.0)) for edge_id in edges)
        crowd = sum(float(n) * float(self._route_zone_load[z]) for z, n in zone_counts.items())
        live_crowd = sum(float(n) * float(live_zone_count[z]) for z, n in zone_counts.items())
        density_crowd = sum(float(n) * self._zone_density(int(z), live_zone_count) for z, n in zone_counts.items())
        central_zones = self._central_route_zones()
        central = sum(float(n) for z, n in zone_counts.items() if z in central_zones)
        if int(self.cfg.num_zones) == 9:
            zones_seen = set(zone_counts)
            perimeter_zones = set(self._routing_edges_by_zone) - central_zones
            corner_zones = self._corner_route_zones()
            target_penalty = 70.0 * float(self._route_target_zone_count[dest_zone])
            live_target_penalty = 55.0 * float(live_zone_count[dest_zone])
            density_target_penalty = 180.0 * self._zone_density(int(dest_zone), live_zone_count)
            center_penalty = 125.0 * central
            crowd_penalty = 1.6 * crowd + 1.9 * live_crowd + 95.0 * density_crowd + 18.0 * edge_jam
            coverage_bonus = 12.0 * float(len(zones_seen & perimeter_zones))
            corner_bonus = 12.0 * float(len(zones_seen & corner_zones))
            underused_bonus = sum(
                20.0 / (1.0 + float(live_zone_count[z]))
                for z in zones_seen & perimeter_zones
            )
            return (
                float(len(edges))
                + crowd_penalty
                + center_penalty
                + target_penalty
                + live_target_penalty
                + density_target_penalty
                - coverage_bonus
                - corner_bonus
                - underused_bonus
            )
        target_penalty = 55.0 * float(self._route_target_zone_count[dest_zone])
        center_penalty = 6.0 * central
        crowd_penalty = 0.35 * crowd + 0.15 * live_crowd + 20.0 * density_crowd + 8.0 * edge_jam
        return float(len(edges)) + crowd_penalty + center_penalty + target_penalty

    def _corner_route_zones(self) -> set[int]:
        side = int(round(float(self.cfg.num_zones) ** 0.5))
        if side * side != int(self.cfg.num_zones) or side < 2:
            return set()
        return {0, side - 1, side * (side - 1), side * side - 1}

    def _candidate_via_zones(
        self,
        current_zone: int | None,
        dest_zone: int,
        live_zone_count: Counter[int] | None = None,
    ) -> list[int]:
        if int(self.cfg.num_zones) != 9:
            return []
        assert self._route_rng is not None
        live_zone_count = live_zone_count or Counter()
        central_zones = self._central_route_zones()
        corner_zones = self._corner_route_zones()
        zones = [
            int(z)
            for z, edges in self._routing_edges_by_zone.items()
            if edges and z not in central_zones and z not in {current_zone, dest_zone}
        ]
        zones.sort(
            key=lambda z: (
                float(live_zone_count[z])
                + 0.35 * float(self._route_zone_load[z])
                + 2.5 * float(self._route_target_zone_count[z]),
                0 if z in corner_zones else 1,
                self._route_rng.random(),
            )
        )
        return zones

    def _compose_random_od_route(
        self,
        current_edge: str,
        current_lane: str,
        dest_edge: str,
        blocked: set[str],
        *,
        via_edge: str | None = None,
    ) -> list[str] | None:
        stops = [current_edge]
        if via_edge is not None:
            stops.append(via_edge)
        stops.append(dest_edge)
        full_edges: list[str] = []
        for idx, (start_edge, end_edge) in enumerate(zip(stops, stops[1:])):
            if start_edge == end_edge:
                return None
            try:
                route = traci.simulation.findRoute(start_edge, end_edge)
            except traci.TraCIException:
                return None
            edges = list(getattr(route, "edges", []) or [])
            if len(edges) < 2 or edges[0] != start_edge or edges[-1] != end_edge:
                return None
            if idx == 0 and current_lane and not self._lane_can_reach_edge(current_lane, edges[1]):
                return None
            if idx == 0:
                full_edges.extend(edges)
            else:
                full_edges.extend(edges[1:])
        if len(full_edges) < 2 or full_edges[0] != current_edge or full_edges[-1] != dest_edge:
            return None
        if blocked.intersection(full_edges[1:]):
            return None
        return full_edges

    def _live_route_zone_counts(self) -> Counter[int]:
        counts: Counter[int] = Counter()
        try:
            vehicle_ids = traci.vehicle.getIDList()
        except traci.TraCIException:
            return counts
        for other_id in vehicle_ids:
            try:
                x, y = traci.vehicle.getPosition(other_id)
                if not _valid_sumo_position(x, y):
                    continue
                nx, ny = self._norm_xy(x, y)
            except (AssertionError, traci.TraCIException):
                continue
            zone = zone_of(nx, ny, self.cfg.map_size, self.cfg.num_zones)
            counts[int(zone)] += 1
        return counts

    def _clear_route_accounting(self, veh_id: str) -> None:
        old_zone = self._route_target_zone.pop(veh_id, None)
        if old_zone is not None:
            self._route_target_zone_count[old_zone] -= 1
            if self._route_target_zone_count[old_zone] <= 0:
                del self._route_target_zone_count[old_zone]
        old_load = self._route_zone_load_by_vehicle.pop(veh_id, None)
        if old_load:
            for z, n in old_load.items():
                self._route_zone_load[z] -= n
                if self._route_zone_load[z] <= 0:
                    del self._route_zone_load[z]
        self._route_target_edge.pop(veh_id, None)
        self._route_exit_side.pop(veh_id, None)

    @staticmethod
    def _opposite_boundary_side(side: str) -> str:
        return {"left": "right", "right": "left", "bottom": "top", "top": "bottom"}[side]

    def _nearest_boundary_side_for_edge(self, edge_id: str) -> str | None:
        midpoint = self._routing_edge_midpoint.get(edge_id)
        if midpoint is None:
            return None
        x, y = midpoint
        dists = {
            "left": float(x),
            "right": float(self.cfg.map_size) - float(x),
            "bottom": float(y),
            "top": float(self.cfg.map_size) - float(y),
        }
        side, dist = min(dists.items(), key=lambda item: item[1])
        if dist <= float(self.cfg.map_size) * self.open_boundary_margin:
            return side
        return None

    def _rank_boundary_sides(self, current_edge: str, live_zone_count: Counter[int]) -> list[str]:
        sides = [side for side, edges in self._routing_boundary_edges_by_side.items() if edges]
        if not sides:
            return []
        nearest = self._nearest_boundary_side_for_edge(current_edge)
        preferred = self._opposite_boundary_side(nearest) if nearest else None

        def side_score(side: str) -> tuple[float, float]:
            zones = [self._routing_edge_zone[e] for e in self._routing_boundary_edges_by_side.get(side, []) if e in self._routing_edge_zone]
            density = min((self._zone_density(int(z), live_zone_count) for z in zones), default=0.0)
            return (0.0 if side == preferred else 1.0, density)

        return sorted(sides, key=side_score)

    def _rank_boundary_edges(self, side: str, live_zone_count: Counter[int], blocked: set[str]) -> list[str]:
        edges = [e for e in self._routing_boundary_edges_by_side.get(side, []) if e not in blocked]
        edges.sort(key=lambda e: (self._zone_density(self._routing_edge_zone.get(e, 0), live_zone_count), self._route_rng.random() if self._route_rng else 0.0))
        return edges

    def _remember_route_accounting(
        self,
        veh_id: str,
        *,
        dest_zone: int,
        dest_edge: str,
        edges: list[str],
        exit_side: str | None = None,
    ) -> None:
        self._clear_route_accounting(veh_id)
        self._route_target_edge[veh_id] = dest_edge
        self._route_target_zone[veh_id] = int(dest_zone)
        self._route_target_zone_count[int(dest_zone)] += 1
        if exit_side is not None:
            self._route_exit_side[veh_id] = str(exit_side)
        new_load: Counter[int] = Counter()
        for edge_id in edges:
            z = self._routing_edge_zone.get(edge_id)
            if z is not None:
                self._route_zone_load[int(z)] += 1
                new_load[int(z)] += 1
        self._route_zone_load_by_vehicle[veh_id] = new_load

    def _reset_respawned_node(
        self, node_idx: int, *, generation: int | None = None
    ) -> None:
        node_idx = int(node_idx)
        if generation is None:
            self._node_generations[node_idx] += 1
        else:
            generation = int(generation)
            if generation < self._node_generations[node_idx]:
                raise ValueError("vehicle generation cannot move backwards")
            self._node_generations[node_idx] = generation
        self._replacement_pending_nodes.add(node_idx)
        ns = self.nodes[node_idx]
        self._node_zone_memory[node_idx].clear()
        self._aux_zone_memory[node_idx].clear()
        ns.reset_all(self.template_state, self.cfg.local_lr)
        self._refresh_node_signatures(ns)
        self._reset_aux_node(node_idx, old_az=None, new_az=None)

    def _live_route_edge_pressure(self) -> Counter[str]:
        pressure: Counter[str] = Counter()
        try:
            vehicle_ids = traci.vehicle.getIDList()
        except traci.TraCIException:
            return pressure
        for veh_id in vehicle_ids:
            try:
                edge_id = traci.vehicle.getRoadID(veh_id)
            except traci.TraCIException:
                continue
            if edge_id not in self._routing_edge_zone:
                continue
            try:
                wait = float(traci.vehicle.getWaitingTime(veh_id))
                speed = float(traci.vehicle.getSpeed(veh_id))
            except traci.TraCIException:
                wait = 0.0
                speed = 0.0
            pressure[edge_id] += 1.0 + min(8.0, wait / 8.0)
            if speed < 0.3:
                pressure[edge_id] += 4.0
        return pressure

    def _vehicle_at_boundary_exit(self, veh_id: str, exit_side: str) -> bool:
        try:
            edge_id = traci.vehicle.getRoadID(veh_id)
            x, y = traci.vehicle.getPosition(veh_id)
        except traci.TraCIException:
            return False
        if edge_id not in self._routing_boundary_edges_by_side.get(exit_side, ()):
            return False
        if not _valid_sumo_position(x, y):
            return False
        nx, ny = self._norm_xy(x, y)
        side_dist = {
            "left": nx,
            "right": float(self.cfg.map_size) - nx,
            "bottom": ny,
            "top": float(self.cfg.map_size) - ny,
        }.get(exit_side)
        if side_dist is None:
            return False
        return side_dist <= float(self.cfg.map_size) * self.open_boundary_exit_margin

    def _distance_to_lane_end(self, veh_id: str) -> float | None:
        try:
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_pos = float(traci.vehicle.getLanePosition(veh_id))
            lane_len = float(traci.lane.getLength(lane_id))
        except traci.TraCIException:
            return None
        return max(0.0, lane_len - lane_pos)

    def _apply_intersection_right_of_way(self, *, step: int) -> None:
        if not self.intersection_control or not self._routing_junction_in_edges:
            return
        active_ids = set(traci.vehicle.getIDList())
        for veh_id in list(self._junction_held_vehicle_ids):
            if veh_id in active_ids:
                try:
                    traci.vehicle.setSpeed(veh_id, -1.0)
                except traci.TraCIException:
                    pass
        self._junction_held_vehicle_ids.clear()

        for junction_id, in_edges in self._routing_junction_in_edges.items():
            queues: dict[str, list[tuple[str, float, float]]] = {}
            for edge_id in in_edges:
                try:
                    veh_ids = traci.edge.getLastStepVehicleIDs(edge_id)
                except traci.TraCIException:
                    continue
                close: list[tuple[str, float, float]] = []
                for veh_id in veh_ids:
                    dist = self._distance_to_lane_end(veh_id)
                    if dist is None or dist > 2.5 * self.intersection_stop_distance:
                        continue
                    try:
                        wait = float(traci.vehicle.getWaitingTime(veh_id))
                        speed = float(traci.vehicle.getSpeed(veh_id))
                    except traci.TraCIException:
                        continue
                    if wait >= self.intersection_wait_seconds or speed < 0.25:
                        close.append((veh_id, dist, wait))
                if close:
                    queues[edge_id] = close

            if len(queues) < 3:
                self._junction_release_state.pop(junction_id, None)
                continue

            state = self._junction_release_state.get(junction_id)
            if state and state[0] in queues and int(step) <= state[1]:
                release_edge = state[0]
            else:
                release_edge = max(
                    queues,
                    key=lambda edge: (
                        max(wait for _veh, _dist, wait in queues[edge]),
                        len(queues[edge]),
                    ),
                )
                self._junction_release_state[junction_id] = (
                    release_edge,
                    int(step) + self.intersection_release_steps,
                )

            for edge_id, rows in queues.items():
                for veh_id, dist, _wait in rows:
                    try:
                        if edge_id == release_edge:
                            traci.vehicle.setSpeed(veh_id, -1.0)
                        elif dist <= self.intersection_stop_distance:
                            traci.vehicle.setSpeed(veh_id, 0.0)
                            self._junction_held_vehicle_ids.add(veh_id)
                    except traci.TraCIException:
                        pass

    def _assign_random_od_route(self, veh_id: str) -> bool:
        if self._route_rng is None or not self._routing_edges_by_zone:
            return False
        try:
            current_edge = traci.vehicle.getRoadID(veh_id)
            current_lane = traci.vehicle.getLaneID(veh_id)
        except traci.TraCIException:
            return False
        if not current_edge or current_edge.startswith(":"):
            return False
        current_zone = self._routing_zone_for_edge(current_edge)
        preferred_zones = self._candidate_destination_zones(current_zone)
        if not preferred_zones:
            return False

        blocked: set[str] = set()
        live_zone_count = self._live_route_zone_counts()
        edge_pressure = self._live_route_edge_pressure()
        candidates: list[tuple[float, int, str, list[str], str | None]] = []
        if (
            self.open_boundary_routing
            and self._routing_boundary_edges_by_side
            and self._route_rng.random() < self.open_boundary_probability
        ):
            for exit_side in self._rank_boundary_sides(current_edge, live_zone_count)[:2]:
                for dest_edge in self._rank_boundary_edges(exit_side, live_zone_count, blocked)[:8]:
                    if dest_edge == current_edge:
                        continue
                    dest_zone = int(self._routing_edge_zone.get(dest_edge, current_zone or 0))
                    edges = self._compose_random_od_route(current_edge, current_lane, dest_edge, blocked)
                    if edges:
                        score = 0.85 * self._route_score(edges, dest_zone, live_zone_count, edge_pressure)
                        candidates.append((score, dest_zone, dest_edge, edges, exit_side))
                if candidates:
                    break
        zone_passes = [preferred_zones]
        relaxed_zones = self._candidate_destination_zones(
            current_zone,
            ignore_min_distance=True,
        )
        if relaxed_zones and set(relaxed_zones) != set(preferred_zones):
            zone_passes.append(sorted(relaxed_zones))

        first_attempts = 10 if self.route_max_zone_distance is not None else 40
        relaxed_attempts = 6 if self.route_max_zone_distance is not None else 24
        for pass_idx, dest_zones in enumerate(zone_passes):
            for _ in range(first_attempts if pass_idx == 0 else relaxed_attempts):
                dest_zone = self._choose_destination_zone(dest_zones)
                dest_edge = self._route_rng.choice(self._routing_edges_by_zone[dest_zone])
                if dest_edge == current_edge or dest_edge in blocked:
                    continue
                edges = self._compose_random_od_route(
                    current_edge,
                    current_lane,
                    dest_edge,
                    blocked,
                )
                if edges:
                    candidates.append((self._route_score(edges, dest_zone, live_zone_count, edge_pressure), dest_zone, dest_edge, edges, None))
                via_zones = self._candidate_via_zones(current_zone, dest_zone, live_zone_count)
                if via_zones:
                    via_sample = via_zones[:1]
                    if len(via_zones) > 1:
                        via_sample.append(self._route_rng.choice(via_zones[1:4]))
                    for via_zone in via_sample:
                        via_edge = self._route_rng.choice(self._routing_edges_by_zone[via_zone])
                        if via_edge in {current_edge, dest_edge} or via_edge in blocked:
                            continue
                        edges = self._compose_random_od_route(
                            current_edge,
                            current_lane,
                            dest_edge,
                            blocked,
                            via_edge=via_edge,
                        )
                        if edges:
                            candidates.append(
                                (self._route_score(edges, dest_zone, live_zone_count, edge_pressure), dest_zone, dest_edge, edges, None)
                            )
            if candidates:
                break
        if not candidates:
            return False
        candidates.sort(key=lambda item: item[0])
        _, dest_zone, dest_edge, edges, exit_side = self._route_rng.choice(candidates[: min(8, len(candidates))])
        try:
            traci.vehicle.setRoute(veh_id, edges)
        except traci.TraCIException:
            return False
        self._remember_route_accounting(
            veh_id,
            dest_zone=dest_zone,
            dest_edge=dest_edge,
            edges=edges,
            exit_side=exit_side,
        )
        self._route_changes += 1
        return True

    def _respawn_vehicle_opposite_side(self, veh_id: str, exit_side: str) -> bool:
        if self._route_rng is None:
            return False
        try:
            node_idx = self._veh_ids.index(veh_id)
        except ValueError:
            return False
        entry_side = self._opposite_boundary_side(exit_side)
        blocked: set[str] = set()
        live_zone_count = self._live_route_zone_counts()
        entries = self._rank_boundary_edges(entry_side, live_zone_count, blocked)[:12]
        exits = self._rank_boundary_edges(exit_side, live_zone_count, blocked)[:12]
        for entry_edge in entries:
            for dest_edge in exits:
                if entry_edge == dest_edge:
                    continue
                edges = self._compose_random_od_route(entry_edge, "", dest_edge, blocked)
                if not edges:
                    continue
                self._respawn_seq += 1
                route_id = f"open_boundary_{int(self.cfg.seed)}_{node_idx}_{self._respawn_seq}"
                new_id = f"veh_ob_{node_idx:03d}_{self._respawn_seq:05d}"
                try:
                    traci.route.add(route_id, edges)
                    traci.vehicle.add(
                        new_id,
                        route_id,
                        depart="now",
                        departLane="best",
                        departPos="0",
                        departSpeed="max",
                    )
                except traci.TraCIException:
                    continue
                try:
                    traci.vehicle.remove(veh_id)
                except traci.TraCIException:
                    pass
                self._clear_route_accounting(veh_id)
                self._missing_tracked_vehicles.discard(veh_id)
                self._missing_tracked_vehicle_since.pop(veh_id, None)
                self._veh_ids[node_idx] = new_id
                dest_zone = int(self._routing_edge_zone.get(dest_edge, 0))
                self._remember_route_accounting(
                    new_id,
                    dest_zone=dest_zone,
                    dest_edge=dest_edge,
                    edges=edges,
                    exit_side=exit_side,
                )
                self._reset_respawned_node(node_idx)
                return True
        return False

    def _respawn_missing_tracked_vehicle(self, veh_id: str) -> bool:
        if self._route_rng is None or not self._routing_edges_by_zone:
            return False
        try:
            node_idx = self._veh_ids.index(veh_id)
        except ValueError:
            return False
        source_zone = int(self.nodes[node_idx].current_az)
        source_edges = [e for e in self._routing_edges_by_zone.get(source_zone, [])]
        if not source_edges:
            source_edges = [e for edges in self._routing_edges_by_zone.values() for e in edges]
        if not source_edges:
            return False
        blocked: set[str] = set()
        live_zone_count = self._live_route_zone_counts()
        edge_pressure = self._live_route_edge_pressure()
        candidates: list[tuple[float, int, str, str, list[str]]] = []
        for start_edge in self._route_rng.sample(source_edges, k=min(10, len(source_edges))):
            if start_edge in blocked:
                continue
            current_zone = self._routing_zone_for_edge(start_edge)
            dest_zones = self._candidate_destination_zones(current_zone)
            if not dest_zones:
                dest_zones = self._candidate_destination_zones(current_zone, ignore_min_distance=True)
            for _ in range(8):
                if not dest_zones:
                    break
                dest_zone = self._choose_destination_zone(dest_zones)
                dest_edge = self._route_rng.choice(self._routing_edges_by_zone[dest_zone])
                if dest_edge == start_edge or dest_edge in blocked:
                    continue
                edges = self._compose_random_od_route(start_edge, "", dest_edge, blocked)
                if edges:
                    candidates.append(
                        (
                            self._route_score(edges, dest_zone, live_zone_count, edge_pressure),
                            dest_zone,
                            start_edge,
                            dest_edge,
                            edges,
                        )
                    )
        if not candidates:
            return False
        candidates.sort(key=lambda item: item[0])
        _score, dest_zone, _start_edge, dest_edge, edges = self._route_rng.choice(
            candidates[: min(8, len(candidates))]
        )
        self._respawn_seq += 1
        route_id = f"tracked_respawn_{int(self.cfg.seed)}_{node_idx}_{self._respawn_seq}"
        new_id = f"veh_rp_{node_idx:03d}_{self._respawn_seq:05d}"
        try:
            traci.route.add(route_id, edges)
            traci.vehicle.add(
                new_id,
                route_id,
                depart="now",
                departLane="best",
                departPos="0",
                departSpeed="max",
            )
        except traci.TraCIException:
            return False
        self._clear_route_accounting(veh_id)
        self._missing_tracked_vehicles.discard(veh_id)
        self._missing_tracked_vehicle_since.pop(veh_id, None)
        self._veh_ids[node_idx] = new_id
        self._remember_route_accounting(
            new_id,
            dest_zone=int(dest_zone),
            dest_edge=dest_edge,
            edges=edges,
            exit_side=None,
        )
        self._reset_respawned_node(node_idx)
        return True

    def _refresh_random_od_routes(self, *, step: int) -> None:
        if not self.random_od_routing:
            return
        active = set(traci.vehicle.getIDList())
        for veh_id in list(self._veh_ids):
            if veh_id not in active:
                exit_side = self._route_exit_side.get(veh_id)
                missing_since = self._missing_tracked_vehicle_since.get(veh_id)
                if missing_since is None:
                    continue
                if int(step) - int(missing_since) < 3:
                    continue
                if self.open_boundary_routing and exit_side:
                    self._respawn_vehicle_opposite_side(veh_id, exit_side)
                else:
                    self._respawn_missing_tracked_vehicle(veh_id)
                continue
            try:
                route = list(traci.vehicle.getRoute(veh_id))
                route_idx = int(traci.vehicle.getRouteIndex(veh_id))
            except traci.TraCIException:
                continue
            if not route or route_idx < 0:
                self._assign_random_od_route(veh_id)
                continue
            remaining = len(route) - route_idx - 1
            target = self._route_target_edge.get(veh_id)
            exit_side = self._route_exit_side.get(veh_id)
            if (
                self.open_boundary_routing
                and exit_side
                and self._vehicle_at_boundary_exit(veh_id, exit_side)
                and self._respawn_vehicle_opposite_side(veh_id, exit_side)
            ):
                continue
            try:
                waiting_time = float(traci.vehicle.getWaitingTime(veh_id))
            except traci.TraCIException:
                waiting_time = 0.0
            jammed = self.jam_reroute_wait_seconds > 0.0 and waiting_time >= self.jam_reroute_wait_seconds
            near_exit = bool(exit_side and self._vehicle_at_boundary_exit(veh_id, exit_side))
            if target is None or jammed or (not exit_side and remaining <= ROUTE_REASSIGN_EDGE_BUFFER):
                if not near_exit:
                    self._assign_random_od_route(veh_id)

    def _attach_sumo_nodes(self) -> None:
        traci.start(["sumo", "-c", self.sumo_config, "--seed", str(int(self.cfg.seed)), "--time-to-teleport", "-1"])
        self._sumo_open = True

        (x0, y0), (x1, y1) = traci.simulation.getNetBoundary()
        self._sumo_bbox = (float(x0), float(y0), float(x1), float(y1))

        # Warm up until enough vehicles are active for fixed node indexing.
        steps = 0
        while len(traci.vehicle.getIDList()) < self.cfg.num_nodes and steps < self._warmup_limit_steps:
            traci.simulationStep()
            steps += 1
            if traci.simulation.getMinExpectedNumber() <= 0:
                break

        active = sorted(traci.vehicle.getIDList())
        if len(active) < self.cfg.num_nodes:
            raise RuntimeError(
                f"SUMO has only {len(active)} active vehicles after warmup, "
                f"but cfg.num_nodes={self.cfg.num_nodes}. Increase demand or lower num-nodes."
            )
        self._veh_ids = active[: self.cfg.num_nodes]

        # Initialize node positions from SUMO and reset per-zone state consistently.
        self._sync_nodes_from_sumo(reset_on_zone_change=True)
        if self.random_od_routing:
            assigned = 0
            for vid in self._veh_ids:
                assigned += int(self._assign_random_od_route(vid))
            print(
                f"[SUMO-RRE] Random OD routing enabled: assigned {assigned}/{len(self._veh_ids)} "
                f"initial cross-zone routes",
                flush=True,
            )

    def _norm_xy(self, x: float, y: float) -> tuple[float, float]:
        assert self._sumo_bbox is not None
        x0, y0, x1, y1 = self._sumo_bbox
        nx = (float(x) - x0) / max(1e-9, (x1 - x0))
        ny = (float(y) - y0) / max(1e-9, (y1 - y0))
        return (
            max(0.0, min(self.cfg.map_size, nx * self.cfg.map_size)),
            max(0.0, min(self.cfg.map_size, ny * self.cfg.map_size)),
        )

    @staticmethod
    def _clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _pack_model_state(self, model: nn.Module) -> _PackedModelState:
        state = model.state_dict()
        spec_key = tuple(
            (name, tuple(int(v) for v in tensor.shape), str(tensor.dtype), int(tensor.numel()))
            for name, tensor in state.items()
        )
        spec = self._packed_state_specs.get(spec_key)
        if spec is None:
            spec = tuple(
                (name, tuple(int(v) for v in tensor.shape), tensor.dtype, int(tensor.numel()))
                for name, tensor in state.items()
            )
            self._packed_state_specs[spec_key] = spec
        pieces = [tensor.detach().to(device="cpu").reshape(-1) for tensor in state.values()]
        flat = torch.cat(pieces).clone() if len(pieces) > 1 else pieces[0].clone()
        return _PackedModelState(flat=flat, spec=spec)

    def _load_model_state(self, model: nn.Module, state: dict[str, torch.Tensor] | _PackedModelState) -> None:
        device = next(model.parameters()).device
        if isinstance(state, _PackedModelState):
            model.load_state_dict(state.to_state_dict(device=device))
            return
        model.load_state_dict({k: v.to(device) for k, v in state.items()})

    def _snapshot_variant(self, v) -> dict[str, object]:
        return {
            "weights": self._pack_model_state(v.model),
        }

    def _restore_variant(self, v, snapshot: dict[str, object]) -> None:
        self._load_model_state(v.model, snapshot["weights"])  # type: ignore[arg-type]
        v.opt = optim.Adam(v.model.parameters(), lr=self.cfg.local_lr)
        # Re-entering a zone reuses the weights but starts fresh metadata.
        v.m_samples = 0
        v.n_samples = 0
        v.quality = 0.0
        v.t_wait = 0
        v.last_rmse = 45.0
        v.last_rmse_available = False
        v.rmse_ema_short = 0.0
        v.rmse_ema_long = 0.0
        v.rmse_batches = 0
        self._refresh_variant_signature(v)
        v.recovery_steps_left = 0
        v.recovery_accepts_left = 0
        v.recovery_cooldown_left = 0

    def _save_node_zone_memory(self, node_idx: int, zone: int) -> None:
        if not self.zone_model_memory:
            return
        ns = self.nodes[node_idx]
        self._node_zone_memory[node_idx][int(zone)] = {
            "visited_bitmap": 0,
            "variants": {
                mode_id: self._snapshot_variant(v)
                for mode_id, v in ns.variants.items()
            },
        }

    def _restore_node_zone_memory(self, node_idx: int, zone: int) -> bool:
        cached = self._node_zone_memory[node_idx].get(int(zone))
        if not cached:
            return False
        ns = self.nodes[node_idx]
        ns.current_az = int(zone)
        ns.pending_slots.clear()
        ns.clear_current_visit_samples()
        ns.visited_bitmap = int(cached.get("visited_bitmap", 0))
        variants = cached.get("variants", {})
        if isinstance(variants, dict):
            for mode_id, snapshot in variants.items():
                if mode_id in ns.variants and isinstance(snapshot, dict):
                    self._restore_variant(ns.variants[mode_id], snapshot)
        return True

    def _reset_node_for_zone_change(self, ns, new_az: int) -> None:
        if not self.zone_model_memory:
            super()._reset_node_for_zone_change(ns, new_az)
            return
        node_idx = next(
            (idx for idx, candidate in enumerate(self.nodes) if candidate is ns),
            -1,
        )
        if node_idx < 0:
            super()._reset_node_for_zone_change(ns, new_az)
            return
        old_az = int(ns.current_az)
        self._save_node_zone_memory(node_idx, old_az)
        if self._restore_node_zone_memory(node_idx, int(new_az)):
            return
        ns.current_az = int(new_az)
        ns.reset_all(self.template_state, self.cfg.local_lr)
        self._refresh_node_signatures(ns)

    def _sync_nodes_from_sumo(self, *, reset_on_zone_change: bool) -> int:
        active = set(traci.vehicle.getIDList())
        missing = [vid for vid in self._veh_ids if vid not in active]
        if missing:
            new_missing = [vid for vid in missing if vid not in self._missing_tracked_vehicles]
            if new_missing:
                preview = ", ".join(new_missing[:8])
                if len(new_missing) > 8:
                    preview += ", ..."
                print(
                    f"[SUMO-RRE] warning: {len(new_missing)} tracked SUMO vehicle(s) "
                    f"disappeared ({preview}); retaining their last node positions",
                    flush=True,
                )
                for vid in new_missing:
                    self._clear_route_accounting(vid)
                    self._missing_tracked_vehicle_since.setdefault(vid, getattr(self, "_current_sumo_step", 0))
                self._missing_tracked_vehicles.update(new_missing)
        recovered = [vid for vid in self._missing_tracked_vehicles if vid in active]
        if recovered:
            preview = ", ".join(recovered[:8])
            if len(recovered) > 8:
                preview += ", ..."
            print(
                f"[SUMO-RRE] warning: tracked vehicle(s) reappeared ({preview}); resuming SUMO sync",
                flush=True,
            )
            self._missing_tracked_vehicles.difference_update(recovered)
            for vid in recovered:
                self._missing_tracked_vehicle_since.pop(vid, None)
        synced = 0
        for i, vid in enumerate(self._veh_ids):
            if vid not in active:
                synced += 1
                continue
            x, y = traci.vehicle.getPosition(vid)
            if not _valid_sumo_position(x, y):
                synced += 1
                continue
            nx, ny = self._norm_xy(x, y)
            ns = self.nodes[i]
            ns.node.x = float(nx)
            ns.node.y = float(ny)
            new_az = zone_of(ns.node.x, ns.node.y, self.cfg.map_size, self.cfg.num_zones)
            if i in self._replacement_pending_nodes:
                ns.current_az = int(new_az)
                self._node_zone_memory[i].clear()
                self._aux_zone_memory[i].clear()
                self._replacement_pending_nodes.discard(i)
            elif new_az != ns.current_az and reset_on_zone_change:
                old_az = int(ns.current_az)
                self._reset_node_for_zone_change(ns, new_az)
                # Match reset semantics for auxiliary baselines on zone transition.
                self._reset_aux_node(i, old_az=old_az, new_az=int(new_az))
            self._update_visited(ns)
            synced += 1
        return synced

    def _snapshot_aux_model(self, model: nn.Module) -> dict[str, object]:
        return {
            "weights": self._pack_model_state(model),
        }

    def _restore_aux_model(
        self,
        model: nn.Module,
        snapshot: dict[str, object],
    ) -> None:
        self._load_model_state(model, snapshot["weights"])  # type: ignore[arg-type]

    def _save_aux_zone_memory(self, i: int, zone: int) -> None:
        if not self.zone_model_memory or self.skip_aux_baselines:
            return
        entry: dict[str, object] = {}
        if "iso" in self.aux_baselines:
            entry["iso"] = self._snapshot_aux_model(self.iso_models[i])
        if "greedy" in self.aux_baselines:
            entry["greedy"] = self._snapshot_aux_model(self.greedy_models[i])
        self._aux_zone_memory[i][int(zone)] = entry

    def _reset_aux_node(
        self,
        i: int,
        *,
        old_az: int | None = None,
        new_az: int | None = None,
    ) -> None:
        if self.skip_aux_baselines:
            return
        if self.zone_model_memory and old_az is not None:
            self._save_aux_zone_memory(i, int(old_az))
        cached = (
            self._aux_zone_memory[i].get(int(new_az))
            if self.zone_model_memory and new_az is not None
            else None
        )
        if "iso" in self.aux_baselines:
            if cached and isinstance(cached.get("iso"), dict):
                self._restore_aux_model(self.iso_models[i], cached["iso"])  # type: ignore[arg-type]
                self.iso_m_samples[i] = 0
                self.iso_n_samples[i] = 0
            else:
                self.iso_models[i].load_state_dict(self._aux_template_state)
                self.iso_m_samples[i] = 0
                self.iso_n_samples[i] = 0
            self.iso_opts[i] = optim.Adam(self.iso_models[i].parameters(), lr=self.cfg.local_lr)
        if "greedy" in self.aux_baselines:
            if cached and isinstance(cached.get("greedy"), dict):
                self._restore_aux_model(self.greedy_models[i], cached["greedy"])  # type: ignore[arg-type]
                self.greedy_m_samples[i] = 0
                self.greedy_n_samples[i] = 0
            else:
                self.greedy_models[i].load_state_dict(self._aux_template_state)
                self.greedy_m_samples[i] = 0
                self.greedy_n_samples[i] = 0
            self.greedy_opts[i] = optim.Adam(self.greedy_models[i].parameters(), lr=self.cfg.local_lr)

    def _zones_for_dynamic_obstacle(self, obs) -> set[int]:
        x0 = float(obs.center[0]) - 0.5 * float(obs.size[0])
        x1 = float(obs.center[0]) + 0.5 * float(obs.size[0])
        y0 = float(obs.center[1]) - 0.5 * float(obs.size[1])
        y1 = float(obs.center[1]) + 0.5 * float(obs.size[1])
        zones: set[int] = set()
        for az in range(int(self.cfg.num_zones)):
            zx0, zx1, zy0, zy1 = zone_bounds(
                az,
                float(self.cfg.map_size),
                int(self.cfg.num_zones),
            )
            if not (x1 < zx0 or zx1 < x0 or y1 < zy0 or zy1 < y0):
                zones.add(int(az))
        if not zones and int(getattr(obs, "zone", -1)) >= 0:
            zones.add(int(obs.zone))
        return zones

    def _changed_dynamic_zones(
        self,
        previous_active: set[str],
        current_active: set[str],
    ) -> set[int]:
        dynamic_schedule = getattr(self.map_engine, "dynamic_schedule", None)
        if dynamic_schedule is None:
            return set()
        changed_ids = previous_active.symmetric_difference(current_active)
        if not changed_ids:
            return set()
        by_id = {obs.scene_id: obs for obs in dynamic_schedule.obstacles}
        zones: set[int] = set()
        for scene_id in changed_ids:
            obs = by_id.get(scene_id)
            if obs is not None:
                zones.update(self._zones_for_dynamic_obstacle(obs))
        return zones

    @staticmethod
    def _pull_model_from_state(
        model_puller: nn.Module,
        exp_puller: float,
        provider_state: dict[str, torch.Tensor],
        exp_provider: float,
        *,
        merge_strategy: str = "average",
    ) -> None:
        rre_sim.weighted_state_pull(
            model_puller,
            exp_puller,
            provider_state,
            exp_provider,
            merge_strategy=merge_strategy,
        )

    def _contact_links_from_measurements(
        self,
        meas: list[tuple[int, int, int, float]],
    ) -> list[tuple[int, int, int]]:
        snr_min_db = float(
            getattr(
                self.cfg,
                "model_transfer_snr_min_db",
                self.cfg.snr_min_db,
            )
        )
        directed_ok: set[tuple[int, int]] = set()
        pair_zone: dict[tuple[int, int], int] = {}
        for az, tx_idx, rx_idx, val in meas:
            tx = int(tx_idx)
            rx = int(rx_idx)
            if tx == rx:
                continue
            a, b = (tx, rx) if tx < rx else (rx, tx)
            pair_zone.setdefault((a, b), int(az))
            if self._snr_from_rx_power_dbm(float(val)) >= snr_min_db:
                directed_ok.add((tx, rx))
        links = [
            (int(az), int(a), int(b))
            for (a, b), az in pair_zone.items()
            if (a, b) in directed_ok and (b, a) in directed_ok
        ]
        links.sort(key=lambda row: (row[0], row[1], row[2]))
        return links

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        events = 0
        links: list[tuple[int, int, int]] = []
        if contact_links is None:
            for az, idxs in zone_nodes.items():
                ids = sorted(idxs)
                for ii in range(len(ids)):
                    for jj in range(ii + 1, len(ids)):
                        links.append((int(az), int(ids[ii]), int(ids[jj])))
        else:
            seen: set[tuple[int, int, int]] = set()
            for az, a, b in contact_links:
                ia = int(a)
                ib = int(b)
                if ia == ib:
                    continue
                if ib < ia:
                    ia, ib = ib, ia
                key = (int(az), ia, ib)
                if key not in seen:
                    links.append(key)
                    seen.add(key)
        links.sort(key=lambda row: (row[0], row[1], row[2]))

        for _az, i, j in links:
            i_state = {
                k: v.detach().clone()
                for k, v in self.greedy_models[i].state_dict().items()
            }
            j_state = {
                k: v.detach()
                for k, v in self.greedy_models[j].state_dict().items()
            }
            i_m = int(self.greedy_m_samples[i])
            j_m = int(self.greedy_m_samples[j])
            i_n = int(self.greedy_n_samples[i])
            j_n = int(self.greedy_n_samples[j])
            i_exp = float(i_n)
            j_exp = float(j_n)

            self._pull_model_from_state(
                self.greedy_models[i],
                i_exp,
                j_state,
                j_exp,
                merge_strategy=str(self.cfg.merge_strategy),
            )
            self._pull_model_from_state(
                self.greedy_models[j],
                j_exp,
                i_state,
                i_exp,
                merge_strategy=str(self.cfg.merge_strategy),
            )
            new_m = bound_raw_samples(i_m + j_m)
            new_n = saturate_n_samples(new_m)
            self.greedy_m_samples[i] = new_m
            self.greedy_m_samples[j] = new_m
            self.greedy_n_samples[i] = new_n
            self.greedy_n_samples[j] = new_n
            events += 2
        return events

    @staticmethod
    def _state_dict_nbytes(state: dict[str, torch.Tensor]) -> int:
        return int(sum(int(t.numel()) * int(t.element_size()) for t in state.values()))

    @staticmethod
    def _state_dict_numel(state: dict[str, torch.Tensor]) -> int:
        return int(sum(int(t.numel()) for t in state.values()))

    def _build_communication_assumptions(self) -> dict[str, int | float | str | bool]:
        rssi_state = self.template.state_dict()
        rssi_bytes = self._state_dict_nbytes(rssi_state)
        rssi_params = self._state_dict_numel(rssi_state)
        policy_bytes = 0
        policy_params = 0
        if self.agents:
            policy_state = next(iter(self.agents.values())).policy.state_dict()
            policy_bytes = self._state_dict_nbytes(policy_state)
            policy_params = self._state_dict_numel(policy_state)
        accepted_merge_meta_bytes = 8
        decision_scalar_meta_bytes = int(rre_sim.DECISION_METADATA_SCALAR_FLOATS) * 4
        model_signature_bytes = int(rre_sim.MODEL_SIGNATURE_FLOATS) * 4
        decision_meta_bytes = decision_scalar_meta_bytes + model_signature_bytes
        accepted_pull_bytes = rssi_bytes + accepted_merge_meta_bytes
        local_policy_pull_bytes = policy_bytes if self.local_policy_share else 0
        return {
            "rssi_model": str(self.cfg.rssi_model),
            "predictor_input_dim": int(self._predictor_input_dim()),
            "predictor_include_time": bool(getattr(self.cfg, "predictor_include_time", False)),
            "predictor_time_encoding": "learnable-multiscale-fourier-with-log-trend",
            "predictor_time_step_duration": float(getattr(self.cfg, "predictor_time_step_duration", 1.0)),
            "predictor_time_unit": float(getattr(self.cfg, "predictor_time_unit", 1.0)),
            "predictor_time_num_frequencies": int(getattr(self.cfg, "predictor_time_num_frequencies", 8)),
            "predictor_time_min_period": float(getattr(self.cfg, "predictor_time_min_period", 2.0)),
            "predictor_time_max_period": float(getattr(self.cfg, "predictor_time_max_period", 1000.0)),
            "predictor_time_clock": "global-absolute-simulation-time",
            "local_sample_weighting": str(getattr(self.cfg, "local_sample_weighting", "uniform")),
            "local_sample_recency_half_life_steps": float(
                getattr(self.cfg, "local_sample_recency_half_life_steps", 0.0)
            ),
            "prediction_target": "propagation_loss_db",
            "measurement_signal": "rssi_dbm",
            "propagation_loss_min_db": float(self._loss_min_db()),
            "propagation_loss_max_db": float(self._loss_max_db()),
            "predictor_prior": str(getattr(self.cfg, "predictor_prior", "none")),
            "predictor_prior_loss_db": float(self._predictor_prior_loss_db()) if self._predictor_prior_loss_db() is not None else "",
            "predictor_prior_normalized_loss": float(self._predictor_prior_normalized_loss()) if self._predictor_prior_normalized_loss() is not None else "",
            "predictor_prior_rssi_dbm": (
                float(self.cfg.tx_power_dbm) - float(self._predictor_prior_loss_db())
                if self._predictor_prior_loss_db() is not None
                else ""
            ),
            "noise_floor_dbm": float(self.cfg.noise_floor_dbm),
            "snr_min_db": float(self.cfg.snr_min_db),
            "model_transfer_snr_min_db": float(
                self.cfg.model_transfer_snr_min_db
            ),
            "online_training_links": (
                "directed-receive-snr-at-or-above-threshold"
            ),
            "contact_threshold_snr_db": float(
                self.cfg.model_transfer_snr_min_db
            ),
            "contact_threshold_rx_power_dbm": float(
                self.cfg.noise_floor_dbm
                + self.cfg.model_transfer_snr_min_db
            ),
            "contact_threshold_rssi_dbm": float(
                self.cfg.noise_floor_dbm
                + self.cfg.model_transfer_snr_min_db
            ),
            "contact_threshold_loss_db": float(
                self.cfg.tx_power_dbm
                - self.cfg.noise_floor_dbm
                - self.cfg.model_transfer_snr_min_db
            ),
            "rssi_model_params": int(rssi_params),
            "B_model_bytes": int(rssi_bytes),
            "policy_params": int(policy_params),
            "B_policy_bytes": int(policy_bytes),
            "B_decision_meta_bytes_per_directed_decision": int(decision_meta_bytes),
            "B_decision_scalar_meta_bytes_per_directed_decision": int(decision_scalar_meta_bytes),
            "B_model_signature_bytes_per_directed_decision": int(model_signature_bytes),
            "B_accepted_merge_meta_bytes_per_pull": int(accepted_merge_meta_bytes),
            "B_accepted_pull_bytes": int(accepted_pull_bytes),
            "B_local_policy_pull_bytes": int(local_policy_pull_bytes),
            "B_local_zramp_accepted_pull_bytes": int(accepted_pull_bytes + local_policy_pull_bytes),
            "local_policy_share": bool(self.local_policy_share),
            "zramp_policy_mode": str(self.zramp_policy_mode),
            "metadata_note": "Z-RAMP pays compact provider metadata for every feasible directed decision; accepted local-policy pulls transfer propagation-loss predictor weights, merge metadata, and optional policy weights.",
        }

    def _resolve_local_initial_pull_probability(self) -> float:
        if self.zramp_policy_mode != "local":
            return 1.0
        if self.local_policy_initial_pull_prob_override is not None:
            return float(self.local_policy_initial_pull_prob_override)
        if self.local_policy_initial_pull == "greedy":
            return 1.0
        if self.local_policy_initial_pull == "fixed":
            return 0.5
        base = float(self._communication_assumptions.get("B_accepted_pull_bytes", 0))
        extra = float(self._communication_assumptions.get("B_local_policy_pull_bytes", 0))
        decision = float(self._communication_assumptions.get("B_decision_meta_bytes_per_directed_decision", 0))
        if base <= 0.0 or extra <= 0.0:
            return 1.0
        return max(0.0, min(1.0, (base - decision) / (base + extra)))

    def _init_local_policy_agents(self) -> None:
        self.local_agents.clear()
        self._local_policy_pending_transitions.clear()
        self._local_policy_versions.clear()
        self._local_policy_initial_rngs.clear()
        for mode_id, template_agent in self.agents.items():
            mode_offset = int(zlib.crc32(str(mode_id).encode("utf-8")) % 10_000_000)
            policy_state = {
                k: v.detach().clone()
                for k, v in template_agent.policy.state_dict().items()
            }
            target_state = {
                k: v.detach().clone()
                for k, v in template_agent.target.state_dict().items()
            }
            agents: list[DQNAgent] = []
            rngs: list[random.Random] = []
            for node_idx in range(int(self.cfg.num_nodes)):
                seed = int(self.cfg.seed) + 1_900_003 + mode_offset + 104_729 * int(node_idx)
                agent = DQNAgent(
                    device=self.device,
                    gamma=self.cfg.gamma,
                    lr=self.cfg.rl_lr,
                    batch_size=self.cfg.rl_batch_size,
                    tau=self.cfg.rl_target_tau,
                    capacity=self.cfg.replay_capacity,
                    epsilon_start=self.cfg.epsilon_start,
                    epsilon_end=self.cfg.epsilon_end,
                    epsilon_decay_steps=self.cfg.epsilon_decay_steps,
                    rng_seed=seed,
                    action_policy=template_agent.action_policy,
                )
                agent.policy.load_state_dict(policy_state)
                agent.target.load_state_dict(target_state)
                agents.append(agent)
                rngs.append(random.Random(seed + 57_911))
            self.local_agents[mode_id] = agents
            self._local_policy_pending_transitions[mode_id] = [0 for _ in agents]
            self._local_policy_versions[mode_id] = [0 for _ in agents]
            self._local_policy_initial_rngs[mode_id] = rngs

    def _local_policy_ready(self, mode_id: str, node_idx: int) -> bool:
        versions = self._local_policy_versions.get(mode_id)
        return bool(versions is not None and 0 <= int(node_idx) < len(versions) and versions[int(node_idx)] > 0)

    def _select_action_from_local_agent(self, mode_id: str, node_idx: int, state: torch.Tensor) -> int:
        agents = self.local_agents.get(mode_id)
        if agents is None or not (0 <= int(node_idx) < len(agents)):
            return super()._select_action(mode_id, state, node_idx=node_idx)
        agent = agents[int(node_idx)]
        policy = str(agent.action_policy).strip().lower()
        if policy in {"reject", "always_reject"}:
            return 0
        if policy in {"accept", "always_accept"}:
            return 1
        if not self._local_policy_ready(mode_id, int(node_idx)):
            rng = self._local_policy_initial_rngs[mode_id][int(node_idx)]
            p = float(self.local_policy_initial_pull_probability)
            self._local_policy_initial_decisions[mode_id] += 1
            action = 1 if rng.random() < p else 0
            if action == 1:
                self._local_policy_initial_accepts[mode_id] += 1
            return action
        return agent.select_action(state)

    def _make_peer_view(self, ns_j, mode: str):
        view = super()._make_peer_view(ns_j, mode)
        if self.zramp_policy_mode == "local" and mode in self.local_agents:
            node_idx = self.node_idx(ns_j)
            if 0 <= node_idx < len(self.local_agents[mode]):
                agent = self.local_agents[mode][node_idx]
                view._policy_state = self._clone_model_state(agent.policy)  # type: ignore[attr-defined]
                view._policy_version = int(self._local_policy_versions[mode][node_idx])  # type: ignore[attr-defined]
        return view

    def _merge_local_policy_from_state(
        self,
        *,
        mode_id: str,
        puller_idx: int,
        exp_puller: float,
        provider_policy_state: dict[str, torch.Tensor],
        exp_provider: float,
        provider_version: int,
    ) -> None:
        if not self.local_policy_share:
            return
        agents = self.local_agents.get(mode_id)
        if agents is None or not (0 <= int(puller_idx) < len(agents)):
            return
        puller = agents[int(puller_idx)]
        # Exchange only the online pulling policy. The target network remains
        # local because current Z-RAMP transitions are terminal/bandit-style.
        rre_sim.weighted_state_pull(
            puller.policy,
            exp_puller,
            provider_policy_state,
            exp_provider,
            merge_strategy="average",
        )
        if int(provider_version) > int(self._local_policy_versions[mode_id][int(puller_idx)]):
            self._local_policy_versions[mode_id][int(puller_idx)] = int(provider_version)
        self._local_policy_pull_updates[mode_id] += 1
        self._last_local_policy_pull_updates += 1

    def perform_merge(self, ns_i, ns_j, mode: str, j_view=None) -> None:
        if mode not in self.local_agents:
            super().perform_merge(ns_i, ns_j, mode, j_view=j_view)
            return
        i_idx = self.node_idx(ns_i)
        j_idx = self.node_idx(ns_j)
        v_i = ns_i.variants[mode]
        exp_puller = float(v_i.experience)
        if j_view is None:
            v_j = ns_j.variants[mode]
            exp_provider = float(v_j.experience)
            if 0 <= j_idx < len(self.local_agents[mode]):
                provider_agent = self.local_agents[mode][j_idx]
                provider_policy_state = {
                    k: v.detach()
                    for k, v in provider_agent.policy.state_dict().items()
                }
                provider_version = int(self._local_policy_versions[mode][j_idx])
            else:
                provider_policy_state = {}
                provider_version = 0
        else:
            exp_provider = float(j_view.experience)
            provider_policy_state = getattr(j_view, "_policy_state", {})
            provider_version = int(getattr(j_view, "_policy_version", 0))

        super().perform_merge(ns_i, ns_j, mode, j_view=j_view)
        if provider_policy_state and i_idx >= 0:
            self._merge_local_policy_from_state(
                mode_id=mode,
                puller_idx=i_idx,
                exp_puller=exp_puller,
                provider_policy_state=provider_policy_state,
                exp_provider=exp_provider,
                provider_version=provider_version,
            )

    def _reset_policy_step_counters(self) -> None:
        self._last_local_policy_queued_transitions = 0
        self._last_local_policy_pull_updates = 0
        self._last_local_policy_train_updates_this_step = 0

    def _ensure_decision_log_stream(self):
        if self._decision_log_stream_writer is not None:
            return self._decision_log_stream_writer
        self._decision_log_stream_path.parent.mkdir(parents=True, exist_ok=True)
        self._decision_log_stream_file = open(
            self._decision_log_stream_path,
            "w",
            newline="",
            encoding="utf-8",
        )
        self._decision_log_stream_writer = csv.DictWriter(
            self._decision_log_stream_file,
            fieldnames=DECISION_LOG_FIELDS,
        )
        self._decision_log_stream_writer.writeheader()
        return self._decision_log_stream_writer

    def _flush_decision_log_stream(self) -> None:
        if self._decision_log_stream_file is not None:
            self._decision_log_stream_file.flush()

    def _close_decision_log_stream(self) -> None:
        if self._decision_log_stream_file is not None:
            self._decision_log_stream_file.flush()
            self._decision_log_stream_file.close()
        self._decision_log_stream_file = None
        self._decision_log_stream_writer = None

    def _decision_log_row_count(self) -> int:
        return int(self._decision_log_stream_count)

    def _record_decision_row(self, row: dict) -> None:
        normalized = {field: row.get(field, "") for field in DECISION_LOG_FIELDS}
        self.decision_log.append(normalized)
        try:
            self._decision_action_counts[str(normalized["mode"])][int(normalized["action"])] += 1
        except Exception:
            pass
        writer = self._ensure_decision_log_stream()
        writer.writerow(normalized)
        self._decision_log_stream_count += 1
        if self._decision_log_stream_count % 50000 == 0:
            self._flush_decision_log_stream()

    def _select_action(
        self,
        mode_id: str,
        state: torch.Tensor,
        node_idx: int | None = None,
    ) -> int:
        if node_idx is None:
            return super()._select_action(mode_id, state, node_idx=node_idx)
        return self._select_action_from_local_agent(mode_id, int(node_idx), state)

    def _queue_rl_transition(self, mode_id: str, transition) -> None:
        if len(transition) < 6:
            return
        s, a, r, s2, d, node_idx = transition
        idx = int(node_idx)
        agents = self.local_agents.get(mode_id)
        pending = self._local_policy_pending_transitions.get(mode_id)
        if agents is None or pending is None or not (0 <= idx < len(agents)):
            return
        agents[idx].push(s, a, r, s2, d)
        pending[idx] += 1
        self._last_local_policy_queued_transitions += 1

    def _local_pending_transition_count(self) -> int:
        return int(sum(sum(rows) for rows in self._local_policy_pending_transitions.values()))

    def _train_rl_agents(self, step: int | None = None) -> dict[str, float]:
        del step
        losses = {m: 0.0 for m in self.agents}
        for mode_id, agents in self.local_agents.items():
            pending = self._local_policy_pending_transitions.get(mode_id, [])
            versions = self._local_policy_versions.get(mode_id, [])
            for node_idx, agent in enumerate(agents):
                if node_idx >= len(pending):
                    continue
                batch_size = int(agent.batch_size)
                if pending[node_idx] < batch_size or len(agent.replay) < batch_size:
                    continue
                batches = pending[node_idx] // batch_size
                updates = max(1, int(batches) * int(self.local_policy_updates_per_batch))
                for _ in range(updates):
                    loss = agent.train_step()
                    losses[mode_id] = float(loss)
                    self._local_policy_train_updates[mode_id] += 1
                    self._last_local_policy_train_updates_this_step += 1
                    if node_idx < len(versions):
                        versions[node_idx] += 1
                pending[node_idx] -= int(batches) * batch_size
        return losses

    def _communication_overhead_row(
        self,
        *,
        feasible_decisions: int,
        greedy_events: int,
        rl_events: Counter,
    ) -> dict[str, int | float]:
        ass = self._communication_assumptions
        decision_meta = int(ass.get("B_decision_meta_bytes_per_directed_decision", 0))
        accepted_pull = int(ass.get("B_accepted_pull_bytes", 0))
        local_policy_pull = int(ass.get("B_local_policy_pull_bytes", 0))
        greedy_bytes = int(greedy_events) * accepted_pull
        self._comm_cumulative_bytes["greedy"] += greedy_bytes
        row: dict[str, int | float] = {
            "greedy_comm_bytes": int(greedy_bytes),
            "greedy_comm_mb": float(greedy_bytes) / 1_000_000.0,
            "greedy_comm_cumulative_mb": float(self._comm_cumulative_bytes["greedy"]) / 1_000_000.0,
            "local_policy_initial_pull_probability": float(self.local_policy_initial_pull_probability),
        }
        for mode_id in self.agents:
            accepted = int(rl_events.get(mode_id, 0))
            policy_bytes = local_policy_pull
            mode_bytes = int(feasible_decisions) * decision_meta + accepted * (accepted_pull + policy_bytes)
            self._comm_cumulative_bytes[mode_id] += mode_bytes
            greedy_cum = float(self._comm_cumulative_bytes["greedy"])
            row[f"{mode_id}_comm_bytes"] = int(mode_bytes)
            row[f"{mode_id}_comm_mb"] = float(mode_bytes) / 1_000_000.0
            row[f"{mode_id}_comm_cumulative_mb"] = float(self._comm_cumulative_bytes[mode_id]) / 1_000_000.0
            row[f"{mode_id}_comm_vs_greedy_ratio"] = (
                float(mode_bytes) / float(greedy_bytes) if greedy_bytes > 0 else float("nan")
            )
            row[f"{mode_id}_comm_cumulative_vs_greedy_ratio"] = (
                float(self._comm_cumulative_bytes[mode_id]) / greedy_cum if greedy_cum > 0.0 else float("nan")
            )
        return row

    def _communication_log_fragment(self) -> str:
        if not self.sharing_rows:
            return ""
        row = self.sharing_rows[-1]
        mode_id = "t2_b0.75" if "t2_b0.75" in self.agents else next(iter(self.agents), "")
        if not mode_id:
            return ""
        mode_mb = float(row.get(f"{mode_id}_comm_cumulative_mb", float("nan")))
        greedy_mb = float(row.get("greedy_comm_cumulative_mb", float("nan")))
        ratio = float(row.get(f"{mode_id}_comm_cumulative_vs_greedy_ratio", float("nan")))
        if not np.isfinite(mode_mb) or not np.isfinite(greedy_mb):
            return ""
        return f" comm {mode_id}/greedy={mode_mb:.1f}/{greedy_mb:.1f}MB ({ratio:.3f}x)"

    @staticmethod
    def _mode_label(mode_id: str) -> str:
        if mode_id == "t2_b0.75":
            return "z-ramp"
        if mode_id == "iso":
            return "Local only"
        if mode_id == "greedy":
            return "Greedy share-all"
        if mode_id == "central":
            return "Central per-zone"
        return mode_id

    def _save_sharing_plot(self, out_dir: Path) -> None:
        if not self.sharing_rows:
            return
        steps = [int(row["step"]) for row in self.sharing_rows]
        plt.figure(figsize=(10, 4.8))
        if any("greedy_events" in row for row in self.sharing_rows):
            ys = [float(row.get("greedy_events", float("nan"))) for row in self.sharing_rows]
            plt.plot(steps, ys, label="Greedy", color="#666666", linewidth=1.8)
        mode_ids = ["t2_b0.75"] if "t2_b0.75" in self.agents else sorted(self.agents)
        for mode_id in mode_ids:
            key = f"{mode_id}_events"
            if not any(key in row for row in self.sharing_rows):
                continue
            ys = [float(row.get(key, float("nan"))) for row in self.sharing_rows]
            plt.plot(steps, ys, label=self._mode_label(mode_id), linewidth=2.0)
        plt.xlabel("Step")
        plt.ylabel("Sharing events")
        plt.title("Sharing events per step")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "sharing_events_over_time.png", dpi=180)
        plt.close()

    def _central_training_arrays(
        self,
        zone: int,
        new_features: list[list[float]],
        new_targets: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = [list(row) for row in new_features]
        targets = [float(value) for value in new_targets]
        if len(rows) != len(targets):
            raise ValueError("central features and targets differ in length")
        if self.central_accumulate_samples:
            stored_x = self._central_accumulated_x[int(zone)]
            stored_y = self._central_accumulated_y[int(zone)]
            stored_x.extend(rows)
            stored_y.extend(targets)
            rows = stored_x
            targets = stored_y
        feature_dim = self._predictor_input_dim()
        if not rows:
            return (
                np.empty((0, feature_dim), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )
        X = np.asarray(rows, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        if int(X.shape[0]) != int(y.shape[0]):
            raise ValueError("central features and targets differ in length")
        return X, y

    def _train_predictors_from_current_measurements(
        self,
        *,
        step: int,
        measurements: list[tuple[int, int, int, float]],
    ) -> None:
        """Train on directed measurements that the receiver can decode.

        Ray tracing records every same-zone directed link.  A record is a
        locally available training sample only when its receive SNR meets the
        same directed-link threshold used to construct bidirectional model-
        sharing contacts.
        """

        self._meas_per_node = defaultdict(list)
        zone_X: dict[int, list[list[float]]] = defaultdict(list)
        zone_y: dict[int, list[float]] = defaultdict(list)
        for zone, tx_idx, rx_idx, value in measurements:
            if self._snr_from_rx_power_dbm(float(value)) < float(
                self.cfg.snr_min_db
            ):
                continue
            tx_node = self.nodes[tx_idx].node
            rx_node = self.nodes[rx_idx].node
            features = self._pair_model_features(
                (tx_node.x, tx_node.y),
                (rx_node.x, rx_node.y),
                step=step,
                zone=int(zone),
            )
            self._meas_per_node[int(rx_idx)].append((features, float(value)))
            zone_X[int(zone)].append(features)
            zone_y[int(zone)].append(float(value))

        for node_idx, node_state in enumerate(self.nodes):
            new_rows = self._meas_per_node.get(node_idx, [])
            if new_rows:
                node_state.current_visit_samples_x.extend(
                    list(features) for features, _value in new_rows
                )
                node_state.current_visit_samples_y.extend(
                    float(value) for _features, value in new_rows
                )
                node_state.current_visit_sample_steps.extend(
                    [int(step)] * len(new_rows)
                )
            if not new_rows:
                continue
            X = np.asarray(node_state.current_visit_samples_x, dtype=np.float32)
            y = np.asarray(
                node_state.current_visit_samples_y, dtype=np.float32
            ).reshape(-1, 1)
            sample_steps = np.asarray(
                node_state.current_visit_sample_steps, dtype=np.float32
            )
            sample_weights = self._sample_recency_weights(
                sample_steps, current_step=step
            )
            n_new = int(len(new_rows))
            self._train_local(
                node_state,
                X,
                y,
                sample_count_increment=n_new,
                sample_weights=sample_weights,
            )
            if "iso" in self.aux_baselines:
                self._train_one_local(
                    self.iso_models[node_idx],
                    self.iso_opts[node_idx],
                    X,
                    y,
                    seed_key=f"iso:{node_idx}",
                    sample_weights=sample_weights,
                )
                if n_new > 0:
                    self.iso_m_samples[node_idx] = bound_raw_samples(
                        int(self.iso_m_samples[node_idx]) + n_new
                    )
                    self.iso_n_samples[node_idx] = saturate_n_samples(
                        self.iso_m_samples[node_idx]
                    )
            if "greedy" in self.aux_baselines:
                self._train_one_local(
                    self.greedy_models[node_idx],
                    self.greedy_opts[node_idx],
                    X,
                    y,
                    seed_key=f"greedy:{node_idx}",
                    sample_weights=sample_weights,
                )
                if n_new > 0:
                    self.greedy_m_samples[node_idx] = bound_raw_samples(
                        int(self.greedy_m_samples[node_idx]) + n_new
                    )
                    self.greedy_n_samples[node_idx] = saturate_n_samples(
                        self.greedy_m_samples[node_idx]
                    )

        if "central" in self.aux_baselines:
            for zone in range(self.cfg.num_zones):
                ZX, Zy = self._central_training_arrays(
                    zone, zone_X.get(zone, []), zone_y.get(zone, [])
                )
                if int(ZX.shape[0]) == 0:
                    continue
                self._train_one_local(
                    self.central_models[zone],
                    self.central_opts[zone],
                    ZX,
                    Zy,
                    seed_key=f"central:{zone}",
                )

    def _train_one_local(
        self,
        model: nn.Module,
        opt: optim.Optimizer,
        X: np.ndarray,
        y_dbm: np.ndarray,
        *,
        seed_key: str = "aux",
        sample_weights: np.ndarray | None = None,
    ) -> None:
        cfg = self.cfg
        y_scaled = self._normalize_target_from_rssi(y_dbm)
        add_evidence = getattr(model, "add_evidence", None)
        reset_evidence = getattr(model, "reset_evidence", None)
        if callable(add_evidence) and callable(reset_evidence):
            reset_evidence()
            add_evidence(
                torch.as_tensor(
                    X, dtype=torch.float32, device=self.aux_device
                ),
                torch.as_tensor(
                    y_scaled, dtype=torch.float32, device=self.aux_device
                ),
                origin=rre_sim._stable_torch_seed(
                    "mergeable-aux", int(cfg.seed), seed_key
                ),
                sample_weights=(
                    None
                    if sample_weights is None
                    else torch.as_tensor(
                        sample_weights,
                        dtype=torch.float32,
                        device=self.aux_device,
                    )
                ),
            )
            return
        xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y_scaled, dtype=torch.float32)
        if yt.ndim == 1:
            yt = yt.unsqueeze(-1)
        if sample_weights is not None:
            wt_np = np.asarray(sample_weights, dtype=np.float32).reshape(-1, 1)
            if int(wt_np.shape[0]) != int(X.shape[0]):
                raise ValueError(
                    f"sample_weights length {wt_np.shape[0]} does not match X rows {X.shape[0]}"
                )
            wt = torch.tensor(wt_np, dtype=torch.float32)
            ds = torch.utils.data.TensorDataset(xt, yt, wt)
        else:
            ds = torch.utils.data.TensorDataset(xt, yt)
        generator = torch.Generator()
        generator.manual_seed(
            rre_sim._stable_torch_seed(
                cfg.seed,
                "aux-local",
                int(getattr(self, "_current_sumo_step", 0)),
                seed_key,
                int(X.shape[0]),
            )
        )
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.local_batch_size,
            shuffle=True,
            generator=generator,
        )
        model.train()
        for _ in range(cfg.local_epochs):
            for batch in loader:
                if len(batch) == 3:
                    bx, by, bw = batch
                else:
                    bx, by = batch
                    bw = None
                bx = bx.to(self.aux_device)
                by = by.to(self.aux_device)
                bw = bw.to(self.aux_device) if bw is not None else None
                opt.zero_grad(set_to_none=True)
                loss = self._weighted_regression_loss(model(bx), by, bw)
                loss.backward()
                opt.step()

    def _predict_model_dbm(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.aux_device)
            yp = model(xt).cpu().numpy().flatten()
        return self._denorm_dbm(yp)

    def _compute_aux_fidelity(self, row: dict[str, float | int]) -> dict[str, float | int]:
        if self.skip_aux_baselines:
            return row
        methods = [
            name
            for name in ("iso", "greedy", "central")
            if name in self.aux_baselines
        ]
        totals = {
            name: {
                "individual_sum_sq": 0.0,
                "individual_count": 0,
                "model_rmse": [],
            }
            for name in methods
        }
        for az, (X, y) in self.fidelity_grid.items():
            members = [i for i, ns in enumerate(self.nodes) if ns.current_az == az]
            if X.shape[0] == 0 or not members:
                for name in methods:
                    row[f"{name}_z{az}"] = float("nan")
                continue

            for name in methods:
                if name == "iso":
                    predictions = [
                        self._predict_model_dbm(self.iso_models[index], X)
                        for index in members
                    ]
                elif name == "greedy":
                    predictions = [
                        self._predict_model_dbm(self.greedy_models[index], X)
                        for index in members
                    ]
                else:
                    predictions = [
                        self._predict_model_dbm(self.central_models[az], X)
                    ]
                stats = rre_sim._prediction_fidelity_statistics(
                    np.stack(predictions), y
                )
                individual = float(stats["pooled_rmse"])
                row[f"{name}_z{az}"] = individual
                row[f"{name}_individual_z{az}"] = individual
                row[f"{name}_mean_model_rmse_z{az}"] = float(
                    stats["mean_model_rmse"]
                )
                totals[name]["individual_sum_sq"] += float(
                    stats["individual_sum_sq"]
                )
                totals[name]["individual_count"] += int(stats["individual_count"])
                totals[name]["model_rmse"].extend(
                    float(value) for value in np.asarray(stats["model_rmse"])
                )

        for name in methods:
            values = totals[name]
            individual_count = int(values["individual_count"])
            model_rmse = list(values["model_rmse"])
            pooled = (
                float(
                    np.sqrt(
                        float(values["individual_sum_sq"]) / individual_count
                    )
                )
                if individual_count > 0
                else float("nan")
            )
            row[f"{name}_total"] = pooled
            row[f"{name}_individual_total"] = pooled
            row[f"{name}_mean_model_rmse_total"] = (
                float(np.mean(model_rmse)) if model_rmse else float("nan")
            )
        return row

    @staticmethod
    def _nanmean(vals: list[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        if not np.any(np.isfinite(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    def _save_method_plot(self, out_dir: Path) -> None:
        if not self.fidelity_history:
            return
        steps = [int(r["step"]) for r in self.fidelity_history]
        rl_keys = sorted(self.agents.keys())
        cmap = plt.get_cmap("tab10")

        plt.figure(figsize=(10, 5.5))

        ref_b = float(self.cfg.beta)

        def _legend_for_mode(mid: str) -> str:
            if mid == "v4":
                return "RL reference (oracle)"
            if mid == "t2_b0.75":
                return "z-ramp"
            if "_b" in mid and mid.startswith("t"):
                pref, bpart = mid.split("_b", 1)
                try:
                    b = float(bpart)
                    return f"RL {pref} β={b:g}"
                except ValueError:
                    pass
            if mid.startswith("t") and mid[1:].isdigit():
                return f"RL {mid.upper()} β={ref_b:g}"
            return f"RL ({mid})"

        for ki, mk in enumerate(rl_keys):
            ys = [float(r.get(f"{mk}_total", float("nan"))) for r in self.fidelity_history]
            plt.plot(steps, ys, label=_legend_for_mode(mk), color=cmap(ki % 10), linewidth=2)
        if "iso" in self.aux_baselines:
            iso = [float(r.get("iso_total", float("nan"))) for r in self.fidelity_history]
            plt.plot(steps, iso, "--", label="Local only", color="#555555", linewidth=2)
        if "greedy" in self.aux_baselines:
            greedy = [float(r.get("greedy_total", float("nan"))) for r in self.fidelity_history]
            plt.plot(steps, greedy, ":", label="Greedy share-all", color="#888888", linewidth=2.2)
        if "central" in self.aux_baselines:
            central = [float(r.get("central_total", float("nan"))) for r in self.fidelity_history]
            plt.plot(steps, central, "-.", label="Central per-zone", color="#aaaaaa", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("RMSE [dB]")
        plt.title("Average RMSE over zones by method")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7, ncol=2, loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "rmse_over_time_methods.png", dpi=180)
        plt.close()

    def _print_gossip_merge_summary(self) -> None:
        """How often each RL mode chose merge (action 1). RL ≈ local-only usually means ~0% accepts."""
        by_mode: dict[str, Counter[int]] = defaultdict(Counter)
        if self._decision_action_counts:
            for mode_id, counts in self._decision_action_counts.items():
                by_mode[str(mode_id)].update(counts)
        else:
            for row in self.decision_log:
                # Future-window modes defer reward but still log the merge decision here.
                by_mode[str(row["mode"])][int(row["action"])] += 1
        if not by_mode:
            print("[SUMO-RRE] Gossip merge summary: no decisions logged", flush=True)
            return
        parts: list[str] = []
        for mode_id in sorted(by_mode):
            c = by_mode[mode_id]
            acc = int(c.get(1, 0))
            n = acc + int(c.get(0, 0))
            pct = (100.0 * acc / n) if n else 0.0
            parts.append(f"{mode_id}: accept {acc}/{n} ({pct:.1f}%)")
        print("[SUMO-RRE] Gossip merge summary: " + "; ".join(parts), flush=True)

    def _write_partial_outputs(
        self,
        *,
        step: int,
        elapsed_s: float,
        reason: str,
        zone_rssi_rows: list[dict[str, float | int]] | None = None,
        dynamic_rows: list[dict[str, int | str]] | None = None,
        sharing_rows: list[dict[str, int | float]] | None = None,
        local_policy_rows: list[dict[str, int | float]] | None = None,
    ) -> None:
        """Persist useful in-run state without cleaning up the active Sionna scene."""
        out = Path(self.cfg.results_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.fidelity_history:
            fieldnames = sorted({k for row in self.fidelity_history for k in row.keys()})
            fieldnames = ["step"] + [f for f in fieldnames if f != "step"]
            with open(out / "fidelity.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in self.fidelity_history:
                    w.writerow(row)
            with open(out / "latest_fidelity.json", "w", encoding="utf-8") as f:
                json.dump(self.fidelity_history[-1], f, indent=2, sort_keys=True)

        self._flush_decision_log_stream()

        if zone_rssi_rows:
            with open(out / "zone_rssi_partial.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        "zone",
                        "n_links",
                        "mean_rssi_dbm",
                        "min_rssi_dbm",
                        "max_rssi_dbm",
                        "mean_propagation_loss_db",
                        "min_propagation_loss_db",
                        "max_propagation_loss_db",
                    ],
                )
                w.writeheader()
                for row in zone_rssi_rows:
                    w.writerow(row)

        if dynamic_rows:
            with open(out / "dynamic_obstacles_partial.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["step", "active_count", "active_ids"])
                w.writeheader()
                for row in dynamic_rows:
                    w.writerow(row)

        if sharing_rows:
            fields = sorted({k for row in sharing_rows for k in row.keys()})
            fields = ["step"] + [f for f in fields if f != "step"]
            with open(out / "sharing_events_partial.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in sharing_rows:
                    w.writerow(row)

        if local_policy_rows:
            fields = sorted({k for row in local_policy_rows for k in row.keys()})
            fields = ["step"] + [f for f in fields if f != "step"]
            with open(out / "local_policy_training_partial.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in local_policy_rows:
                    w.writerow(row)

        progress = {
            "step": int(step),
            "requested_steps": int(self.cfg.sim_steps),
            "elapsed_s": float(elapsed_s),
            "reason": str(reason),
            "fidelity_rows": int(len(self.fidelity_history)),
            "decision_rows": int(self._decision_log_row_count()),
            "sharing_rows": int(len(self.sharing_rows)),
            "local_policy_rows": int(len(self.local_policy_rows)),
            "zramp_policy_mode": str(self.zramp_policy_mode),
            "local_policy_share": bool(self.local_policy_share),
            "local_policy_initial_pull": str(self.local_policy_initial_pull),
            "local_policy_initial_pull_probability": float(self.local_policy_initial_pull_probability),
            "local_policy_pending_transitions": int(self._local_pending_transition_count()),
            "local_policy_train_updates": dict(self._local_policy_train_updates),
            "local_policy_pull_updates": dict(self._local_policy_pull_updates),
            "local_policy_initial_decisions": dict(self._local_policy_initial_decisions),
            "local_policy_initial_accepts": dict(self._local_policy_initial_accepts),
            "modes": list(self.agents.keys()),
        }
        if self.fidelity_history:
            progress["latest_fidelity_step"] = int(self.fidelity_history[-1]["step"])
        with open(out / "progress.json", "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, sort_keys=True)

    def _run_trace_record_only(self) -> None:
        if self._trace_record_path is None:
            raise RuntimeError("--trace-record-only requires --measurement-trace-out")
        cfg = self.cfg
        os.makedirs(cfg.results_dir, exist_ok=True)
        cfg.save(os.path.join(cfg.results_dir, "config.json"))
        zone_rssi_rows: list[dict[str, float | int]] = []
        dynamic_rows: list[dict[str, int | str]] = []
        dynamic_schedule = getattr(self.map_engine, "dynamic_schedule", None)
        using_mobility_trace = self._mobility_trace is not None
        source_desc = (
            f"mobility trace {self.mobility_trace_in}"
            if using_mobility_trace
            else "live SUMO"
        )
        print(
            f"[SUMO-RRE] Recording Sionna measurement trace for {cfg.sim_steps} steps "
            f"from {source_desc} -> {self._trace_record_path}",
            flush=True,
        )
        total_start = time.time()
        last_completed_step = 0
        stopped_reason = "completed"
        try:
            for step in range(1, int(cfg.sim_steps) + 1):
                t0 = time.time()
                self._current_sumo_step = int(step)
                if using_mobility_trace:
                    synced = self._apply_mobility_trace_node_state(
                        step,
                        reset_on_zone_change=True,
                    )
                else:
                    if self.random_od_routing:
                        self._refresh_random_od_routes(step=step)
                        self._apply_intersection_right_of_way(step=step)
                    traci.simulationStep()
                    synced = self._sync_nodes_from_sumo(reset_on_zone_change=True)
                self._record_trace_node_state(step, synced=synced)

                fidelity_refresh_zones: set[int] = set()
                if dynamic_schedule is not None:
                    previous_dynamic = set(getattr(self.map_engine, "_last_dynamic_active", ()))
                    active_dynamic = self.map_engine.apply_dynamic_step(step)
                    fidelity_refresh_zones = self._changed_dynamic_zones(
                        previous_dynamic,
                        set(active_dynamic),
                    )
                    if fidelity_refresh_zones and not self._explicit_fidelity_schedule_enabled():
                        self._trace_fidelity_step = int(step)
                        self._build_fidelity_grid(
                            n_pairs=cfg.fidelity_grid_per_zone,
                            zones=fidelity_refresh_zones,
                        )
                    active_ids = ";".join(active_dynamic)
                    self._trace_dynamic_by_step[int(step)] = active_ids
                    if fidelity_refresh_zones and not self._explicit_fidelity_schedule_enabled():
                        self._trace_refresh_zones_by_step[int(step)] = sorted(fidelity_refresh_zones)
                    dynamic_rows.append(
                        {
                            "step": int(step),
                            "active_count": int(len(active_dynamic)),
                            "active_ids": active_ids,
                        }
                    )
                elif step % max(1, int(cfg.fidelity_refresh_every)) == 0:
                    self._trace_fidelity_step = int(step)
                    self._build_fidelity_grid(n_pairs=cfg.fidelity_grid_per_zone)
                    self._trace_refresh_zones_by_step[int(step)] = list(range(int(cfg.num_zones)))

                zone_nodes: dict[int, list[int]] = defaultdict(list)
                for i, ns in enumerate(self.nodes):
                    if not bool(self._current_node_active[i]):
                        continue
                    zone_nodes[int(ns.current_az)].append(i)

                meas = self.tracer.step_measurements([ns.node for ns in self.nodes], zone_nodes)
                self._record_trace_measurements(step, meas)

                zone_vals: dict[int, list[float]] = defaultdict(list)
                for az, _tx_idx, _rx_idx, val in meas:
                    zone_vals[int(az)].append(float(val))
                for az in range(int(cfg.num_zones)):
                    vals = zone_vals.get(az, [])
                    zone_rssi_rows.append(
                        {
                            "step": int(step),
                            "zone": int(az),
                            "n_links": int(len(vals)),
                            "mean_rssi_dbm": float(sum(vals) / len(vals)) if vals else float("nan"),
                            "min_rssi_dbm": float(min(vals)) if vals else float("nan"),
                            "max_rssi_dbm": float(max(vals)) if vals else float("nan"),
                            "mean_propagation_loss_db": float(self.cfg.tx_power_dbm - (sum(vals) / len(vals))) if vals else float("nan"),
                            "min_propagation_loss_db": float(self.cfg.tx_power_dbm - max(vals)) if vals else float("nan"),
                            "max_propagation_loss_db": float(self.cfg.tx_power_dbm - min(vals)) if vals else float("nan"),
                        }
                    )

                scheduled = self._fidelity_schedule_spec(step)
                if scheduled is not None:
                    n_pairs, is_final = scheduled
                    self._trace_fidelity_step = int(step)
                    self._build_fidelity_grid(n_pairs=n_pairs)
                    self._trace_refresh_zones_by_step[int(step)] = list(range(int(cfg.num_zones)))

                last_completed_step = int(step)
                if cfg.verbose or (self.progress_every and step % self.progress_every == 0):
                    print(
                        f"[SUMO-RRE] trace step {step:03d}/{cfg.sim_steps} "
                        f"dt={time.time() - t0:.1f}s elapsed={time.time() - total_start:.1f}s "
                        f"synced={synced}/{cfg.num_nodes} samples={len(meas)}",
                        flush=True,
                    )
                if (not using_mobility_trace) and traci.simulation.getMinExpectedNumber() <= 0:
                    stopped_reason = "sumo_finished"
                    print(f"[SUMO-RRE] SUMO finished early at step {step}", flush=True)
                    break

            if (
                stopped_reason == "completed"
                and last_completed_step >= int(cfg.sim_steps)
                and self._fidelity_schedule_spec(int(cfg.sim_steps)) is None
            ):
                self._trace_fidelity_step = int(cfg.sim_steps)
                self._build_fidelity_grid(n_pairs=cfg.final_fidelity_grid_per_zone)
            self._save_measurement_trace(last_step=last_completed_step, reason=stopped_reason)

            zone_rssi_csv = Path(cfg.results_dir) / "zone_rssi.csv"
            with open(zone_rssi_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        "zone",
                        "n_links",
                        "mean_rssi_dbm",
                        "min_rssi_dbm",
                        "max_rssi_dbm",
                        "mean_propagation_loss_db",
                        "min_propagation_loss_db",
                        "max_propagation_loss_db",
                    ],
                )
                w.writeheader()
                for row in zone_rssi_rows:
                    w.writerow(row)
            if dynamic_rows:
                dynamic_csv = Path(cfg.results_dir) / "dynamic_obstacles.csv"
                with open(dynamic_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["step", "active_count", "active_ids"])
                    w.writeheader()
                    for row in dynamic_rows:
                        w.writerow(row)
            self._write_partial_outputs(
                step=last_completed_step,
                elapsed_s=time.time() - total_start,
                reason=f"trace_{stopped_reason}",
                zone_rssi_rows=zone_rssi_rows,
                dynamic_rows=dynamic_rows,
                sharing_rows=[],
                local_policy_rows=[],
            )
        finally:
            self._close_decision_log_stream()
            if self._sumo_open:
                traci.close()
                self._sumo_open = False

    def run(self) -> None:
        if self.trace_record_only:
            self._run_trace_record_only()
            return
        cfg = self.cfg
        os.makedirs(cfg.results_dir, exist_ok=True)
        cfg.save(os.path.join(cfg.results_dir, "config.json"))
        zone_rssi_rows: list[dict[str, float | int]] = []
        dynamic_rows: list[dict[str, int | str]] = []
        dynamic_schedule = getattr(self.map_engine, "dynamic_schedule", None)
        if dynamic_schedule is not None:
            dyn_summary = dynamic_schedule.summary()
            dyn_summary["requested_sim_steps"] = int(cfg.sim_steps)
            dyn_summary["requested_source"] = str(self.dynamic_map)
            with open(
                os.path.join(cfg.results_dir, "dynamic_schedule_summary.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(dyn_summary, f, indent=2, sort_keys=True)
        if cfg.verbose:
            print(
                f"[SUMO-RRE] Running {cfg.sim_steps} steps with RL modes {cfg.active_modes}",
                flush=True,
            )
            print(
                f"[SUMO-RRE] Z-RAMP decentralized local-policy mode "
                f"policy_share={int(self.local_policy_share)} "
                f"initial_pull={self.local_policy_initial_pull} "
                f"p={self.local_policy_initial_pull_probability:.4f}",
                flush=True,
            )
            if dynamic_schedule is not None:
                gaps = dynamic_schedule.coverage_gaps(sim_steps=int(cfg.sim_steps))
                gap_msg = "none" if not gaps else str(gaps)
                print(
                    f"[SUMO-RRE] Dynamic map: {len(dynamic_schedule.obstacles)} obstacles, "
                    f"coverage gaps={gap_msg}",
                    flush=True,
                )

        total_start = time.time()
        last_completed_step = 0
        stopped_reason = "completed"
        try:
            for step in range(1, cfg.sim_steps + 1):
                t0 = time.time()
                self._current_sumo_step = int(step)
                self._reset_policy_step_counters()

                # 1) Mobility from SUMO, or cached node states during replay.
                for ns in self.nodes:
                    for v in ns.variants.values():
                        v.t_wait += 1
                    self._tick_spike_recovery(ns)
                fidelity_refresh_zones: set[int] = set()
                if self._trace_replay is not None:
                    synced = self._apply_trace_node_state(step, reset_on_zone_change=True)
                    refresh_map = self._trace_replay.get("refresh_zones_by_step", {})
                    if isinstance(refresh_map, dict):
                        fidelity_refresh_zones = {int(z) for z in refresh_map.get(int(step), [])}
                    if fidelity_refresh_zones and not self._explicit_fidelity_schedule_enabled():
                        self._trace_fidelity_step = int(step)
                        self._build_fidelity_grid(
                            n_pairs=cfg.fidelity_grid_per_zone,
                            zones=fidelity_refresh_zones,
                        )
                    dynamic_map = self._trace_replay.get("dynamic_by_step", {})
                    active_ids = ""
                    if isinstance(dynamic_map, dict):
                        active_ids = str(dynamic_map.get(int(step), ""))
                    if active_ids or self.dynamic_map:
                        dynamic_rows.append(
                            {
                                "step": int(step),
                                "active_count": int(len([x for x in active_ids.split(";") if x])),
                                "active_ids": active_ids,
                            }
                        )
                else:
                    if self.random_od_routing:
                        self._refresh_random_od_routes(step=step)
                        self._apply_intersection_right_of_way(step=step)
                    traci.simulationStep()
                    synced = self._sync_nodes_from_sumo(reset_on_zone_change=True)
                    self._record_trace_node_state(step, synced=synced)
                    if dynamic_schedule is not None:
                        previous_dynamic = set(getattr(self.map_engine, "_last_dynamic_active", ()))
                        active_dynamic = self.map_engine.apply_dynamic_step(step)
                        fidelity_refresh_zones = self._changed_dynamic_zones(
                            previous_dynamic,
                            set(active_dynamic),
                        )
                        if fidelity_refresh_zones and not self._explicit_fidelity_schedule_enabled():
                            self._trace_fidelity_step = int(step)
                            self._build_fidelity_grid(
                                n_pairs=cfg.fidelity_grid_per_zone,
                                zones=fidelity_refresh_zones,
                            )
                        active_ids = ";".join(active_dynamic)
                        self._trace_dynamic_by_step[int(step)] = active_ids
                        if fidelity_refresh_zones and not self._explicit_fidelity_schedule_enabled():
                            self._trace_refresh_zones_by_step[int(step)] = sorted(fidelity_refresh_zones)
                        dynamic_rows.append(
                            {
                                "step": int(step),
                                "active_count": int(len(active_dynamic)),
                                "active_ids": active_ids,
                            }
                        )

                # 2) Zone partitioning.
                zone_nodes: dict[int, list[int]] = defaultdict(list)
                for i, ns in enumerate(self.nodes):
                    if not bool(self._current_node_active[i]):
                        continue
                    zone_nodes[ns.current_az].append(i)

                # 3) Oracle sets for v4 (not used here, but left semantic-equivalent).
                if "v4" in self.reward_modes and step % cfg.oracle_every_k == 0:
                    self._build_oracle_sets()

                for mode in self.reward_modes.values():
                    mode.on_step_start(self, step)

                # 4) Ground-truth measurements via Sionna, or cached trace replay.
                if self._trace_replay is not None:
                    meas = self._trace_measurements_for_step(step)
                else:
                    meas = self.tracer.step_measurements([ns.node for ns in self.nodes], zone_nodes)
                    self._record_trace_measurements(step, meas)
                self._current_link_rssi_dbm = {
                    (int(tx_idx), int(rx_idx)): float(value)
                    for _zone, tx_idx, rx_idx, value in meas
                    if int(tx_idx) != int(rx_idx)
                }
                self._network_step_stats = {}
                contact_links = self._contact_links_from_measurements(meas)
                # Current observations update every predictor before model
                # sharing, so decisions use all locally available information.
                self._train_predictors_from_current_measurements(
                    step=int(step),
                    measurements=meas,
                )

                # 5) Gossip + RL decisions over feasible bidirectional contacts.
                decision_start = len(self.decision_log)
                before_gossip = getattr(self, "_before_gossip_step", None)
                if callable(before_gossip):
                    before_gossip(int(step))
                self._gossip_step(step, zone_nodes, contact_links=contact_links)
                after_gossip = getattr(
                    self, "_after_gossip_step_complete", None
                )
                if callable(after_gossip):
                    after_gossip(int(step))
                rl_events = Counter()
                for row in self.decision_log[decision_start:]:
                    if int(row.get("action", 0)) == 1:
                        rl_events[str(row.get("mode"))] += 1
                self.decision_log.clear()
                greedy_events = 0
                if "greedy" in self.aux_baselines:
                    greedy_events = self._greedy_share_step(zone_nodes, contact_links=contact_links)
                sharing_row: dict[str, int | float] = {
                    "step": int(step),
                    "feasible_contact_pairs": int(len(contact_links)),
                    "feasible_contact_decisions": int(2 * len(contact_links)),
                    "noise_floor_dbm": float(self.cfg.noise_floor_dbm),
                    "snr_min_db": float(self.cfg.snr_min_db),
                    "model_transfer_snr_min_db": float(self.cfg.model_transfer_snr_min_db),
                    "contact_threshold_snr_db": float(self.cfg.model_transfer_snr_min_db),
                    "contact_threshold_rx_power_dbm": float(self.cfg.noise_floor_dbm + self.cfg.model_transfer_snr_min_db),
                    "rssi_gossip_threshold_dbm": float(self.cfg.noise_floor_dbm + self.cfg.model_transfer_snr_min_db),
                    "propagation_loss_contact_threshold_db": float(self.cfg.tx_power_dbm - self.cfg.noise_floor_dbm - self.cfg.model_transfer_snr_min_db),
                    "greedy_events": int(greedy_events),
                }
                for mode_id in self.agents:
                    sharing_row[f"{mode_id}_events"] = int(rl_events.get(mode_id, 0))
                for key, value in dict(
                    getattr(self, "_network_step_stats", {})
                ).items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        sharing_row[str(key)] = (
                            int(value)
                            if isinstance(value, (int, np.integer))
                            else float(value)
                        )
                sharing_row.update(
                    self._communication_overhead_row(
                        feasible_decisions=int(2 * len(contact_links)),
                        greedy_events=int(greedy_events),
                        rl_events=rl_events,
                    )
                )
                self.sharing_rows.append(sharing_row)

                zone_vals: dict[int, list[float]] = defaultdict(list)
                for az, _tx_idx, _rx_idx, val in meas:
                    zone_vals[int(az)].append(float(val))
                for az in range(cfg.num_zones):
                    vals = zone_vals.get(az, [])
                    if vals:
                        mean_v = sum(vals) / float(len(vals))
                        min_v = min(vals)
                        max_v = max(vals)
                    else:
                        mean_v = float("nan")
                        min_v = float("nan")
                        max_v = float("nan")
                    zone_rssi_rows.append(
                        {
                            "step": int(step),
                            "zone": int(az),
                            "n_links": int(len(vals)),
                            "mean_rssi_dbm": float(mean_v),
                            "min_rssi_dbm": float(min_v),
                            "max_rssi_dbm": float(max_v),
                            "mean_propagation_loss_db": float(self.cfg.tx_power_dbm - mean_v) if vals else float("nan"),
                            "min_propagation_loss_db": float(self.cfg.tx_power_dbm - max_v) if vals else float("nan"),
                            "max_propagation_loss_db": float(self.cfg.tx_power_dbm - min_v) if vals else float("nan"),
                        }
                    )


                # 7) Feed newly received feasible samples into open t-window slots.
                stream_modes = [mode for mode in self.reward_modes.values() if hasattr(mode, "ingest_sample")]
                if stream_modes:
                    for i, rows in self._meas_per_node.items():
                        ns = self.nodes[i]
                        for feat, val in rows:
                            for mode in stream_modes:
                                mode.ingest_sample(ns, feat, val)  # type: ignore[attr-defined]

                # 8) Drain matured transitions.
                for mode_id, mode in self.reward_modes.items():
                    ready = mode.on_step_end(self, step)
                    for transition in ready:
                        self._queue_rl_transition(mode_id, transition)

                # 10) Decentralized per-vehicle DQN policy optimization.
                losses = self._train_rl_agents(step)
                self.local_policy_rows.append(
                    {
                        "step": int(step),
                        "pending_transitions": int(self._local_pending_transition_count()),
                        "queued_transitions": int(self._last_local_policy_queued_transitions),
                        "pull_updates_this_step": int(self._last_local_policy_pull_updates),
                        "train_updates_this_step": int(self._last_local_policy_train_updates_this_step),
                        **{
                            f"{mode_id}_train_updates": int(self._local_policy_train_updates[mode_id])
                            for mode_id in self.agents
                        },
                    }
                )

                # 11) Fidelity logging.
                fid_row_for_log = None
                scheduled = self._fidelity_schedule_spec(step)
                if scheduled is not None:
                    n_pairs, is_final = scheduled
                    fid_row_for_log = self._evaluate_fidelity_now(
                        step,
                        n_pairs=n_pairs,
                        is_final=is_final,
                    )
                else:
                    if (
                        self._trace_replay is None
                        and dynamic_schedule is None
                        and step % max(1, int(cfg.fidelity_refresh_every)) == 0
                    ):
                        self._trace_fidelity_step = int(step)
                        self._build_fidelity_grid(n_pairs=cfg.fidelity_grid_per_zone)
                        self._trace_refresh_zones_by_step[int(step)] = list(range(int(cfg.num_zones)))
                    if int(cfg.fidelity_log_every) > 0 and step % int(cfg.fidelity_log_every) == 0:
                        row = self._compute_fidelity_row(step)
                        row = self._compute_aux_fidelity(row)
                        self.fidelity_history.append(row)
                        fid_row_for_log = row

                dt = time.time() - t0
                elapsed = time.time() - total_start
                last_completed_step = int(step)
                should_log_rmse = self.log_rmse_every and step % self.log_rmse_every == 0
                if should_log_rmse and fid_row_for_log is None:
                    fid_row_for_log = self._compute_fidelity_row(step)
                    fid_row_for_log = self._compute_aux_fidelity(fid_row_for_log)
                if cfg.verbose or should_log_rmse:
                    fid_row = (
                        fid_row_for_log
                        if fid_row_for_log is not None
                        else (
                            self.fidelity_history[-1]
                            if (
                                self.fidelity_history
                                and int(self.fidelity_history[-1].get("step", -1)) == int(step)
                            )
                            else self._compute_aux_fidelity(self._compute_fidelity_row(step))
                        )
                    )
                    rmse_str = " ".join(
                        [f"{m}:{float(fid_row.get(f'{m}_total', float('nan'))):.2f}" for m in self.agents]
                        + ([f"iso:{float(fid_row.get('iso_total', float('nan'))):.2f}"] if "iso" in self.aux_baselines else [])
                        + ([f"greedy:{float(fid_row.get('greedy_total', float('nan'))):.2f}"] if "greedy" in self.aux_baselines else [])
                        + ([f"central:{float(fid_row.get('central_total', float('nan'))):.2f}"] if "central" in self.aux_baselines else [])
                    )
                    loss_str = " ".join(f"{m}:{losses[m]:.4f}" for m in self.agents)
                    print(
                        f"[SUMO-RRE] step {step:03d}/{cfg.sim_steps} dt={dt:.1f}s "
                        f"synced={synced}/{cfg.num_nodes} "
                        f"policy_pending={self._local_pending_transition_count()} "
                        f"policy_trained={self._last_local_policy_train_updates_this_step} rmse {rmse_str} "
                        f"{self._communication_log_fragment()} "
                        f"losses {loss_str}",
                        flush=True,
                    )
                elif self.progress_every and step % self.progress_every == 0:
                    latest = (
                        f" latest_fidelity_step={int(self.fidelity_history[-1]['step'])}"
                        if self.fidelity_history
                        else ""
                    )
                    print(
                        f"[SUMO-RRE] progress step {step:03d}/{cfg.sim_steps} "
                        f"dt={dt:.1f}s elapsed={elapsed:.1f}s synced={synced}/{cfg.num_nodes}"
                        f"{latest}",
                        flush=True,
                    )

                if self.flush_every and step % self.flush_every == 0:
                    self._write_partial_outputs(
                        step=step,
                        elapsed_s=elapsed,
                        reason="periodic",
                        zone_rssi_rows=zone_rssi_rows,
                        dynamic_rows=dynamic_rows,
                        sharing_rows=self.sharing_rows,
                        local_policy_rows=self.local_policy_rows,
                    )

                if self.max_wall_seconds is not None and elapsed >= self.max_wall_seconds:
                    stopped_reason = "max_wall_seconds"
                    print(
                        f"[SUMO-RRE] stopping early at step {step}/{cfg.sim_steps} "
                        f"after {elapsed:.1f}s to save outputs before wall limit",
                        flush=True,
                    )
                    break

                if self._trace_replay is None and traci.simulation.getMinExpectedNumber() <= 0:
                    stopped_reason = "sumo_finished"
                    print(f"[SUMO-RRE] SUMO finished early at step {step}", flush=True)
                    break

            # Flush deferred transitions.
            for mode_id, mode in self.reward_modes.items():
                ready = mode.on_sim_end(self)
                for transition in ready:
                    self._queue_rl_transition(mode_id, transition)

            if cfg.verbose:
                self._print_gossip_merge_summary()

            # Final evaluation + outputs (same as base experiment).
            if stopped_reason == "completed" and last_completed_step >= cfg.sim_steps:
                scheduled_final = self._latest_fidelity_at_step(cfg.sim_steps)
                if scheduled_final is not None:
                    self.final_fidelity_snapshot = scheduled_final
                else:
                    self._trace_fidelity_step = int(cfg.sim_steps)
                    self._build_fidelity_grid(n_pairs=cfg.final_fidelity_grid_per_zone)
                    self.final_fidelity_snapshot = self._compute_fidelity_row(cfg.sim_steps)
                    self.final_fidelity_snapshot = self._compute_aux_fidelity(self.final_fidelity_snapshot)
            elif self.fidelity_history:
                self.final_fidelity_snapshot = dict(self.fidelity_history[-1])
            else:
                self.final_fidelity_snapshot = self._compute_fidelity_row(last_completed_step)
                self.final_fidelity_snapshot = self._compute_aux_fidelity(self.final_fidelity_snapshot)
            self._save_outputs()
            self._write_partial_outputs(
                step=last_completed_step,
                elapsed_s=time.time() - total_start,
                reason=stopped_reason,
                zone_rssi_rows=zone_rssi_rows,
                dynamic_rows=dynamic_rows,
                sharing_rows=self.sharing_rows,
                local_policy_rows=self.local_policy_rows,
            )
            zone_rssi_csv = Path(cfg.results_dir) / "zone_rssi.csv"
            with open(zone_rssi_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        "zone",
                        "n_links",
                        "mean_rssi_dbm",
                        "min_rssi_dbm",
                        "max_rssi_dbm",
                        "mean_propagation_loss_db",
                        "min_propagation_loss_db",
                        "max_propagation_loss_db",
                    ],
                )
                w.writeheader()
                for row in zone_rssi_rows:
                    w.writerow(row)
            if dynamic_rows:
                dynamic_csv = Path(cfg.results_dir) / "dynamic_obstacles.csv"
                with open(dynamic_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=["step", "active_count", "active_ids"],
                    )
                    w.writeheader()
                    for row in dynamic_rows:
                        w.writerow(row)
            if self.sharing_rows:
                sharing_csv = Path(cfg.results_dir) / "sharing_events.csv"
                fields = sorted({k for row in self.sharing_rows for k in row.keys()})
                fields = ["step"] + [f for f in fields if f != "step"]
                with open(sharing_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for row in self.sharing_rows:
                        w.writerow(row)
            if self._communication_assumptions:
                with open(Path(cfg.results_dir) / "communication_overhead_assumptions.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            **self._communication_assumptions,
                            "local_policy_initial_pull": str(self.local_policy_initial_pull),
                            "local_policy_initial_pull_probability": float(self.local_policy_initial_pull_probability),
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                    )
            if self.local_policy_rows:
                local_policy_csv = Path(cfg.results_dir) / "local_policy_training.csv"
                fields = sorted({k for row in self.local_policy_rows for k in row.keys()})
                fields = ["step"] + [f for f in fields if f != "step"]
                with open(local_policy_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for row in self.local_policy_rows:
                        w.writerow(row)
                with open(Path(cfg.results_dir) / "local_policy_summary.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "zramp_policy_mode": str(self.zramp_policy_mode),
                            "local_policy_share": bool(self.local_policy_share),
                            "local_policy_initial_pull": str(self.local_policy_initial_pull),
                            "local_policy_initial_pull_probability": float(self.local_policy_initial_pull_probability),
                            "local_policy_pending_transitions": int(self._local_pending_transition_count()),
                            "local_policy_train_updates": dict(self._local_policy_train_updates),
                            "local_policy_pull_updates": dict(self._local_policy_pull_updates),
                            "local_policy_initial_decisions": dict(self._local_policy_initial_decisions),
                            "local_policy_initial_accepts": dict(self._local_policy_initial_accepts),
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                    )
            if cfg.verbose:
                print(f"[SUMO-RRE] total runtime {time.time() - total_start:.1f}s", flush=True)
        finally:
            self._close_decision_log_stream()
            if self._sumo_open:
                traci.close()
                self._sumo_open = False


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SUMO + multi-mode RL vs local/greedy/central in one rollout"
    )
    p.add_argument("--sumo-config", default="city_grid.sumocfg")
    p.add_argument("--sumo-net", default="city_grid.net.xml")
    p.add_argument(
        "--dynamic-map",
        default=None,
        help="Optional dynamic radio-obstacle schedule JSON applied before each Sionna measurement",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-nodes", type=int, default=70)
    p.add_argument("--num-zones", type=int, default=4)
    p.add_argument("--sim-steps", type=int, default=300)
    p.add_argument("--results-dir", default="SUMO/results_city_grid_multi_rl")
    p.add_argument("--num-rays", type=int, default=100_000)
    p.add_argument("--trace-tx-batch-size", type=int, default=None)
    p.add_argument(
        "--reward-t",
        type=int,
        default=2,
        help="Future-window T used for heads tT_bβ (see --rl-betas)",
    )
    p.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Explicit active mode list; overrides --reward-t/--rl-betas/--also-windows.",
    )
    p.add_argument(
        "--rl-betas",
        type=str,
        default="1,1.25,1.5,1.75,2",
        help="Comma-separated β values → one RL head each: t<RewardT>_bβ",
    )
    p.add_argument(
        "--reference-beta",
        type=float,
        default=2.0,
        help="β for plain auxiliary heads (--also-windows and oracle v4 use ExperimentConfig.beta)",
    )
    p.add_argument(
        "--also-windows",
        type=str,
        default="1,3,5,7",
        help='Comma-separated extra horizons without β suffix (canonical tW). Omit reward-t to avoid duplication. '
        'Pass "none" to disable.',
    )
    p.add_argument(
        "--rl-action-policy",
        choices=["softmax", "argmax", "reject", "accept"],
        default=None,
        help="Default action policy for RL heads without a per-mode suffix.",
    )
    p.add_argument(
        "--diagnostic-reject",
        action="store_true",
        help="Add a forced-reject t<RewardT>_b0_reject head; should track local training.",
    )
    p.add_argument(
        "--oracle",
        action="store_true",
        help="Include reference mode v4 (extra ray-oracle evaluations per zone per step)",
    )
    p.add_argument(
        "--tx-power-dbm",
        type=float,
        default=None,
        help="Transmit power used in RSSI=mapping (see RRE_TX_POWER_DBM)",
    )
    p.add_argument(
        "--noise-floor-dbm",
        type=float,
        default=None,
        help="Receiver noise floor used for SNR feasibility, in dBm",
    )
    p.add_argument(
        "--snr-min-db",
        type=float,
        default=None,
        help="Minimum SNR in both directions for a same-zone contact to be feasible",
    )
    p.add_argument(
        "--rssi-gossip-threshold",
        type=float,
        default=None,
        help="Deprecated compatibility alias: received-power threshold in dBm; converted to snr_min_db",
    )
    p.add_argument(
        "--predictor-prior",
        choices=["snr-threshold", "max-loss", "none"],
        default=None,
        help="Initial propagation-loss prior for fresh predictors before any received samples are observed",
    )
    p.add_argument("--predictor-time", dest="predictor_include_time", action="store_true", default=None)
    p.add_argument("--no-predictor-time", dest="predictor_include_time", action="store_false")
    p.add_argument("--predictor-time-step-duration", type=float, default=None)
    p.add_argument("--predictor-time-unit", "--predictor-time-scale-steps", dest="predictor_time_unit", type=float, default=None)
    p.add_argument("--predictor-time-frequencies", type=int, default=None)
    p.add_argument("--predictor-time-min-period", type=float, default=None)
    p.add_argument("--predictor-time-max-period", type=float, default=None)
    p.set_defaults(predictor_include_time=None)
    p.add_argument(
        "--merge-strategy",
        choices=["average", "ot"],
        default=None,
        help="Model-weight merge rule for accepted pulls: average or sliced OT alignment",
    )
    p.add_argument(
        "--rl-only",
        action="store_true",
        help="Skip auxiliary local/greedy/central baselines for short RL-only experiments.",
    )
    p.add_argument(
        "--aux-baselines",
        default=None,
        help="Auxiliary baselines to train/evaluate, e.g. none, iso, greedy, central, all, or comma-combinations. Overrides --rl-only when provided.",
    )
    p.add_argument("--spike-recovery", action="store_true")
    p.add_argument("--spike-short-alpha", type=float, default=None)
    p.add_argument("--spike-long-alpha", type=float, default=None)
    p.add_argument("--spike-ratio", type=float, default=None)
    p.add_argument("--spike-abs-db", type=float, default=None)
    p.add_argument("--spike-min-batches", type=int, default=None)
    p.add_argument("--spike-window-steps", type=int, default=None)
    p.add_argument("--spike-accept-budget", type=int, default=None)
    p.add_argument("--spike-cooldown-steps", type=int, default=None)
    p.add_argument("--spike-beta-scale", type=float, default=None)
    p.add_argument("--spike-accept-prob", type=float, default=None)
    p.add_argument("--fidelity-grid-per-zone", type=int, default=None)
    p.add_argument("--fidelity-grid-n-tx", type=int, default=None)
    p.add_argument("--fidelity-refresh-every", type=int, default=None)
    p.add_argument("--fidelity-eval-every", type=int, default=None)
    p.add_argument("--final-fidelity-grid-per-zone", type=int, default=None)
    p.add_argument("--fidelity-final-steps", default=None)
    p.add_argument("--fidelity-log-every", type=int, default=None)
    p.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="When --quiet is used, print cheap progress every N steps without computing RMSE.",
    )
    p.add_argument(
        "--log-rmse-every",
        type=int,
        default=0,
        help="Print a SUMO-RRE RMSE/loss line every N steps for active RL heads.",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=0,
        help="Write partial CSV/JSON outputs every N steps so notebook timeouts keep usable data.",
    )
    p.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="Stop the simulation loop after roughly this many seconds and save partial outputs.",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--random-od-routing",
        action="store_true",
        help="Continuously reroute tracked vehicles to seeded random destinations in different zones",
    )
    p.add_argument(
        "--route-min-zone-distance",
        type=int,
        default=1,
        help="Minimum Manhattan zone distance for neighboring random OD destinations",
    )
    p.add_argument(
        "--route-max-zone-distance",
        type=int,
        default=1,
        help="Maximum Manhattan zone distance for neighboring random OD destinations",
    )
    p.add_argument(
        "--open-boundary-routing",
        action="store_true",
        help="Let vehicles exit at boundary edges and respawn from the opposite side",
    )
    p.add_argument(
        "--open-boundary-probability",
        type=float,
        default=0.45,
        help="Probability that a new random-OD assignment targets an open-boundary exit",
    )
    p.add_argument(
        "--open-boundary-margin",
        type=float,
        default=0.12,
        help="Map-fraction margin used to classify boundary entry/exit edges",
    )
    p.add_argument(
        "--open-boundary-respawn-buffer",
        type=int,
        default=2,
        help="Deprecated safety buffer; respawn now also requires physical boundary proximity",
    )
    p.add_argument(
        "--open-boundary-exit-margin",
        type=float,
        default=0.035,
        help="Map-fraction distance from the border required before open-boundary respawn",
    )
    p.add_argument(
        "--jam-reroute-wait-seconds",
        type=float,
        default=25.0,
        help="Reroute vehicles that have been waiting this long instead of feeding a jam",
    )
    p.add_argument(
        "--intersection-control",
        action="store_true",
        help="Meter queued four-way junction approaches to provide explicit right-of-way",
    )
    p.add_argument(
        "--intersection-wait-seconds",
        type=float,
        default=12.0,
        help="Waiting-time threshold before a junction approach participates in deadlock control",
    )
    p.add_argument(
        "--intersection-release-steps",
        type=int,
        default=8,
        help="Number of SUMO steps one queued approach keeps right-of-way",
    )
    p.add_argument(
        "--intersection-stop-distance",
        type=float,
        default=24.0,
        help="Distance before a junction where non-priority approaches are held",
    )
    p.add_argument(
        "--zone-model-memory",
        action="store_true",
        help="Reuse each node's latest saved model when it re-enters a zone",
    )
    p.add_argument(
        "--local-policy-share",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In local policy mode, accepted RSSI pulls also pull and merge the provider DQN policy",
    )
    p.add_argument(
        "--local-policy-initial-pull",
        choices=["greedy", "byte-match", "fixed"],
        default="byte-match",
        help="Initial action rule before a car has trained/received a DQN policy",
    )
    p.add_argument(
        "--local-policy-initial-pull-prob",
        type=float,
        default=None,
        help="Override initial pull probability; mainly for --local-policy-initial-pull fixed",
    )
    p.add_argument(
        "--local-policy-updates-per-batch",
        type=int,
        default=1,
        help="Local DQN train updates to run per newly accumulated full batch of transitions",
    )
    p.add_argument(
        "--mobility-trace-in",
        default=None,
        help="Replay precomputed SUMO mobility positions from this JSON while recording Sionna measurements",
    )
    p.add_argument(
        "--measurement-trace-in",
        default=None,
        help="Replay cached SUMO/Sionna measurements from this .npz trace instead of ray tracing",
    )
    p.add_argument(
        "--measurement-trace-out",
        default=None,
        help="Write a cached SUMO/Sionna measurement trace to this .npz file",
    )
    p.add_argument(
        "--trace-record-only",
        action="store_true",
        help="Only record the measurement trace; skip RL training and policy evaluation",
    )
    return p


def _parse_positive_int_list(raw: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if p:
            out.append(int(p))
    return tuple(out)


def _parse_beta_list(raw: str) -> tuple[float, ...]:
    out: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if p:
            out.append(float(p))
    return tuple(out)


def _results_complete(results_dir: Path) -> bool:
    progress = results_dir / "progress.json"
    fidelity = results_dir / "fidelity.csv"
    if not progress.is_file() or not fidelity.is_file():
        return False
    try:
        with open(progress, encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("reason", "")).lower() == "completed"
    except Exception:
        return False


def _acquire_results_lock(results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    if _results_complete(results_dir):
        print(f"[SUMO-RRE] Results already complete in {results_dir}; skipping", flush=True)
        return None

    lock_path = results_dir / ".run.lock"
    wait_start = time.time()
    last_notice = 0.0
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "pid": os.getpid(),
                    "host": os.uname().nodename,
                    "created_at_unix": time.time(),
                    "results_dir": str(results_dir),
                }, sort_keys=True) + "\n")
            return lock_path
        except FileExistsError:
            if _results_complete(results_dir):
                print(f"[SUMO-RRE] Results completed while waiting for {results_dir}; skipping", flush=True)
                return None
            now = time.time()
            if now - last_notice >= 300.0:
                print(
                    f"[SUMO-RRE] Waiting for active writer lock {lock_path} "
                    f"for {(now - wait_start) / 60.0:.1f} min",
                    flush=True,
                )
                last_notice = now
            time.sleep(30.0)


def _release_results_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    results_lock = _acquire_results_lock(Path(args.results_dir))
    if results_lock is None:
        return 0
    try:
        return _main_locked(args)
    finally:
        _release_results_lock(results_lock)


def _main_locked(args: argparse.Namespace) -> int:
    if args.modes:
        active_modes = parse_modes(str(args.modes))
        ref_beta = float(args.reference_beta)
    else:
        reward_t = int(args.reward_t)
        if f"t{reward_t}" not in WINDOW_T_BY_MODE:
            raise SystemExit(f"--reward-t {reward_t} must be one of {tuple(WINDOW_T_BY_MODE.keys())}")
        betas = _parse_beta_list(args.rl_betas)
        if not betas:
            raise SystemExit("Need at least one value in --rl-betas")
        modes: list[str] = []
        modes.extend(f"t{reward_t}_b{b:g}" for b in betas)
        if args.diagnostic_reject:
            modes.append(f"t{reward_t}_b0_reject")

        ref_beta = float(args.reference_beta)
        aw = str(args.also_windows).strip()
        if aw.lower() in {"none", "off"}:
            extra_horizons = ()
        else:
            extra_horizons = _parse_positive_int_list(aw)
            for tw in extra_horizons:
                if tw not in WINDOW_T_VALUES:
                    raise SystemExit(f"--also-windows contained {tw}; allowed {WINDOW_T_VALUES}")
        plain_added: set[str] = set()
        for tw in sorted({w for w in extra_horizons if w != reward_t}):
            tid = f"t{tw}"
            if tid not in plain_added:
                plain_added.add(tid)
                modes.append(tid)
        if args.oracle:
            if "v4" not in modes:
                modes.append("v4")

        active_modes = tuple(modes)

    net_bounds = read_net_bounds(args.sumo_net)
    map_size = max(net_bounds.width, net_bounds.height)
    cfg_kwargs: dict = dict(
        seed=int(args.seed),
        num_nodes=int(args.num_nodes),
        num_zones=int(args.num_zones),
        sim_steps=int(args.sim_steps),
        map_size=float(map_size),
        beta=ref_beta,
        active_modes=active_modes,
        results_dir=str(args.results_dir),
        num_rays=int(args.num_rays),
        verbose=not args.quiet,
    )
    if args.spike_recovery:
        cfg_kwargs["spike_recovery_enabled"] = True
    spike_overrides = {
        "spike_recovery_short_alpha": args.spike_short_alpha,
        "spike_recovery_long_alpha": args.spike_long_alpha,
        "spike_recovery_ratio": args.spike_ratio,
        "spike_recovery_abs_db": args.spike_abs_db,
        "spike_recovery_min_batches": args.spike_min_batches,
        "spike_recovery_window_steps": args.spike_window_steps,
        "spike_recovery_accept_budget": args.spike_accept_budget,
        "spike_recovery_cooldown_steps": args.spike_cooldown_steps,
        "spike_recovery_beta_scale": args.spike_beta_scale,
        "spike_recovery_accept_prob": args.spike_accept_prob,
    }
    cfg_kwargs.update({k: v for k, v in spike_overrides.items() if v is not None})
    if args.rl_action_policy is not None:
        cfg_kwargs["rl_action_policy"] = str(args.rl_action_policy)
    if args.trace_tx_batch_size is not None:
        cfg_kwargs["trace_tx_batch_size"] = int(args.trace_tx_batch_size)
    if args.fidelity_grid_per_zone is not None:
        cfg_kwargs["fidelity_grid_per_zone"] = int(args.fidelity_grid_per_zone)
    if args.fidelity_grid_n_tx is not None:
        cfg_kwargs["fidelity_grid_n_tx"] = int(args.fidelity_grid_n_tx)
    if args.fidelity_refresh_every is not None:
        cfg_kwargs["fidelity_refresh_every"] = int(args.fidelity_refresh_every)
    if args.fidelity_eval_every is not None:
        cfg_kwargs["fidelity_eval_every"] = int(args.fidelity_eval_every)
    if args.final_fidelity_grid_per_zone is not None:
        cfg_kwargs["final_fidelity_grid_per_zone"] = int(args.final_fidelity_grid_per_zone)
    if args.fidelity_final_steps is not None:
        raw_steps = str(args.fidelity_final_steps).strip()
        cfg_kwargs["fidelity_final_steps"] = (
            tuple() if raw_steps.lower() in {"", "none", "off"} else _parse_positive_int_list(raw_steps)
        )
    if args.fidelity_log_every is not None:
        cfg_kwargs["fidelity_log_every"] = int(args.fidelity_log_every)
    if args.tx_power_dbm is not None:
        cfg_kwargs["tx_power_dbm"] = float(args.tx_power_dbm)
    if args.noise_floor_dbm is not None:
        cfg_kwargs["noise_floor_dbm"] = float(args.noise_floor_dbm)
    if args.snr_min_db is not None:
        cfg_kwargs["snr_min_db"] = float(args.snr_min_db)
    if args.rssi_gossip_threshold is not None:
        cfg_kwargs["rssi_gossip_threshold"] = float(args.rssi_gossip_threshold)
    if args.predictor_prior is not None:
        cfg_kwargs["predictor_prior"] = str(args.predictor_prior)
    if args.predictor_include_time is not None:
        cfg_kwargs["predictor_include_time"] = bool(args.predictor_include_time)
    if args.predictor_time_step_duration is not None:
        cfg_kwargs["predictor_time_step_duration"] = float(args.predictor_time_step_duration)
    if args.predictor_time_unit is not None:
        cfg_kwargs["predictor_time_unit"] = float(args.predictor_time_unit)
    if args.predictor_time_frequencies is not None:
        cfg_kwargs["predictor_time_num_frequencies"] = int(args.predictor_time_frequencies)
    if args.predictor_time_min_period is not None:
        cfg_kwargs["predictor_time_min_period"] = float(args.predictor_time_min_period)
    if args.predictor_time_max_period is not None:
        cfg_kwargs["predictor_time_max_period"] = float(args.predictor_time_max_period)
    elif bool(args.predictor_include_time):
        duration = float(args.predictor_time_step_duration or 1.0)
        unit = float(args.predictor_time_unit or 1.0)
        minimum = float(args.predictor_time_min_period or 2.0)
        cfg_kwargs["predictor_time_max_period"] = max(
            float(args.sim_steps) * duration / unit,
            minimum * 2.0,
        )
    if args.merge_strategy is not None:
        cfg_kwargs["merge_strategy"] = str(args.merge_strategy)
    cfg = build_config_from_env(**cfg_kwargs)
    sim = SumoT2Simulation(
        cfg,
        sumo_config=args.sumo_config,
        sumo_net=args.sumo_net,
        dynamic_map=args.dynamic_map,
        skip_aux_baselines=bool(args.rl_only),
        aux_baselines=args.aux_baselines,
        progress_every=int(args.progress_every),
        log_rmse_every=int(args.log_rmse_every),
        flush_every=int(args.flush_every),
        max_wall_seconds=args.max_wall_seconds,
        random_od_routing=bool(args.random_od_routing),
        route_min_zone_distance=int(args.route_min_zone_distance),
        route_max_zone_distance=args.route_max_zone_distance,
        open_boundary_routing=bool(args.open_boundary_routing),
        open_boundary_probability=float(args.open_boundary_probability),
        open_boundary_margin=float(args.open_boundary_margin),
        open_boundary_exit_margin=float(args.open_boundary_exit_margin),
        open_boundary_respawn_buffer=int(args.open_boundary_respawn_buffer),
        jam_reroute_wait_seconds=float(args.jam_reroute_wait_seconds),
        intersection_control=bool(args.intersection_control),
        intersection_wait_seconds=float(args.intersection_wait_seconds),
        intersection_release_steps=int(args.intersection_release_steps),
        intersection_stop_distance=float(args.intersection_stop_distance),
        zone_model_memory=bool(args.zone_model_memory),
        local_policy_share=bool(args.local_policy_share),
        local_policy_initial_pull=str(args.local_policy_initial_pull),
        local_policy_initial_pull_prob=args.local_policy_initial_pull_prob,
        local_policy_updates_per_batch=int(args.local_policy_updates_per_batch),
        mobility_trace_in=args.mobility_trace_in,
        measurement_trace_in=args.measurement_trace_in,
        measurement_trace_out=args.measurement_trace_out,
        trace_record_only=bool(args.trace_record_only),
    )
    sim.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
