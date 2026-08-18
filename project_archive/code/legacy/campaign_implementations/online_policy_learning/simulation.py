"""SUMO zRAMP variant with local policy sharing.

The base SUMO simulator provides decentralized predictor pulls and delayed
policy rewards. This subclass defines the current six-feature contact policy
surface and the deterministic policy-transfer gate used by the bootstrap
experiments.
"""

from __future__ import annotations

import csv
import math
import random
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rl_reward_experiment.sim as rre_sim  # noqa: E402
from rl_reward_experiment.rl_agent import DQNAgent, QNet, ReplayBuffer  # noqa: E402
from SUMO.sumo_rl import SumoT2Simulation  # noqa: E402

from .metadata import PROBE_FREE14_OBSERVATION_FEATURES


POLICY_STATE_FEATURES = (
    "relative_recent_error",
    "receiver_error_unavailable",
    "relative_model_experience",
    "normalized_model_signature_distance",
    "receiver_model_age",
    "local_contact_availability",
)
COMPACT4_POLICY_STATE_FEATURES = (
    "relative_model_experience",
    "normalized_model_signature_distance",
    "receiver_model_age",
    "local_contact_availability",
)
POLICY_FEATURE_SETS = {
    "current6": POLICY_STATE_FEATURES,
    "compact4": COMPACT4_POLICY_STATE_FEATURES,
    "probe_free14": PROBE_FREE14_OBSERVATION_FEATURES,
}
POLICY_STATE_DIM = len(POLICY_STATE_FEATURES)


def _make_policy_agent(
    *,
    state_dim: int,
    device: torch.device,
    gamma: float,
    lr: float,
    batch_size: int,
    tau: float,
    capacity: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    rng_seed: int,
    action_policy: str,
) -> DQNAgent:
    """Create a DQNAgent with the configured contact-policy surface."""
    state_dim = int(state_dim)
    agent = DQNAgent(
        device=device,
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        tau=tau,
        capacity=capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        rng_seed=rng_seed,
        action_policy=action_policy,
    )
    torch_seed = int(rng_seed) & 0x7FFFFFFFFFFFFFFF
    fork_devices: list[int] = []
    if getattr(device, "type", "") == "cuda":
        fork_devices = [torch.cuda.current_device() if device.index is None else int(device.index)]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(torch_seed)
        if getattr(device, "type", "") == "cuda":
            torch.cuda.manual_seed_all(torch_seed)
        agent.policy = QNet(state_dim=state_dim).to(device)
        final = agent.policy.net[-1]
        if isinstance(final, torch.nn.Linear):
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)
        agent.target = QNet(state_dim=state_dim).to(device)
    agent.target.load_state_dict(agent.policy.state_dict())
    agent.target.eval()
    agent.opt = torch.optim.Adam(agent.policy.parameters(), lr=lr)
    agent.loss_fn = torch.nn.MSELoss()
    agent.replay = ReplayBuffer(capacity=capacity, state_dim=state_dim)
    return agent


class BootstrapPolicySharingSimulation(SumoT2Simulation):
    """Same simulator as SUMO zRAMP, with the current local pull policy."""

    policy_transfer_rule = "accepted_predictor_pull_if_provider_policy_more_experienced"

    def __init__(
        self,
        *args,
        policy_state_features: object = None,
        policy_temperature: float = 1.0,
        **kwargs,
    ) -> None:
        temperature = float(policy_temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("policy_temperature must be finite and positive")
        self.policy_temperature = temperature
        self.policy_state_features = self._resolve_policy_state_features(policy_state_features)
        self.policy_state_dim = len(self.policy_state_features)
        self._feature_log_path: Path | None = None
        self._feature_log_file = None
        self._feature_log_writer: csv.DictWriter | None = None
        self._feature_log_count = 0
        self._last_local_policy_pull_updates_by_mode: Counter[str] = Counter()
        super().__init__(*args, **kwargs)
        self.local_policy_initial_pull = "dqn-softmax-from-first-contact"
        self.local_policy_initial_pull_probability = 0.5
        self._feature_log_path = Path(self.cfg.results_dir) / "feature_transitions.csv"

    @staticmethod
    def _resolve_policy_state_features(raw: object = None) -> tuple[str, ...]:
        if raw is None:
            return tuple(POLICY_STATE_FEATURES)
        if isinstance(raw, str):
            if raw not in POLICY_FEATURE_SETS:
                allowed = ", ".join(sorted(POLICY_FEATURE_SETS))
                raise ValueError(f"Unknown policy feature set {raw!r}; expected one of: {allowed}")
            return tuple(POLICY_FEATURE_SETS[raw])
        try:
            features = tuple(str(name) for name in raw)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("policy_state_features must be None, a feature-set name, or an iterable") from exc
        if not features:
            raise ValueError("policy_state_features must contain at least one feature")
        known = {
            feature
            for features in POLICY_FEATURE_SETS.values()
            for feature in features
        }
        unknown = [name for name in features if name not in known]
        if unknown:
            raise ValueError(f"Unsupported policy feature(s): {unknown}")
        return features

    def _init_local_policy_agents(self) -> None:
        self.local_agents.clear()
        self._local_policy_pending_transitions.clear()
        self._local_policy_versions.clear()
        self._local_policy_initial_rngs.clear()
        for mode_id, template_agent in list(self.agents.items()):
            mode_offset = int(zlib.crc32(str(mode_id).encode("utf-8")) % 10_000_000)
            feature_offset = int(
                zlib.crc32(",".join(self.policy_state_features).encode("utf-8")) % 10_000_000
            )
            template_seed = int(self.cfg.seed) + 501_173 + mode_offset + feature_offset
            action_policy = str(template_agent.action_policy)
            template_agent = _make_policy_agent(
                state_dim=self.policy_state_dim,
                device=self.device,
                gamma=self.cfg.gamma,
                lr=self.cfg.rl_lr,
                batch_size=self.cfg.rl_batch_size,
                tau=self.cfg.rl_target_tau,
                capacity=self.cfg.replay_capacity,
                epsilon_start=self.cfg.epsilon_start,
                epsilon_end=self.cfg.epsilon_end,
                epsilon_decay_steps=self.cfg.epsilon_decay_steps,
                rng_seed=template_seed,
                action_policy=action_policy,
            )
            template_agent.softmax_temperature = self.policy_temperature
            self.agents[mode_id] = template_agent
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
                seed = (
                    int(self.cfg.seed)
                    + 1_900_003
                    + mode_offset
                    + feature_offset
                    + 104_729 * int(node_idx)
                )
                agent = _make_policy_agent(
                    state_dim=self.policy_state_dim,
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
                    action_policy=action_policy,
                )
                agent.softmax_temperature = self.policy_temperature
                agent.policy.load_state_dict(policy_state)
                agent.target.load_state_dict(target_state)
                agents.append(agent)
                rngs.append(random.Random(seed + 57_911))
            self.local_agents[mode_id] = agents
            self._local_policy_pending_transitions[mode_id] = [0 for _ in agents]
            self._local_policy_versions[mode_id] = [0 for _ in agents]
            self._local_policy_initial_rngs[mode_id] = rngs

    def _state_features(
        self,
        mode: str,
        ns_i,
        ns_j,
        az: int,
        neighbor_count: int,
        j_view: Optional[object] = None,
    ) -> torch.Tensor:
        del az
        eps = 1e-8
        v_i = ns_i.variants[mode]
        i_rmse = float(v_i.last_rmse)
        i_rmse_available = bool(getattr(v_i, "last_rmse_available", False))
        e_i = float(v_i.experience)
        sig_i = self._variant_signature(v_i)

        if j_view is None:
            v_j = ns_j.variants[mode]
            j_rmse = float(v_j.last_rmse)
            j_rmse_available = bool(getattr(v_j, "last_rmse_available", False))
            e_j = float(v_j.experience)
            sig_j = self._variant_signature(v_j)
        else:
            j_rmse = float(j_view.last_rmse)
            j_rmse_available = bool(getattr(j_view, "last_rmse_available", False))
            e_j = float(j_view.experience)
            sig_j = j_view.model_signature.detach().to(dtype=torch.float32, device="cpu")

        rmse_sum = i_rmse + j_rmse
        if i_rmse_available and j_rmse_available and rmse_sum > eps:
            rel_recent_error = (i_rmse - j_rmse) / rmse_sum
        else:
            rel_recent_error = 0.0
        receiver_error_unavailable = 0.0 if i_rmse_available else 1.0

        exp_sum = e_i + e_j
        rel_model_experience = (e_j - e_i) / exp_sum if exp_sum > eps else 0.0

        sig_den = float(torch.norm(sig_i).item() + torch.norm(sig_j).item())
        model_dissimilarity = (
            rre_sim._signature_diff_norm(sig_i, sig_j) / sig_den
            if sig_den > eps
            else 0.0
        )

        age_steps = max(0.0, float(v_i.t_wait))
        receiver_model_age = age_steps / (1.0 + age_steps)
        contact_count = max(0.0, float(neighbor_count))
        local_contact_availability = contact_count / (1.0 + contact_count)

        feature_values = {
            "relative_recent_error": rel_recent_error,
            "receiver_error_unavailable": receiver_error_unavailable,
            "relative_model_experience": rel_model_experience,
            "normalized_model_signature_distance": model_dissimilarity,
            "receiver_model_age": receiver_model_age,
            "local_contact_availability": local_contact_availability,
        }
        return torch.tensor(
            [feature_values[name] for name in self.policy_state_features],
            dtype=torch.float32,
        )

    def _policy_experience(self, mode_id: str, node_idx: int) -> int:
        agents = self.local_agents.get(mode_id)
        if agents is None or not (0 <= int(node_idx) < len(agents)):
            return 0
        return int(len(agents[int(node_idx)].replay))

    def _policy_replay_capacity(self, mode_id: str, node_idx: int) -> int:
        agents = self.local_agents.get(mode_id)
        if agents is None or not (0 <= int(node_idx) < len(agents)):
            return 0
        return int(agents[int(node_idx)].replay.capacity)

    def _policy_replay_full(self, mode_id: str, node_idx: int) -> bool:
        cap = self._policy_replay_capacity(mode_id, node_idx)
        return cap <= 0 or self._policy_experience(mode_id, node_idx) >= cap

    def _resolve_local_initial_pull_probability(self) -> float:
        return 0.5

    def _select_action_from_local_agent(self, mode_id: str, node_idx: int, state: torch.Tensor) -> int:
        agents = self.local_agents.get(mode_id)
        if agents is None or not (0 <= int(node_idx) < len(agents)):
            return super()._select_action(mode_id, state, node_idx=node_idx)
        return agents[int(node_idx)].select_action(state)

    def _ensure_feature_log_stream(self) -> csv.DictWriter:
        if self._feature_log_writer is not None:
            return self._feature_log_writer
        if self._feature_log_path is None:
            self._feature_log_path = Path(self.cfg.results_dir) / "feature_transitions.csv"
        self._feature_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._feature_log_file = open(
            self._feature_log_path,
            "w",
            newline="",
            encoding="utf-8",
        )
        fields = [
            "mode",
            "node_idx",
            "action",
            "reward",
            "done",
            *self.policy_state_features,
        ]
        self._feature_log_writer = csv.DictWriter(self._feature_log_file, fieldnames=fields)
        self._feature_log_writer.writeheader()
        return self._feature_log_writer

    def _flush_feature_log_stream(self) -> None:
        if self._feature_log_file is not None:
            self._feature_log_file.flush()

    def _close_feature_log_stream(self) -> None:
        if self._feature_log_file is not None:
            self._feature_log_file.flush()
            self._feature_log_file.close()
        self._feature_log_file = None
        self._feature_log_writer = None

    def _record_feature_transition(self, mode_id: str, transition) -> None:
        if len(transition) < 6:
            return
        state, action, reward, _next_state, done, node_idx = transition
        values = state.detach().to(device="cpu", dtype=torch.float32).reshape(-1).tolist()
        if len(values) != len(self.policy_state_features):
            return
        row = {
            "mode": str(mode_id),
            "node_idx": int(node_idx),
            "action": int(action),
            "reward": float(reward),
            "done": bool(done),
            **{
                name: float(value)
                for name, value in zip(self.policy_state_features, values)
            },
        }
        self._ensure_feature_log_stream().writerow(row)
        self._feature_log_count += 1
        if self._feature_log_count % 50000 == 0:
            self._flush_feature_log_stream()

    def _queue_rl_transition(self, mode_id: str, transition) -> None:
        self._record_feature_transition(mode_id, transition)
        super()._queue_rl_transition(mode_id, transition)

    def _write_partial_outputs(self, *args, **kwargs) -> None:
        super()._write_partial_outputs(*args, **kwargs)
        self._flush_feature_log_stream()

    def run(self) -> None:
        try:
            super().run()
        finally:
            self._close_feature_log_stream()

    def _reset_policy_step_counters(self) -> None:
        super()._reset_policy_step_counters()
        self._last_local_policy_pull_updates_by_mode.clear()

    def _make_peer_view(self, ns_j, mode: str):
        view = rre_sim.Simulation._make_peer_view(self, ns_j, mode)
        if self.zramp_policy_mode == "local" and mode in self.local_agents:
            node_idx = self.node_idx(ns_j)
            if 0 <= node_idx < len(self.local_agents[mode]):
                agent = self.local_agents[mode][node_idx]
                view._policy_state = self._clone_model_state(agent.policy)  # type: ignore[attr-defined]
                view._policy_experience = self._policy_experience(mode, node_idx)  # type: ignore[attr-defined]
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
        if float(exp_provider) <= float(exp_puller):
            return
        agents = self.local_agents.get(mode_id)
        if agents is None or not (0 <= int(puller_idx) < len(agents)):
            return
        if not provider_policy_state:
            return

        puller = agents[int(puller_idx)]
        rre_sim.weighted_state_pull(
            puller.policy,
            float(exp_puller),
            provider_policy_state,
            float(exp_provider),
            merge_strategy="average",
        )
        if int(provider_version) > int(self._local_policy_versions[mode_id][int(puller_idx)]):
            self._local_policy_versions[mode_id][int(puller_idx)] = int(provider_version)
        self._local_policy_pull_updates[mode_id] += 1
        self._last_local_policy_pull_updates += 1
        self._last_local_policy_pull_updates_by_mode[mode_id] += 1

    def perform_merge(self, ns_i, ns_j, mode: str, j_view: Optional[object] = None) -> None:
        """Pull predictor weights, then maybe pull more-experienced policy weights."""
        if mode not in self.local_agents:
            rre_sim.Simulation.perform_merge(self, ns_i, ns_j, mode, j_view=j_view)
            return

        i_idx = self.node_idx(ns_i)
        j_idx = self.node_idx(ns_j)
        exp_puller = float(self._policy_experience(mode, i_idx))

        if j_view is None:
            if 0 <= j_idx < len(self.local_agents[mode]):
                provider_agent = self.local_agents[mode][j_idx]
                provider_policy_state = {
                    k: v.detach()
                    for k, v in provider_agent.policy.state_dict().items()
                }
                exp_provider = float(self._policy_experience(mode, j_idx))
                provider_version = int(self._local_policy_versions[mode][j_idx])
            else:
                provider_policy_state = {}
                exp_provider = 0.0
                provider_version = 0
        else:
            provider_policy_state = getattr(j_view, "_policy_state", {})
            exp_provider = float(getattr(j_view, "_policy_experience", 0.0))
            provider_version = int(getattr(j_view, "_policy_version", 0))

        rre_sim.Simulation.perform_merge(self, ns_i, ns_j, mode, j_view=j_view)
        if i_idx >= 0:
            self._merge_local_policy_from_state(
                mode_id=mode,
                puller_idx=i_idx,
                exp_puller=exp_puller,
                provider_policy_state=provider_policy_state,
                exp_provider=exp_provider,
                provider_version=provider_version,
            )

    def _build_communication_assumptions(self) -> dict[str, int | float | str | bool]:
        ass = super()._build_communication_assumptions()
        scalar_meta = int(rre_sim.DECISION_METADATA_SCALAR_FLOATS) * 4 + 1
        signature_meta = int(rre_sim.MODEL_SIGNATURE_FLOATS) * 4
        accepted_pull = int(ass.get("B_accepted_pull_bytes", 0))
        policy_pull = int(ass.get("B_local_policy_pull_bytes", 0))
        ass["policy_observation_dim"] = int(self.policy_state_dim)
        ass["policy_observation_features"] = ",".join(self.policy_state_features)
        ass["B_decision_meta_bytes_per_directed_decision"] = int(scalar_meta + signature_meta)
        ass["B_decision_scalar_meta_bytes_per_directed_decision"] = int(scalar_meta)
        ass["B_model_signature_bytes_per_directed_decision"] = int(signature_meta)
        ass["B_local_zramp_accepted_pull_bytes"] = int(accepted_pull)
        ass["B_bootstrap_policy_pull_bytes"] = int(policy_pull)
        ass["policy_transfer_rule"] = self.policy_transfer_rule
        ass["policy_experience"] = "matured local policy replay-buffer transitions"
        ass["policy_action_selection"] = (
            "temperature_softmax_from_first_contact_neutral_q_initialization"
        )
        ass["policy_softmax_temperature"] = float(self.policy_temperature)
        ass["prediction_target"] = "propagation_loss_db"
        ass["metadata_note"] = (
            "Z-RAMP pays compact provider metadata for every feasible directed "
            "decision: provider predictor experience, provider recent-error "
            "availability/RMSE, and compact predictor signature. Accepted pulls "
            "transfer propagation-loss predictor weights and merge metadata. "
            "Policy weights are transferred only when the provider policy has "
            "more matured replay transitions than the receiver policy."
        )
        return ass

    def _communication_overhead_row(
        self,
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
            policy_pulls = int(self._last_local_policy_pull_updates_by_mode.get(mode_id, 0))
            mode_bytes = (
                int(feasible_decisions) * decision_meta
                + accepted * accepted_pull
                + policy_pulls * local_policy_pull
            )
            self._comm_cumulative_bytes[mode_id] += mode_bytes
            greedy_cum = float(self._comm_cumulative_bytes["greedy"])
            row[f"{mode_id}_policy_pull_events"] = int(policy_pulls)
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
