"""Corrected policy-distribution variants for the 150 m role sweep."""

from __future__ import annotations

from dataclasses import replace

import torch

from online_policy_learning.online_local_validation_policy import (
    ExactSequentialBidirectionalSimulation,
    SampleSharingExactSequentialSimulation,
    _TrainingExample,
)


class _ClippedTargetMixin:
    """Use a bounded target while preserving the provider ordering."""

    policy_target_clip: float

    def _bounded_example(self, example: _TrainingExample) -> _TrainingExample:
        if getattr(self, "policy_training_target", "validation-gain") == (
            "information-gain"
        ):
            return example
        limit = float(self.policy_target_clip)
        target = max(-limit, min(limit, float(example.target_gain)))
        return replace(example, target_gain=target)

    def _train_exact_pair(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        example: _TrainingExample,
        sample_multiplier: int,
    ) -> None:
        super()._train_exact_pair(
            step=step,
            mode=mode,
            receiver_idx=receiver_idx,
            example=self._bounded_example(example),
            sample_multiplier=sample_multiplier,
        )


class FrozenEncoderSampleSharingSimulation(
    _ClippedTargetMixin,
    SampleSharingExactSequentialSimulation
):
    """Stable seeded embeddings, bounded sample gossip, and private heads."""

    policy_transfer_rule = (
        "frozen-seeded-encoder-recent-balanced-sample-gossip"
    )

    def __init__(
        self,
        *args,
        recent_sample_capacity: int = 256,
        predecision_head_batches: int = 4,
        policy_target_clip: float = 1.0,
        **kwargs,
    ) -> None:
        if bool(kwargs.get("align_policy_encoders", False)):
            raise ValueError(
                "the corrected sample method freezes encoders and does not "
                "exchange encoder parameters"
            )
        self.recent_sample_capacity = int(recent_sample_capacity)
        self.predecision_head_batches = int(predecision_head_batches)
        self.policy_target_clip = float(policy_target_clip)
        if self.recent_sample_capacity <= 0:
            raise ValueError("recent sample capacity must be positive")
        if self.predecision_head_batches <= 0:
            raise ValueError("pre-decision head batches must be positive")
        if self.policy_target_clip <= 0.0:
            raise ValueError("policy target clip must be positive")
        super().__init__(*args, **kwargs)
        self._configure_agents()
        self.zramp_policy_mode = (
            "private-head-frozen-seeded-encoder-balanced-sample-gossip"
        )
        self._communication_assumptions = (
            self._build_communication_assumptions()
        )

    def _configure_agent(self, agent) -> None:
        agent.configure_hybrid_samples(self.recent_sample_capacity)
        agent.freeze_encoder()

    def _configure_agents(self) -> None:
        for agents in self.local_agents.values():
            for agent in agents:
                self._configure_agent(agent)

    def _reset_respawned_node(
        self, node_idx: int, *, generation: int | None = None
    ) -> None:
        super()._reset_respawned_node(node_idx, generation=generation)
        for agents in self.local_agents.values():
            self._configure_agent(agents[int(node_idx)])

    def _training_sample_wire_bytes(self) -> int:
        # Two quantized embeddings, gain, sample id, and uint32 event step.
        return super()._training_sample_wire_bytes() + 4

    def _ordered_snapshot_samples(
        self,
        samples: dict[
            object, tuple[torch.Tensor, torch.Tensor, float]
        ],
        steps: dict[object, int],
        limit: int,
    ) -> list[
        tuple[
            object,
            tuple[torch.Tensor, torch.Tensor, float],
            int,
        ]
    ]:
        """Put recent samples first, then balance positive and non-positive."""

        count = max(0, int(limit))
        if count == 0:
            return []
        priority = self.local_agents[next(iter(self.local_agents))][
            0
        ].sample_priority
        newest = sorted(
            samples,
            key=lambda key: (
                -int(steps.get(key, 0)),
                priority(key),
                repr(key),
            ),
        )
        selected = newest[: min(len(newest), count // 2)]
        selected_set = set(selected)
        remaining = count - len(selected)
        nonpositive_slots = remaining // 2
        positive_slots = remaining - nonpositive_slots
        positive = [
            key
            for key in newest
            if key not in selected_set and float(samples[key][2]) > 0.0
        ][:positive_slots]
        selected.extend(positive)
        selected_set.update(positive)
        nonpositive = [
            key
            for key in newest
            if key not in selected_set and float(samples[key][2]) <= 0.0
        ][:nonpositive_slots]
        selected.extend(nonpositive)
        selected_set.update(nonpositive)
        selected.extend(
            key for key in newest if key not in selected_set
        )
        return [
            (key, samples[key], int(steps.get(key, 0)))
            for key in selected
        ]

    def _train_exact_pair(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        example: _TrainingExample,
        sample_multiplier: int,
    ) -> None:
        bounded = _ClippedTargetMixin._bounded_example(self, example)
        receiver_agent = self.local_agents[mode][int(receiver_idx)]
        receiver_agent.remember_shared_sample(
            bounded.sample_id,
            bounded.receiver_embedding,
            bounded.provider_embedding,
            float(bounded.target_gain),
            sample_step=int(step),
        )
        # The shared embedding basis is fixed. Only the private gain head is
        # trainable, while raw state and trajectories remain local.
        ExactSequentialBidirectionalSimulation._train_exact_pair(
            self,
            step=step,
            mode=mode,
            receiver_idx=receiver_idx,
            example=bounded,
            sample_multiplier=sample_multiplier,
        )

    def _train_heads_before_decision(
        self, zone_nodes: dict[int, list[int]]
    ) -> None:
        active = sorted(
            {
                int(node_idx)
                for indices in zone_nodes.values()
                for node_idx in indices
            }
        )
        for agents in self.local_agents.values():
            for node_idx in active:
                agents[node_idx].train_head_batches(
                    num_batches=self.predecision_head_batches
                )

    def _gossip_step(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> None:
        links = self._normalized_contact_links(zone_nodes, contact_links)
        sample_network_stats = (
            self._share_samples_with_all_feasible_neighbors(links)
        )
        # A newcomer learns from the received bundle before it ranks the
        # providers in this same contact opportunity.
        self._train_heads_before_decision(zone_nodes)
        ExactSequentialBidirectionalSimulation._gossip_step(
            self, step, zone_nodes, contact_links=links
        )
        if sample_network_stats:
            self._network_step_stats.update(sample_network_stats)

    def _build_communication_assumptions(
        self,
    ) -> dict[str, int | float | str | bool]:
        assumptions = super()._build_communication_assumptions()
        assumptions.update(
            {
                "zramp_policy_mode": (
                    "private-head-frozen-seeded-encoder-balanced-sample-gossip"
                ),
                "policy_transfer_rule": self.policy_transfer_rule,
                "policy_encoder_frozen": True,
                "policy_encoder_alignment_enabled": False,
                "policy_encoder_alignment_payload_bytes_per_direction": 0,
                "encoder_consensus_bytes_per_direction": 0,
                "policy_recent_sample_capacity": int(
                    self.recent_sample_capacity
                ),
                "policy_historical_sample_capacity": int(
                    self.policy_sample_capacity
                    - self.recent_sample_capacity
                ),
                "policy_predecision_head_batches": int(
                    self.predecision_head_batches
                ),
                "policy_target_clip": float(self.policy_target_clip),
                "B_training_sample_bytes": int(
                    self._training_sample_wire_bytes()
                ),
                "training_sample_payload": (
                    "two-signed-int8-embeddings-with-scales-plus-clipped-"
                    "float32-gain-16-byte-id-and-uint32-event-step"
                ),
                "metadata_note": (
                    "All vehicles use the same permanently frozen seeded "
                    "encoder. No encoder or gain-head parameters are sent. "
                    "Each private head trains on a 256-recent/256-historical "
                    "reservoir and fits received balanced bundles before "
                    "making a decision."
                ),
            }
        )
        return assumptions


class AllNeighborPolicySharingSimulation(
    _ClippedTargetMixin,
    ExactSequentialBidirectionalSimulation,
):
    """Experience-average the complete policy with every current neighbor."""


    policy_transfer_rule = (
        "synchronous-experience-weighted-full-policy-all-neighbors"
    )

    def __init__(
        self,
        *args,
        policy_target_clip: float = 1.0,
        **kwargs,
    ) -> None:
        self.policy_target_clip = float(policy_target_clip)
        self._suppress_policy_gossip = False
        kwargs["share_policy_every_contact"] = True
        kwargs["align_policy_encoders"] = False
        super().__init__(*args, **kwargs)
        self.zramp_policy_mode = (
            "experience-weighted-full-policy-all-neighbors"
        )
        self._communication_assumptions = (
            self._build_communication_assumptions()
        )

    def _share_policies_with_all_feasible_neighbors(
        self, links: list[tuple[int, int, int]]
    ) -> None:
        if self._suppress_policy_gossip:
            return
        super()._share_policies_with_all_feasible_neighbors(links)

    def _gossip_step(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> None:
        links = self._normalized_contact_links(zone_nodes, contact_links)
        self._share_policies_with_all_feasible_neighbors(links)
        # The all-neighbor exchange above is the only policy exchange. Avoid
        # repeating it on only the subsequently selected model-transfer links.
        old_share = self.share_policy_every_contact
        old_local = self.local_policy_share
        self._suppress_policy_gossip = True
        self.share_policy_every_contact = False
        self.local_policy_share = False
        try:
            ExactSequentialBidirectionalSimulation._gossip_step(
                self, step, zone_nodes, contact_links=links
            )
        finally:
            self.share_policy_every_contact = old_share
            self.local_policy_share = old_local
            self._suppress_policy_gossip = False

    def _build_communication_assumptions(
        self,
    ) -> dict[str, int | float | str | bool]:
        assumptions = super()._build_communication_assumptions()
        assumptions.update(
            {
                "zramp_policy_mode": (
                    "experience-weighted-full-policy-all-neighbors"
                ),
                "policy_transfer_rule": self.policy_transfer_rule,
                "policy_target_clip": float(self.policy_target_clip),
                "policy_neighbor_exchange": (
                    "synchronous snapshot-based aggregation over every "
                    "current physical contact before provider selection"
                ),
            }
        )
        return assumptions
