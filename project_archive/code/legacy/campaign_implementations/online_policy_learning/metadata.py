"""Fixed-size model metadata and utility-policy observation schemas."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


CURRENT_OBSERVATION_FEATURES = (
    "relative_recent_error",
    "receiver_error_unavailable",
    "relative_model_experience",
    "normalized_model_signature_distance",
    "receiver_model_age",
    "local_contact_availability",
)
STUDY10_OBSERVATION_FEATURES = (
    "relative_recent_error",
    "receiver_error_unavailable",
    "provider_error_unavailable",
    "relative_model_experience",
    "normalized_model_signature_distance",
    "representation_cka_dissimilarity",
    "steps_since_last_model_aggregation",
    "local_contact_availability",
    "node_density",
    "zone_buffer_maturity",
)
STUDY_OBSERVATION_FEATURES = (
    "relative_recent_error",
    "receiver_error_unavailable",
    "provider_error_unavailable",
    "relative_model_experience",
    "relative_provider_freshness",
    "normalized_model_signature_distance",
    "representation_cka_dissimilarity",
    "normalized_prediction_disagreement",
    "steps_since_last_model_aggregation",
    "provider_pull_novelty",
    "local_contact_availability",
    "node_density",
    "zone_buffer_maturity",
)
BEST5_OBSERVATION_FEATURES = (
    "zone_buffer_maturity",
    "receiver_error_unavailable",
    "relative_model_experience",
    "normalized_prediction_disagreement",
    "provider_error_unavailable",
)
PROBE_FREE14_OBSERVATION_FEATURES = (
    "zone_buffer_maturity",
    "relative_model_experience",
    "normalized_model_signature_distance",
    "relative_recent_error",
    "receiver_error_unavailable",
    "provider_error_unavailable",
    "relative_opt_validation_quality",
    "total_opt_validation_quality",
    "relative_reward_validation_quality",
    "total_reward_validation_quality",
    "steps_since_last_model_aggregation",
    "receiver_aggregations_this_step",
    "relative_provider_freshness",
    "provider_pull_novelty",
)
OBSERVATION_FEATURES = CURRENT_OBSERVATION_FEATURES
FEATURE_SETS = {
    "current6": CURRENT_OBSERVATION_FEATURES,
    "study10": STUDY10_OBSERVATION_FEATURES,
    "study13": STUDY_OBSERVATION_FEATURES,
    "best5": BEST5_OBSERVATION_FEATURES,
    "probe_free14": PROBE_FREE14_OBSERVATION_FEATURES,
}

CKA_PROBE_COUNT = 16
CKA_SIGNATURE_FLOATS = CKA_PROBE_COUNT * (CKA_PROBE_COUNT + 1) // 2


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable pre-pull metadata for one active zone predictor."""

    experience: float
    raw_experience: int
    recent_error: float
    recent_error_available: bool
    model_age: int
    signature: torch.Tensor
    representation_signature: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float32)
    )
    prediction_signature: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float32)
    )
    model_age_shared: bool = False
    validation_opt_quality: float = 0.0
    validation_reward_quality: float = 0.0
    validation_quality_shared: bool = False

    @property
    def wire_nbytes(self) -> int:
        # Two float32 scalars, one availability byte, and fixed float32 signatures.
        return (
            2 * 4
            + 1
            + int(self.signature.numel()) * 4
            + int(self.representation_signature.numel()) * 4
            + int(self.prediction_signature.numel()) * 4
            + (4 if self.model_age_shared else 0)
            + (2 * 4 if self.validation_quality_shared else 0)
        )


def build_cka_signature(model: nn.Module, probes: torch.Tensor) -> torch.Tensor:
    """Return a normalized upper-triangular Gram signature for linear CKA."""

    linear_layers = [
        module for module in model.modules() if isinstance(module, nn.Linear)
    ]
    if not linear_layers:
        raise ValueError("CKA signature requires a predictor with a Linear layer")
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        captured.append(inputs[0].detach())

    final_linear = linear_layers[-1]
    handle = final_linear.register_forward_pre_hook(capture)
    was_training = bool(model.training)
    model.eval()
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            model(probes.to(device=device, dtype=torch.float32))
    finally:
        handle.remove()
        model.train(was_training)
    if not captured:
        raise RuntimeError("failed to capture penultimate predictor activations")

    activations = captured[-1].to(device="cpu", dtype=torch.float32)
    activations = activations - activations.mean(dim=0, keepdim=True)
    gram = activations @ activations.T
    norm = torch.linalg.vector_norm(gram)
    if float(norm) > 1.0e-12:
        gram = gram / norm
    else:
        gram = torch.zeros_like(gram)
    rows, cols = torch.triu_indices(gram.shape[0], gram.shape[1])
    values = gram[rows, cols].clone()
    # Weight off-diagonal entries so a vector dot product equals the full
    # symmetric Gram-matrix Frobenius inner product used by linear CKA.
    values[rows != cols] *= math.sqrt(2.0)
    return values


def build_prediction_signature(
    model: nn.Module, probes: torch.Tensor
) -> torch.Tensor:
    """Return predictor outputs on deterministic public same-zone probes."""

    was_training = bool(model.training)
    model.eval()
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            predictions = model(
                probes.to(device=device, dtype=torch.float32)
            )
    finally:
        model.train(was_training)
    return predictions.detach().to(device="cpu", dtype=torch.float32).reshape(-1)


def build_metadata(
    variant,
    *,
    representation_signature: torch.Tensor | None = None,
    prediction_signature: torch.Tensor | None = None,
    share_model_age: bool = False,
    validation_opt_quality: float = 0.0,
    validation_reward_quality: float = 0.0,
    share_validation_quality: bool = False,
) -> ModelMetadata:
    """Snapshot the compact metadata used by provider observations."""

    signature = variant.model_signature.detach().to(
        device="cpu", dtype=torch.float32
    ).reshape(-1).clone()
    return ModelMetadata(
        experience=float(variant.experience),
        raw_experience=max(0, int(variant.m_samples)),
        recent_error=float(variant.last_rmse),
        recent_error_available=bool(variant.last_rmse_available),
        model_age=max(0, int(variant.t_wait)),
        signature=signature,
        representation_signature=(
            torch.empty(0, dtype=torch.float32)
            if representation_signature is None
            else representation_signature.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1)
            .clone()
        ),
        prediction_signature=(
            torch.empty(0, dtype=torch.float32)
            if prediction_signature is None
            else prediction_signature.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1)
            .clone()
        ),
        model_age_shared=bool(share_model_age),
        validation_opt_quality=max(0.0, float(validation_opt_quality)),
        validation_reward_quality=max(0.0, float(validation_reward_quality)),
        validation_quality_shared=bool(share_validation_quality),
    )


def build_observation(
    receiver: ModelMetadata,
    provider: ModelMetadata,
    *,
    neighbor_count: int,
    zone_neighbor_count: int | None = None,
    zone_buffer_samples: int = 0,
    steps_since_provider_pull: int | None = None,
    receiver_aggregations_this_step: int = 0,
    feature_names: tuple[str, ...] = OBSERVATION_FEATURES,
) -> torch.Tensor:
    """Construct one selected bootstrap observation schema."""

    eps = 1.0e-8
    error_sum = float(receiver.recent_error + provider.recent_error)
    if (
        receiver.recent_error_available
        and provider.recent_error_available
        and error_sum > eps
    ):
        relative_recent_error = (
            float(receiver.recent_error) - float(provider.recent_error)
        ) / error_sum
    else:
        relative_recent_error = 0.0

    receiver_error_unavailable = (
        0.0 if receiver.recent_error_available else 1.0
    )
    provider_error_unavailable = (
        0.0 if provider.recent_error_available else 1.0
    )
    experience_sum = float(receiver.experience + provider.experience)
    relative_model_experience = (
        (float(provider.experience) - float(receiver.experience)) / experience_sum
        if experience_sum > eps
        else 0.0
    )

    receiver_age_steps = float(max(0, receiver.model_age))
    provider_age_steps = float(max(0, provider.model_age))
    age_sum = receiver_age_steps + provider_age_steps
    relative_provider_freshness = (
        (receiver_age_steps - provider_age_steps) / age_sum
        if age_sum > eps
        else 0.0
    )

    receiver_signature = receiver.signature.to(dtype=torch.float32)
    provider_signature = provider.signature.to(dtype=torch.float32)
    if receiver_signature.numel() != provider_signature.numel():
        size = min(receiver_signature.numel(), provider_signature.numel())
        receiver_signature = receiver_signature[:size]
        provider_signature = provider_signature[:size]
    signature_denominator = float(
        torch.norm(receiver_signature).item()
        + torch.norm(provider_signature).item()
    )
    model_dissimilarity = (
        float(torch.norm(receiver_signature - provider_signature).item())
        / signature_denominator
        if signature_denominator > eps
        else 0.0
    )

    receiver_representation = receiver.representation_signature.to(dtype=torch.float32)
    provider_representation = provider.representation_signature.to(dtype=torch.float32)
    representation_cka_dissimilarity = 0.0
    if receiver_representation.numel() and provider_representation.numel():
        size = min(receiver_representation.numel(), provider_representation.numel())
        similarity = float(
            torch.dot(
                receiver_representation[:size], provider_representation[:size]
            ).item()
        )
        representation_cka_dissimilarity = 1.0 - max(
            0.0, min(1.0, similarity)
        )


    receiver_prediction = receiver.prediction_signature.to(dtype=torch.float32)
    provider_prediction = provider.prediction_signature.to(dtype=torch.float32)
    if receiver_prediction.numel() != provider_prediction.numel():
        size = min(receiver_prediction.numel(), provider_prediction.numel())
        receiver_prediction = receiver_prediction[:size]
        provider_prediction = provider_prediction[:size]
    prediction_denominator = float(
        torch.norm(receiver_prediction).item()
        + torch.norm(provider_prediction).item()
    )
    normalized_prediction_disagreement = (
        float(torch.norm(receiver_prediction - provider_prediction).item())
        / prediction_denominator
        if prediction_denominator > eps
        else 0.0
    )
    age = float(max(0, receiver.model_age))
    receiver_model_age = age / (1.0 + age)
    contacts = float(max(0, int(neighbor_count)))
    local_contact_availability = contacts / (1.0 + contacts)
    zone_neighbors = float(
        max(
            0,
            int(
                neighbor_count
                if zone_neighbor_count is None
                else zone_neighbor_count
            ),
        )
    )
    node_density = zone_neighbors / (1.0 + zone_neighbors)
    buffered = float(max(0, int(zone_buffer_samples)))
    zone_buffer_maturity = buffered / (1.0 + buffered)

    if steps_since_provider_pull is None:
        provider_pull_novelty = 1.0
    else:
        pull_age = float(max(0, int(steps_since_provider_pull)))
        provider_pull_novelty = pull_age / (1.0 + pull_age)

    def relative_quality(receiver_quality: float, provider_quality: float) -> float:
        receiver_value = max(0.0, float(receiver_quality))
        provider_value = max(0.0, float(provider_quality))
        total = receiver_value + provider_value
        return (
            (provider_value - receiver_value) / total
            if total > eps
            else 0.0
        )

    def normalized_total_quality(receiver_quality: float, provider_quality: float) -> float:
        # Validation quality is unbounded (quantity times diversity). Log
        # compression preserves resolution after the first few samples.
        total = max(0.0, float(receiver_quality)) + max(
            0.0, float(provider_quality)
        )
        compressed = math.log1p(total)
        return compressed / (1.0 + compressed)

    relative_opt_validation_quality = relative_quality(
        receiver.validation_opt_quality, provider.validation_opt_quality
    )
    total_opt_validation_quality = normalized_total_quality(
        receiver.validation_opt_quality, provider.validation_opt_quality
    )
    relative_reward_validation_quality = relative_quality(
        receiver.validation_reward_quality, provider.validation_reward_quality
    )
    total_reward_validation_quality = normalized_total_quality(
        receiver.validation_reward_quality, provider.validation_reward_quality
    )
    aggregations = float(max(0, int(receiver_aggregations_this_step)))
    normalized_receiver_aggregations = aggregations / (1.0 + aggregations)

    values = {
        "relative_recent_error": relative_recent_error,
        "receiver_error_unavailable": receiver_error_unavailable,
        "provider_error_unavailable": provider_error_unavailable,
        "relative_model_experience": relative_model_experience,
        "normalized_model_signature_distance": model_dissimilarity,
        "representation_cka_dissimilarity": representation_cka_dissimilarity,
        "receiver_model_age": receiver_model_age,
        "relative_provider_freshness": relative_provider_freshness,
        "steps_since_last_model_aggregation": receiver_model_age,
        "local_contact_availability": local_contact_availability,
        "normalized_prediction_disagreement": normalized_prediction_disagreement,
        "node_density": node_density,
        "zone_buffer_maturity": zone_buffer_maturity,
        "provider_pull_novelty": provider_pull_novelty,
        "relative_opt_validation_quality": relative_opt_validation_quality,
        "total_opt_validation_quality": total_opt_validation_quality,
        "relative_reward_validation_quality": relative_reward_validation_quality,
        "total_reward_validation_quality": total_reward_validation_quality,
        "receiver_aggregations_this_step": normalized_receiver_aggregations,
    }
    unknown = [name for name in feature_names if name not in values]
    if unknown:
        raise ValueError(f"unknown observation features: {unknown}")
    return torch.tensor(
        [values[name] for name in feature_names], dtype=torch.float32
    )
