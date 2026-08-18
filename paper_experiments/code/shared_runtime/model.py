"""Propagation-loss predictor architectures and temporal encoding."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableFourierTimeEncoding(nn.Module):
    """Encode non-negative global time with drift and learnable frequencies."""

    def __init__(
        self,
        num_frequencies: int,
        min_period: float,
        max_period: float,
        time_unit: float = 1.0,
    ) -> None:
        super().__init__()
        if int(num_frequencies) < 1:
            raise ValueError("num_frequencies must be positive")
        if float(min_period) <= 0.0 or float(max_period) <= float(min_period):
            raise ValueError("Require 0 < min_period < max_period")
        if float(time_unit) <= 0.0:
            raise ValueError("time_unit must be positive")
        self.num_frequencies = int(num_frequencies)
        self.time_unit = float(time_unit)
        initial_periods = torch.logspace(
            math.log10(float(min_period)),
            math.log10(float(max_period)),
            steps=self.num_frequencies,
        )
        initial_frequencies = 2.0 * math.pi / initial_periods
        self.log_omega = nn.Parameter(torch.log(initial_frequencies))

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.num_frequencies

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        elif t.ndim != 2 or int(t.shape[-1]) != 1:
            raise ValueError("t must have shape [batch] or [batch, 1]")
        u = t / self.time_unit
        omega = torch.exp(self.log_omega).unsqueeze(0)
        phase = u * omega
        trend = torch.log1p(torch.clamp_min(u, 0.0))
        periodic = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return torch.cat([trend, periodic], dim=-1)


class LearnedScalarTimeEncoding(nn.Module):
    """Encode raw scalar time with a bounded learned MLP and no fixed basis.

    Bounding the representation prevents an absolute timestamp from
    overwhelming normalized spatial coordinates when training spans
    substantially different time horizons.
    """

    def __init__(
        self,
        output_dim: int = 16,
        hidden_dim: int = 16,
        time_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if int(output_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("learned time dimensions must be positive")
        if not math.isfinite(float(time_scale)) or float(time_scale) <= 0.0:
            raise ValueError("time_scale must be finite and positive")
        self._output_dim = int(output_dim)
        self.time_scale = float(time_scale)
        self.network = nn.Sequential(
            nn.Linear(1, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self._output_dim),
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        elif t.ndim != 2 or int(t.shape[-1]) != 1:
            raise ValueError("t must have shape [batch] or [batch, 1]")
        return torch.tanh(self.network(t / self.time_scale))


class _CoordinateTimePredictor(nn.Module):
    """MLP that optionally transforms its final raw input as global time."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        include_time: bool = False,
        num_time_frequencies: int = 8,
        min_time_period: float = 2.0,
        max_time_period: float = 1000.0,
        time_unit: float = 1.0,
        time_encoding: str = "fourier",
        learned_time_dim: int = 16,
        learned_time_hidden_dim: int = 16,
        learned_time_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.raw_input_dim = int(input_dim)
        self.include_time = bool(include_time)
        if self.raw_input_dim <= int(self.include_time):
            raise ValueError("input_dim does not leave room for coordinates")
        if self.include_time:
            encoding = str(time_encoding).strip().lower()
            if encoding == "fourier":
                self.time_encoder: nn.Module | None = LearnableFourierTimeEncoding(
                    num_frequencies=num_time_frequencies,
                    min_period=min_time_period,
                    max_period=max_time_period,
                    time_unit=time_unit,
                )
            elif encoding in {"learned", "mlp", "scalar-mlp"}:
                self.time_encoder = LearnedScalarTimeEncoding(
                    output_dim=int(learned_time_dim),
                    hidden_dim=int(learned_time_hidden_dim),
                    time_scale=float(learned_time_scale),
                )
            else:
                raise ValueError(
                    "time_encoding must be 'fourier' or 'learned'"
                )
            network_input_dim = (
                self.raw_input_dim
                - 1
                + int(getattr(self.time_encoder, "output_dim"))
            )
        else:
            self.time_encoder = None
            network_input_dim = self.raw_input_dim
        layers: list[nn.Module] = []
        previous = network_input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, int(width)), nn.ReLU()))
            previous = int(width)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def _network_input(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[-1]) != self.raw_input_dim:
            raise ValueError(
                f"expected predictor input dimension {self.raw_input_dim}, "
                f"got {int(x.shape[-1])}"
            )
        if self.time_encoder is not None:
            coordinates = x[..., :-1]
            global_time = x[..., -1:]
            x = torch.cat(
                (coordinates, self.time_encoder(global_time)), dim=-1
            )
        return x

    def hidden_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final hidden representation before scalar prediction."""

        return self.network[:-1](self._network_input(x))

    def output_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network[-1](hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_from_hidden(self.hidden_features(x))


class RSSIPredictor(_CoordinateTimePredictor):
    """Deeper and wider MLP for single-position radio maps."""

    def __init__(self, input_dim: int = 2, **time_kwargs) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dims=(256, 256, 256, 256),
            **time_kwargs,
        )


class DualRSSIPredictor(_CoordinateTimePredictor):
    """Large MLP for arbitrary transmitter/receiver pairs."""

    def __init__(self, input_dim: int = 4, **time_kwargs) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dims=(512, 512, 512, 512),
            **time_kwargs,
        )


class TinyRSSIPredictor(_CoordinateTimePredictor):
    """Lightweight MLP for fast training."""

    def __init__(self, input_dim: int = 4, **time_kwargs) -> None:
        super().__init__(
            input_dim=input_dim, hidden_dims=(64, 64), **time_kwargs
        )


class LogDistanceRSSIPredictor(nn.Module):
    """Two-parameter learned log-distance propagation-loss model."""

    def __init__(
        self,
        input_dim: int = 4,
        *,
        include_time: bool = False,
        **_unused_time_kwargs,
    ) -> None:
        super().__init__()
        self.raw_input_dim = int(input_dim)
        self.include_time = bool(include_time)
        if self.raw_input_dim not in {4, 5}:
            raise ValueError("log-distance predictor expects 4 or 5 inputs")
        if self.include_time != (self.raw_input_dim == 5):
            raise ValueError("include_time and input_dim are inconsistent")
        self.intercept = nn.Parameter(torch.zeros(1))
        self.slope = nn.Parameter(torch.zeros(1))

    def set_normalized_prior(self, value: float) -> None:
        with torch.no_grad():
            self.intercept.fill_(float(value))
            self.slope.zero_()

    @staticmethod
    def distance_feature(x: torch.Tensor) -> torch.Tensor:
        delta = x[..., :2] - x[..., 2:4]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        return torch.log1p(distance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.intercept + self.slope * self.distance_feature(x)


class LogDistanceResidualRSSIPredictor(nn.Module):
    """Tiny coordinate MLP added to a learned log-distance baseline.

    The baseline and residual are expressed in normalized propagation-loss
    units. ``set_normalized_prior`` initializes the complete predictor to the
    established constant prior while leaving both components trainable.
    """

    def __init__(
        self,
        input_dim: int = 4,
        *,
        include_time: bool = False,
        gate_residual_with_support: bool = False,
        support_vectors: int = 32,
        **time_kwargs,
    ) -> None:
        super().__init__()
        self.raw_input_dim = int(input_dim)
        self.include_time = bool(include_time)
        if self.raw_input_dim not in {4, 5}:
            raise ValueError("distance-residual predictor expects 4 or 5 inputs")
        if self.include_time != (self.raw_input_dim == 5):
            raise ValueError("include_time and input_dim are inconsistent")
        self.residual = TinyRSSIPredictor(
            input_dim=self.raw_input_dim,
            include_time=self.include_time,
            **time_kwargs,
        )
        self.distance_intercept = nn.Parameter(torch.zeros(1))
        self.distance_slope = nn.Parameter(torch.zeros(1))
        self.gate_residual_with_support = bool(gate_residual_with_support)
        if self.gate_residual_with_support:
            if int(support_vectors) < 2:
                raise ValueError("RBF gating needs at least two support vectors")
            self.support_capacity = int(support_vectors)
            self.register_buffer(
                "support_vectors",
                torch.zeros((self.support_capacity, 4), dtype=torch.float32),
            )
            self.register_buffer(
                "support_filled", torch.zeros(1, dtype=torch.float32)
            )
            self.register_buffer(
                "support_seen", torch.zeros(1, dtype=torch.float32)
            )
            self.register_buffer(
                "support_bandwidth", torch.ones(1, dtype=torch.float32)
            )
        else:
            self.support_capacity = 0

    def set_normalized_prior(self, value: float) -> None:
        """Start at the constant deployment prior without disabling learning."""

        with torch.no_grad():
            self.distance_intercept.fill_(float(value))
            self.distance_slope.zero_()
            final = self.residual.network[-1]
            final.weight.zero_()
            final.bias.zero_()

    @staticmethod
    def _splitmix64(value: int) -> int:
        """Deterministic integer mixer used by uniform reservoir sampling."""

        mask = (1 << 64) - 1
        value = (int(value) + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return (value ^ (value >> 31)) & mask

    @torch.no_grad()
    def record_support(
        self,
        x: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> None:
        """Maintain a fixed-size uniform reservoir of positive observations."""

        if not self.gate_residual_with_support:
            return
        rows = torch.as_tensor(
            x, dtype=torch.float32, device=self.support_vectors.device
        )
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        if rows.numel() == 0:
            return
        if int(rows.shape[1]) != self.raw_input_dim:
            raise ValueError("support rows have the wrong feature dimension")
        # Feasible observations are the positive samples in this simulation.
        # Recency weights affect fitting, not whether a location was observed.
        del sample_weights
        filled = min(
            self.support_capacity,
            max(0, int(round(float(self.support_filled.item())))),
        )
        seen = max(0, int(round(float(self.support_seen.item()))))
        for row in rows[:, :4]:
            seen += 1
            if filled < self.support_capacity:
                slot = filled
                filled += 1
            else:
                candidate = self._splitmix64(seen) % seen
                if candidate >= self.support_capacity:
                    continue
                slot = int(candidate)
            self.support_vectors[slot].copy_(row)
        self.support_filled.fill_(float(filled))
        self.support_seen.fill_(float(seen))
        self._refresh_support_bandwidth(filled)

    @torch.no_grad()
    def _refresh_support_bandwidth(self, filled: int | None = None) -> None:
        if not self.gate_residual_with_support:
            return
        count = (
            int(round(float(self.support_filled.item())))
            if filled is None
            else int(filled)
        )
        count = min(self.support_capacity, max(0, count))
        if count < 2:
            # Dimension-derived fallback used only for the first observation.
            self.support_bandwidth.fill_(1.0 / math.sqrt(4.0))
            return
        centres = self.support_vectors[:count]
        distances = torch.cdist(centres, centres)
        distances.fill_diagonal_(float("inf"))
        nearest = distances.min(dim=1).values
        positive = nearest[torch.isfinite(nearest) & (nearest > 1.0e-8)]
        if int(positive.numel()) == 0:
            bandwidth = 1.0 / math.sqrt(4.0)
        else:
            bandwidth = float(torch.median(positive).item())
        self.support_bandwidth.fill_(max(bandwidth, 1.0e-6))

    def support_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Maximum RBF similarity to a retained positive support vector."""

        if not self.gate_residual_with_support:
            return torch.ones(
                (*x.shape[:-1], 1), dtype=x.dtype, device=x.device
            )
        filled = min(
            self.support_capacity,
            max(0, int(round(float(self.support_filled.item())))),
        )
        if filled == 0:
            return torch.zeros(
                (*x.shape[:-1], 1), dtype=x.dtype, device=x.device
            )
        coordinates = x[..., :4].to(dtype=torch.float32)
        centres = self.support_vectors[:filled].to(
            device=x.device, dtype=coordinates.dtype
        )
        squared_distance = torch.sum(
            torch.square(coordinates.unsqueeze(-2) - centres), dim=-1
        )
        bandwidth = self.support_bandwidth.to(
            device=x.device, dtype=coordinates.dtype
        ).clamp_min(1.0e-6)
        confidence = torch.exp(
            -squared_distance / (2.0 * bandwidth.square())
        ).amax(dim=-1, keepdim=True)
        return confidence.to(dtype=x.dtype)

    def distance_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """Learned monotone log-distance component with no fitted scale knob."""

        delta = x[..., :2] - x[..., 2:4]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        return self.distance_intercept + self.distance_slope * torch.log1p(
            distance
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        if self.gate_residual_with_support:
            residual = residual * self.support_confidence(x)
        return self.distance_baseline(x) + residual


class MicroRSSIPredictor(_CoordinateTimePredictor):
    """Minimal MLP with one 64-unit hidden layer."""

    def __init__(self, input_dim: int = 4, **time_kwargs) -> None:
        super().__init__(
            input_dim=input_dim, hidden_dims=(64,), **time_kwargs
        )


class SmallRSSIPredictor(_CoordinateTimePredictor):
    """Medium-small propagation-loss MLP."""

    def __init__(self, input_dim: int = 4, **time_kwargs) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dims=(128, 128, 128),
            **time_kwargs,
        )


class CensoredRSSIPredictor(_CoordinateTimePredictor):
    """One predictor with separate link-feasibility and feasible-loss heads.

    The public ``forward`` method remains a scalar normalized propagation
    loss, so aggregation and evaluation code can treat this like every other
    predictor.  Training can additionally use ``supervised_loss`` to balance
    feasible/artificial-unavailable classification while regressing RSSI only
    for genuine feasible links.
    """

    def __init__(
        self,
        input_dim: int = 4,
        *,
        hidden_dims: tuple[int, ...] = (64, 64),
        feasible_loss_ceiling: float = 0.86,
        classification_weight: float = 0.10,
        unavailable_class_fraction: float = 0.50,
        hard_decision: bool = False,
        feasibility_threshold: float = 0.50,
        **time_kwargs,
    ) -> None:
        if not hidden_dims:
            raise ValueError("censored predictor needs at least one hidden layer")
        super().__init__(
            input_dim=input_dim,
            hidden_dims=tuple(int(value) for value in hidden_dims),
            **time_kwargs,
        )
        layers = list(self.network.children())
        self.network = nn.Sequential(*layers[:-1])
        width = int(hidden_dims[-1])
        self.feasibility_head = nn.Linear(width, 1)
        self.feasible_loss_head = nn.Linear(width, 1)
        self.feasible_loss_ceiling = float(feasible_loss_ceiling)
        self.classification_weight = float(classification_weight)
        self.unavailable_class_fraction = float(unavailable_class_fraction)
        self.hard_decision = bool(hard_decision)
        self.feasibility_threshold = float(feasibility_threshold)
        self.censoring_boundary = 1.0
        if not 0.0 < self.feasible_loss_ceiling < 1.0:
            raise ValueError("feasible_loss_ceiling must be in (0, 1)")
        if self.classification_weight <= 0.0:
            raise ValueError("classification_weight must be positive")
        if not 0.0 < self.unavailable_class_fraction < 1.0:
            raise ValueError(
                "unavailable_class_fraction must be strictly between zero and one"
            )
        if not 0.0 < self.feasibility_threshold < 1.0:
            raise ValueError(
                "feasibility_threshold must be strictly between zero and one"
            )

    def forward_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = super().forward(x)
        feasibility_logit = self.feasibility_head(hidden)
        feasible_loss = self.feasible_loss_ceiling * torch.sigmoid(
            self.feasible_loss_head(hidden)
        )
        return feasibility_logit, feasible_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feasibility_logit, feasible_loss = self.forward_components(x)
        probability = torch.sigmoid(feasibility_logit)
        if self.hard_decision:
            # Normalized loss 1.0 maps exactly to the configured RSSI floor.
            # The discontinuous decision is inference-only: supervised_loss()
            # trains the classifier and conditional regressor separately.
            return torch.where(
                probability >= self.feasibility_threshold,
                feasible_loss,
                torch.full_like(feasible_loss, self.censoring_boundary),
            )
        # Expected normalized loss: unavailable links use the propagation-loss
        # floor (1.0), feasible links use the conditional regression head.
        return probability * feasible_loss + (1.0 - probability)

    def set_censoring_boundary(self, normalized_loss: float) -> None:
        """Use one observable reception boundary for labels and hard output."""

        boundary = float(normalized_loss)
        if not 0.0 < boundary <= 1.0:
            raise ValueError("censoring boundary must be in (0, 1]")
        self.censoring_boundary = boundary
        self.feasible_loss_ceiling = min(
            self.feasible_loss_ceiling, boundary
        )

    def set_normalized_prior(self, prior: float) -> None:
        """Initialize a spatially constant conservative scalar prediction."""

        value = float(np.clip(prior, 0.0, 1.0))
        conditional = min(0.5 * self.feasible_loss_ceiling, value)
        denominator = max(1.0e-6, 1.0 - conditional)
        probability = float(np.clip((1.0 - value) / denominator, 1.0e-4, 1.0 - 1.0e-4))
        with torch.no_grad():
            self.feasibility_head.weight.zero_()
            self.feasibility_head.bias.fill_(math.log(probability / (1.0 - probability)))
            self.feasible_loss_head.weight.zero_()
            ratio = float(
                np.clip(
                    conditional / self.feasible_loss_ceiling,
                    1.0e-4,
                    1.0 - 1.0e-4,
                )
            )
            self.feasible_loss_head.bias.fill_(math.log(ratio / (1.0 - ratio)))

    def supervised_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Balanced classification plus conditional feasible-link regression."""

        target = target.reshape(-1, 1)
        logits, feasible_loss = self.forward_components(x)
        # Real received links are strictly above the normalized floor.  The
        # reversible artificial unavailable links are exactly 1.0.
        feasible = target < (self.censoring_boundary - 1.0e-6)
        labels = feasible.to(dtype=target.dtype)
        weights = (
            torch.ones_like(target)
            if sample_weights is None
            else sample_weights.reshape(-1, 1).to(
                device=target.device, dtype=target.dtype
            )
        ).clamp_min(0.0)

        feasible_mass = torch.sum(weights * labels)
        unavailable_mass = torch.sum(weights * (1.0 - labels))
        if feasible_mass > 0.0 and unavailable_mass > 0.0:
            unavailable_fraction = float(self.unavailable_class_fraction)
            feasible_fraction = 1.0 - unavailable_fraction
            class_weights = (
                weights
                * (
                    labels * (feasible_fraction / feasible_mass)
                    + (1.0 - labels)
                    * (unavailable_fraction / unavailable_mass)
                )
            )
            class_loss = torch.sum(
                F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                * class_weights
            )
        else:
            class_loss = torch.sum(
                F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                * weights
            ) / torch.sum(weights).clamp_min(1.0e-8)

        regression_weights = weights * labels
        regression_loss = torch.sum(
            torch.square(feasible_loss - target) * regression_weights
        ) / torch.sum(regression_weights).clamp_min(1.0e-8)
        return regression_loss + self.classification_weight * class_loss


class IndependentCensoredRSSIPredictor(nn.Module):
    """Two-head predictor whose feasibility and RSSI features cannot interfere."""

    def __init__(
        self,
        input_dim: int = 4,
        *,
        hidden_dims: tuple[int, ...] = (128, 128, 128),
        feasible_loss_ceiling: float = 0.86,
        classification_weight: float = 0.20,
        unavailable_class_fraction: float = 0.50,
        initial_logit_scale: float = 2.0,
        hard_decision: bool = False,
        feasibility_threshold: float = 0.50,
        **time_kwargs,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("independent censored predictor needs hidden layers")
        if not math.isfinite(float(initial_logit_scale)) or float(
            initial_logit_scale
        ) <= 0.0:
            raise ValueError("initial_logit_scale must be finite and positive")
        branch_kwargs = dict(
            input_dim=int(input_dim),
            hidden_dims=tuple(int(value) for value in hidden_dims),
            **time_kwargs,
        )
        self.feasibility_network = _CoordinateTimePredictor(**branch_kwargs)
        self.feasible_loss_network = _CoordinateTimePredictor(**branch_kwargs)
        self.raw_input_dim = int(input_dim)
        self.include_time = bool(time_kwargs.get("include_time", False))
        self.feasible_loss_ceiling = float(feasible_loss_ceiling)
        self.classification_weight = float(classification_weight)
        self.unavailable_class_fraction = float(unavailable_class_fraction)
        self.hard_decision = bool(hard_decision)
        self.feasibility_threshold = float(feasibility_threshold)
        self.censoring_boundary = 1.0
        self.calibration_log_scale = nn.Parameter(
            torch.tensor(
                math.log(math.expm1(float(initial_logit_scale))),
                dtype=torch.float32,
            )
        )
        self.calibration_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        support_width = int(hidden_dims[-1])
        self.register_buffer(
            "feasibility_support_precision",
            torch.ones((support_width,), dtype=torch.float32),
        )
        self.register_buffer(
            "rssi_support_precision",
            torch.ones((support_width,), dtype=torch.float32),
        )
        self.register_buffer(
            "feasibility_support_mass",
            torch.zeros((), dtype=torch.float32),
        )
        self.register_buffer(
            "feasibility_support_feasible_mass",
            torch.zeros((), dtype=torch.float32),
        )
        self.register_buffer(
            "feasibility_support_unavailable_mass",
            torch.zeros((), dtype=torch.float32),
        )
        self.register_buffer(
            "rssi_support_mass",
            torch.zeros((), dtype=torch.float32),
        )
        if not 0.0 < self.feasible_loss_ceiling < 1.0:
            raise ValueError("feasible_loss_ceiling must be in (0, 1)")
        if self.classification_weight <= 0.0:
            raise ValueError("classification_weight must be positive")
        if not 0.0 < self.unavailable_class_fraction < 1.0:
            raise ValueError(
                "unavailable_class_fraction must be strictly between zero and one"
            )
        if not 0.0 < self.feasibility_threshold < 1.0:
            raise ValueError(
                "feasibility_threshold must be strictly between zero and one"
            )

    def _logit_scale(self) -> torch.Tensor:
        return torch.clamp(
            F.softplus(self.calibration_log_scale), min=0.25, max=8.0
        )

    def _features_and_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feasibility_hidden = self.feasibility_network.hidden_features(x)
        rssi_hidden = self.feasible_loss_network.hidden_features(x)
        raw_logit = self.feasibility_network.output_from_hidden(
            feasibility_hidden
        )
        feasibility_logit = (
            self._logit_scale() * raw_logit + self.calibration_bias
        )
        feasible_loss = self.feasible_loss_ceiling * torch.sigmoid(
            self.feasible_loss_network.output_from_hidden(rssi_hidden)
        )
        return feasibility_logit, feasible_loss, feasibility_hidden, rssi_hidden

    def forward_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feasibility_logit, feasible_loss, _feasibility_hidden, _rssi_hidden = (
            self._features_and_components(x)
        )
        return feasibility_logit, feasible_loss

    @staticmethod
    def _diagonal_support_risk(
        hidden: torch.Tensor, precision: torch.Tensor
    ) -> torch.Tensor:
        diagonal = precision.to(device=hidden.device, dtype=hidden.dtype)
        leverage = torch.mean(
            torch.square(hidden) / diagonal.clamp_min(1.0), dim=-1, keepdim=True
        )
        return torch.sqrt(leverage.clamp_min(0.0))

    def forward_components_with_support(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, conditional, feasibility_hidden, rssi_hidden = (
            self._features_and_components(x)
        )
        probability = torch.sigmoid(logits)
        feasibility_risk = self._diagonal_support_risk(
            feasibility_hidden, self.feasibility_support_precision
        )
        rssi_risk = self._diagonal_support_risk(
            rssi_hidden, self.rssi_support_precision
        )
        # Every decision needs classifier support. Conditional-RSSI support is
        # relevant in proportion to the model's own probability of feasibility.
        support_risk = feasibility_risk + probability * rssi_risk
        return logits, conditional, support_risk

    def _accumulate_support(
        self,
        feasibility_hidden: torch.Tensor,
        rssi_hidden: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            detached_weights = weights.detach().clamp_min(0.0)
            detached_labels = labels.detach()
            classifier_weight = detached_weights
            regression_weight = detached_weights * detached_labels
            self.feasibility_support_precision.add_(
                torch.sum(
                    torch.square(feasibility_hidden.detach())
                    * classifier_weight,
                    dim=0,
                ).to(self.feasibility_support_precision)
            )
            self.rssi_support_precision.add_(
                torch.sum(
                    torch.square(rssi_hidden.detach()) * regression_weight,
                    dim=0,
                ).to(self.rssi_support_precision)
            )
            feasible_mass = torch.sum(
                detached_weights * detached_labels
            ).to(self.feasibility_support_mass)
            unavailable_mass = torch.sum(
                detached_weights * (1.0 - detached_labels)
            ).to(self.feasibility_support_mass)
            self.feasibility_support_mass.add_(feasible_mass + unavailable_mass)
            self.feasibility_support_feasible_mass.add_(feasible_mass)
            self.feasibility_support_unavailable_mass.add_(unavailable_mass)
            self.rssi_support_mass.add_(feasible_mass)

    def support_summary(self) -> tuple[float, float]:
        """Return model-only evidence strength and class balance in [0, 1]."""

        feasible = max(
            0.0, float(self.feasibility_support_feasible_mass.detach().cpu())
        )
        unavailable = max(
            0.0, float(self.feasibility_support_unavailable_mass.detach().cpu())
        )
        total = feasible + unavailable
        strength = 1.0 - math.exp(-total / 1024.0)
        balance = (
            0.0
            if total <= 0.0
            else 2.0 * min(feasible, unavailable) / total
        )
        return float(np.clip(strength, 0.0, 1.0)), float(
            np.clip(balance, 0.0, 1.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feasibility_logit, feasible_loss = self.forward_components(x)
        probability = torch.sigmoid(feasibility_logit)
        if self.hard_decision:
            return torch.where(
                probability >= self.feasibility_threshold,
                feasible_loss,
                torch.full_like(feasible_loss, self.censoring_boundary),
            )
        return probability * feasible_loss + (1.0 - probability)

    def set_censoring_boundary(self, normalized_loss: float) -> None:
        """Use one observable reception boundary for labels and hard output."""

        boundary = float(normalized_loss)
        if not 0.0 < boundary <= 1.0:
            raise ValueError("censoring boundary must be in (0, 1]")
        self.censoring_boundary = boundary
        self.feasible_loss_ceiling = min(
            self.feasible_loss_ceiling, boundary
        )

    @staticmethod
    def _output_layer(branch: _CoordinateTimePredictor) -> nn.Linear:
        layer = branch.network[-1]
        if not isinstance(layer, nn.Linear):
            raise TypeError("predictor branch must end in a linear layer")
        return layer

    def set_normalized_prior(self, prior: float) -> None:
        """Initialize a constant soft prediction without coupling the branches."""

        value = float(np.clip(prior, 0.0, 1.0))
        conditional = min(0.5 * self.feasible_loss_ceiling, value)
        denominator = max(1.0e-6, 1.0 - conditional)
        probability = float(
            np.clip(
                (1.0 - value) / denominator,
                1.0e-4,
                1.0 - 1.0e-4,
            )
        )
        with torch.no_grad():
            self.calibration_bias.zero_()
            scale = float(self._logit_scale().item())
            feasibility = self._output_layer(self.feasibility_network)
            feasibility.weight.zero_()
            feasibility.bias.fill_(
                math.log(probability / (1.0 - probability)) / scale
            )
            feasible = self._output_layer(self.feasible_loss_network)
            feasible.weight.zero_()
            ratio = float(
                np.clip(
                    conditional / self.feasible_loss_ceiling,
                    1.0e-4,
                    1.0 - 1.0e-4,
                )
            )
            feasible.bias.fill_(math.log(ratio / (1.0 - ratio)))

    def supervised_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Train feasibility only with both classes, preserving transferred maps."""

        target = target.reshape(-1, 1)
        logits, feasible_loss, feasibility_hidden, rssi_hidden = (
            self._features_and_components(x)
        )
        labels = (
            target < (self.censoring_boundary - 1.0e-6)
        ).to(dtype=target.dtype)
        weights = (
            torch.ones_like(target)
            if sample_weights is None
            else sample_weights.reshape(-1, 1).to(
                device=target.device, dtype=target.dtype
            )
        ).clamp_min(0.0)
        if self.training and torch.is_grad_enabled():
            self._accumulate_support(
                feasibility_hidden, rssi_hidden, labels, weights
            )

        feasible_mass = torch.sum(weights * labels)
        unavailable_mass = torch.sum(weights * (1.0 - labels))
        if feasible_mass > 0.0 and unavailable_mass > 0.0:
            unavailable_fraction = float(self.unavailable_class_fraction)
            feasible_fraction = 1.0 - unavailable_fraction
            class_weights = weights * (
                labels * (feasible_fraction / feasible_mass)
                + (1.0 - labels)
                * (unavailable_fraction / unavailable_mass)
            )
            class_loss = torch.sum(
                F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                * class_weights
            )
        else:
            # A feasible-only newcomer must not erase a transferred blockage
            # classifier. The independent RSSI branch can still adapt locally.
            class_loss = torch.sum(logits * 0.0)

        regression_weights = weights * labels
        regression_loss = torch.sum(
            torch.square(feasible_loss - target) * regression_weights
        ) / torch.sum(regression_weights).clamp_min(1.0e-8)
        return regression_loss + self.classification_weight * class_loss


class EnsembleIndependentCensoredRSSIPredictor(nn.Module):
    """Grid-free bootstrapped ensemble for hard censored RSSI prediction.

    Corresponding members share the repository's common initialization across
    vehicles, so member-wise FedAvg remains meaningful. Members train on
    independently bootstrapped minibatch weights. Their disagreement supplies
    per-query epistemic uncertainty without storing or transmitting coordinates.
    """

    def __init__(
        self,
        input_dim: int = 4,
        *,
        hidden_dims: tuple[int, ...] = (64, 64),
        members: int = 3,
        bootstrap_keep_probability: float = 0.80,
        feasibility_threshold: float = 0.50,
        **time_kwargs,
    ) -> None:
        super().__init__()
        if int(members) < 2:
            raise ValueError("ensemble needs at least two members")
        if not 0.0 < float(bootstrap_keep_probability) <= 1.0:
            raise ValueError("bootstrap_keep_probability must be in (0, 1]")
        self.members = nn.ModuleList(
            [
                IndependentCensoredRSSIPredictor(
                    input_dim=int(input_dim),
                    hidden_dims=tuple(int(value) for value in hidden_dims),
                    hard_decision=False,
                    feasibility_threshold=float(feasibility_threshold),
                    **time_kwargs,
                )
                for _ in range(int(members))
            ]
        )
        self.raw_input_dim = int(input_dim)
        self.include_time = bool(time_kwargs.get("include_time", False))
        self.bootstrap_keep_probability = float(bootstrap_keep_probability)
        self.register_buffer(
            "decision_threshold",
            torch.tensor(float(feasibility_threshold), dtype=torch.float32),
        )
        self.censoring_boundary = 1.0
        self._unavailable_class_fraction = 0.50

    @property
    def unavailable_class_fraction(self) -> float:
        return float(self._unavailable_class_fraction)

    @property
    def feasibility_threshold(self) -> float:
        return float(self.decision_threshold.detach().cpu().item())

    def set_feasibility_threshold(self, value: float) -> None:
        threshold = float(value)
        if not 0.0 < threshold < 1.0:
            raise ValueError("feasibility threshold must be in (0, 1)")
        with torch.no_grad():
            self.decision_threshold.fill_(threshold)

    @unavailable_class_fraction.setter
    def unavailable_class_fraction(self, value: float) -> None:
        fraction = float(value)
        if not 0.0 < fraction < 1.0:
            raise ValueError("unavailable_class_fraction must be in (0, 1)")
        self._unavailable_class_fraction = fraction
        for member in getattr(self, "members", []):
            member.unavailable_class_fraction = fraction

    def set_censoring_boundary(self, normalized_loss: float) -> None:
        boundary = float(normalized_loss)
        if not 0.0 < boundary <= 1.0:
            raise ValueError("censoring boundary must be in (0, 1]")
        self.censoring_boundary = boundary
        for member in self.members:
            member.set_censoring_boundary(boundary)

    def set_normalized_prior(self, prior: float) -> None:
        for member in self.members:
            member.set_normalized_prior(prior)

    def forward_member_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [member.forward_components(x) for member in self.members]
        logits = torch.stack([row[0] for row in rows], dim=0)
        conditional = torch.stack([row[1] for row in rows], dim=0)
        return logits, conditional

    def forward_member_components_with_support(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = [
            member.forward_components_with_support(x)
            for member in self.members
        ]
        logits = torch.stack([row[0] for row in rows], dim=0)
        conditional = torch.stack([row[1] for row in rows], dim=0)
        support_risk = torch.stack([row[2] for row in rows], dim=0)
        return logits, conditional, support_risk

    def support_summary(self) -> tuple[float, float]:
        summaries = [member.support_summary() for member in self.members]
        return (
            float(np.mean([row[0] for row in summaries])),
            float(np.mean([row[1] for row in summaries])),
        )

    def forward_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, conditional = self.forward_member_components(x)
        probability = torch.sigmoid(logits).mean(dim=0)
        probability = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        mean_logit = torch.logit(probability)
        return mean_logit, conditional.mean(dim=0)

    def prediction_with_uncertainty(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, conditional, support_risk = (
            self.forward_member_components_with_support(x)
        )
        probabilities = torch.sigmoid(logits)
        mean_probability = probabilities.mean(dim=0)
        mean_conditional = conditional.mean(dim=0)
        boundary = torch.full_like(mean_conditional, self.censoring_boundary)
        prediction = torch.where(
            mean_probability >= self.decision_threshold,
            mean_conditional,
            boundary,
        )
        # Disagreement is epistemic; ambiguity penalizes decisions near the
        # feasibility boundary. Both are dimensionless normalized-loss risks.
        probability_std = probabilities.std(dim=0, unbiased=False)
        conditional_std = conditional.std(dim=0, unbiased=False)
        ambiguity = 4.0 * mean_probability * (1.0 - mean_probability)
        uncertainty = (
            conditional_std
            + 0.35 * probability_std
            + 0.10 * ambiguity
            + 0.25 * support_risk.mean(dim=0)
        )
        return prediction, uncertainty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, _uncertainty = self.prediction_with_uncertainty(x)
        return prediction

    def supervised_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        count = int(x.shape[0])
        base_weights = (
            torch.ones((count, 1), device=x.device, dtype=target.dtype)
            if sample_weights is None
            else sample_weights.reshape(count, 1).to(
                device=x.device, dtype=target.dtype
            )
        )
        losses: list[torch.Tensor] = []
        for index, member in enumerate(self.members):
            if self.bootstrap_keep_probability >= 1.0 or count <= 2:
                member_weights = base_weights
            else:
                mask = (
                    torch.rand((count, 1), device=x.device)
                    < self.bootstrap_keep_probability
                ).to(dtype=target.dtype)
                # Guarantee a gradient for every member in very sparse batches.
                if float(mask.sum().item()) <= 0.0:
                    mask[index % count] = 1.0
                member_weights = base_weights * mask
            losses.append(member.supervised_loss(x, target, member_weights))
        return torch.stack(losses).mean()


class CompactRSSIPredictor(_CoordinateTimePredictor):
    """Slightly smaller counterpart to the 128-wide online predictor."""

    def __init__(self, input_dim: int = 4, **time_kwargs) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dims=(112, 112, 112),
            **time_kwargs,
        )


class MediumRSSIPredictor(_CoordinateTimePredictor):
    """Moderate capacity step above the 128-wide online predictor."""

    def __init__(self, input_dim: int = 4, **time_kwargs) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dims=(192, 192, 192),
            **time_kwargs,
        )


class _GridParameter(nn.Module):
    """Store a flat interpolation grid as modest-width exact-policy rows."""

    def __init__(self, size: int) -> None:
        super().__init__()
        if int(size) <= 0:
            raise ValueError("grid size must be positive")
        rows = int(math.isqrt(int(size)))
        while int(size) % rows:
            rows -= 1
        self.weight = nn.Parameter(torch.zeros(rows, int(size) // rows))


class _GridSupport(nn.Module):
    """Non-trainable effective observation mass for an interpolation grid."""

    def __init__(self, size: int) -> None:
        super().__init__()
        if int(size) <= 0:
            raise ValueError("grid size must be positive")
        rows = int(math.isqrt(int(size)))
        while int(size) % rows:
            rows -= 1
        self.register_buffer(
            "weight", torch.zeros(rows, int(size) // rows, dtype=torch.float32)
        )


class ConservativeLocalSupportRSSIPredictor(nn.Module):
    """Conservative multilinear map whose updates have only local support.

    The four endpoint coordinates use a 9-point grid per dimension.  When
    enabled, time uses five points across the complete simulation.  Every
    observation therefore updates at most 32 neighboring coefficients, while
    unobserved cells remain exactly at the configured propagation-loss prior.
    """

    def __init__(
        self,
        input_dim: int = 4,
        *,
        include_time: bool = False,
        learned_time_scale: float = 1000.0,
        spatial_grid_points: int = 9,
        time_grid_points: int = 5,
        support_prior_strength: float = 0.0,
        **_unused_time_kwargs,
    ) -> None:
        super().__init__()
        self.raw_input_dim = int(input_dim)
        self.include_time = bool(include_time)
        expected = 5 if self.include_time else 4
        if self.raw_input_dim != expected:
            raise ValueError(
                f"local-support predictor expects input_dim={expected}, "
                f"got {self.raw_input_dim}"
            )
        if int(spatial_grid_points) < 2 or int(time_grid_points) < 2:
            raise ValueError("local-support grids need at least two points")
        if not math.isfinite(float(learned_time_scale)) or float(learned_time_scale) <= 0.0:
            raise ValueError("learned_time_scale must be finite and positive")
        if not math.isfinite(float(support_prior_strength)) or float(support_prior_strength) < 0.0:
            raise ValueError("support_prior_strength must be finite and nonnegative")
        self.support_prior_strength = float(support_prior_strength)

        self.learned_time_scale = float(learned_time_scale)
        self.grid_shape = (int(spatial_grid_points),) * 4 + (
            (int(time_grid_points),) if self.include_time else ()
        )
        grid_size = math.prod(self.grid_shape)
        self.grid = _GridParameter(grid_size)
        self.support = _GridSupport(grid_size)
        self.register_buffer("prior", torch.ones(1, dtype=torch.float32))

        strides: list[int] = []
        for dimension in range(len(self.grid_shape)):
            strides.append(math.prod(self.grid_shape[dimension + 1 :]))
        self.register_buffer(
            "grid_strides",
            torch.tensor(strides, dtype=torch.long),
            persistent=False,
        )

    def set_normalized_prior(self, value: float) -> None:
        with torch.no_grad():
            self.prior.fill_(float(value))

    def _interpolation_corners(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Return normalized coordinates and (flat-index, basis-weight) pairs."""

        if int(x.shape[-1]) != self.raw_input_dim:
            raise ValueError(
                f"expected predictor input dimension {self.raw_input_dim}, "
                f"got {int(x.shape[-1])}"
            )
        coordinates = x.to(dtype=torch.float32)
        if self.include_time:
            coordinates = torch.cat(
                (
                    coordinates[..., :4],
                    coordinates[..., 4:5] / self.learned_time_scale,
                ),
                dim=-1,
            )
        coordinates = torch.clamp(coordinates, 0.0, 1.0)
        sizes = coordinates.new_tensor(self.grid_shape)
        scaled = coordinates * (sizes - 1.0)
        lower = torch.floor(scaled).to(dtype=torch.long)
        upper = torch.minimum(
            lower + 1,
            torch.as_tensor(
                self.grid_shape,
                dtype=torch.long,
                device=coordinates.device,
            )
            - 1,
        )
        fraction = scaled - lower.to(dtype=scaled.dtype)

        corners: list[tuple[torch.Tensor, torch.Tensor]] = []
        dimensions = len(self.grid_shape)
        strides = self.grid_strides.to(device=coordinates.device)
        for corner in range(1 << dimensions):
            index = torch.zeros_like(lower[..., 0])
            weight = torch.ones_like(coordinates[..., 0])
            for dimension in range(dimensions):
                use_upper = bool(corner & (1 << dimension))
                point = upper[..., dimension] if use_upper else lower[..., dimension]
                component = (
                    fraction[..., dimension]
                    if use_upper
                    else 1.0 - fraction[..., dimension]
                )
                index = index + point * strides[dimension]
                weight = weight * component
            corners.append((index, weight))
        return coordinates, corners

    @torch.no_grad()
    def record_support(
        self,
        x: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> None:
        """Record local support from newly received training observations."""

        device = self.support.weight.device
        rows = torch.as_tensor(x, dtype=torch.float32, device=device)
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        if rows.numel() == 0:
            return
        weights = None
        if sample_weights is not None:
            weights = torch.as_tensor(
                sample_weights, dtype=torch.float32, device=device
            ).reshape(-1)
            if int(weights.numel()) != int(rows.shape[0]):
                raise ValueError("sample_weights and support rows differ in length")
        _coordinates, corners = self._interpolation_corners(rows)
        flat_support = self.support.weight.reshape(-1)
        for index, basis_weight in corners:
            contribution = basis_weight.reshape(-1).square()
            if weights is not None:
                contribution = contribution * weights
            flat_support.scatter_add_(0, index.reshape(-1), contribution)

    @torch.no_grad()
    def support_at(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate the stored support at query rows."""

        device = self.support.weight.device
        rows = torch.as_tensor(x, dtype=torch.float32, device=device)
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        coordinates, corners = self._interpolation_corners(rows)
        flat_support = self.support.weight.reshape(-1)
        support = torch.zeros_like(coordinates[..., 0])
        for index, weight in corners:
            support = support + weight * flat_support[index]
        return support.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coordinates, corners = self._interpolation_corners(x)
        flat_grid = self.grid.weight.reshape(-1)
        delta = torch.zeros_like(coordinates[..., 0])
        for index, weight in corners:
            delta = delta + weight * flat_grid[index]
        if self.support_prior_strength > 0.0:
            flat_support = self.support.weight.reshape(-1)
            local_support = torch.zeros_like(delta)
            for index, weight in corners:
                local_support = local_support + weight * flat_support[index]
            confidence = local_support / (
                local_support + float(self.support_prior_strength)
            )
            delta = delta * confidence
        return (
            self.prior.to(device=delta.device, dtype=delta.dtype) + delta
        ).unsqueeze(-1)


MERGEABLE_EVIDENCE_FORMAT = 1


class MergeableRFFRidgePredictor(nn.Module):
    """Fixed random features with provenance-aware additive ridge evidence."""

    def __init__(
        self,
        *,
        input_dim: int = 4,
        include_time: bool = False,
        learned_time_scale: float = 1000.0,
        basis_dim: int = 192,
        ridge: float = 1.0,
        **_unused,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(basis_dim) < 2:
            raise ValueError("basis_dim must be at least two")
        if float(ridge) <= 0.0 or not math.isfinite(float(ridge)):
            raise ValueError("ridge must be finite and positive")
        self.raw_input_dim = int(input_dim)
        self.include_time = bool(include_time)
        self.time_scale = float(learned_time_scale)
        if self.include_time and self.time_scale <= 0.0:
            raise ValueError("learned_time_scale must be positive")
        self.basis_dim = int(basis_dim)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            91_739 + 131 * self.raw_input_dim + 17 * self.basis_dim
        )
        rows = self.basis_dim - 1
        projection = torch.randn(
            (rows, self.raw_input_dim), generator=generator
        )
        projection = projection / projection.norm(
            dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        bandwidth = torch.logspace(-0.2, 1.1, steps=rows).unsqueeze(1)
        projection = projection * bandwidth * (2.0 * math.pi)
        phase = torch.rand((rows,), generator=generator) * (2.0 * math.pi)
        self.register_buffer("projection", projection.to(torch.float32))
        self.register_buffer("phase", phase.to(torch.float32))
        self.register_buffer(
            "mergeable_format_version",
            torch.tensor(MERGEABLE_EVIDENCE_FORMAT, dtype=torch.int64),
        )
        self.register_buffer(
            "ridge", torch.tensor(float(ridge), dtype=torch.float32)
        )
        self.register_buffer(
            "prior_value", torch.tensor(0.0, dtype=torch.float32)
        )
        self.register_buffer(
            "evidence_keys", torch.empty((0,), dtype=torch.int64)
        )
        self.register_buffer(
            "evidence_versions", torch.empty((0,), dtype=torch.int64)
        )
        self.register_buffer(
            "evidence_precision",
            torch.empty(
                (0, self.basis_dim, self.basis_dim), dtype=torch.float32
            ),
        )
        self.register_buffer(
            "evidence_information",
            torch.empty((0, self.basis_dim), dtype=torch.float32),
        )
        self.register_buffer(
            "evidence_mass", torch.empty((0,), dtype=torch.float32)
        )
        self.weight = nn.Parameter(
            torch.zeros(self.basis_dim, dtype=torch.float32)
        )
        self._refresh_weight()

    def _scaled_inputs(self, x: torch.Tensor) -> torch.Tensor:
        values = x.to(dtype=torch.float32)
        if (
            values.ndim != 2
            or int(values.shape[1]) != self.raw_input_dim
        ):
            raise ValueError(
                f"expected predictor input [N, {self.raw_input_dim}]"
            )
        if self.include_time:
            values = values.clone()
            values[:, -1] = values[:, -1] / self.time_scale
        return values

    def features(self, x: torch.Tensor) -> torch.Tensor:
        values = self._scaled_inputs(x)
        angles = values @ self.projection.transpose(0, 1) + self.phase
        random_features = math.sqrt(2.0) * torch.cos(angles)
        return torch.cat(
            (
                torch.ones(
                    (int(values.shape[0]), 1),
                    dtype=values.dtype,
                    device=values.device,
                ),
                random_features,
            ),
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.features(x) @ self.weight).unsqueeze(-1)

    @torch.no_grad()
    def set_normalized_prior(self, value: float) -> None:
        self.prior_value.fill_(float(value))
        self._refresh_weight()

    @torch.no_grad()
    def reset_evidence(self) -> None:
        device = self.weight.device
        self.evidence_keys = torch.empty(
            (0,), dtype=torch.int64, device=device
        )
        self.evidence_versions = torch.empty(
            (0,), dtype=torch.int64, device=device
        )
        self.evidence_precision = torch.empty(
            (0, self.basis_dim, self.basis_dim),
            dtype=torch.float32,
            device=device,
        )
        self.evidence_information = torch.empty(
            (0, self.basis_dim), dtype=torch.float32, device=device
        )
        self.evidence_mass = torch.empty(
            (0,), dtype=torch.float32, device=device
        )
        self._refresh_weight()

    @torch.no_grad()
    def add_evidence(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        *,
        origin: int,
        sample_weights: torch.Tensor | None = None,
    ) -> None:
        if int(x.shape[0]) == 0:
            return
        z = self.features(x.to(device=self.weight.device))
        y = target.to(
            device=self.weight.device, dtype=torch.float32
        ).reshape(-1)
        if int(y.numel()) != int(z.shape[0]):
            raise ValueError(
                "evidence features and targets differ in length"
            )
        if sample_weights is None:
            weights = torch.ones_like(y)
        else:
            weights = sample_weights.to(
                device=self.weight.device, dtype=torch.float32
            ).reshape(-1)
            if int(weights.numel()) != int(y.numel()):
                raise ValueError(
                    "evidence weights and targets differ in length"
                )
            weights = weights.clamp_min(0.0)
        precision = z.transpose(0, 1) @ (
            z * weights.unsqueeze(1)
        )
        information = z.transpose(0, 1) @ (y * weights)
        key = int(origin)
        matches = torch.nonzero(
            self.evidence_keys == key
        ).reshape(-1)
        if int(matches.numel()) == 0:
            self.evidence_keys = torch.cat(
                (
                    self.evidence_keys,
                    torch.tensor(
                        [key], dtype=torch.int64, device=z.device
                    ),
                )
            )
            self.evidence_versions = torch.cat(
                (
                    self.evidence_versions,
                    torch.tensor(
                        [int(z.shape[0])],
                        dtype=torch.int64,
                        device=z.device,
                    ),
                )
            )
            self.evidence_precision = torch.cat(
                (self.evidence_precision, precision.unsqueeze(0)), dim=0
            )
            self.evidence_information = torch.cat(
                (
                    self.evidence_information,
                    information.unsqueeze(0),
                ),
                dim=0,
            )
            self.evidence_mass = torch.cat(
                (self.evidence_mass, weights.sum().reshape(1)), dim=0
            )
        else:
            index = int(matches[0].item())
            self.evidence_versions[index] += int(z.shape[0])
            self.evidence_precision[index].add_(precision)
            self.evidence_information[index].add_(information)
            self.evidence_mass[index].add_(weights.sum())
        order = torch.argsort(self.evidence_keys)
        self.evidence_keys = self.evidence_keys[order]
        self.evidence_versions = self.evidence_versions[order]
        self.evidence_precision = self.evidence_precision[order]
        self.evidence_information = self.evidence_information[order]
        self.evidence_mass = self.evidence_mass[order]
        self._refresh_weight()

    @torch.no_grad()
    def _refresh_weight(self) -> None:
        precision = torch.eye(
            self.basis_dim,
            dtype=torch.float32,
            device=self.weight.device,
        ) * self.ridge.to(device=self.weight.device)
        information = torch.zeros(
            self.basis_dim,
            dtype=torch.float32,
            device=self.weight.device,
        )
        information[0] = (
            self.ridge.to(device=self.weight.device)
            * self.prior_value.to(device=self.weight.device)
        )
        if int(self.evidence_precision.shape[0]):
            precision = precision + self.evidence_precision.sum(dim=0)
            information = (
                information + self.evidence_information.sum(dim=0)
            )
        self.weight.copy_(torch.linalg.solve(precision, information))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in (
            "evidence_keys",
            "evidence_versions",
            "evidence_precision",
            "evidence_information",
            "evidence_mass",
        ):
            key = prefix + name
            incoming = state_dict.get(key)
            if (
                torch.is_tensor(incoming)
                and tuple(getattr(self, name).shape)
                != tuple(incoming.shape)
            ):
                setattr(
                    self,
                    name,
                    torch.empty_like(
                        incoming, device=getattr(self, name).device
                    ),
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class MergeableLogDistanceRidgePredictor(MergeableRFFRidgePredictor):
    """Exact provenance-aware ridge regression on [1, log(1+d)]."""

    def __init__(
        self,
        *,
        input_dim: int = 4,
        include_time: bool = False,
        learned_time_scale: float = 1000.0,
        ridge: float = 1.0,
        **_unused,
    ) -> None:
        super().__init__(
            input_dim=int(input_dim),
            include_time=bool(include_time),
            learned_time_scale=float(learned_time_scale),
            basis_dim=2,
            ridge=float(ridge),
        )
        # The inherited random projection is not part of this basis. Empty
        # deterministic buffers retain the common mergeable-state contract.
        self.projection = torch.empty(
            (0, self.raw_input_dim), dtype=torch.float32
        )
        self.phase = torch.empty((0,), dtype=torch.float32)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        values = self._scaled_inputs(x)
        distance_feature = LogDistanceRSSIPredictor.distance_feature(values)
        return torch.cat((torch.ones_like(distance_feature), distance_feature), dim=1)


def is_mergeable_evidence_state(
    state: object,
) -> bool:
    return (
        hasattr(state, "keys")
        and "mergeable_format_version" in state
    )


def mergeable_evidence_newer_count(
    base: dict[str, torch.Tensor],
    other: dict[str, torch.Tensor],
) -> int:
    base_versions = {
        int(key): int(version)
        for key, version in zip(
            base["evidence_keys"], base["evidence_versions"]
        )
    }
    return sum(
        int(int(version) > base_versions.get(int(key), -1))
        for key, version in zip(
            other["evidence_keys"], other["evidence_versions"]
        )
    )


def mergeable_evidence_delta_indices(
    base: dict[str, torch.Tensor],
    other: dict[str, torch.Tensor],
    *,
    max_rows: int = 0,
) -> list[int]:
    """Select the most informative newer provenance rows to transmit."""

    if (
        not is_mergeable_evidence_state(base)
        or not is_mergeable_evidence_state(other)
    ):
        raise ValueError("both states must contain mergeable evidence")
    base_rows = {
        int(key): (int(version), index)
        for index, (key, version) in enumerate(
            zip(base["evidence_keys"], base["evidence_versions"])
        )
    }
    base_precision = base["evidence_precision"].detach().to(
        device="cpu", dtype=torch.float64
    )
    other_precision = other["evidence_precision"].detach().to(
        device="cpu", dtype=torch.float64
    )
    ridge = max(float(base["ridge"].reshape(-1)[0]), 1.0e-12)
    before = torch.full(
        (int(base["weight"].numel()),), ridge, dtype=torch.float64
    )
    if int(base_precision.shape[0]):
        before.add_(
            torch.diagonal(base_precision, dim1=1, dim2=2).sum(dim=0)
        )
    candidates: list[tuple[float, float, int, int]] = []
    for index, (raw_key, raw_version) in enumerate(
        zip(other["evidence_keys"], other["evidence_versions"])
    ):
        key = int(raw_key)
        version = int(raw_version)
        previous = base_rows.get(key)
        if previous is not None and version <= previous[0]:
            continue
        delta = torch.diagonal(
            other_precision[index], dim1=0, dim2=1
        )
        if previous is not None:
            delta = delta - torch.diagonal(
                base_precision[previous[1]], dim1=0, dim2=1
            )
        delta = delta.clamp_min(0.0)
        score = torch.log1p(delta / before.clamp_min(1.0e-12)).sum()
        mass = float(other["evidence_mass"][index])
        candidates.append((float(score), mass, -key, index))
    candidates.sort(reverse=True)
    limit = int(max_rows)
    if limit > 0:
        candidates = candidates[:limit]
    return sorted(
        (row[3] for row in candidates),
        key=lambda index: int(other["evidence_keys"][index]),
    )


def mergeable_evidence_delta_state(
    base: dict[str, torch.Tensor],
    other: dict[str, torch.Tensor],
    *,
    max_rows: int = 0,
) -> dict[str, torch.Tensor]:
    """Return only newer rows selected for one capacity-bounded transfer."""

    indices = mergeable_evidence_delta_indices(
        base, other, max_rows=max_rows
    )
    output = {
        name: value.detach().cpu().clone()
        for name, value in other.items()
    }
    basis_dim = int(other["weight"].numel())
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    if indices:
        output["evidence_keys"] = other["evidence_keys"][
            tensor_indices
        ].detach().cpu().clone()
        output["evidence_versions"] = other["evidence_versions"][
            tensor_indices
        ].detach().cpu().clone()
        output["evidence_precision"] = other["evidence_precision"][
            tensor_indices
        ].detach().cpu().clone()
        output["evidence_information"] = other[
            "evidence_information"
        ][tensor_indices].detach().cpu().clone()
        output["evidence_mass"] = other["evidence_mass"][
            tensor_indices
        ].detach().cpu().clone()
    else:
        output["evidence_keys"] = torch.empty((0,), dtype=torch.int64)
        output["evidence_versions"] = torch.empty((0,), dtype=torch.int64)
        output["evidence_precision"] = torch.empty(
            (0, basis_dim, basis_dim), dtype=other["weight"].dtype
        )
        output["evidence_information"] = torch.empty(
            (0, basis_dim), dtype=other["weight"].dtype
        )
        output["evidence_mass"] = torch.empty(
            (0,), dtype=other["weight"].dtype
        )
    return output


def mergeable_evidence_summary_nbytes(
    state: dict[str, torch.Tensor],
) -> int:
    """Wire bytes for row count plus provenance key/version pairs."""

    if not is_mergeable_evidence_state(state):
        raise ValueError("state must contain mergeable evidence")
    return 4 + 16 * int(state["evidence_keys"].numel())


def mergeable_evidence_direction_nbytes(
    base: dict[str, torch.Tensor],
    other: dict[str, torch.Tensor],
    *,
    max_rows: int = 0,
) -> int:
    """Bytes sent by ``other`` to let ``base`` install selected deltas."""

    indices = mergeable_evidence_delta_indices(
        base, other, max_rows=max_rows
    )
    total = mergeable_evidence_summary_nbytes(other) + 4
    for index in indices:
        total += 16
        for name in (
            "evidence_precision",
            "evidence_information",
            "evidence_mass",
        ):
            value = other[name][index]
            total += int(value.numel()) * int(value.element_size())
    return int(total)


def mergeable_evidence_diagonal_information_gain(
    base: dict[str, torch.Tensor],
    other: dict[str, torch.Tensor],
) -> float:
    """Return cheap directional feature-space novelty in the other state.

    The exact log-determinant gain would require a cubic matrix factorization
    for every candidate contact. This diagonal approximation is linear in
    the ledger size, is zero for redundant pulls, and remains positive when
    the other state carries a newer cumulative origin row.
    """

    if (
        not is_mergeable_evidence_state(base)
        or not is_mergeable_evidence_state(other)
    ):
        return float("-inf")
    base_precision = base["evidence_precision"].detach().to(
        device="cpu", dtype=torch.float64
    )
    other_precision = other["evidence_precision"].detach().to(
        device="cpu", dtype=torch.float64
    )
    ridge = max(float(base["ridge"].reshape(-1)[0]), 1.0e-12)
    basis_dim = int(base["weight"].numel())
    before = torch.full((basis_dim,), ridge, dtype=torch.float64)
    if int(base_precision.shape[0]):
        before.add_(
            torch.diagonal(base_precision, dim1=1, dim2=2).sum(dim=0)
        )
    after = before.clone()
    base_rows = {
        int(key): (int(version), index)
        for index, (key, version) in enumerate(
            zip(base["evidence_keys"], base["evidence_versions"])
        )
    }
    other_diagonal = torch.diagonal(
        other_precision, dim1=1, dim2=2
    )
    for index, (raw_key, raw_version) in enumerate(
        zip(other["evidence_keys"], other["evidence_versions"])
    ):
        key = int(raw_key)
        version = int(raw_version)
        previous = base_rows.get(key)
        if previous is not None and version <= previous[0]:
            continue
        delta = other_diagonal[index]
        if previous is not None:
            delta = delta - torch.diagonal(
                base_precision[previous[1]], dim1=0, dim2=1
            )
        after.add_(delta.clamp_min(0.0))
    gain = torch.log(after.clamp_min(1.0e-12)).sub(
        torch.log(before.clamp_min(1.0e-12))
    ).sum()
    return float(max(0.0, float(gain)))


def mergeable_evidence_union_states(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the associative, commutative, idempotent evidence union."""

    if (
        not is_mergeable_evidence_state(left)
        or not is_mergeable_evidence_state(right)
    ):
        raise ValueError("both states must contain mergeable evidence")
    for name in (
        "mergeable_format_version",
        "projection",
        "phase",
        "ridge",
        "prior_value",
    ):
        if (
            tuple(left[name].shape) != tuple(right[name].shape)
            or not torch.allclose(
                left[name].to(torch.float64),
                right[name].to(torch.float64),
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise ValueError(
                f"mergeable predictor basis mismatch for {name}"
            )
    rows: dict[
        int,
        tuple[int, torch.Tensor, torch.Tensor, torch.Tensor],
    ] = {}
    for state in (left, right):
        for index, raw_key in enumerate(state["evidence_keys"]):
            key = int(raw_key)
            candidate = (
                int(state["evidence_versions"][index]),
                state["evidence_precision"][index],
                state["evidence_information"][index],
                state["evidence_mass"][index],
            )
            existing = rows.get(key)
            if existing is None or candidate[0] > existing[0]:
                rows[key] = candidate
            elif candidate[0] == existing[0]:
                if not all(
                    torch.equal(a, b)
                    for a, b in zip(candidate[1:], existing[1:])
                ):
                    raise ValueError(
                        "conflicting evidence for origin "
                        f"{key} at version {candidate[0]}"
                    )
    output = {
        name: value.detach().cpu().clone()
        for name, value in left.items()
    }
    keys = sorted(rows)
    basis_dim = int(left["weight"].numel())
    output["evidence_keys"] = torch.tensor(keys, dtype=torch.int64)
    output["evidence_versions"] = torch.tensor(
        [rows[key][0] for key in keys], dtype=torch.int64
    )
    output["evidence_precision"] = (
        torch.stack(
            [rows[key][1].detach().cpu() for key in keys]
        )
        if keys
        else torch.empty(
            (0, basis_dim, basis_dim),
            dtype=left["weight"].dtype,
        )
    )
    output["evidence_information"] = (
        torch.stack(
            [rows[key][2].detach().cpu() for key in keys]
        )
        if keys
        else torch.empty(
            (0, basis_dim), dtype=left["weight"].dtype
        )
    )
    output["evidence_mass"] = (
        torch.stack(
            [rows[key][3].detach().cpu() for key in keys]
        )
        if keys
        else torch.empty((0,), dtype=left["weight"].dtype)
    )
    precision = (
        torch.eye(basis_dim, dtype=left["weight"].dtype)
        * output["ridge"]
    )
    information = torch.zeros(
        basis_dim, dtype=left["weight"].dtype
    )
    information[0] = output["ridge"] * output["prior_value"]
    if keys:
        precision = (
            precision + output["evidence_precision"].sum(dim=0)
        )
        information = (
            information + output["evidence_information"].sum(dim=0)
        )
    output["weight"] = torch.linalg.solve(precision, information)
    return output


def make_rssi_predictor(
    arch: str = "tiny",
    *,
    input_dim: int = 4,
    include_time: bool = False,
    num_time_frequencies: int = 8,
    min_time_period: float = 2.0,
    max_time_period: float = 1000.0,
    time_unit: float = 1.0,
    time_encoding: str = "fourier",
    learned_time_dim: int = 16,
    learned_time_hidden_dim: int = 16,
    learned_time_scale: float = 1000.0,
    spatial_grid_points: int = 9,
    time_grid_points: int = 5,
    support_prior_strength: float = 0.0,
    mergeable_basis_dim: int = 192,
    mergeable_ridge: float = 1.0,
) -> nn.Module:
    """Create a propagation-loss predictor from TX/RX coordinates."""

    if int(input_dim) != 4 or bool(include_time):
        raise ValueError(
            "propagation-loss predictors require exactly four inputs: "
            "tx_x, tx_y, rx_x, rx_y"
        )

    kwargs = {
        "input_dim": int(input_dim),
        "include_time": bool(include_time),
        "num_time_frequencies": int(num_time_frequencies),
        "min_time_period": float(min_time_period),
        "max_time_period": float(max_time_period),
        "time_unit": float(time_unit),
        "time_encoding": str(time_encoding),
        "learned_time_dim": int(learned_time_dim),
        "learned_time_hidden_dim": int(learned_time_hidden_dim),
        "learned_time_scale": float(learned_time_scale),
    }
    key = str(arch or "tiny").strip().lower().replace("_", "-")
    if key in {
        "mergeable-rff",
        "mergeable-evidence",
        "evidence-ridge",
    }:
        return MergeableRFFRidgePredictor(
            **kwargs,
            basis_dim=int(mergeable_basis_dim),
            ridge=float(mergeable_ridge),
        )
    if key in {
        "mergeable-log-distance",
        "exact-log-distance",
        "sufficient-statistics-log-distance",
    }:
        return MergeableLogDistanceRidgePredictor(
            **kwargs,
            ridge=float(mergeable_ridge),
        )
    if key in {
        "local-support",
        "conservative-local-support",
        "conservative-grid",
        "local-grid",
    }:
        return ConservativeLocalSupportRSSIPredictor(
            **kwargs,
            spatial_grid_points=int(spatial_grid_points),
            time_grid_points=int(time_grid_points),
            support_prior_strength=float(support_prior_strength),
        )
    if key in {"micro", "single-64", "4-64-1"}:
        return MicroRSSIPredictor(**kwargs)
    if key in {"tiny", "4-64-64-1", "64"}:
        return TinyRSSIPredictor(**kwargs)
    if key in {
        "log-distance-only",
        "learned-log-distance",
        "distance-only-learned",
    }:
        return LogDistanceRSSIPredictor(**kwargs)
    if key in {
        "log-distance-residual",
        "distance-residual",
        "tiny-distance-residual",
    }:
        return LogDistanceResidualRSSIPredictor(**kwargs)
    if key.startswith("rbf-distance-residual-k"):
        count_text = key.removeprefix("rbf-distance-residual-k")
        if not count_text.isdigit():
            raise ValueError(f"invalid RBF support-vector count in {arch!r}")
        return LogDistanceResidualRSSIPredictor(
            **kwargs,
            gate_residual_with_support=True,
            support_vectors=int(count_text),
        )
    if key in {"censored-tiny", "two-head-tiny", "hurdle-tiny"}:
        return CensoredRSSIPredictor(
            **kwargs,
            hidden_dims=(64, 64),
        )
    if key in {
        "hard-censored-tiny",
        "hard-two-head-tiny",
        "hard-hurdle-tiny",
    }:
        return CensoredRSSIPredictor(
            **kwargs,
            hidden_dims=(64, 64),
            hard_decision=True,
        )
    if key in {
        "compact",
        "small-112",
        "4-112-112-112-1",
        "5-112-112-112-1",
        "112",
    }:
        return CompactRSSIPredictor(**kwargs)
    if key in {
        "small",
        "medium-small",
        "4-128-128-128-1",
        "128",
    }:
        return SmallRSSIPredictor(**kwargs)
    if key in {"censored-small", "two-head-small"}:
        return CensoredRSSIPredictor(
            **kwargs,
            hidden_dims=(128, 128, 128),
        )
    if key in {"hard-censored-small", "hard-two-head-small"}:
        return CensoredRSSIPredictor(
            **kwargs,
            hidden_dims=(128, 128, 128),
            hard_decision=True,
        )
    if key == "hurdle-small":
        return IndependentCensoredRSSIPredictor(
            **kwargs,
            hidden_dims=(128, 128, 128),
        )
    if key == "hard-hurdle-small":
        return IndependentCensoredRSSIPredictor(
            **kwargs,
            hidden_dims=(128, 128, 128),
            hard_decision=True,
        )
    if key in {"hard-ensemble-hurdle", "ensemble-hurdle"}:
        return EnsembleIndependentCensoredRSSIPredictor(
            **kwargs,
            hidden_dims=(64, 64),
            members=3,
        )
    if key in {"blocked-calibrated-ensemble", "hard-blockage-ensemble"}:
        return EnsembleIndependentCensoredRSSIPredictor(
            **kwargs,
            hidden_dims=(64, 64),
            members=3,
            bootstrap_keep_probability=0.95,
            classification_weight=1.0,
            feasibility_threshold=0.20,
        )
    if key in {
        "medium",
        "medium-192",
        "4-192-192-192-1",
        "192",
    }:
        return MediumRSSIPredictor(**kwargs)

    if key in {
        "dual",
        "large",
        "4-512-512-512-512-1",
        "512",
    }:
        return DualRSSIPredictor(**kwargs)
    raise ValueError(f"Unknown RSSI predictor architecture: {arch!r}")
