"""Equal-finetuning comparison used by the legacy one-directional baseline."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class FinetunedPairResult:
    baseline_rmse: float
    merged_rmse: float
    num_samples: int
    fine_tune_batches: int


def _train_from_common_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    device: torch.device,
    lr: float,
    epochs: int,
    batch_size: int,
    random_seed: int,
) -> int:
    if batch is None or int(batch[0].shape[0]) == 0 or int(epochs) <= 0:
        return 0
    features = batch[0].to(device=device, dtype=torch.float32)
    targets = batch[1].to(device=device, dtype=torch.float32).reshape(-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    updates = 0
    model.train()
    size = int(features.shape[0])
    width = max(1, int(batch_size))
    for _ in range(int(epochs)):
        order = torch.randperm(size, generator=generator)
        for start in range(0, size, width):
            indices = order[start : start + width].to(device=features.device)
            prediction = model(features.index_select(0, indices)).reshape(-1)
            loss = torch.nn.functional.mse_loss(
                prediction, targets.index_select(0, indices)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
    return updates


def _rmse(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    metric_scale: float,
) -> tuple[float, int]:
    squared_error = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for features, targets in batches:
            x = features.to(device=device, dtype=torch.float32)
            y = targets.to(device=device, dtype=torch.float32).reshape(-1)
            prediction = model(x).reshape(-1)
            squared_error += float(torch.sum((prediction - y) ** 2).item())
            count += int(y.numel())
    value = 0.0 if count == 0 else math.sqrt(squared_error / float(count))
    return float(value * float(metric_scale)), count


def evaluate_finetuned_pair(
    *,
    model_factory: Callable[[], nn.Module],
    baseline_state: Mapping[str, torch.Tensor],
    merged_state: Mapping[str, torch.Tensor],
    local_batch: tuple[torch.Tensor, torch.Tensor] | None,
    evaluation_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    lr: float,
    epochs: int,
    batch_size: int,
    metric_scale: float,
    random_seed: int,
) -> FinetunedPairResult:
    """Fine-tune both initializations identically, then evaluate a common window."""

    baseline = model_factory().to(device)
    merged = model_factory().to(device)
    baseline.load_state_dict(copy.deepcopy(dict(baseline_state)))
    merged.load_state_dict(copy.deepcopy(dict(merged_state)))

    baseline_updates = _train_from_common_batch(
        baseline,
        local_batch,
        device=device,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        random_seed=random_seed,
    )
    merged_updates = _train_from_common_batch(
        merged,
        local_batch,
        device=device,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        random_seed=random_seed,
    )
    if baseline_updates != merged_updates:
        raise RuntimeError("common fine-tuning produced unequal update counts")

    baseline_rmse, baseline_count = _rmse(
        baseline,
        evaluation_batches,
        device=device,
        metric_scale=metric_scale,
    )
    merged_rmse, merged_count = _rmse(
        merged,
        evaluation_batches,
        device=device,
        metric_scale=metric_scale,
    )
    if baseline_count != merged_count:
        raise RuntimeError("common evaluation produced unequal sample counts")
    return FinetunedPairResult(
        baseline_rmse=baseline_rmse,
        merged_rmse=merged_rmse,
        num_samples=baseline_count,
        fine_tune_batches=baseline_updates,
    )
