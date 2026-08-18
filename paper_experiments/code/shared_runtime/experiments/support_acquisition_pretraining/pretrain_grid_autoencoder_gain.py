#!/usr/bin/env python3
"""Pretrain a full-grid autoencoder and encoding-only acquisition regressor."""

from __future__ import annotations

import argparse
from concurrent.futures import Executor, ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.support_acquisition_pretraining.grid_autoencoder_gain_model import (  # noqa: E402
    GridAutoencoder,
    GridEncodingGainModel,
    normalized_reconstruction_loss,
)
from experiments.support_acquisition_pretraining.grid_gain import (  # noqa: E402
    GRID_LAYOUT_REGULAR,
)
from experiments.support_acquisition_pretraining.pretrain_shared_frame_gain import (  # noqa: E402
    UnitSquareConfig,
)
from experiments.support_acquisition_pretraining.pretrain_spatial_grid_gain import (  # noqa: E402
    GridBatch,
    balanced_regression_loss,
    make_batch,
)
from experiments.support_acquisition_pretraining.pretrain_union_gain import (  # noqa: E402
    atomic_json,
)


DEFAULT_OUTPUT = ROOT / "artifacts/support_acquisition_pretraining/grid_autoencoder_gain_v1"


@dataclass
class EncodedBatch:
    embeddings: torch.Tensor
    candidate_indices: torch.Tensor
    bank_indices: torch.Tensor
    targets: torch.Tensor
    group_indices: torch.Tensor

    def to(self, device: torch.device) -> "EncodedBatch":
        return EncodedBatch(
            embeddings=self.embeddings.to(device),
            candidate_indices=self.candidate_indices.to(device),
            bank_indices=self.bank_indices.to(device),
            targets=self.targets.to(device),
            group_indices=self.group_indices.to(device),
        )


def self_test() -> None:
    autoencoder = GridAutoencoder(latent_dim=32, base_channels=2)
    acquisition = GridEncodingGainModel(latent_dim=32, hidden_dim=32)
    profiles = torch.zeros((3, 300, 300))
    profiles[0, 20:80, 30:35] = 3.0
    profiles[1, 20:80, 30:35] = 8.0
    profiles[2, 180:240, 200:208] = 5.0
    reconstructed, target, _mass = autoencoder.reconstruct(profiles)
    assert reconstructed.shape == target.shape == (3, 1, 300, 300)
    embedding = autoencoder(profiles)
    assert embedding.shape == (3, 33)
    prediction = acquisition(
        embedding, torch.tensor([1, 2]), torch.tensor([0, 0])
    )
    assert prediction.shape == (2,)
    loss = normalized_reconstruction_loss(reconstructed, target)
    loss = loss + prediction.mean()
    loss.backward()
    assert bool(torch.isfinite(loss))


def _log_uniform_int(
    rng: np.random.Generator, lower: int, upper: int
) -> int:
    if int(lower) == int(upper):
        return int(lower)
    return int(round(math.exp(rng.uniform(math.log(lower), math.log(upper)))))


def prepare_batches(
    *,
    seed: int,
    count: int,
    cfg: UnitSquareConfig,
    groups: int,
    sample_count_min: int,
    sample_count_max: int,
    executor: Executor | None,
    label: str,
    profile_kind: str = "plane-envelope",
) -> list[GridBatch]:
    rng = np.random.default_rng(int(seed))
    batches: list[GridBatch] = []
    progress = max(1, int(count) // 10)
    for index in range(int(count)):
        sample_count = _log_uniform_int(
            rng, int(sample_count_min), int(sample_count_max)
        )
        batches.append(make_batch(
            rng, cfg, int(groups), sample_count, executor,
            profile_kind=profile_kind,
        ))
        if (index + 1) % progress == 0 or index + 1 == int(count):
            print(
                f"[GRID-AE] prepared {label}={index + 1}/{count}",
                flush=True,
            )
    return batches


def reconstruction_metrics(
    model: GridAutoencoder,
    batches: list[GridBatch],
    *,
    device: torch.device,
    microbatch: int,
) -> dict[str, float | int]:
    model.eval()
    error = 0.0
    energy = 0.0
    absolute = 0.0
    values = 0
    profiles = 0
    with torch.no_grad():
        for batch in batches:
            for start in range(0, len(batch.profiles), int(microbatch)):
                raw = batch.profiles[start : start + int(microbatch)].to(
                    device=device, dtype=torch.float32
                )
                reconstruction, target, _mass = model.reconstruct(raw)
                difference = reconstruction - target
                error += float(torch.square(difference).sum())
                energy += float(torch.square(target).sum())
                absolute += float(torch.abs(difference).sum())
                values += int(target.numel())
                profiles += int(target.shape[0])
    return {
        "profiles": int(profiles),
        "relative_rmse": float(math.sqrt(error / max(energy, 1.0e-12))),
        "mean_absolute_error": float(absolute / max(values, 1)),
    }


def encode_batches(
    model: GridAutoencoder,
    batches: list[GridBatch],
    *,
    device: torch.device,
    microbatch: int,
    label: str,
) -> list[EncodedBatch]:
    model.eval()
    encoded: list[EncodedBatch] = []
    progress = max(1, len(batches) // 10)
    with torch.no_grad():
        for index, batch in enumerate(batches):
            parts = []
            for start in range(0, len(batch.profiles), int(microbatch)):
                raw = batch.profiles[start : start + int(microbatch)].to(
                    device=device, dtype=torch.float32
                )
                parts.append(model(raw).cpu())
            encoded.append(EncodedBatch(
                embeddings=torch.cat(parts),
                candidate_indices=batch.candidate_indices,
                bank_indices=batch.bank_indices,
                targets=batch.targets,
                group_indices=batch.group_indices,
            ))
            if label and (
                (index + 1) % progress == 0 or index + 1 == len(batches)
            ):
                print(
                    f"[GRID-AE] encoded {label}={index + 1}/{len(batches)}",
                    flush=True,
                )
    return encoded


def gain_metrics(
    model: GridEncodingGainModel,
    batches: list[EncodedBatch],
    *,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    top1 = 0
    groups_total = 0
    regret = 0.0
    with torch.no_grad():
        for prepared in batches:
            batch = prepared.to(device)
            prediction = model(
                batch.embeddings,
                batch.candidate_indices,
                batch.bank_indices,
            ).cpu().numpy()
            target = batch.targets.cpu().numpy()
            predictions.append(prediction)
            targets.append(target)
            groups = batch.group_indices.cpu().numpy()
            for group in np.unique(groups):
                mask = groups == group
                choice = int(np.argmax(prediction[mask]))
                true_values = target[mask]
                optimum = float(np.max(true_values))
                chosen = float(true_values[choice])
                top1 += int(chosen >= optimum - 1.0e-6)
                regret += optimum - chosen
                groups_total += 1
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    result: dict[str, float | int] = {
        "examples": int(len(target)),
        "groups": int(groups_total),
        "gain_log_mae": float(np.mean(np.abs(prediction - target))),
        "gain_log_rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "gain_prediction_bias_log": float(np.mean(prediction - target)),
        "top1_accuracy_tie_aware": float(top1 / max(groups_total, 1)),
        "mean_selection_regret_log": float(regret / max(groups_total, 1)),
        "positive_target_fraction": float(np.mean(target > 0.0)),
    }
    predicted_gain = np.expm1(prediction)
    true_gain = np.expm1(target)
    for percent in (2, 5, 10, 50, 100, 200, 400):
        threshold = float(percent) / 100.0
        predicted_pass = predicted_gain > threshold
        true_pass = true_gain > threshold
        true_positive = int(np.sum(predicted_pass & true_pass))
        result[f"threshold_{percent}_precision"] = float(
            true_positive / max(int(np.sum(predicted_pass)), 1)
        )
        result[f"threshold_{percent}_recall"] = float(
            true_positive / max(int(np.sum(true_pass)), 1)
        )
        result[f"threshold_{percent}_predicted_pass_fraction"] = float(
            np.mean(predicted_pass)
        )
        result[f"threshold_{percent}_true_pass_fraction"] = float(
            np.mean(true_pass)
        )
    return result


def full_grid_gain_metrics(
    encoder: GridAutoencoder,
    acquisition: GridEncodingGainModel,
    batches: list[GridBatch],
    *,
    device: torch.device,
    microbatch: int,
) -> dict[str, float | int]:
    """Evaluate gain directly from full profiles without caching gradients."""

    encoded = encode_batches(
        encoder,
        batches,
        device=device,
        microbatch=int(microbatch),
        label="",
    )
    return gain_metrics(acquisition, encoded, device=device)


def production_encoder_state(model: GridAutoencoder) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("from_latent.")
        and not key.startswith("decoder_conv.")
    }


def save_pretraining_state(
    output: Path,
    autoencoder: GridAutoencoder,
    *,
    step: int,
    metrics: dict[str, float | int],
) -> None:
    torch.save({
        "format": "synthetic_grid_autoencoder_pretraining_v1",
        "state_dict": autoencoder.state_dict(),
        "step": int(step),
        "metrics": metrics,
    }, output / "autoencoder.pt")


def save_bundle(
    output: Path,
    autoencoder: GridAutoencoder,
    acquisition: GridEncodingGainModel,
    *,
    args: argparse.Namespace,
    cfg: UnitSquareConfig,
    stress_cfg: UnitSquareConfig,
    step: int,
    validation: dict[str, float | int],
    stress_validation: dict[str, float | int],
) -> None:
    torch.save({
        "format": "synthetic_grid_autoencoder_gain_bundle_v4",
        "encoder_state_dict": production_encoder_state(autoencoder),
        "acquisition_state_dict": {
            key: value.detach().cpu()
            for key, value in acquisition.state_dict().items()
        },
        "grid_resolution": int(args.grid_resolution),
        "grid_layout": GRID_LAYOUT_REGULAR,
        "latent_dim": int(args.latent_dim),
        "advertisement_dim": int(args.latent_dim) + 1,
        "base_channels": int(args.base_channels),
        "hidden_dim": int(args.hidden_dim),
        "best_step": int(step),
        "synthetic_config": asdict(cfg),
        "stress_config": asdict(stress_cfg),
        "validation": validation,
        "stress_validation": stress_validation,
        "advertisement": "autoencoder latent, log1p total grid intensity, model id",
        "target": "exact 300x300 log1p relative intensity-envelope gain",
        "decoder_deployed": False,
        "map_data_used": False,
        "radio_measurements_used": False,
    }, output / "bundle.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--grid-resolution", type=int, default=300)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--groups-per-batch", type=int, default=4)
    parser.add_argument("--candidates-per-bank", type=int, default=16)
    parser.add_argument("--training-cache-batches", type=int, default=96)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--stress-validation-batches", type=int, default=8)
    parser.add_argument("--ae-steps", type=int, default=2500)
    parser.add_argument("--ae-min-steps", type=int, default=1000)
    parser.add_argument("--ae-validation-every", type=int, default=100)
    parser.add_argument("--ae-patience", type=int, default=12)
    parser.add_argument("--ae-batch-size", type=int, default=16)
    parser.add_argument("--gain-steps", type=int, default=6000)
    parser.add_argument("--gain-min-steps", type=int, default=1500)
    parser.add_argument("--gain-validation-every", type=int, default=100)
    parser.add_argument("--gain-patience", type=int, default=20)
    parser.add_argument("--ae-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gain-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--finetune-steps", type=int, default=2000)
    parser.add_argument("--finetune-min-steps", type=int, default=500)
    parser.add_argument("--finetune-validation-every", type=int, default=100)
    parser.add_argument("--finetune-patience", type=int, default=12)
    parser.add_argument("--finetune-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--max-planes", type=int, default=512)
    parser.add_argument("--max-bank-size", type=int, default=24)
    parser.add_argument("--max-axes", type=int, default=96)
    parser.add_argument("--stress-max-planes", type=int, default=1024)
    parser.add_argument("--stress-max-bank-size", type=int, default=48)
    parser.add_argument("--stress-max-axes", type=int, default=160)
    parser.add_argument("--sample-count-min", type=int, default=64)
    parser.add_argument("--sample-count-max", type=int, default=65536)
    parser.add_argument("--stress-sample-count-max", type=int, default=1048576)
    parser.add_argument("--target-workers", type=int, default=8)
    parser.add_argument("--encoder-microbatch", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("grid-autoencoder acquisition self-test passed", flush=True)
        return 0
    if int(args.grid_resolution) != 300:
        raise ValueError("this experiment uses the definitive 300x300 grid")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    cfg = UnitSquareConfig(
        min_axes=6,
        max_axes=int(args.max_axes),
        min_planes=1,
        max_planes=int(args.max_planes),
        min_bank_size=1,
        max_bank_size=int(args.max_bank_size),
        candidates_per_bank=int(args.candidates_per_bank),
        grid_resolution=300,
        grid_layout=GRID_LAYOUT_REGULAR,
        redundant_fraction=0.50,
        subset_fraction=0.25,
        evolved_fraction=0.20,
    )
    stress_cfg = UnitSquareConfig(
        min_axes=12,
        max_axes=max(int(args.stress_max_axes), int(args.max_axes)),
        min_planes=1,
        max_planes=max(int(args.stress_max_planes), int(args.max_planes)),
        min_bank_size=1,
        max_bank_size=max(
            int(args.stress_max_bank_size), int(args.max_bank_size)
        ),
        candidates_per_bank=int(args.candidates_per_bank),
        grid_resolution=300,
        grid_layout=GRID_LAYOUT_REGULAR,
        redundant_fraction=0.50,
        subset_fraction=0.25,
        evolved_fraction=0.20,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executor: Executor | None = (
        ProcessPoolExecutor(max_workers=int(args.target_workers))
        if int(args.target_workers) > 1 else None
    )
    started = time.monotonic()
    training = prepare_batches(
        seed=int(args.seed) + 100000,
        count=int(args.training_cache_batches),
        cfg=cfg,
        groups=int(args.groups_per_batch),
        sample_count_min=int(args.sample_count_min),
        sample_count_max=int(args.sample_count_max),
        executor=executor,
        label="training",
    )
    validation = prepare_batches(
        seed=int(args.seed) + 200000,
        count=int(args.validation_batches),
        cfg=cfg,
        groups=int(args.groups_per_batch),
        sample_count_min=int(args.sample_count_min),
        sample_count_max=int(args.sample_count_max),
        executor=executor,
        label="validation",
    )
    stress_validation = prepare_batches(
        seed=int(args.seed) + 300000,
        count=int(args.stress_validation_batches),
        cfg=stress_cfg,
        groups=int(args.groups_per_batch),
        sample_count_min=int(args.sample_count_min),
        sample_count_max=int(args.stress_sample_count_max),
        executor=executor,
        label="stress-validation",
    )
    autoencoder = GridAutoencoder(
        grid_resolution=300,
        latent_dim=int(args.latent_dim),
        base_channels=int(args.base_channels),
    ).to(device)
    ae_optimizer = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=float(args.ae_learning_rate),
        weight_decay=1.0e-5,
    )
    order_rng = np.random.default_rng(int(args.seed) + 400000)
    ae_order = np.arange(len(training), dtype=np.int64)
    best_ae = float("inf")
    best_ae_step = 0
    ae_stale = 0
    ae_history: list[dict[str, float | int]] = []
    for step in range(1, int(args.ae_steps) + 1):
        offset = (step - 1) % len(ae_order)
        if offset == 0:
            order_rng.shuffle(ae_order)
        profiles = training[int(ae_order[offset])].profiles
        selected = order_rng.choice(
            len(profiles),
            size=min(int(args.ae_batch_size), len(profiles)),
            replace=False,
        )
        raw = profiles[selected].to(device=device, dtype=torch.float32)
        autoencoder.train()
        reconstruction, target, _mass = autoencoder.reconstruct(raw)
        ae_loss = normalized_reconstruction_loss(reconstruction, target)
        ae_optimizer.zero_grad(set_to_none=True)
        ae_loss.backward()
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 5.0)
        ae_optimizer.step()
        if step == 1 or step % int(args.ae_validation_every) == 0:
            metrics = reconstruction_metrics(
                autoencoder,
                validation,
                device=device,
                microbatch=int(args.encoder_microbatch),
            )
            score = float(metrics["relative_rmse"])
            ae_history.append({
                "step": step,
                "training_loss": float(ae_loss.detach()),
                **metrics,
            })
            if score < best_ae - 1.0e-5:
                best_ae = score
                best_ae_step = step
                ae_stale = 0
                save_pretraining_state(
                    output, autoencoder, step=step, metrics=metrics
                )
            else:
                ae_stale += 1
            print(
                f"[GRID-AE] ae step={step:05d}/{args.ae_steps} "
                f"loss={float(ae_loss.detach()):.5f} val-rRMSE={score:.5f} "
                f"best={best_ae_step} stale={ae_stale}/{args.ae_patience}",
                flush=True,
            )
            if (
                step >= int(args.ae_min_steps)
                and ae_stale >= int(args.ae_patience)
            ):
                break
    saved_ae = torch.load(
        output / "autoencoder.pt", map_location=device, weights_only=False
    )
    autoencoder.load_state_dict(saved_ae["state_dict"])
    matched_reconstruction = reconstruction_metrics(
        autoencoder, validation, device=device,
        microbatch=int(args.encoder_microbatch)
    )
    stress_reconstruction = reconstruction_metrics(
        autoencoder, stress_validation, device=device,
        microbatch=int(args.encoder_microbatch)
    )
    encoded_training = encode_batches(
        autoencoder, training, device=device,
        microbatch=int(args.encoder_microbatch), label="training"
    )
    encoded_validation = encode_batches(
        autoencoder, validation, device=device,
        microbatch=int(args.encoder_microbatch), label="validation"
    )
    encoded_stress_validation = encode_batches(
        autoencoder, stress_validation, device=device,
        microbatch=int(args.encoder_microbatch), label="stress-validation"
    )
    acquisition = GridEncodingGainModel(
        latent_dim=int(args.latent_dim), hidden_dim=int(args.hidden_dim)
    ).to(device)
    gain_optimizer = torch.optim.AdamW(
        acquisition.parameters(),
        lr=float(args.gain_learning_rate),
        weight_decay=1.0e-5,
    )
    gain_order = np.arange(len(encoded_training), dtype=np.int64)
    best_gain = float("inf")
    best_gain_step = 0
    gain_stale = 0
    gain_history: list[dict[str, float | int]] = []
    for step in range(1, int(args.gain_steps) + 1):
        offset = (step - 1) % len(gain_order)
        if offset == 0:
            order_rng.shuffle(gain_order)
        batch = encoded_training[int(gain_order[offset])].to(device)
        acquisition.train()
        prediction = acquisition(
            batch.embeddings, batch.candidate_indices, batch.bank_indices
        )
        gain_loss = balanced_regression_loss(prediction, batch.targets)
        gain_optimizer.zero_grad(set_to_none=True)
        gain_loss.backward()
        torch.nn.utils.clip_grad_norm_(acquisition.parameters(), 5.0)
        gain_optimizer.step()
        if step == 1 or step % int(args.gain_validation_every) == 0:
            matched = gain_metrics(
                acquisition, encoded_validation, device=device
            )
            stress = gain_metrics(
                acquisition, encoded_stress_validation, device=device
            )
            example_total = int(matched["examples"]) + int(stress["examples"])
            score = math.sqrt((
                int(matched["examples"]) * float(matched["gain_log_rmse"]) ** 2
                + int(stress["examples"]) * float(stress["gain_log_rmse"]) ** 2
            ) / max(example_total, 1))
            gain_history.append({
                "step": step,
                "training_loss": float(gain_loss.detach()),
                "selection_rmse": float(score),
                **{f"matched_{key}": value for key, value in matched.items()},
                **{f"stress_{key}": value for key, value in stress.items()},
            })
            if score < best_gain - 1.0e-5:
                best_gain = score
                best_gain_step = step
                gain_stale = 0
                save_bundle(
                    output, autoencoder, acquisition,
                    args=args, cfg=cfg, stress_cfg=stress_cfg,
                    step=step, validation=matched,
                    stress_validation=stress,
                )
            else:
                gain_stale += 1
            print(
                f"[GRID-AE] gain step={step:05d}/{args.gain_steps} "
                f"loss={float(gain_loss.detach()):.5f} score={score:.5f} "
                f"matched-p10={100*float(matched['threshold_10_precision']):.1f}/"
                f"{100*float(matched['threshold_10_recall']):.1f}% "
                f"stress-p10={100*float(stress['threshold_10_precision']):.1f}/"
                f"{100*float(stress['threshold_10_recall']):.1f}% "
                f"top1={100*float(matched['top1_accuracy_tie_aware']):.1f}/"
                f"{100*float(stress['top1_accuracy_tie_aware']):.1f}% "
                f"best={best_gain_step} stale={gain_stale}/{args.gain_patience}",
                flush=True,
            )
            if (
                step >= int(args.gain_min_steps)
                and gain_stale >= int(args.gain_patience)
            ):
                break
    # Start end-to-end optimization from the best frozen-encoder checkpoint.
    # The decoder is excluded and there is only one objective: the same exact
    # relative grid gain used for post-pull validation.
    bundle = torch.load(
        output / "bundle.pt", map_location=device, weights_only=False
    )
    acquisition.load_state_dict(bundle["acquisition_state_dict"])
    finetune_parameters = [
        *autoencoder.encoder_conv.parameters(),
        *autoencoder.to_latent.parameters(),
        *acquisition.parameters(),
    ]
    finetune_optimizer = torch.optim.AdamW(
        finetune_parameters,
        lr=float(args.finetune_learning_rate),
        weight_decay=1.0e-5,
    )
    finetune_order = np.arange(len(training), dtype=np.int64)
    finetune_stale = 0
    best_finetune_step = 0
    finetune_history: list[dict[str, float | int]] = []
    for step in range(1, int(args.finetune_steps) + 1):
        offset = (step - 1) % len(finetune_order)
        if offset == 0:
            order_rng.shuffle(finetune_order)
        batch = training[int(finetune_order[offset])].to(device)
        autoencoder.train()
        acquisition.train()
        embeddings = autoencoder(batch.profiles)
        prediction = acquisition(
            embeddings, batch.candidate_indices, batch.bank_indices
        )
        finetune_loss = balanced_regression_loss(prediction, batch.targets)
        finetune_optimizer.zero_grad(set_to_none=True)
        finetune_loss.backward()
        torch.nn.utils.clip_grad_norm_(finetune_parameters, 5.0)
        finetune_optimizer.step()
        if step == 1 or step % int(args.finetune_validation_every) == 0:
            matched = full_grid_gain_metrics(
                autoencoder, acquisition, validation,
                device=device, microbatch=int(args.encoder_microbatch)
            )
            stress = full_grid_gain_metrics(
                autoencoder, acquisition, stress_validation,
                device=device, microbatch=int(args.encoder_microbatch)
            )
            example_total = int(matched["examples"]) + int(stress["examples"])
            score = math.sqrt((
                int(matched["examples"]) * float(matched["gain_log_rmse"]) ** 2
                + int(stress["examples"]) * float(stress["gain_log_rmse"]) ** 2
            ) / max(example_total, 1))
            finetune_history.append({
                "step": step,
                "training_loss": float(finetune_loss.detach()),
                "selection_rmse": float(score),
                **{f"matched_{key}": value for key, value in matched.items()},
                **{f"stress_{key}": value for key, value in stress.items()},
            })
            if score < best_gain - 1.0e-5:
                best_gain = score
                best_finetune_step = step
                finetune_stale = 0
                save_bundle(
                    output, autoencoder, acquisition,
                    args=args, cfg=cfg, stress_cfg=stress_cfg,
                    step=int(args.gain_steps) + step,
                    validation=matched, stress_validation=stress,
                )
            else:
                finetune_stale += 1
            print(
                f"[GRID-AE] finetune step={step:05d}/{args.finetune_steps} "
                f"loss={float(finetune_loss.detach()):.5f} score={score:.5f} "
                f"matched-p2={100*float(matched['threshold_2_precision']):.1f}/"
                f"{100*float(matched['threshold_2_recall']):.1f}% "
                f"matched-p10={100*float(matched['threshold_10_precision']):.1f}/"
                f"{100*float(matched['threshold_10_recall']):.1f}% "
                f"stress-p10={100*float(stress['threshold_10_precision']):.1f}/"
                f"{100*float(stress['threshold_10_recall']):.1f}% "
                f"best={best_finetune_step} stale={finetune_stale}/"
                f"{args.finetune_patience}",
                flush=True,
            )
            if (
                step >= int(args.finetune_min_steps)
                and finetune_stale >= int(args.finetune_patience)
            ):
                break
    bundle = torch.load(
        output / "bundle.pt", map_location=device, weights_only=False
    )
    autoencoder.load_state_dict(bundle["encoder_state_dict"], strict=False)
    acquisition.load_state_dict(bundle["acquisition_state_dict"])
    holdout = prepare_batches(
        seed=int(args.seed) + 500000,
        count=int(args.validation_batches), cfg=cfg,
        groups=int(args.groups_per_batch),
        sample_count_min=int(args.sample_count_min),
        sample_count_max=int(args.sample_count_max),
        executor=executor, label="independent-holdout",
    )
    stress_holdout = prepare_batches(
        seed=int(args.seed) + 600000,
        count=int(args.stress_validation_batches), cfg=stress_cfg,
        groups=int(args.groups_per_batch),
        sample_count_min=int(args.sample_count_min),
        sample_count_max=int(args.stress_sample_count_max),
        executor=executor, label="independent-stress-holdout",
    )
    holdout_reconstruction = reconstruction_metrics(
        autoencoder, holdout, device=device,
        microbatch=int(args.encoder_microbatch)
    )
    stress_holdout_reconstruction = reconstruction_metrics(
        autoencoder, stress_holdout, device=device,
        microbatch=int(args.encoder_microbatch)
    )
    encoded_holdout = encode_batches(
        autoencoder, holdout, device=device,
        microbatch=int(args.encoder_microbatch), label="independent-holdout"
    )
    encoded_stress_holdout = encode_batches(
        autoencoder, stress_holdout, device=device,
        microbatch=int(args.encoder_microbatch), label="independent-stress-holdout"
    )
    holdout_gain = gain_metrics(acquisition, encoded_holdout, device=device)
    stress_holdout_gain = gain_metrics(
        acquisition, encoded_stress_holdout, device=device
    )
    if executor is not None:
        executor.shutdown()
    with (output / "autoencoder_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ae_history[0]))
        writer.writeheader()
        writer.writerows(ae_history)
    with (output / "gain_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gain_history[0]))
        writer.writeheader()
        writer.writerows(gain_history)
    with (output / "finetune_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(finetune_history[0]))
        writer.writeheader()
        writer.writerows(finetune_history)
    atomic_json(output / "metrics.json", {
        "schema": "synthetic_grid_autoencoder_gain_pretraining_v1",
        "status": "complete",
        "map_data_used": False,
        "radio_measurements_used": False,
        "best_autoencoder_step": int(best_ae_step),
        "best_gain_step": int(best_gain_step),
        "best_finetune_step": int(best_finetune_step),
        "elapsed_seconds": float(time.monotonic() - started),
        "training_unique_pairs": int(
            len(encoded_training)
            * int(args.groups_per_batch)
            * int(args.candidates_per_bank)
        ),
        "matched_reconstruction": matched_reconstruction,
        "stress_reconstruction": stress_reconstruction,
        "holdout_reconstruction": holdout_reconstruction,
        "stress_holdout_reconstruction": stress_holdout_reconstruction,
        "best_validation": bundle["validation"],
        "best_stress_validation": bundle["stress_validation"],
        "holdout": holdout_gain,
        "stress_holdout": stress_holdout_gain,
        "bundle": str(output / "bundle.pt"),
        "configuration": {
            key: value for key, value in vars(args).items() if key != "output"
        },
    })
    print(
        f"[GRID-AE] complete ae-best={best_ae_step} "
        f"gain-best={best_gain_step} saved={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
