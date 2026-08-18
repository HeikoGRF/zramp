#!/usr/bin/env python3
"""Pretrain a sparse spatial grid codec for deterministic acquisition gain."""

from __future__ import annotations

import argparse
from concurrent.futures import Executor, ProcessPoolExecutor
import csv
from dataclasses import asdict
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.vq import kmeans2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.support_acquisition_pretraining.grid_gain import GRID_LAYOUT_REGULAR  # noqa: E402
from experiments.support_acquisition_pretraining.patch_grid_codec_model import (  # noqa: E402
    PatchGridCodec,
    PatchGridGainModel,
    normalized_reconstruction_loss,
)
from experiments.support_acquisition_pretraining.pretrain_grid_autoencoder_gain import (  # noqa: E402
    encode_batches,
    gain_metrics,
    prepare_batches,
    reconstruction_metrics,
)
from experiments.support_acquisition_pretraining.pretrain_shared_frame_gain import (  # noqa: E402
    UnitSquareConfig,
)
from experiments.support_acquisition_pretraining.pretrain_spatial_grid_gain import (  # noqa: E402
    GridBatch,
    PROFILE_KIND_CELL_TRAVERSAL,
    PROFILE_KIND_PLANE_ENVELOPE,
    balanced_regression_loss,
)
from experiments.support_acquisition_pretraining.pretrain_union_gain import atomic_json  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts/support_acquisition_pretraining/patch_grid_codec_v1"


@torch.no_grad()
def fit_patch_codebook(
    codec: PatchGridCodec,
    batches: list[GridBatch],
    *,
    device: torch.device,
    seed: int,
    maximum_codes: int,
    iterations: int,
) -> None:
    """Fit one shared codebook to nonempty learned patch vectors."""

    if codec.codebook_size < 2:
        return
    rng = np.random.default_rng(int(seed))
    collected: list[np.ndarray] = []
    order = rng.permutation(len(batches))
    for batch_index in order:
        profiles = batches[int(batch_index)].profiles
        profile_order = rng.permutation(len(profiles))
        for start in range(0, len(profile_order), 16):
            selected = profile_order[start : start + 16]
            raw = profiles[selected].to(device=device, dtype=torch.float32)
            transformed, _mass = codec.split_profile(raw)
            codes = codec.raw_patch_codes(transformed)
            active = torch.any(codes != 0.0, dim=2)
            values = codes[active].detach().cpu().numpy()
            if len(values):
                collected.append(values)
            if sum(len(item) for item in collected) >= int(maximum_codes):
                break
        if sum(len(item) for item in collected) >= int(maximum_codes):
            break
    values = np.concatenate(collected, axis=0).astype(np.float32, copy=False)
    if len(values) > int(maximum_codes):
        values = values[rng.choice(
            len(values), size=int(maximum_codes), replace=False
        )]
    clusters = codec.codebook_size - 1
    if len(values) < clusters:
        raise ValueError("not enough nonempty patches to fit the codebook")
    group_codebooks: list[np.ndarray] = []
    for group in range(codec.codebook_groups):
        start = group * codec.codebook_subdim
        stop = start + codec.codebook_subdim
        centroids, _labels = kmeans2(
            values[:, start:stop],
            clusters,
            iter=int(iterations),
            minit="++",
            missing="raise",
            check_finite=False,
            seed=rng,
        )
        group_codebooks.append(np.concatenate((
            np.zeros((1, codec.codebook_subdim), dtype=np.float32),
            np.asarray(centroids, dtype=np.float32),
        )))
    codebook = np.stack(group_codebooks)
    codec.set_codebook(torch.from_numpy(codebook))
    print(
        f"[PATCH-CODEC] fitted codebook={codec.codebook_size} "
        f"groups={codec.codebook_groups} "
        f"from nonempty-patches={len(values)}",
        flush=True,
    )


def self_test() -> None:
    codec = PatchGridCodec(latent_channels=2, hidden_dim=8)
    acquisition = PatchGridGainModel(
        patch_count=codec.patch_count, latent_channels=2, hidden_dim=8
    )
    profiles = torch.zeros(3, 300, 300)
    profiles[0, 10:25, 20:28] = 3.0
    profiles[1, 10:25, 20:28] = 8.0
    profiles[2, 200:220, 210:240] = 5.0
    advertisements = codec(profiles)
    assert advertisements.shape == (3, 1801)
    empty = codec(torch.zeros(1, 300, 300))
    assert bool(torch.all(empty == 0.0))
    decoded = codec.decode_advertisements(advertisements)
    assert decoded.shape == profiles.shape
    assert torch.allclose(
        decoded.sum(dim=(1, 2)), profiles.sum(dim=(1, 2)), rtol=1.0e-4
    )
    reconstruction, target, _mass = codec.reconstruct(profiles)
    loss = normalized_reconstruction_loss(reconstruction, target)
    prediction = acquisition(
        advertisements, torch.tensor([1, 2]), torch.tensor([0, 0])
    )
    assert prediction.shape == (2,)
    loss = loss + prediction.mean()
    loss.backward()
    assert bool(torch.isfinite(loss))


def predict_log_gain(codec: PatchGridCodec, batch: GridBatch) -> torch.Tensor:
    advertisements = codec(batch.profiles)
    candidates = codec.decode_advertisements(
        advertisements[batch.candidate_indices]
    )
    banks = batch.profiles[batch.bank_indices]
    gain = torch.relu(candidates - banks).sum(dim=(1, 2))
    gain = gain / banks.sum(dim=(1, 2)).clamp_min(1.0)
    return torch.log1p(gain)


@torch.no_grad()
def codec_gain_metrics(
    codec: PatchGridCodec,
    batches: list[GridBatch],
    *,
    device: torch.device,
) -> dict[str, float | int]:
    codec.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    active_patches: list[np.ndarray] = []
    top1 = 0
    group_count = 0
    regret = 0.0
    for prepared in batches:
        batch = prepared.to(device)
        advertisements = codec(batch.profiles)
        candidates = codec.decode_advertisements(
            advertisements[batch.candidate_indices]
        )
        banks = batch.profiles[batch.bank_indices]
        relative = torch.relu(candidates - banks).sum(dim=(1, 2))
        relative = relative / banks.sum(dim=(1, 2)).clamp_min(1.0)
        prediction = torch.log1p(relative).cpu().numpy()
        target = batch.targets.cpu().numpy()
        predictions.append(prediction)
        targets.append(target)
        active_patches.append(
            codec.active_patch_counts(advertisements).cpu().numpy()
        )
        groups = batch.group_indices.cpu().numpy()
        for group in np.unique(groups):
            mask = groups == group
            choice = int(np.argmax(prediction[mask]))
            values = target[mask]
            optimum = float(np.max(values))
            chosen = float(values[choice])
            top1 += int(chosen >= optimum - 1.0e-6)
            regret += optimum - chosen
            group_count += 1
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    active = np.concatenate(active_patches)
    result: dict[str, float | int] = {
        "examples": int(len(target)),
        "groups": int(group_count),
        "gain_log_mae": float(np.mean(np.abs(prediction - target))),
        "gain_log_rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "gain_prediction_bias_log": float(np.mean(prediction - target)),
        "top1_accuracy_tie_aware": float(top1 / max(group_count, 1)),
        "mean_selection_regret_log": float(regret / max(group_count, 1)),
        "positive_target_fraction": float(np.mean(target > 0.0)),
        "active_patches_mean": float(np.mean(active)),
        "active_patches_max": int(np.max(active)),
    }
    predicted_gain = np.expm1(prediction)
    true_gain = np.expm1(target)
    for percent in (2, 5, 10, 50, 100, 200, 400):
        threshold = float(percent) / 100.0
        predicted_pass = predicted_gain > threshold
        true_pass = true_gain > threshold
        hit = predicted_pass & true_pass
        result[f"threshold_{percent}_precision"] = float(
            int(np.sum(hit)) / max(int(np.sum(predicted_pass)), 1)
        )
        result[f"threshold_{percent}_recall"] = float(
            int(np.sum(hit)) / max(int(np.sum(true_pass)), 1)
        )
        result[f"threshold_{percent}_predicted_pass_fraction"] = float(
            np.mean(predicted_pass)
        )
        result[f"threshold_{percent}_true_pass_fraction"] = float(
            np.mean(true_pass)
        )
    return result


def combined_rmse(
    first: dict[str, float | int], second: dict[str, float | int]
) -> float:
    total = int(first["examples"]) + int(second["examples"])
    return math.sqrt((
        int(first["examples"]) * float(first["gain_log_rmse"]) ** 2
        + int(second["examples"]) * float(second["gain_log_rmse"]) ** 2
    ) / max(total, 1))


def save_bundle(
    output: Path,
    codec: PatchGridCodec,
    acquisition: PatchGridGainModel,
    *,
    args: argparse.Namespace,
    cfg: UnitSquareConfig,
    stress_cfg: UnitSquareConfig,
    phase: str,
    step: int,
    validation: dict[str, float | int],
    stress_validation: dict[str, float | int],
) -> None:
    torch.save({
        "format": (
            "synthetic_product_quantized_patch_grid_acquisition_bundle_v8"
            if codec.codebook_size and codec.codebook_groups > 1
            else "synthetic_quantized_patch_grid_acquisition_bundle_v7"
            if codec.codebook_size
            else "synthetic_sparse_patch_grid_acquisition_bundle_v6"
        ),
        "codec_state_dict": {
            key: value.detach().cpu() for key, value in codec.state_dict().items()
        },
        "acquisition_state_dict": {
            key: value.detach().cpu()
            for key, value in acquisition.state_dict().items()
        },
        "grid_resolution": 300,
        "grid_layout": GRID_LAYOUT_REGULAR,
        "support_profile_semantics": str(args.support_profile_kind),
        "patch_size": int(args.patch_size),
        "patches_per_axis": int(300 // args.patch_size),
        "latent_channels": int(args.latent_channels),
        "codebook_size": int(codec.codebook_size),
        "codebook_groups": int(codec.codebook_groups),
        "code_index_bits": int(math.ceil(math.log2(codec.codebook_size)))
        if codec.codebook_size else 0,
        "latent_dim": int(codec.latent_dim),
        "advertisement_dim": int(codec.advertisement_dim),
        "hidden_dim": int(args.hidden_dim),
        "acquisition_hidden_dim": int(args.acquisition_hidden_dim),
        "best_phase": str(phase),
        "best_step": int(step),
        "validation": validation,
        "stress_validation": stress_validation,
        "synthetic_config": asdict(cfg),
        "stress_config": asdict(stress_cfg),
        "advertisement": (
            "aligned one-byte patch-code indices, log1p total intensity, model id"
            if codec.codebook_size
            else "sparse nonzero aligned patch codes, log1p total intensity, model id"
        ),
        "acquisition": "aligned patchwise scalar relative-gain regressor",
        "decoder_deployed": False,
        "target": (
            "exact 300x300 relative intensity-envelope gain over traversed cells"
            if args.support_profile_kind == PROFILE_KIND_CELL_TRAVERSAL
            else "exact 300x300 relative intensity-envelope gain"
        ),
        "map_data_used": False,
        "radio_measurements_used": False,
    }, output / "bundle.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--autoencoder-state", type=Path)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--support-profile-kind",
        choices=(PROFILE_KIND_PLANE_ENVELOPE, PROFILE_KIND_CELL_TRAVERSAL),
        default=PROFILE_KIND_PLANE_ENVELOPE,
    )
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=0)
    parser.add_argument("--codebook-groups", type=int, default=1)
    parser.add_argument("--codebook-maximum-codes", type=int, default=50000)
    parser.add_argument("--codebook-iterations", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--acquisition-hidden-dim", type=int, default=96)
    parser.add_argument("--groups-per-batch", type=int, default=4)
    parser.add_argument("--candidates-per-bank", type=int, default=16)
    parser.add_argument("--training-cache-batches", type=int, default=96)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--stress-validation-batches", type=int, default=8)
    parser.add_argument("--ae-steps", type=int, default=3000)
    parser.add_argument("--ae-min-steps", type=int, default=1000)
    parser.add_argument("--ae-validation-every", type=int, default=100)
    parser.add_argument("--ae-patience", type=int, default=15)
    parser.add_argument("--ae-batch-size", type=int, default=16)
    parser.add_argument("--gain-steps", type=int, default=3000)
    parser.add_argument("--gain-min-steps", type=int, default=750)
    parser.add_argument("--gain-validation-every", type=int, default=100)
    parser.add_argument("--gain-patience", type=int, default=12)
    parser.add_argument("--gain-pairs-per-bin", type=int, default=16)
    parser.add_argument("--gain-natural-pairs", type=int, default=96)
    parser.add_argument("--ae-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gain-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--max-planes", type=int, default=512)
    parser.add_argument("--max-bank-size", type=int, default=24)
    parser.add_argument("--max-axes", type=int, default=96)
    parser.add_argument("--stress-max-planes", type=int, default=1024)
    parser.add_argument("--stress-max-bank-size", type=int, default=48)
    parser.add_argument("--stress-max-axes", type=int, default=160)
    parser.add_argument("--sample-count-min", type=int, default=64)
    parser.add_argument("--sample-count-max", type=int, default=1048576)
    parser.add_argument("--stress-sample-count-max", type=int, default=16777216)
    parser.add_argument("--target-workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("patch-grid codec self-test passed", flush=True)
        return 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    cfg = UnitSquareConfig(
        min_axes=6, max_axes=int(args.max_axes),
        min_planes=1, max_planes=int(args.max_planes),
        min_bank_size=1, max_bank_size=int(args.max_bank_size),
        candidates_per_bank=int(args.candidates_per_bank),
        grid_resolution=300, grid_layout=GRID_LAYOUT_REGULAR,
        redundant_fraction=.50, subset_fraction=.25, evolved_fraction=.20,
    )
    stress_cfg = UnitSquareConfig(
        min_axes=12, max_axes=max(int(args.stress_max_axes), int(args.max_axes)),
        min_planes=1, max_planes=max(int(args.stress_max_planes), int(args.max_planes)),
        min_bank_size=1,
        max_bank_size=max(int(args.stress_max_bank_size), int(args.max_bank_size)),
        candidates_per_bank=int(args.candidates_per_bank),
        grid_resolution=300, grid_layout=GRID_LAYOUT_REGULAR,
        redundant_fraction=.50, subset_fraction=.25, evolved_fraction=.20,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executor: Executor | None = (
        ProcessPoolExecutor(max_workers=int(args.target_workers))
        if int(args.target_workers) > 1 else None
    )
    started = time.monotonic()
    common = dict(
        groups=int(args.groups_per_batch),
        sample_count_min=int(args.sample_count_min),
        executor=executor,
        profile_kind=str(args.support_profile_kind),
    )
    training = prepare_batches(
        seed=int(args.seed) + 100000,
        count=int(args.training_cache_batches), cfg=cfg,
        sample_count_max=int(args.sample_count_max), label="training", **common
    )
    validation = prepare_batches(
        seed=int(args.seed) + 200000,
        count=int(args.validation_batches), cfg=cfg,
        sample_count_max=int(args.sample_count_max), label="validation", **common
    )
    stress_validation = prepare_batches(
        seed=int(args.seed) + 300000,
        count=int(args.stress_validation_batches), cfg=stress_cfg,
        sample_count_max=int(args.stress_sample_count_max),
        label="stress-validation", **common
    )
    codec = PatchGridCodec(
        patch_size=int(args.patch_size),
        latent_channels=int(args.latent_channels),
        hidden_dim=int(args.hidden_dim),
        codebook_size=int(args.codebook_size),
        codebook_groups=int(args.codebook_groups),
    ).to(device)
    if args.autoencoder_state is not None:
        codec.load_state_dict(torch.load(
            args.autoencoder_state.resolve(),
            map_location=device,
            weights_only=True,
        ), strict=False)
    optimizer = torch.optim.AdamW(
        codec.parameters(), lr=float(args.ae_learning_rate), weight_decay=1.0e-5
    )
    rng = np.random.default_rng(int(args.seed) + 400000)
    order = np.arange(len(training), dtype=np.int64)
    best_reconstruction = float("inf")
    best_ae_step = 0
    stale = 0
    ae_history: list[dict[str, float | int]] = []
    for step in range(1, int(args.ae_steps) + 1):
        offset = (step - 1) % len(order)
        if offset == 0:
            rng.shuffle(order)
        profiles = training[int(order[offset])].profiles
        selected = rng.choice(
            len(profiles), min(int(args.ae_batch_size), len(profiles)), replace=False
        )
        raw = profiles[selected].to(device=device, dtype=torch.float32)
        codec.train()
        reconstruction, target, _mass = codec.reconstruct(raw)
        loss = normalized_reconstruction_loss(reconstruction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % int(args.ae_validation_every) == 0:
            matched_reconstruction = reconstruction_metrics(
                codec, validation, device=device, microbatch=32
            )
            stress_reconstruction = reconstruction_metrics(
                codec, stress_validation, device=device, microbatch=32
            )
            score = math.sqrt(
                .5 * float(matched_reconstruction["relative_rmse"]) ** 2
                + .5 * float(stress_reconstruction["relative_rmse"]) ** 2
            )
            ae_history.append({
                "step": step, "training_loss": float(loss.detach()),
                "selection_reconstruction_rmse": score,
                "matched_relative_rmse": matched_reconstruction["relative_rmse"],
                "stress_relative_rmse": stress_reconstruction["relative_rmse"],
            })
            if score < best_reconstruction - 1.0e-5:
                best_reconstruction = score
                best_ae_step = step
                stale = 0
                torch.save(codec.state_dict(), output / "autoencoder.pt")
            else:
                stale += 1
            print(
                f"[PATCH-CODEC] ae step={step:05d}/{args.ae_steps} "
                f"loss={float(loss.detach()):.5f} rRMSE={score:.5f} "
                f"best={best_ae_step} stale={stale}/{args.ae_patience}",
                flush=True,
            )
            if step >= int(args.ae_min_steps) and stale >= int(args.ae_patience):
                break
    codec.load_state_dict(torch.load(
        output / "autoencoder.pt", map_location=device, weights_only=True
    ))
    fit_patch_codebook(
        codec,
        training,
        device=device,
        seed=int(args.seed) + 450000,
        maximum_codes=int(args.codebook_maximum_codes),
        iterations=int(args.codebook_iterations),
    )
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    encoded_training = encode_batches(
        codec, training, device=device, microbatch=32, label="training"
    )
    encoded_validation = encode_batches(
        codec, validation, device=device, microbatch=32, label="validation"
    )
    encoded_stress_validation = encode_batches(
        codec, stress_validation, device=device, microbatch=32,
        label="stress-validation"
    )
    acquisition = PatchGridGainModel(
        patch_count=codec.patch_count,
        latent_channels=int(args.latent_channels),
        hidden_dim=int(args.acquisition_hidden_dim),
    ).to(device)
    embedding_parts: list[torch.Tensor] = []
    candidate_parts: list[torch.Tensor] = []
    bank_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    embedding_offset = 0
    for batch in encoded_training:
        embedding_parts.append(batch.embeddings)
        candidate_parts.append(batch.candidate_indices + embedding_offset)
        bank_parts.append(batch.bank_indices + embedding_offset)
        target_parts.append(batch.targets)
        embedding_offset += len(batch.embeddings)
    training_embeddings = torch.cat(embedding_parts).to(device)
    training_candidates = torch.cat(candidate_parts).to(device)
    training_banks = torch.cat(bank_parts).to(device)
    training_targets = torch.cat(target_parts).to(device)
    gain_boundaries = np.asarray([
        0.0,
        math.log1p(0.02),
        math.log1p(0.10),
        math.log1p(0.50),
        math.log1p(1.00),
    ])
    target_values = training_targets.cpu().numpy()
    gain_bins = np.digitize(target_values, gain_boundaries, right=True)
    gain_bins = np.where(target_values <= 0.0, 0, gain_bins + 1)
    bin_members = [
        np.flatnonzero(gain_bins == index)
        for index in np.unique(gain_bins)
    ]
    best_gain = float("inf")
    optimizer = torch.optim.AdamW(
        acquisition.parameters(),
        lr=float(args.gain_learning_rate),
        weight_decay=1.0e-5,
    )
    best_gain_step = 0
    stale = 0
    gain_history: list[dict[str, float | int]] = []
    for step in range(1, int(args.gain_steps) + 1):
        selected_parts = [
            rng.choice(
                members,
                size=int(args.gain_pairs_per_bin),
                replace=len(members) < int(args.gain_pairs_per_bin),
            )
            for members in bin_members
        ]
        selected = np.concatenate(selected_parts)
        rng.shuffle(selected)
        natural = rng.choice(
            len(training_targets),
            size=int(args.gain_natural_pairs),
            replace=len(training_targets) < int(args.gain_natural_pairs),
        )
        all_selected = np.concatenate((natural, selected))
        selected_tensor = torch.as_tensor(
            all_selected, dtype=torch.long, device=device
        )
        acquisition.train()
        prediction = acquisition(
            training_embeddings,
            training_candidates[selected_tensor],
            training_banks[selected_tensor],
        )
        target = training_targets[selected_tensor]
        natural_count = len(natural)
        population_loss = (
            F.smooth_l1_loss(
                prediction[:natural_count],
                target[:natural_count],
                beta=0.02,
            )
            + 0.25 * F.mse_loss(
                prediction[:natural_count], target[:natural_count]
            )
        )
        stratified_loss = balanced_regression_loss(
            prediction[natural_count:], target[natural_count:]
        )
        loss = 0.5 * population_loss + 0.5 * stratified_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(acquisition.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % int(args.gain_validation_every) == 0:
            matched = gain_metrics(
                acquisition, encoded_validation, device=device
            )
            stress = gain_metrics(
                acquisition, encoded_stress_validation, device=device
            )
            score = combined_rmse(matched, stress)
            gain_history.append({
                "step": step, "training_loss": float(loss.detach()),
                "selection_rmse": score,
                **{f"matched_{key}": value for key, value in matched.items()},
                **{f"stress_{key}": value for key, value in stress.items()},
            })
            if score < best_gain - 1.0e-5:
                best_gain = score
                best_gain_step = step
                stale = 0
                save_bundle(
                    output, codec, acquisition,
                    args=args, cfg=cfg, stress_cfg=stress_cfg,
                    phase="aligned-patch-acquisition", step=step,
                    validation=matched, stress_validation=stress,
                )
            else:
                stale += 1
            print(
                f"[PATCH-CODEC] gain step={step:05d}/{args.gain_steps} "
                f"loss={float(loss.detach()):.5f} score={score:.5f} "
                f"p2={100*float(matched['threshold_2_precision']):.1f}/"
                f"{100*float(matched['threshold_2_recall']):.1f}% "
                f"p10={100*float(matched['threshold_10_precision']):.1f}/"
                f"{100*float(matched['threshold_10_recall']):.1f}% "
                f"stress-p10={100*float(stress['threshold_10_precision']):.1f}/"
                f"{100*float(stress['threshold_10_recall']):.1f}% "
                f"best={best_gain_step} stale={stale}/{args.gain_patience}",
                flush=True,
            )
            if step >= int(args.gain_min_steps) and stale >= int(args.gain_patience):
                break
    bundle = torch.load(output / "bundle.pt", map_location=device, weights_only=False)
    codec.load_state_dict(bundle["codec_state_dict"])
    acquisition.load_state_dict(bundle["acquisition_state_dict"])
    holdout = prepare_batches(
        seed=int(args.seed) + 500000,
        count=int(args.validation_batches), cfg=cfg,
        sample_count_max=int(args.sample_count_max), label="holdout", **common
    )
    stress_holdout = prepare_batches(
        seed=int(args.seed) + 600000,
        count=int(args.stress_validation_batches), cfg=stress_cfg,
        sample_count_max=int(args.stress_sample_count_max),
        label="stress-holdout", **common
    )
    encoded_holdout = encode_batches(
        codec, holdout, device=device, microbatch=32, label="holdout"
    )
    encoded_stress_holdout = encode_batches(
        codec, stress_holdout, device=device, microbatch=32,
        label="stress-holdout"
    )
    holdout_metrics = gain_metrics(
        acquisition, encoded_holdout, device=device
    )
    stress_holdout_metrics = gain_metrics(
        acquisition, encoded_stress_holdout, device=device
    )
    holdout_reconstruction = reconstruction_metrics(
        codec, holdout, device=device, microbatch=32
    )
    stress_holdout_reconstruction = reconstruction_metrics(
        codec, stress_holdout, device=device, microbatch=32
    )
    if executor is not None:
        executor.shutdown()
    for name, history in (("autoencoder_history.csv", ae_history), ("gain_history.csv", gain_history)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    atomic_json(output / "metrics.json", {
        "schema": "synthetic_sparse_patch_grid_acquisition_pretraining_v2",
        "status": "complete",
        "map_data_used": False,
        "radio_measurements_used": False,
        "best_autoencoder_step": int(best_ae_step),
        "best_gain_step": int(best_gain_step),
        "elapsed_seconds": float(time.monotonic() - started),
        "best_validation": bundle["validation"],
        "best_stress_validation": bundle["stress_validation"],
        "holdout": holdout_metrics,
        "stress_holdout": stress_holdout_metrics,
        "holdout_reconstruction": holdout_reconstruction,
        "stress_holdout_reconstruction": stress_holdout_reconstruction,
        "bundle": str(output / "bundle.pt"),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
    })
    print(
        f"[PATCH-CODEC] complete ae-best={best_ae_step} "
        f"gain-best={best_gain_step} saved={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
