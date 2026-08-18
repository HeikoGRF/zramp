"""
generate_urban_dataset.py — Offline urban ray-tracing dataset generator.

Generates a dataset of (feature, RSSI) pairs from a configurable Sionna
scene with randomised TX/RX positions and TX power levels.  The resulting
dataset can be used by pretrain_backbone_urban.py to warm‑start the MLP
backbone on *other* cities than Munich (which should be reserved for
evaluation).

Usage
-----
    # Example: pretrain on some non‑Munich scene
    python generate_urban_dataset.py \\
        --scene    other_city_name \\
        --bounds-x 0 200 \\
        --bounds-y -200 0 \\
        --output   urban_other_city.npz \\
        --n-steps  500 \\
        --n-tx     20  \\
        --n-rx     20

The script re-uses the same transmitter/receiver objects for every step,
only repositioning them, so the Sionna scene graph is built once.

Output NPZ keys
---------------
  features : float32 (N, 8)  – RadioFeatureEncoder output
  targets  : float32 (N,)    – RSSI in dBm
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

import sionna.rt as rt
from sionna.rt import PlanarArray, Transmitter, Receiver, PathSolver

sys.path.insert(0, str(Path(__file__).parent))
from sionna_baseline.radio_mlp_lora import MapConfig, RadioFeatureEncoder
from sionna_baseline.backbone_utils import (
    TX_POWER_MIN_DBM, TX_POWER_MAX_DBM, compute_rssi,
)

# ---------------------------------------------------------------------------
# Scene bounds (overridable via CLI)
# Defaults are chosen for a generic Berlin‑like downtown area for backbone
# pretraining; Munich is reserved for evaluation in the benchmarks.
# ---------------------------------------------------------------------------
DEFAULT_BOUNDS_X   = (0.0, 300.0)
DEFAULT_BOUNDS_Y   = (0.0, 300.0)
HEIGHT             = 1.5          # node height above ground (m)
OBS_RANGE_DEFAULT  = 250.0        # only keep links within this range (m)
MIN_DIST           = 2.0          # discard self-close links (m)
NOISE_FLOOR        = -150.0       # dBm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output",   default="urban_dataset.npz")
    p.add_argument("--n-steps",  type=int, default=500,
                   help="Number of batched ray-tracing calls")
    p.add_argument("--n-tx",     type=int, default=20,
                   help="TX nodes per step")
    p.add_argument("--n-rx",     type=int, default=20,
                   help="RX nodes per step")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--max-depth",type=int, default=3)
    p.add_argument("--spp",      type=int, default=100_000,
                   help="Sionna samples_per_src")
    # Scene selection: for *general* backbone pretraining we default to 'etoile'
    # (another urban scene shipped with Sionna RT) so that Munich is reserved
    # for evaluation only.
    p.add_argument(
        "--scene",
        type=str,
        default="etoile",
        help="Name of sionna.rt.scene.<name> to load (e.g. 'etoile', 'uma').",
    )
    # Bounds override (in metres, consistent with evaluation scripts)
    p.add_argument(
        "--bounds-x",
        type=float,
        nargs=2,
        metavar=("X_MIN", "X_MAX"),
        default=list(DEFAULT_BOUNDS_X),
        help="X coordinate range to sample TX/RX from.",
    )
    p.add_argument(
        "--bounds-y",
        type=float,
        nargs=2,
        metavar=("Y_MIN", "Y_MAX"),
        default=list(DEFAULT_BOUNDS_Y),
        help="Y coordinate range to sample TX/RX from.",
    )
    p.add_argument(
        "--obs-range",
        type=float,
        default=OBS_RANGE_DEFAULT,
        help="Maximum TX–RX distance (m) to keep as a valid link.",
    )
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    rng   = np.random.default_rng(args.seed)

    # Resolve scene object dynamically so we can pretrain on non‑Munich cities.
    if args.scene == "munich":
        scene_obj = rt.scene.munich
    elif args.scene == "etoile":
        scene_obj = rt.scene.etoile
    else:
        try:
            scene_obj = getattr(rt.scene, args.scene)
        except AttributeError as exc:
            raise SystemExit(
                f"Unknown scene '{args.scene}'. Please pass a valid name from "
                "sionna.rt.scene.*"
            ) from exc

    print(f"Loading scene '{args.scene}' …", flush=True)
    scene = rt.load_scene(scene_obj)
    scene.frequency = 3.5e9
    scene.tx_array  = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array  = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    # Add TX / RX nodes once; reposition each step
    for i in range(args.n_tx):
        scene.add(Transmitter(name=f"tx_{i}", position=[0.0, 0.0, HEIGHT]))
    for i in range(args.n_rx):
        scene.add(Receiver(name=f"rx_{i}", position=[0.0, 0.0, HEIGHT]))

    bounds_x = (float(args.bounds_x[0]), float(args.bounds_x[1]))
    bounds_y = (float(args.bounds_y[0]), float(args.bounds_y[1]))

    map_cfg = MapConfig(x_min=bounds_x[0], x_max=bounds_x[1],
                        y_min=bounds_y[0], y_max=bounds_y[1])
    encoder = RadioFeatureEncoder(map_cfg)
    solver  = PathSolver()

    gpus = tf.config.list_physical_devices("GPU")
    dev  = "/GPU:0" if gpus else "/CPU:0"
    print(f"Compute device: {gpus[0].name if gpus else 'CPU'}", flush=True)
    print(f"Running {args.n_steps} steps  "
          f"({args.n_tx} TX × {args.n_rx} RX per step) …", flush=True)

    feats_buf:  list[np.ndarray] = []
    target_buf: list[float]      = []
    n_accepted = 0

    for step in range(args.n_steps):
        # --- randomise positions ------------------------------------------------
        tx_pos = np.column_stack([
            rng.uniform(*bounds_x, args.n_tx),
            rng.uniform(*bounds_y, args.n_tx),
            np.full(args.n_tx, HEIGHT),
        ])
        rx_pos = np.column_stack([
            rng.uniform(*bounds_x, args.n_rx),
            rng.uniform(*bounds_y, args.n_rx),
            np.full(args.n_rx, HEIGHT),
        ])
        # each TX gets its own fixed power for this step (varies across steps)
        tx_pw = rng.uniform(TX_POWER_MIN_DBM, TX_POWER_MAX_DBM, args.n_tx)

        for i in range(args.n_tx):
            scene.get(f"tx_{i}").position = tx_pos[i]
        for i in range(args.n_rx):
            scene.get(f"rx_{i}").position = rx_pos[i]

        # --- ray-trace ----------------------------------------------------------
        with tf.device(dev):
            paths = solver(
                scene,
                max_depth=args.max_depth,
                samples_per_src=args.spp,
                specular_reflection=True,
                diffraction=True,
                edge_diffraction=False,
            )
        try:
            a_cplx   = np.array(paths.a[0]) + 1j * np.array(paths.a[1])
            a_cplx   = np.squeeze(a_cplx, axis=(1, 3))
            gain_lin = np.sum(np.abs(a_cplx) ** 2, axis=-1)   # (n_rx, n_tx)
        except Exception:
            gain_lin = np.zeros((args.n_rx, args.n_tx))

        # --- build samples ------------------------------------------------------
        for tx_i in range(args.n_tx):
            tx_xy = tx_pos[tx_i, :2]
            for rx_i in range(args.n_rx):
                rx_xy  = rx_pos[rx_i, :2]
                d2d    = float(np.linalg.norm(rx_xy - tx_xy))
                if d2d < MIN_DIST or d2d > args.obs_range:
                    continue
                rssi = compute_rssi(float(gain_lin[rx_i, tx_i]),
                                    float(tx_pw[tx_i]), NOISE_FLOOR)
                if rssi <= NOISE_FLOOR:
                    continue
                feat = encoder.encode(
                    tuple(tx_xy), tuple(rx_xy),
                    tx_power_dbm=float(tx_pw[tx_i]),
                )
                feats_buf.append(feat.numpy())
                target_buf.append(rssi)
                n_accepted += 1

        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d}/{args.n_steps}  "
                  f"samples so far: {n_accepted:8,}", flush=True)

    # --- save -------------------------------------------------------------------
    feats   = np.array(feats_buf,  dtype=np.float32)
    targets = np.array(target_buf, dtype=np.float32)
    np.savez_compressed(args.output, features=feats, targets=targets)
    print(f"\nSaved {n_accepted:,} samples → {args.output}")
    print(f"Feature matrix: {feats.shape}  "
          f"RSSI range: [{targets.min():.1f}, {targets.max():.1f}] dBm")


if __name__ == "__main__":
    main()
