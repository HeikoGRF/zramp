"""
run_benchmarks.py — Radio Map Benchmark Comparison (GPU-ready SLURM script)

Evaluates four strategies on the same Sionna Munich ray-tracing simulation:
  1. FSPL Baseline         — pure physics, no learning
  2. Local-only LoRA       — lightweight adapter, zero sharing
  3. Greedy Flooding LoRA  — adapter shared whenever nodes are within share_range
  4. Centralized LoRA      — SAME architecture as devices, adapters-only, pooled data

Key design decisions
--------------------
* RSSI is measured up to obs_range=250 m (radio link budget).
* Adapters are only EXCHANGED within share_range=30 m (Bluetooth / WiFi-Direct).
  These two concepts are deliberately separated.
* All approaches TRAIN on every step (online learning).
  Evaluation metrics are computed on the last eval_steps steps only,
  so the cumulative RMSE curve monotonically decreases rather than turning
  upward when frozen adapters encounter new geometry.
* Centralized baseline uses the SAME architecture as devices and trains all
  weights centrally on pooled observations (fair comparison).

Why small buffer + periodic sharing?
-------------------------------------
With a large buffer (e.g. 10 000 entries) every adapter sees the node's
ENTIRE trajectory, effectively learning a global radio map of the whole
zone.  All adapters then converge to the same global solution, so merging
them does nothing useful — or worse, disrupts a well-converged local model.

With max_buf=500 (FIFO, ~26 steps of recent data per node) each adapter
represents the node's CURRENT local geometry.  When two nearby nodes
(<= share_range) exchange adapters, they both gain fresh knowledge from a
slightly different vantage point in the same area.  A node that just
entered a new sub-area recovers far faster by receiving an adapter from a
neighbour that has been there for 20+ steps (cold-start recovery benefit).

Sharing is gated behind share_every=5: nodes train locally for 5 steps
before each merge so the local gradient steps can settle before the
adapter is broadcast.  Without this cooldown, continuous merging prevents
local convergence and can make Greedy LoRA worse than Local LoRA.
"""

import math
import random
import time
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.path import Path as MplPath

import torch
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf
import sionna.rt as rt
from sionna.rt import PlanarArray, Transmitter, Receiver, PathSolver

sys.path.insert(0, str(Path(__file__).parent))
from sionna_baseline.radio_mlp_lora import (  # type: ignore
    MapConfig,
    RadioFeatureEncoder,
    RadioMLPWithLoRA,
)
from sionna_baseline.mobility import RandomWaypoint, get_anchor_zone  # type: ignore
from sionna_baseline.backbone_utils import (  # type: ignore
    sample_node_tx_powers,
    compute_rssi,
    load_or_pretrain_lora_backbone,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_building_grid(scene, bounds_x, bounds_y, res: float = 1.0) -> np.ndarray:
    xs = np.arange(bounds_x[0], bounds_x[1] + res, res)
    ys = np.arange(bounds_y[0], bounds_y[1] + res, res)
    xx, yy = np.meshgrid(xs, ys)
    points = np.c_[xx.ravel(), yy.ravel()]
    mask = np.zeros(len(points), dtype=bool)
    for _, obj in scene.objects.items():
        mesh = obj.mi_mesh
        try:
            vertices = np.array(mesh.vertex_positions_buffer()).reshape(-1, 3)
            faces = np.array(mesh.faces_buffer()).reshape(-1, 3)
        except Exception:
            continue
        verts2d = vertices[:, :2]
        for face in faces:
            poly = verts2d[face]
            p = MplPath(poly)
            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            in_bbox = (
                (points[:, 0] >= xmin - res) & (points[:, 0] <= xmax + res)
                & (points[:, 1] >= ymin - res) & (points[:, 1] <= ymax + res)
            )
            if np.any(in_bbox):
                idx = np.where(in_bbox)[0]
                mask[idx[in_bbox[idx] if False else p.contains_points(points[idx])]] = True
    return mask.reshape(xx.shape)


def _build_grid_safe(scene, bounds_x, bounds_y, res=1.0):
    """Wrapper that handles the contains_points logic correctly."""
    xs = np.arange(bounds_x[0], bounds_x[1] + res, res)
    ys = np.arange(bounds_y[0], bounds_y[1] + res, res)
    xx, yy = np.meshgrid(xs, ys)
    points = np.c_[xx.ravel(), yy.ravel()]
    mask = np.zeros(len(points), dtype=bool)
    for _, obj in scene.objects.items():
        mesh = obj.mi_mesh
        try:
            vertices = np.array(mesh.vertex_positions_buffer()).reshape(-1, 3)
            faces = np.array(mesh.faces_buffer()).reshape(-1, 3)
        except Exception:
            continue
        verts2d = vertices[:, :2]
        for face in faces:
            poly = verts2d[face]
            p = MplPath(poly)
            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            in_bbox = (
                (points[:, 0] >= xmin - res) & (points[:, 0] <= xmax + res)
                & (points[:, 1] >= ymin - res) & (points[:, 1] <= ymax + res)
            )
            if np.any(in_bbox):
                idx = np.where(in_bbox)[0]
                inside = p.contains_points(points[idx])
                mask[idx[inside]] = True
    return mask.reshape(xx.shape)


def count_building_cells(tx_xy, rx_xy, grid_mask, bounds_x, bounds_y, res=1.0) -> int:
    dist = np.linalg.norm(rx_xy - tx_xy)
    if dist < 1e-6:
        return 0
    n_samp = max(2, int(np.ceil(dist / (res / 2.0))))
    ts = np.linspace(0.0, 1.0, n_samp)
    pts = tx_xy[None, :] + ts[:, None] * (rx_xy - tx_xy)[None, :]
    idx_x = np.round((pts[:, 0] - bounds_x[0]) / res).astype(int)
    idx_y = np.round((pts[:, 1] - bounds_y[0]) / res).astype(int)
    valid = (
        (idx_x >= 0) & (idx_x < grid_mask.shape[1])
        & (idx_y >= 0) & (idx_y < grid_mask.shape[0])
    )
    unique = set(zip(idx_y[valid], idx_x[valid]))
    return sum(1 for y, x in unique if grid_mask[y, x])


def compute_fspl(d: float, fspl_const: float,
                 tx_power_dbm: float = 0.0, eps: float = 1e-6) -> float:
    """FSPL prediction: P_tx + path-loss (dBm)."""
    return tx_power_dbm - 20.0 * math.log10(d + eps) + fspl_const


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Centralized LoRA baseline
# ---------------------------------------------------------------------------

def _freeze_base_train_adapters_only(model: nn.Module) -> None:
    """Centralized LoRA baseline trains adapters only (like devices)."""
    for name, p in model.named_parameters():
        if name.startswith(("fc1.", "fc2.", "fc3.")):
            p.requires_grad = False
        else:
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Shared replay utilities
# ---------------------------------------------------------------------------

def _cum_rmse(sse: float, n: int) -> float:
    return math.sqrt(sse / n) if n > 0 else 0.0


def _metrics(preds, truths):
    p, t = np.array(preds), np.array(truths)
    return float(np.sqrt(np.mean((p - t) ** 2))), float(np.mean(np.abs(p - t)))


def _fit_lora(model, opt, buf, loss_fn, noise_floor, min_samples,
              max_train, grad_steps, device):
    unc = [(f, r) for f, r in buf if r != noise_floor]
    if len(unc) < min_samples:
        return None
    samp = random.sample(unc, min(len(unc), max_train))
    x = torch.stack([f for f, _ in samp]).to(device)
    y = torch.tensor([r for _, r in samp], dtype=torch.float32).to(device)
    model.train()
    for _ in range(grad_steps):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        resid = y - model(x)
    return float(torch.clamp(resid.std(), min=3.0).item())


def _fit_large(model, opt, buf, loss_fn, noise_floor, min_samples,
               max_train, grad_steps, device):
    unc = [(f, r) for f, r in buf if r != noise_floor]
    if len(unc) < min_samples:
        return
    samp = random.sample(unc, min(len(unc), max_train))
    x = torch.stack([f for f, _ in samp]).to(device)
    y = torch.tensor([r for _, r in samp], dtype=torch.float32).to(device)
    model.train()
    for _ in range(grad_steps):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    model.eval()


def _pred_lora(model, feat, sigma, n_unc, device):
    model.eval()
    with torch.no_grad():
        mean = float(model(feat.unsqueeze(0).to(device)).item())
    inflate = 1.0 + 1.0 / max(1.0, math.sqrt(max(1.0, float(n_unc))))
    return mean, float(max(3.0, sigma * inflate))


def _pred_large(model, feat, device):
    model.eval()
    with torch.no_grad():
        return float(model(feat.unsqueeze(0).to(device)).item())


def _adapter_exchange_bytes(model) -> int:
    """2× because both nodes send their full adapter before computing the merge."""
    return sum(p.numel() for p in model.adapter_parameters()) * 4 * 2


_OBS_UPLOAD_BYTES = (8 + 1) * 4   # 8 features + 1 RSSI, float32


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmarks(results_dir: Path) -> None:
    set_seeds(42)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Configuration ───────────────────────────────────────────────────
    num_nodes   = 20
    # Step 7: train continuously; compute eval over last N steps
    num_steps   = 100
    eval_steps  = 20

    dt          = 1.0
    obs_range   = 250.0   # max distance for RSSI measurement (radio link)
    share_range = 30.0    # max distance for adapter exchange (Bluetooth / WiFi-Direct)
    zone_size   = 100
    bounds_x    = (0, 99)
    bounds_y    = (-199, -100)
    freq_hz     = 3.5e9
    c_light     = 3e8
    fspl_const  = -20.0 * math.log10(4.0 * math.pi * freq_hz / c_light)
    noise_floor = -150.0

    lr_lora     = 1e-2
    lr_large    = 1e-3
    grad_steps_lora  = 20
    grad_steps_large = 50   # larger model needs more updates per step
    min_samples = 8
    # FIFO sliding window: keep only the most recent max_buf observations.
    # This keeps each adapter specialised to the node's CURRENT local area.
    # With ~19 obs/step/node, 500 entries ≈ last 26 steps ≈ 50–80 m of travel.
    max_buf     = 500
    max_train   = 500       # train on the full recent buffer each step
    # Only exchange adapters every share_every steps so local gradient descent
    # can settle between merges (prevents "merge churn" that worsens convergence).
    share_every = 5

    device = torch.device("cpu")

    map_cfg = MapConfig(
        x_min=bounds_x[0], x_max=bounds_x[1],
        y_min=bounds_y[0], y_max=bounds_y[1],
    )
    encoder = RadioFeatureEncoder(map_cfg)

    print(f"FSPL constant at {freq_hz/1e9:.2f} GHz: {fspl_const:.2f} dB")
    print(f"Steps: {num_steps}  (train all, eval on last {eval_steps})")
    print(f"Observation range: {obs_range} m  |  Sharing range: {share_range} m")

    # ── Scene & mobility ─────────────────────────────────────────────────
    print("Loading Munich scene …", flush=True)
    scene = rt.load_scene(rt.scene.munich)
    scene.frequency = freq_hz
    scene.tx_array  = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array  = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    nodes: dict[int, np.ndarray] = {}
    for i in range(num_nodes):
        nodes[i] = np.array([
            np.random.uniform(*bounds_x),
            np.random.uniform(*bounds_y),
            1.5,
        ])
    for i in range(num_nodes):
        scene.add(Transmitter(name=f"tx_{i}", position=nodes[i]))
        scene.add(Receiver(name=f"rx_{i}", position=nodes[i]))

    # Each node has a fixed TX power for the full run (heterogeneous hardware)
    node_tx_power_dbm: dict[int, float] = sample_node_tx_powers(num_nodes)

    mobility = RandomWaypoint(
        bounds_x=bounds_x, bounds_y=bounds_y, velocity=(1.0, 3.0), pause_time=2.0
    )

    gpus = tf.config.list_physical_devices("GPU")
    compute_device = "/GPU:0" if gpus else "/CPU:0"
    print(f"Compute device: {gpus[0].name if gpus else 'CPU'}", flush=True)

    print("Building 2-D building grid …", flush=True)
    grid_mask = _build_grid_safe(scene, bounds_x, bounds_y, res=1.0)
    print(f"Building grid: {np.sum(grid_mask)} occupied cells", flush=True)

    # ── Collect simulation data once ────────────────────────────────────
    solver = PathSolver()
    simulation_data: list[dict] = []
    t0 = time.time()

    for step in range(num_steps):
        mobility.step(nodes, dt)
        for i in range(num_nodes):
            scene.get(f"tx_{i}").position = nodes[i]
            scene.get(f"rx_{i}").position = nodes[i]

        with tf.device(compute_device):
            paths = solver(
                scene, max_depth=3, samples_per_src=int(1e5),
                specular_reflection=True, diffraction=True, edge_diffraction=False,
            )
        try:
            a_cplx = np.array(paths.a[0]) + 1j * np.array(paths.a[1])
            a_cplx = np.squeeze(a_cplx, axis=(1, 3))
            power  = np.sum(np.abs(a_cplx) ** 2, axis=-1)
        except Exception:
            power = np.zeros((num_nodes, num_nodes))

        obs: list[dict] = []
        share_contacts: set[tuple[int, int]] = set()   # within share_range → adapter exchange

        for tx_i in range(num_nodes):
            for rx_i in range(num_nodes):
                if tx_i == rx_i:
                    continue
                ptx, prx = nodes[tx_i], nodes[rx_i]
                d2d = float(np.linalg.norm(ptx[:2] - prx[:2]))
                if d2d > obs_range:
                    continue
                ztx = get_anchor_zone(ptx[0], ptx[1], square_size=zone_size)
                zrx = get_anchor_zone(prx[0], prx[1], square_size=zone_size)
                if ztx != zrx:
                    continue

                tx_xy = ptx[:2].copy()
                rx_xy = prx[:2].copy()
                pwr   = float(power[rx_i, tx_i])
                rssi  = compute_rssi(pwr, node_tx_power_dbm[tx_i], noise_floor)
                nc    = count_building_cells(tx_xy, rx_xy, grid_mask, bounds_x, bounds_y)
                d     = float(np.linalg.norm(rx_xy - tx_xy))
                feat  = encoder.encode(tuple(tx_xy), tuple(rx_xy),
                                       tx_power_dbm=node_tx_power_dbm[tx_i],
                                       has_obstacle=(nc > 0))

                obs.append({
                    "tx_i": tx_i, "rx_i": rx_i,
                    "tx_xy": tx_xy, "rx_xy": rx_xy,
                    "rssi": rssi, "nc": nc, "d": d,
                    "fspl": compute_fspl(d, fspl_const,
                                        tx_power_dbm=node_tx_power_dbm[tx_i]),
                    "feat": feat,
                })

                # sharing only possible within Bluetooth/WiFi-Direct range
                if d2d <= share_range:
                    share_contacts.add((min(tx_i, rx_i), max(tx_i, rx_i)))

        simulation_data.append({
            "step": step,
            "obs": obs,
            "share_contacts": share_contacts,
        })

        if (step + 1) % 40 == 0:
            total_obs = sum(len(s["obs"]) for s in simulation_data)
            print(
                f"  Step {step+1:3d}/{num_steps}  "
                f"obs={total_obs:6d}  "
                f"elapsed={time.time()-t0:5.1f}s",
                flush=True,
            )

    total_obs = sum(len(s["obs"]) for s in simulation_data)
    print(
        f"\nData collection done in {time.time()-t0:.1f}s  |  "
        f"total observations: {total_obs:,}",
        flush=True,
    )

    # ── Pre-train / load shared LoRA backbone ────────────────────────────
    # Tries backbone_urban.pt (urban ray-tracing); falls back to FSPL pretraining.
    base_state, eff_fspl = load_or_pretrain_lora_backbone(
        encoder, fspl_const,
        backbone_path="backbone_etoile.pt",
        hidden_sizes=(128, 128), rank=16, alpha=1.0,
        n_samples=50_000, epochs=200, lr=1e-3,
    )

    # ── Centralized LoRA baseline: same model as devices ───────────────
    print("Initializing centralized baseline (centralized LoRA; adapters-only) …", flush=True)
    central_model = RadioMLPWithLoRA(
        input_dim=8, hidden_sizes=(128, 128), rank=16, alpha=1.0, fspl_const=eff_fspl
    )
    central_model.load_base_state(base_state)
    _freeze_base_train_adapters_only(central_model)
    central_model.to(device)
    central_model.eval()
    print("Centralized baseline initialized.", flush=True)

    def new_lora_adapter() -> RadioMLPWithLoRA:
        m = RadioMLPWithLoRA(
            input_dim=8, hidden_sizes=(128, 128), rank=16, alpha=1.0, fspl_const=eff_fspl
        )
        m.load_base_state(base_state)
        m.to(device)
        return m

    # ====================================================================
    # Approach 1 — FSPL Baseline
    # ====================================================================
    def run_fspl(sim_data):
        print("Running Approach 1: FSPL Baseline …", flush=True)
        cum_sse = cum_n = 0.0
        curve: list[float] = []
        bytes_curve: list[float] = []
        eval_p, eval_t = [], []
        for sd in sim_data:
            sp, st = [], []
            for o in sd["obs"]:
                sp.append(o["fspl"]); st.append(o["rssi"])
                if sd["step"] >= num_steps - eval_steps:
                    eval_p.append(o["fspl"]); eval_t.append(o["rssi"])
            if st:
                arr_p, arr_t = np.array(sp), np.array(st)
                cum_sse += float(np.sum((arr_p - arr_t) ** 2))
                cum_n   += len(st)
            curve.append(_cum_rmse(cum_sse, int(cum_n)))
            bytes_curve.append(0.0)
        rmse, mae = _metrics(eval_p, eval_t)
        print(f"  → RMSE: {rmse:.2f} dB  MAE: {mae:.2f} dB  comm: 0 B")
        return dict(name="FSPL\nBaseline", cum_rmse=curve, eval_rmse=rmse, eval_mae=mae,
                    adapter_kb=0.0, shareable=True, bytes_curve=bytes_curve, total_bytes=0)

    # ====================================================================
    # Approach 2 — Local-only LoRA  (no adapter sharing)
    # ====================================================================
    def run_local_lora(sim_data):
        print("Running Approach 2: Local-only LoRA (no sharing) …", flush=True)
        loss_fn  = nn.MSELoss()
        adapters: dict[int, RadioMLPWithLoRA] = {}
        opts:     dict[int, optim.Optimizer]  = {}
        sigmas:   dict[int, float]            = {}
        bufs:     dict[int, list]             = {}

        def _get(rx):
            if rx not in adapters:
                adapters[rx] = new_lora_adapter()
                opts[rx]     = optim.Adam(adapters[rx].adapter_parameters(),
                                          lr=lr_lora, weight_decay=1e-4)
                sigmas[rx]   = 25.0
                bufs[rx]     = []
            return adapters[rx]

        cum_sse = cum_n = 0.0
        curve, bytes_curve, eval_p, eval_t = [], [], [], []

        for sd in sim_data:
            step = sd["step"]
            sp, st = [], []
            for o in sd["obs"]:
                rx = o["rx_i"]; _get(rx)
                n_unc = sum(1 for _, r in bufs[rx] if r != noise_floor)
                pred, _ = _pred_lora(adapters[rx], o["feat"], sigmas[rx], n_unc, device)
                sp.append(pred); st.append(o["rssi"])
                if step >= num_steps - eval_steps:
                    eval_p.append(pred); eval_t.append(o["rssi"])
                # FIFO: drop oldest entry so the buffer stays local/recent
                bufs[rx].append((o["feat"].detach().clone(), o["rssi"]))
                bufs[rx] = bufs[rx][-max_buf:]
            # train after every step
            for rx in adapters:
                s = _fit_lora(adapters[rx], opts[rx], bufs[rx], loss_fn,
                              noise_floor, min_samples, max_train, grad_steps_lora, device)
                if s is not None:
                    sigmas[rx] = s
            if st:
                arr_p, arr_t = np.array(sp), np.array(st)
                cum_sse += float(np.sum((arr_p - arr_t) ** 2))
                cum_n   += len(st)
            curve.append(_cum_rmse(cum_sse, int(cum_n)))
            bytes_curve.append(0.0)

        rmse, mae = _metrics(eval_p, eval_t)
        n_adapter = sum(p.numel() for p in next(iter(adapters.values())).adapter_parameters())
        print(f"  → RMSE: {rmse:.2f} dB  MAE: {mae:.2f} dB  comm: 0 B  "
              f"adapter: {n_adapter*4/1024:.1f} KB")
        return dict(name="Local LoRA\n(no sharing)", cum_rmse=curve,
                    eval_rmse=rmse, eval_mae=mae,
                    adapter_kb=n_adapter * 4 / 1024, shareable=True,
                    bytes_curve=bytes_curve, total_bytes=0)

    # ====================================================================
    # Approach 3 — Greedy Flooding LoRA  (ours)
    #   Adapters exchanged whenever two nodes come within share_range (30 m)
    # ====================================================================
    def run_greedy_lora(sim_data):
        print("Running Approach 3: Greedy Flooding LoRA (ours) …", flush=True)
        loss_fn  = nn.MSELoss()
        adapters: dict[int, RadioMLPWithLoRA] = {}
        opts:     dict[int, optim.Optimizer]  = {}
        sigmas:   dict[int, float]            = {}
        bufs:     dict[int, list]             = {}

        def _get(rx):
            if rx not in adapters:
                adapters[rx] = new_lora_adapter()
                opts[rx]     = optim.Adam(adapters[rx].adapter_parameters(),
                                          lr=lr_lora, weight_decay=1e-4)
                sigmas[rx]   = 25.0
                bufs[rx]     = []
            return adapters[rx]

        cum_sse = cum_n = cum_bytes = 0.0
        curve, bytes_curve, eval_p, eval_t = [], [], [], []

        for sd in sim_data:
            step = sd["step"]
            sp, st, step_bytes = [], [], 0

            for o in sd["obs"]:
                rx = o["rx_i"]; _get(rx)
                n_unc = sum(1 for _, r in bufs[rx] if r != noise_floor)
                pred, _ = _pred_lora(adapters[rx], o["feat"], sigmas[rx], n_unc, device)
                sp.append(pred); st.append(o["rssi"])
                if step >= num_steps - eval_steps:
                    eval_p.append(pred); eval_t.append(o["rssi"])
                # FIFO: drop oldest entry so the buffer stays local/recent
                bufs[rx].append((o["feat"].detach().clone(), o["rssi"]))
                bufs[rx] = bufs[rx][-max_buf:]

            # local training (every step)
            for rx in adapters:
                s = _fit_lora(adapters[rx], opts[rx], bufs[rx], loss_fn,
                              noise_floor, min_samples, max_train, grad_steps_lora, device)
                if s is not None:
                    sigmas[rx] = s

            # weighted federated averaging — only every share_every steps
            # so local gradient descent can settle between merges
            if step % share_every != 0:
                if st:
                    arr_p, arr_t = np.array(sp), np.array(st)
                    cum_sse += float(np.sum((arr_p - arr_t) ** 2))
                    cum_n   += len(st)
                curve.append(_cum_rmse(cum_sse, int(cum_n)))
                bytes_curve.append(float(cum_bytes))
                continue

            for (a, b) in sd["share_contacts"]:
                _get(a); _get(b)
                n_a = max(1, sum(1 for _, r in bufs[a] if r != noise_floor))
                n_b = max(1, sum(1 for _, r in bufs[b] if r != noise_floor))
                w_a = n_a / (n_a + n_b); w_b = 1.0 - w_a
                sa  = adapters[a].get_adapter_state()
                sb  = adapters[b].get_adapter_state()
                avg = {k: w_a * sa[k] + w_b * sb[k] for k in sa}
                adapters[a].load_adapter_state(avg)
                adapters[b].load_adapter_state(avg)
                sigmas[a] = sigmas[b] = w_a * sigmas[a] + w_b * sigmas[b]
                step_bytes += _adapter_exchange_bytes(adapters[a])

            cum_bytes += step_bytes
            if st:
                arr_p, arr_t = np.array(sp), np.array(st)
                cum_sse += float(np.sum((arr_p - arr_t) ** 2))
                cum_n   += len(st)
            curve.append(_cum_rmse(cum_sse, int(cum_n)))
            bytes_curve.append(float(cum_bytes))

        rmse, mae = _metrics(eval_p, eval_t)
        n_adapter = sum(p.numel() for p in next(iter(adapters.values())).adapter_parameters())
        print(f"  → RMSE: {rmse:.2f} dB  MAE: {mae:.2f} dB  "
              f"comm: {cum_bytes/1024**2:.3f} MB  adapter: {n_adapter*4/1024:.1f} KB")
        return dict(name="Greedy Flooding\nLoRA (ours)", cum_rmse=curve,
                    eval_rmse=rmse, eval_mae=mae,
                    adapter_kb=n_adapter * 4 / 1024, shareable=True,
                    bytes_curve=bytes_curve, total_bytes=float(cum_bytes))

    # ====================================================================
    # Approach 4 — Centralized LoRA (adapters-only)
    # ====================================================================
    def run_centralized_mlp(sim_data):
        print("Running Approach 4: Centralized LoRA (adapters-only) …", flush=True)
        model = central_model
        opt   = optim.Adam(model.adapter_parameters(), lr=lr_lora, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        buf: list = []

        cum_sse = cum_n = cum_bytes = 0.0
        curve, bytes_curve, eval_p, eval_t = [], [], [], []

        for sd in sim_data:
            step = sd["step"]
            sp, st, step_bytes = [], [], 0

            for o in sd["obs"]:
                pred = _pred_large(model, o["feat"], device)
                sp.append(pred); st.append(o["rssi"])
                if step >= num_steps - eval_steps:
                    eval_p.append(pred); eval_t.append(o["rssi"])
                buf.append((o["feat"].detach().clone(), o["rssi"]))
                if len(buf) > max_buf * num_nodes:
                    buf = random.sample(buf, max_buf * num_nodes)
                step_bytes += _OBS_UPLOAD_BYTES   # every obs uploaded to server

            # train every step
            _fit_lora(model, opt, buf, loss_fn, noise_floor, min_samples,
                      max_train * 4, grad_steps_lora, device)

            cum_bytes += step_bytes
            if st:
                arr_p, arr_t = np.array(sp), np.array(st)
                cum_sse += float(np.sum((arr_p - arr_t) ** 2))
                cum_n   += len(st)
            curve.append(_cum_rmse(cum_sse, int(cum_n)))
            bytes_curve.append(float(cum_bytes))

        rmse, mae = _metrics(eval_p, eval_t)
        kb = sum(p.numel() for p in model.adapter_parameters()) * 4 / 1024
        print(f"  → RMSE: {rmse:.2f} dB  MAE: {mae:.2f} dB  "
              f"comm: {cum_bytes/1024**2:.3f} MB  model: {kb:.1f} KB")
        return dict(name="Centralized\nLoRA", cum_rmse=curve,
                    eval_rmse=rmse, eval_mae=mae,
                    adapter_kb=kb, shareable=False,
                    bytes_curve=bytes_curve, total_bytes=float(cum_bytes))

    # ── Run all ──────────────────────────────────────────────────────────
    t_replay = time.time()
    res_fspl    = run_fspl(simulation_data)
    res_local   = run_local_lora(simulation_data)
    res_greedy  = run_greedy_lora(simulation_data)
    res_central = run_centralized_mlp(simulation_data)
    results = [res_fspl, res_local, res_greedy, res_central]
    print(f"\nAll approaches done in {time.time()-t_replay:.1f}s", flush=True)

    # ── Plotting ─────────────────────────────────────────────────────────
    colors    = ["#9E9E9E", "#2196F3", "#4CAF50", "#F44336"]
    lstyles   = ["--", "-.", "-", ":"]
    markers   = ["s", "^", "o", "D"]
    steps_ax  = list(range(1, num_steps + 1))
    xlabels   = [r["name"] for r in results]

    fig = plt.figure(figsize=(22, 17))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.36)

    # (a) learning curves
    ax1 = fig.add_subplot(gs[0, :])
    for i, r in enumerate(results):
        ax1.plot(steps_ax, r["cum_rmse"], lstyles[i], color=colors[i],
                 linewidth=2.2, label=r["name"].replace("\n", " "))
    _, yhi = ax1.get_ylim()
    ax1.axvspan(num_steps - eval_steps + 0.5, num_steps + 0.5,
                alpha=0.07, color="orange", label="eval window")
    ax1.axvline(num_steps - eval_steps + 0.5, color="black",
                linestyle=":", linewidth=1.2, alpha=0.5)
    ax1.text(num_steps - eval_steps + 1.5, yhi * 0.97,
             "eval window (last 40 steps)", fontsize=9, va="top", alpha=0.65)
    ax1.set_xlabel("Simulation step", fontsize=11)
    ax1.set_ylabel("Cumulative RMSE (dB)", fontsize=11)
    ax1.set_title(
        "(a) Learning curves — all approaches train on every step, "
        "eval window = last 40 steps", fontsize=12)
    ax1.legend(fontsize=10, ncol=5)
    ax1.grid(alpha=0.3)

    # (b) eval RMSE
    ax2 = fig.add_subplot(gs[1, 0])
    rmses = [r["eval_rmse"] for r in results]
    bars  = ax2.bar(xlabels, rmses, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, rmses):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                 f"{v:.1f} dB", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Eval RMSE (dB)", fontsize=11)
    ax2.set_title("(b) Eval RMSE — last 40 steps", fontsize=12)
    ax2.set_ylim(0, max(rmses) * 1.25)
    ax2.tick_params(axis="x", labelsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # (c) eval MAE
    ax3 = fig.add_subplot(gs[1, 1])
    maes = [r["eval_mae"] for r in results]
    bars = ax3.bar(xlabels, maes, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, maes):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                 f"{v:.1f} dB", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Eval MAE (dB)", fontsize=11)
    ax3.set_title("(c) Eval MAE — last 40 steps", fontsize=12)
    ax3.set_ylim(0, max(maes) * 1.25)
    ax3.tick_params(axis="x", labelsize=9)
    ax3.grid(axis="y", alpha=0.3)

    # (d) model / adapter size
    ax4 = fig.add_subplot(gs[1, 2])
    kbs = [r["adapter_kb"] for r in results]
    bars = ax4.bar(xlabels, kbs, color=colors, edgecolor="white", width=0.55)
    for i, (bar, v) in enumerate(zip(bars, kbs)):
        if v > 0:
            ax4.text(bar.get_x() + bar.get_width() / 2, v + 2,
                     f"{v:.0f} KB", ha="center", va="bottom", fontsize=10, fontweight="bold")
        lbl = "✓ distributable" if results[i]["shareable"] else "✗ not shareable"
        col = "#2e7d32" if results[i]["shareable"] else "#c62828"
        ax4.text(bar.get_x() + bar.get_width() / 2, max(v / 2, 10),
                 lbl, ha="center", va="center", fontsize=7.5,
                 color=col, fontweight="bold", rotation=90)
    ax4.set_ylabel("Exchangeable size (KB, fp32)", fontsize=10)
    ax4.set_title(f"(d) Per-device model size\n"
                  f"(sharing range: {share_range:.0f} m)", fontsize=12)
    ax4.tick_params(axis="x", labelsize=9)
    ax4.grid(axis="y", alpha=0.3)

    # (e) cumulative bytes
    ax5 = fig.add_subplot(gs[2, :2])
    for i, r in enumerate(results):
        mb = r["total_bytes"] / 1024**2
        label = f"{r['name'].replace(chr(10), ' ')}  ({mb:.2f} MB total)"
        ax5.plot(steps_ax, [b / 1024**2 for b in r["bytes_curve"]],
                 lstyles[i], color=colors[i], linewidth=2.2, label=label)
    ax5.axvline(num_steps - eval_steps + 0.5, color="black",
                linestyle=":", linewidth=1.2, alpha=0.5)
    ax5.set_xlabel("Simulation step", fontsize=11)
    ax5.set_ylabel("Cumulative data exchanged (MB)", fontsize=11)
    ax5.set_title(
        f"(e) Communication overhead\n"
        f"Greedy LoRA: adapter swaps at ≤{share_range:.0f} m  |  "
        f"Centralized: all observations uploaded", fontsize=11)
    ax5.legend(fontsize=9, ncol=2)
    ax5.grid(alpha=0.3)

    # (f) efficiency scatter — bytes vs RMSE (the "bang for buck" chart)
    ax6 = fig.add_subplot(gs[2, 2])
    for i, r in enumerate(results):
        mb   = r["total_bytes"] / 1024**2
        rmse = r["eval_rmse"]
        ax6.scatter(mb, rmse, s=180, color=colors[i], marker=markers[i],
                    zorder=5, edgecolors="white", linewidths=1.2)
        ax6.annotate(r["name"].replace("\n", " "), (mb, rmse),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=8, color=colors[i], fontweight="bold")
    ax6.set_xlabel("Total data exchanged (MB)", fontsize=11)
    ax6.set_ylabel("Eval RMSE (dB)", fontsize=11)
    ax6.set_title("(f) Accuracy–communication tradeoff\n"
                  "(lower-left = better)", fontsize=12)
    ax6.grid(alpha=0.3)

    fig.suptitle(
        "Radio Map Prediction — Strategy Comparison\n"
        f"Munich 3D Ray-Tracing Scene, 3.5 GHz, {num_nodes} Mobile Nodes, "
        f"{num_steps} Steps  |  obs range: {obs_range:.0f} m  "
        f"share range: {share_range:.0f} m",
        fontsize=13, fontweight="bold",
    )

    fig_path = results_dir / "benchmark_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {fig_path}", flush=True)

    # ── Summary text ─────────────────────────────────────────────────────
    summary_path = results_dir / "benchmark_summary.txt"
    W = 90
    with summary_path.open("w") as f:
        f.write(
            f"Benchmark Comparison — Munich 3D, {num_steps} steps "
            f"(obs_range={obs_range}m, share_range={share_range}m)\n"
        )
        f.write("=" * W + "\n")
        f.write(
            f"{'Strategy':<32} {'Eval RMSE':>10} {'Eval MAE':>9} "
            f"{'Size (KB)':>10} {'Comm (MB)':>10} {'Shareable':>10}\n"
        )
        f.write("-" * W + "\n")
        for r in results:
            name    = r["name"].replace("\n", " ")
            comm_mb = r["total_bytes"] / 1024**2
            f.write(
                f"{name:<32} {r['eval_rmse']:>9.2f}  {r['eval_mae']:>8.2f}  "
                f"{r['adapter_kb']:>9.1f}  {comm_mb:>9.3f}  "
                f"{'Yes' if r['shareable'] else 'No':>9}\n"
            )
        f.write("=" * W + "\n")

        d_fspl    = res_fspl["eval_rmse"]    - res_greedy["eval_rmse"]
        d_local   = res_local["eval_rmse"]   - res_greedy["eval_rmse"]
        d_central = res_greedy["eval_rmse"]  - res_central["eval_rmse"]
        g_mb      = res_greedy["total_bytes"]  / 1024**2
        c_mb      = res_central["total_bytes"] / 1024**2

        f.write("\nAccuracy gains (Greedy LoRA as pivot):\n")
        f.write(f"  vs FSPL Baseline:    {d_fspl:+.2f} dB\n")
        f.write(f"  vs Local-only LoRA:  {d_local:+.2f} dB  ← value of greedy sharing\n")
        f.write(f"  vs Centralized LoRA:  {d_central:+.2f} dB  ← gap vs centralized adapters-only baseline\n")
        f.write("\nCommunication cost:\n")
        f.write(f"  Greedy LoRA (adapter swaps ≤{share_range:.0f}m):  {g_mb:.3f} MB\n")
        f.write(f"  Centralized (all obs uploaded):          {c_mb:.3f} MB\n")
        if c_mb > 0 and g_mb > 0:
            f.write(f"  Centralized uses {c_mb/g_mb:.1f}× more bandwidth\n")
        elif g_mb == 0:
            f.write("  Greedy LoRA: 0 bytes (no contacts within share_range)\n")

    print(f"Saved summary → {summary_path}", flush=True)


def main() -> None:
    run_benchmarks(Path("benchmark_results"))


if __name__ == "__main__":
    main()
