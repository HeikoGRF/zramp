"""
run_benchmarks_zones.py — Zone-based Radio Map Benchmark

ARCHITECTURE
============
The 100×100 m Munich simulation area is divided into a 2×2 grid of
four 50×50 m ANCHOR ZONES (numbered 0–3, row-major).

    x: 0–49   │ x: 50–99
    ──────────┼──────────
    Zone 0    │ Zone 1     y: -199 → -149
    ──────────┼──────────
    Zone 2    │ Zone 3     y: -149 → -100

Each LoRA adapter IS the radio map of one anchor zone.  A node always
carries the adapter for its CURRENT zone; it cannot and does not use
that adapter to predict links outside that zone.

ZONE TRANSITION (Proposition 3)
================================
When node N crosses from Zone A into Zone B:
  1. Buffer for Zone A is evicted (no old observations carried forward).
  2. Adapter for Zone A is saved to the zone cache.
  3. Adapter for Zone B is loaded from the zone cache if it exists
     (warm start), otherwise a fresh base-model adapter is used (cold start).
  4. The base MLP backbone (FSPL-pretrained) is NEVER reset — only the
     LoRA delta weights change.

SHARING GATE (Proposition 2)
==============================
Two nodes may exchange adapters ONLY when ALL of the following hold:
  (a) Both are in the SAME anchor zone (zone adapters are zone-specific).
  (b) Physical distance ≤ share_range (30 m): Bluetooth / WiFi-Direct range.
  (c) Bidirectional link RSSI > rssi_share_threshold (-100 dBm).

After a successful exchange, BOTH nodes receive the IDENTICAL merged adapter.

ANALYSIS METRICS (second output figure)
=========================================
  1. Per-zone RMSE — shows which zones benefit most from sharing.
  2. Cold-start recovery — RMSE vs steps-since-zone-entry; the "smoking gun"
     for the value of adapter sharing.
  3. RMSE vs distance bin — shows whether sharing helps mainly short/long links.
  4. LOS vs NLOS breakdown — sharing benefit may be concentrated on NLOS links.
  5. Error CDF — "90th-percentile error" story; more meaningful than mean RMSE.
  6. Per-step RMSE stability — mean ± std bands to show variance reduction.
"""

import math
import random
import time
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
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
    MapConfig, RadioFeatureEncoder, RadioMLPWithLoRA,
)
from sionna_baseline.mobility import RandomWaypoint  # type: ignore
from sionna_baseline.backbone_utils import (  # type: ignore
    sample_node_tx_powers,
    compute_rssi,
    load_or_pretrain_lora_backbone,
)


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------

N_COL, N_ROW = 2, 2
N_ZONES = N_COL * N_ROW
ZONE_NAMES  = {0: "Zone 0\n(SW)", 1: "Zone 1\n(SE)", 2: "Zone 2\n(NW)", 3: "Zone 3\n(NE)"}
ZONE_LABELS = {0: "Z0 SW", 1: "Z1 SE", 2: "Z2 NW", 3: "Z3 NE"}


def get_zone_id(x: float, y: float, bounds_x: tuple, bounds_y: tuple) -> int:
    zone_w = (bounds_x[1] - bounds_x[0]) / N_COL
    zone_h = (bounds_y[1] - bounds_y[0]) / N_ROW
    col = min(int((x - bounds_x[0]) / zone_w), N_COL - 1)
    row = min(int((y - bounds_y[0]) / zone_h), N_ROW - 1)
    return row * N_COL + col


# ---------------------------------------------------------------------------
# Scene / building helpers
# ---------------------------------------------------------------------------

def _build_grid_safe(scene, bounds_x, bounds_y, res=1.0):
    xs = np.arange(bounds_x[0], bounds_x[1] + res, res)
    ys = np.arange(bounds_y[0], bounds_y[1] + res, res)
    xx, yy = np.meshgrid(xs, ys)
    points = np.c_[xx.ravel(), yy.ravel()]
    mask = np.zeros(len(points), dtype=bool)
    for _, obj in scene.objects.items():
        mesh = obj.mi_mesh
        try:
            vertices = np.array(mesh.vertex_positions_buffer()).reshape(-1, 3)
            faces    = np.array(mesh.faces_buffer()).reshape(-1, 3)
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
                mask[idx[p.contains_points(points[idx])]] = True
    return mask.reshape(xx.shape)


def count_building_cells(tx_xy, rx_xy, grid_mask, bounds_x, bounds_y, res=1.0) -> int:
    dist = np.linalg.norm(rx_xy - tx_xy)
    if dist < 1e-6:
        return 0
    n_samp = max(2, int(np.ceil(dist / (res / 2.0))))
    ts  = np.linspace(0.0, 1.0, n_samp)
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


def _freeze_base_train_adapters_only(model: nn.Module) -> None:
    """Centralized LoRA baseline trains adapters only (like devices)."""
    for name, p in model.named_parameters():
        if name.startswith(("fc1.", "fc2.", "fc3.")):
            p.requires_grad = False
        else:
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Shared training / inference utilities
# ---------------------------------------------------------------------------

def _cum_rmse(sse: float, n: int) -> float:
    return math.sqrt(sse / n) if n > 0 else 0.0


def _metrics(preds, truths):
    p, t = np.array(preds), np.array(truths)
    return float(np.sqrt(np.mean((p - t) ** 2))), float(np.mean(np.abs(p - t)))


def _ema(values: list[float], alpha: float = 0.2) -> list[float]:
    out = []
    s = values[0] if values else 0.0
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def _eval_rec(pred: float, true_rssi: float, d: float, nc: int, zone: int,
              delta_t: int = -1) -> dict:
    """Build a rich per-observation record for downstream analysis."""
    return {
        "pred":    pred,
        "true":    true_rssi,
        "abs_err": abs(pred - true_rssi),
        "sq_err":  (pred - true_rssi) ** 2,
        "d":       d,
        "nc":      nc,
        "is_los":  nc == 0,
        "zone":    zone,
        "delta_t": delta_t,   # steps since this node last entered the current zone
    }


def _fit_lora(model, opt, buf, loss_fn, noise_floor, min_samples,
              max_train, grad_steps, device):
    unc = [(f, r) for f, r in buf if r != noise_floor]
    if len(unc) < min_samples:
        return None
    samp = unc if len(unc) <= max_train else random.sample(unc, max_train)
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
    samp = unc if len(unc) <= max_train else random.sample(unc, max_train)
    x = torch.stack([f for f, _ in samp]).to(device)
    y = torch.tensor([r for _, r in samp], dtype=torch.float32).to(device)
    model.train()
    for _ in range(grad_steps):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    model.eval()


def _pred_lora(model, feat, device):
    model.eval()
    with torch.no_grad():
        return float(model(feat.unsqueeze(0).to(device)).item())


def _pred_large(model, feat, device):
    model.eval()
    with torch.no_grad():
        return float(model(feat.unsqueeze(0).to(device)).item())


def _adapter_exchange_bytes(model) -> int:
    return sum(p.numel() for p in model.adapter_parameters()) * 4 * 2


_OBS_UPLOAD_BYTES = (8 + 1) * 4


# ---------------------------------------------------------------------------
# Analysis helpers (post-run)
# ---------------------------------------------------------------------------

DIST_BINS   = [(0, 15), (15, 30), (30, 50), (50, 999)]
DIST_LABELS = ["0–15 m", "15–30 m", "30–50 m", ">50 m"]


def _per_zone_rmse(eval_records: list[dict]) -> dict[int, float]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for r in eval_records:
        buckets[r["zone"]].append(r["sq_err"])
    return {z: math.sqrt(np.mean(v)) for z, v in buckets.items() if v}


def _coldstart_curve(cold_records: list[dict], max_delta: int = 25) -> list[float]:
    """Mean abs error per step-since-zone-entry (0 … max_delta)."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for r in cold_records:
        dt = min(r["delta_t"], max_delta)
        buckets[dt].append(r["abs_err"])
    return [float(np.mean(buckets[dt])) if buckets[dt] else float("nan")
            for dt in range(max_delta + 1)]


def _dist_breakdown(eval_records: list[dict]) -> list[float]:
    """Mean abs error per distance bin."""
    out = []
    for lo, hi in DIST_BINS:
        vals = [r["abs_err"] for r in eval_records if lo <= r["d"] < hi]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


def _nlos_breakdown(eval_records: list[dict]) -> tuple[float, float]:
    """(mean_abs_error_LOS, mean_abs_error_NLOS)."""
    los  = [r["abs_err"] for r in eval_records if     r["is_los"]]
    nlos = [r["abs_err"] for r in eval_records if not r["is_los"]]
    return (float(np.mean(los)) if los else float("nan"),
            float(np.mean(nlos)) if nlos else float("nan"))


def _cdf(eval_records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    errs = np.sort([r["abs_err"] for r in eval_records])
    cdf  = np.arange(1, len(errs) + 1) / len(errs)
    return errs, cdf


def _per_zone_bytes(zone_bytes: dict[int, float],
                    eval_records: list[dict]) -> dict[int, tuple[float, float]]:
    """Returns {zone: (mb_exchanged, eval_rmse)} for scatter."""
    zone_rmse = _per_zone_rmse(eval_records)
    return {z: (zone_bytes.get(z, 0.0) / 1024**2, zone_rmse.get(z, float("nan")))
            for z in range(N_ZONES)}


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmarks_zones(results_dir: Path) -> None:
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────
    num_nodes   = 20
    # Step 7: train continuously; compute eval over last N steps
    num_steps   = 100
    eval_steps  = 20

    dt             = 1.0
    obs_range      = 250.0
    share_range    = 30.0
    rssi_share_thr = -100.0
    bounds_x       = (0, 99)
    bounds_y       = (-199, -100)
    freq_hz        = 3.5e9
    c_light        = 3e8
    fspl_const     = -20.0 * math.log10(4.0 * math.pi * freq_hz / c_light)
    noise_floor    = -150.0

    lr_lora          = 1e-2
    lr_large         = 1e-3
    grad_steps_lora  = 20
    grad_steps_large = 50
    min_samples      = 8
    max_buf          = 500
    max_train        = 500
    share_every      = 5

    # Step 5: data sparsity (collect fewer observations)
    # - obs_every_k_steps: only collect observations every k steps (1 = no sparsity)
    # - obs_keep_prob: per-observation subsampling probability (1.0 = keep all)
    obs_every_k_steps = 1
    obs_keep_prob = 1.0

    device = torch.device("cpu")
    map_cfg = MapConfig(x_min=bounds_x[0], x_max=bounds_x[1],
                        y_min=bounds_y[0], y_max=bounds_y[1])
    encoder = RadioFeatureEncoder(map_cfg)

    print(f"Zone layout: {N_ROW}×{N_COL} grid  →  {N_ZONES} zones of "
          f"{(bounds_x[1]-bounds_x[0])//N_COL}×{(bounds_y[1]-bounds_y[0])//N_ROW} m each")
    print(f"Steps: {num_steps}  |  obs_range: {obs_range} m  "
          f"|  share_range: {share_range} m  |  RSSI gate: {rssi_share_thr} dBm")
    print(f"Data sparsity: obs_every_k_steps={obs_every_k_steps}, obs_keep_prob={obs_keep_prob}", flush=True)

    # ── Scene & mobility ──────────────────────────────────────────────────
    print("Loading Munich scene …", flush=True)
    scene = rt.load_scene(rt.scene.munich)
    scene.frequency = freq_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

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
        bounds_x=bounds_x, bounds_y=bounds_y,
        velocity=(2.0, 5.0), pause_time=1.0,
    )

    gpus = tf.config.list_physical_devices("GPU")
    compute_dev = "/GPU:0" if gpus else "/CPU:0"
    print(f"Compute device: {gpus[0].name if gpus else 'CPU'}", flush=True)

    print("Building 2-D building grid …", flush=True)
    grid_mask = _build_grid_safe(scene, bounds_x, bounds_y, res=1.0)
    print(f"Building grid: {int(np.sum(grid_mask))} occupied cells", flush=True)

    # ── Collect simulation data ────────────────────────────────────────────
    solver = PathSolver()
    simulation_data: list[dict] = []
    total_zone_transitions = 0
    prev_node_zones: dict[int, int] = {}
    t0 = time.time()

    for step in range(num_steps):
        mobility.step(nodes, dt)
        for i in range(num_nodes):
            scene.get(f"tx_{i}").position = nodes[i]
            scene.get(f"rx_{i}").position = nodes[i]

        with tf.device(compute_dev):
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

        node_zones: dict[int, int] = {
            i: get_zone_id(nodes[i][0], nodes[i][1], bounds_x, bounds_y)
            for i in range(num_nodes)
        }
        step_transitions = sum(
            1 for i in range(num_nodes)
            if i in prev_node_zones and prev_node_zones[i] != node_zones[i]
        )
        total_zone_transitions += step_transitions
        prev_node_zones = dict(node_zones)

        obs: list[dict] = []
        link_rssi: dict[tuple[int, int], float] = {}

        collect_obs = (step % obs_every_k_steps == 0)
        for tx_i in range(num_nodes):
            for rx_i in range(num_nodes):
                if tx_i == rx_i:
                    continue
                ptx, prx = nodes[tx_i], nodes[rx_i]
                d2d = float(np.linalg.norm(ptx[:2] - prx[:2]))
                if d2d > obs_range:
                    continue
                tx_xy = ptx[:2].copy()
                rx_xy = prx[:2].copy()
                pwr  = float(power[rx_i, tx_i])
                rssi = compute_rssi(pwr, node_tx_power_dbm[tx_i], noise_floor)
                nc   = count_building_cells(tx_xy, rx_xy, grid_mask, bounds_x, bounds_y)
                d    = float(np.linalg.norm(rx_xy - tx_xy))
                feat = encoder.encode(tuple(tx_xy), tuple(rx_xy),
                                      tx_power_dbm=node_tx_power_dbm[tx_i],
                                      has_obstacle=(nc > 0))

                tx_zone = node_zones[tx_i]
                rx_zone = node_zones[rx_i]
                # Apply sparsity at collection time (so local buffers have less data)
                if collect_obs and (np.random.rand() <= obs_keep_prob):
                    obs.append({
                        "tx_i": tx_i, "rx_i": rx_i,
                        "tx_zone": tx_zone, "rx_zone": rx_zone,
                        "rssi": rssi, "nc": nc, "d": d,
                        "fspl": compute_fspl(d, fspl_const,
                                            tx_power_dbm=node_tx_power_dbm[tx_i]),
                        "feat": feat,
                    })
                key = (min(tx_i, rx_i), max(tx_i, rx_i))
                if key not in link_rssi or rssi < link_rssi[key]:
                    link_rssi[key] = rssi

        share_contacts: set[tuple[int, int]] = set()
        for tx_i in range(num_nodes):
            for rx_i in range(tx_i + 1, num_nodes):
                if node_zones[tx_i] != node_zones[rx_i]:
                    continue
                d2d = float(np.linalg.norm(nodes[tx_i][:2] - nodes[rx_i][:2]))
                if d2d > share_range:
                    continue
                key = (tx_i, rx_i)
                if link_rssi.get(key, noise_floor) > rssi_share_thr:
                    share_contacts.add(key)

        simulation_data.append({
            "step": step,
            "obs": obs,
            "node_zones": node_zones,
            "link_rssi": link_rssi,
            "share_contacts": share_contacts,
            "n_transitions": step_transitions,
        })

        if (step + 1) % 20 == 0:
            n_total = sum(len(s["obs"]) for s in simulation_data)
            n_intra = sum(sum(1 for o in s["obs"] if o["tx_zone"] == o["rx_zone"])
                          for s in simulation_data)
            print(f"  Step {step+1:3d}/{num_steps}  obs={n_total:5d} "
                  f"(intra={n_intra})  trans={total_zone_transitions}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    total_obs = sum(len(s["obs"]) for s in simulation_data)
    intra_obs = sum(sum(1 for o in s["obs"] if o["tx_zone"] == o["rx_zone"])
                    for s in simulation_data)
    print(f"\nData done in {time.time()-t0:.1f}s  |  "
          f"total obs: {total_obs:,}  intra-zone: {intra_obs:,}  "
          f"transitions: {total_zone_transitions}", flush=True)

    # ── Pre-train / load shared LoRA backbone ─────────────────────────────
    # Tries backbone_urban.pt (urban ray-tracing); falls back to FSPL pretraining.
    base_state, eff_fspl = load_or_pretrain_lora_backbone(
        encoder, fspl_const,
        backbone_path="backbone_etoile.pt",
        hidden_sizes=(128, 128), rank=16, alpha=1.0,
        n_samples=50_000, epochs=200, lr=1e-3,
    )
    _tmp = RadioMLPWithLoRA(input_dim=8, hidden_sizes=(128, 128), rank=16, alpha=1.0)
    adapter_kb = sum(p.numel() for p in _tmp.adapter_parameters()) * 4 / 1024
    print(f"LoRA backbone ready  ({adapter_kb:.1f} KB adapter)", flush=True)

    print("Initializing centralized baseline (centralized LoRA; adapters-only) …", flush=True)
    central_model = RadioMLPWithLoRA(
        input_dim=8, hidden_sizes=(128, 128), rank=16, alpha=1.0, fspl_const=eff_fspl
    )
    central_model.load_base_state(base_state)
    _freeze_base_train_adapters_only(central_model)
    central_model.to(device)
    central_model.eval()
    central_adapter_kb = sum(p.numel() for p in central_model.adapter_parameters()) * 4 / 1024
    print(f"Centralized baseline initialized  ({central_adapter_kb:.1f} KB adapter)", flush=True)

    loss_fn = nn.MSELoss()

    def new_lora_adapter() -> RadioMLPWithLoRA:
        m = RadioMLPWithLoRA(
            input_dim=8, hidden_sizes=(128, 128), rank=16, alpha=1.0, fspl_const=eff_fspl)
        m.load_base_state(base_state)
        m.to(device)
        return m

    # ====================================================================
    # Approach 1 — FSPL Baseline
    # ====================================================================
    def run_fspl(sim_data):
        print("Running Approach 1: FSPL Baseline …", flush=True)
        cum_sse = cum_n = 0.0
        step_rmse, cum_curve, bytes_curve = [], [], []
        eval_p, eval_t = [], []
        eval_records: list[dict] = []

        for sd in sim_data:
            sp, st = [], []
            for o in sd["obs"]:
                if o["tx_zone"] != o["rx_zone"]:
                    continue
                pred = o["fspl"]
                sp.append(pred); st.append(o["rssi"])
                if sd["step"] >= num_steps - eval_steps:
                    eval_p.append(pred); eval_t.append(o["rssi"])
                    eval_records.append(_eval_rec(pred, o["rssi"], o["d"], o["nc"],
                                                  o["rx_zone"]))
            if st:
                arr_p, arr_t = np.array(sp), np.array(st)
                sse = float(np.sum((arr_p - arr_t) ** 2))
                cum_sse += sse; cum_n += len(st)
                step_rmse.append(math.sqrt(sse / len(st)))
            else:
                step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
            cum_curve.append(_cum_rmse(cum_sse, int(cum_n)))
            bytes_curve.append(0.0)

        rmse, mae = _metrics(eval_p, eval_t)
        print(f"  → RMSE: {rmse:.2f} dB  MAE: {mae:.2f} dB  comm: 0 B")
        return dict(name="FSPL\nBaseline", cum_rmse=cum_curve, step_rmse=step_rmse,
                    eval_rmse=rmse, eval_mae=mae, adapter_kb=0.0, shareable=True,
                    bytes_curve=bytes_curve, total_bytes=0.0,
                    eval_records=eval_records, cold_records=[],
                    zone_bytes={z: 0.0 for z in range(N_ZONES)})

    # ====================================================================
    # Approach 2 — Local-only LoRA  (cold start on every zone entry)
    # ====================================================================
    def run_local_lora(sim_data):
        print("Running Approach 2: Local-only LoRA …", flush=True)
        adapters: dict[int, RadioMLPWithLoRA] = {}
        opts:     dict[int, optim.Optimizer]  = {}
        bufs:     dict[int, list]             = {}
        node_zones_rt:  dict[int, int] = {}
        zone_entry_step: dict[int, int] = {}

        def cold_start(node_id: int, new_zone: int, step: int) -> None:
            adapters[node_id] = new_lora_adapter()
            opts[node_id]     = optim.Adam(adapters[node_id].adapter_parameters(),
                                           lr=lr_lora, weight_decay=1e-4)
            bufs[node_id]         = []
            node_zones_rt[node_id]  = new_zone
            zone_entry_step[node_id] = step

        for i in range(num_nodes):
            cold_start(i, sim_data[0]["node_zones"][i], 0)

        cum_sse = cum_n = 0.0
        step_rmse, cum_curve, bytes_curve = [], [], []
        eval_p, eval_t = [], []
        eval_records:  list[dict] = []
        cold_records:  list[dict] = []

        for sd in sim_data:
            step = sd["step"]
            for i in range(num_nodes):
                if sd["node_zones"][i] != node_zones_rt[i]:
                    cold_start(i, sd["node_zones"][i], step)

            sp, st = [], []
            for o in sd["obs"]:
                rx_i = o["rx_i"]
                if o["tx_zone"] != node_zones_rt[rx_i]:
                    continue
                pred = _pred_lora(adapters[rx_i], o["feat"], device)
                sp.append(pred); st.append(o["rssi"])
                dt = step - zone_entry_step.get(rx_i, 0)
                cold_records.append({"delta_t": dt, "abs_err": abs(pred - o["rssi"])})
                if step >= num_steps - eval_steps:
                    eval_p.append(pred); eval_t.append(o["rssi"])
                    eval_records.append(_eval_rec(pred, o["rssi"], o["d"], o["nc"],
                                                  node_zones_rt[rx_i], dt))
                bufs[rx_i].append((o["feat"].detach().clone(), o["rssi"]))
                bufs[rx_i] = bufs[rx_i][-max_buf:]

            for i in adapters:
                _fit_lora(adapters[i], opts[i], bufs[i], loss_fn,
                          noise_floor, min_samples, max_train, grad_steps_lora, device)

            if st:
                arr_p, arr_t = np.array(sp), np.array(st)
                sse = float(np.sum((arr_p - arr_t) ** 2))
                cum_sse += sse; cum_n += len(st)
                step_rmse.append(math.sqrt(sse / len(st)))
            else:
                step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
            cum_curve.append(_cum_rmse(cum_sse, int(cum_n)))
            bytes_curve.append(0.0)

        rmse, mae = _metrics(eval_p, eval_t)
        print(f"  → RMSE: {rmse:.2f} dB  MAE: {mae:.2f} dB  comm: 0 B")
        return dict(name="Local LoRA\n(no sharing)", cum_rmse=cum_curve, step_rmse=step_rmse,
                    eval_rmse=rmse, eval_mae=mae, adapter_kb=adapter_kb, shareable=True,
                    bytes_curve=bytes_curve, total_bytes=0.0,
                    eval_records=eval_records, cold_records=cold_records,
                    zone_bytes={z: 0.0 for z in range(N_ZONES)})

    # ====================================================================
    # Approach 3 — Greedy Flooding LoRA  (zone cache + RSSI gate)
    # ====================================================================
    def run_greedy_lora(sim_data):
        print("Running Approach 3: Greedy Flooding LoRA …", flush=True)
        adapters: dict[int, RadioMLPWithLoRA] = {}
        opts:     dict[int, optim.Optimizer]  = {}
        bufs:     dict[int, list]             = {}
        node_zones_rt:  dict[int, int] = {}
        zone_entry_step: dict[int, int] = {}
        zone_cache: dict[int, dict] = {}
        zone_bytes: dict[int, float] = defaultdict(float)

        def save_to_cache(node_id: int, old_zone: int) -> None:
            if len(bufs.get(node_id, [])) < min_samples:
                return
            state = adapters[node_id].get_adapter_state()
            if old_zone not in zone_cache:
                zone_cache[old_zone] = state
            else:
                n_node = max(1, len(bufs[node_id]))
                w = n_node / (n_node + max_buf)
                zone_cache[old_zone] = {
                    k: w * state[k] + (1 - w) * zone_cache[old_zone][k]
                    for k in state
                }

        def enter_zone(node_id: int, new_zone: int, step: int) -> None:
            adapter = new_lora_adapter()
            if new_zone in zone_cache:
                adapter.load_adapter_state(zone_cache[new_zone])
            adapters[node_id]        = adapter
            opts[node_id]            = optim.Adam(adapter.adapter_parameters(),
                                                   lr=lr_lora, weight_decay=1e-4)
            bufs[node_id]            = []
            node_zones_rt[node_id]   = new_zone
            zone_entry_step[node_id] = step

        for i in range(num_nodes):
            enter_zone(i, sim_data[0]["node_zones"][i], 0)

        cum_sse = cum_n = cum_bytes = 0.0
        step_rmse, cum_curve, bytes_curve = [], [], []
        eval_p, eval_t = [], []
        eval_records: list[dict] = []
        cold_records: list[dict] = []

        for sd in sim_data:
            step = sd["step"]
            for i in range(num_nodes):
                new_z = sd["node_zones"][i]
                if new_z != node_zones_rt[i]:
                    save_to_cache(i, node_zones_rt[i])
                    enter_zone(i, new_z, step)

            sp, st, step_bytes = [], [], 0
            for o in sd["obs"]:
                rx_i = o["rx_i"]
                if o["tx_zone"] != node_zones_rt[rx_i]:
                    continue
                pred = _pred_lora(adapters[rx_i], o["feat"], device)
                sp.append(pred); st.append(o["rssi"])
                dt = step - zone_entry_step.get(rx_i, 0)
                cold_records.append({"delta_t": dt, "abs_err": abs(pred - o["rssi"])})
                if step >= num_steps - eval_steps:
                    eval_p.append(pred); eval_t.append(o["rssi"])
                    eval_records.append(_eval_rec(pred, o["rssi"], o["d"], o["nc"],
                                                  node_zones_rt[rx_i], dt))
                bufs[rx_i].append((o["feat"].detach().clone(), o["rssi"]))
                bufs[rx_i] = bufs[rx_i][-max_buf:]

            for i in adapters:
                _fit_lora(adapters[i], opts[i], bufs[i], loss_fn,
                          noise_floor, min_samples, max_train, grad_steps_lora, device)

            if step % share_every == 0:
                for (a, b) in sd["share_contacts"]:
                    if a not in adapters or b not in adapters:
                        continue
                    n_a = max(1, len(bufs[a]))
                    n_b = max(1, len(bufs[b]))
                    w_a = n_a / (n_a + n_b)
                    w_b = 1.0 - w_a
                    sa  = adapters[a].get_adapter_state()
                    sb  = adapters[b].get_adapter_state()
                    avg = {k: w_a * sa[k] + w_b * sb[k] for k in sa}
                    adapters[a].load_adapter_state(avg)
                    adapters[b].load_adapter_state(avg)
                    shared_zone = node_zones_rt[a]
                    zone_cache[shared_zone] = avg
                    b_cost = _adapter_exchange_bytes(adapters[a])
                    step_bytes += b_cost
                    zone_bytes[shared_zone] += b_cost

            cum_bytes += step_bytes
            if st:
                arr_p, arr_t = np.array(sp), np.array(st)
                sse = float(np.sum((arr_p - arr_t) ** 2))
                cum_sse += sse; cum_n += len(st)
                step_rmse.append(math.sqrt(sse / len(st)))
            else:
                step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
            cum_curve.append(_cum_rmse(cum_sse, int(cum_n)))
            bytes_curve.append(float(cum_bytes))

        rmse, mae = _metrics(eval_p, eval_t)
        print(f"  → RMSE: {rmse:.2f} dB  MAE: {mae:.2f} dB  "
              f"comm: {cum_bytes/1024**2:.3f} MB")
        print(f"     Zone cache: {sorted(zone_cache.keys())}  "
              f"zone_bytes: { {z: f'{v/1024**2:.2f} MB' for z,v in zone_bytes.items()} }")
        return dict(name="Greedy Flooding\nLoRA (ours)", cum_rmse=cum_curve,
                    step_rmse=step_rmse, eval_rmse=rmse, eval_mae=mae,
                    adapter_kb=adapter_kb, shareable=True,
                    bytes_curve=bytes_curve, total_bytes=float(cum_bytes),
                    eval_records=eval_records, cold_records=cold_records,
                    zone_bytes=dict(zone_bytes))

    # ====================================================================
    # Approach 4 — Centralized LoRA (adapters-only; all zones)
    # ====================================================================
    def run_centralized_mlp(sim_data):
        print("Running Approach 4: Centralized LoRA (adapters-only) …", flush=True)
        model = central_model
        opt   = optim.Adam(model.adapter_parameters(), lr=lr_lora, weight_decay=1e-4)
        buf: list = []
        cum_sse = cum_n = cum_bytes = 0.0
        step_rmse, cum_curve, bytes_curve = [], [], []
        eval_p, eval_t = [], []
        eval_records: list[dict] = []

        for sd in sim_data:
            step = sd["step"]
            sp, st, step_bytes = [], [], 0
            for o in sd["obs"]:
                pred = _pred_large(model, o["feat"], device)
                sp.append(pred); st.append(o["rssi"])
                buf.append((o["feat"].detach().clone(), o["rssi"]))
                if len(buf) > max_buf * num_nodes:
                    buf = random.sample(buf, max_buf * num_nodes)
                step_bytes += _OBS_UPLOAD_BYTES
                if step >= num_steps - eval_steps and o["tx_zone"] == o["rx_zone"]:
                    eval_p.append(pred); eval_t.append(o["rssi"])
                    eval_records.append(_eval_rec(pred, o["rssi"], o["d"], o["nc"],
                                                  o["rx_zone"]))

            _fit_lora(model, opt, buf, loss_fn, noise_floor, min_samples,
                      max_train * 4, grad_steps_lora, device)

            cum_bytes += step_bytes
            intra = [(p, t) for p, t, o in zip(sp, st, sd["obs"])
                     if o["tx_zone"] == o["rx_zone"]]
            if intra:
                arr_p = np.array([x[0] for x in intra])
                arr_t = np.array([x[1] for x in intra])
                sse = float(np.sum((arr_p - arr_t) ** 2))
                cum_sse += sse; cum_n += len(intra)
                step_rmse.append(math.sqrt(sse / len(intra)))
            else:
                step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
            cum_curve.append(_cum_rmse(cum_sse, int(cum_n)))
            bytes_curve.append(float(cum_bytes))

        rmse, mae = _metrics(eval_p, eval_t)
        print(f"  → RMSE: {rmse:.2f} dB  MAE: {mae:.2f} dB  "
              f"comm: {cum_bytes/1024**2:.3f} MB")
        return dict(name="Centralized\nLoRA", cum_rmse=cum_curve,
                    step_rmse=step_rmse, eval_rmse=rmse, eval_mae=mae,
                    adapter_kb=central_adapter_kb, shareable=False,
                    bytes_curve=bytes_curve, total_bytes=float(cum_bytes),
                    eval_records=eval_records, cold_records=[],
                    zone_bytes={z: 0.0 for z in range(N_ZONES)})

    # ── Run all ────────────────────────────────────────────────────────────
    t_replay = time.time()
    res_fspl    = run_fspl(simulation_data)
    res_local   = run_local_lora(simulation_data)
    res_greedy  = run_greedy_lora(simulation_data)
    res_central = run_centralized_mlp(simulation_data)
    results = [res_fspl, res_local, res_greedy, res_central]
    print(f"\nAll done in {time.time()-t_replay:.1f}s", flush=True)

    # ── Pre-compute all analysis metrics ──────────────────────────────────
    colors  = ["#9E9E9E", "#2196F3", "#4CAF50", "#F44336"]
    lstyles = ["--",      "-.",      "-",       ":"]
    markers = ["s",       "^",       "o",       "D"]
    steps_x = list(range(1, num_steps + 1))
    short_names = ["FSPL", "Local LoRA", "Greedy LoRA", "Centralized LoRA"]

    per_zone_rmse = {r["name"]: _per_zone_rmse(r["eval_records"]) for r in results}
    dist_break    = {r["name"]: _dist_breakdown(r["eval_records"]) for r in results}
    nlos_break    = {r["name"]: _nlos_breakdown(r["eval_records"]) for r in results}
    cdfs          = {r["name"]: _cdf(r["eval_records"]) for r in results if r["eval_records"]}
    # Cold-start curves only for LoRA methods
    coldstart_curves = {
        r["name"]: _coldstart_curve(r["cold_records"])
        for r in [res_local, res_greedy]
        if r["cold_records"]
    }
    per_zone_bytes_greedy = _per_zone_bytes(res_greedy["zone_bytes"], res_greedy["eval_records"])
    # Per-step RMSE std across runs (use existing step_rmse lists, compute rolling std)
    transition_steps = [s["step"] + 1 for s in simulation_data if s["n_transitions"] > 0]

    # ── Figure 1: Overview (7 panels) ─────────────────────────────────────
    fig1 = plt.figure(figsize=(22, 18))
    gs1  = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.50, wspace=0.36)

    # (a) Cumulative RMSE
    ax = fig1.add_subplot(gs1[0, :2])
    for i, r in enumerate(results):
        ax.plot(steps_x, r["cum_rmse"], lstyles[i], color=colors[i],
                linewidth=2.2, label=short_names[i])
    ax.axvspan(num_steps - eval_steps + 0.5, num_steps + 0.5, alpha=0.08, color="orange")
    ax.axvline(num_steps - eval_steps + 0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Step"); ax.set_ylabel("Cumulative RMSE (dB)")
    ax.set_title("(a) Cumulative RMSE — intra-zone links", fontsize=12)
    ax.legend(fontsize=10, ncol=4); ax.grid(alpha=0.3)

    # (b) Per-step RMSE (EMA) + zone transitions
    ax = fig1.add_subplot(gs1[0, 2])
    for i, r in enumerate(results):
        ax.plot(steps_x, _ema(r["step_rmse"]), lstyles[i], color=colors[i],
                linewidth=2.0, label=short_names[i])
    if transition_steps:
        ax.vlines(transition_steps, *ax.get_ylim(), colors="black",
                  linewidths=0.4, alpha=0.2, label=f"transitions ({len(transition_steps)})")
    ax.set_xlabel("Step"); ax.set_ylabel("Per-step RMSE (dB, EMA)")
    ax.set_title("(b) Per-step RMSE\n(spikes = cold starts)", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (c) Eval RMSE bars
    ax = fig1.add_subplot(gs1[1, 0])
    rmses = [r["eval_rmse"] for r in results]
    bars = ax.bar(short_names, rmses, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Eval RMSE (dB)"); ax.set_ylim(0, max(rmses) * 1.25)
    ax.set_title(f"(c) Eval RMSE — last {eval_steps} steps", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # (d) Model size
    ax = fig1.add_subplot(gs1[1, 1])
    kbs  = [r["adapter_kb"] for r in results]
    bars = ax.bar(short_names, kbs, color=colors, edgecolor="white", width=0.55)
    for i, (bar, v) in enumerate(zip(bars, kbs)):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                    f"{v:.0f} KB", ha="center", va="bottom", fontsize=10, fontweight="bold")
        lbl = "zone-shareable" if results[i]["shareable"] else "server only"
        col = "#2e7d32" if results[i]["shareable"] else "#c62828"
        ax.text(bar.get_x() + bar.get_width() / 2, max(v / 2, 8),
                lbl, ha="center", va="center", fontsize=7,
                color=col, fontweight="bold", rotation=90)
    ax.set_ylabel("Exchangeable size (KB)"); ax.set_title("(d) Per-device model size", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # (e) Communication bars
    ax = fig1.add_subplot(gs1[1, 2])
    mbs  = [r["total_bytes"] / 1024**2 for r in results]
    bars = ax.bar(short_names, mbs, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, mbs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f"{v:.3f} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Total data exchanged (MB)")
    ax.set_title("(e) Communication overhead", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # (f) Cumulative bytes curve
    ax = fig1.add_subplot(gs1[2, :2])
    for i, r in enumerate(results):
        ax.plot(steps_x, [b / 1024**2 for b in r["bytes_curve"]],
                lstyles[i], color=colors[i], linewidth=2.2,
                label=f"{short_names[i]} ({mbs[i]:.3f} MB)")
    ax.axvline(num_steps - eval_steps + 0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Step"); ax.set_ylabel("Cumulative data exchanged (MB)")
    ax.set_title(
        f"(f) Communication — gate: same zone + ≤{share_range:.0f} m + RSSI > {rssi_share_thr:.0f} dBm",
        fontsize=11)
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)

    # (g) Accuracy–bandwidth tradeoff
    ax = fig1.add_subplot(gs1[2, 2])
    for i, r in enumerate(results):
        ax.scatter(mbs[i], r["eval_rmse"], s=200, color=colors[i],
                   marker=markers[i], zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(short_names[i], (mbs[i], r["eval_rmse"]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=8, color=colors[i], fontweight="bold")
    ax.set_xlabel("Total data exchanged (MB)"); ax.set_ylabel("Eval RMSE (dB)")
    ax.set_title("(g) Accuracy–bandwidth tradeoff\n(lower-left = better)", fontsize=12)
    ax.grid(alpha=0.3)

    fig1.suptitle(
        "Zone-based Radio Map Benchmark — Munich 3D Ray-Tracing\n"
        f"3.5 GHz  |  {num_nodes} nodes  |  {N_ZONES} anchor zones (50×50 m)  |  "
        f"{num_steps} steps  |  {total_zone_transitions} zone transitions",
        fontsize=13, fontweight="bold",
    )
    p1 = results_dir / "benchmark_zones.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig1)
    print(f"Saved → {p1}", flush=True)

    # ── Figure 2: Deep analysis (6 panels) ────────────────────────────────
    fig2 = plt.figure(figsize=(22, 16))
    gs2  = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.50, wspace=0.38)

    # (h) Per-zone RMSE
    ax = fig2.add_subplot(gs2[0, 0])
    x_zones = np.arange(N_ZONES)
    width = 0.2
    for i, (r, name) in enumerate(zip(results, short_names)):
        vals = [per_zone_rmse[r["name"]].get(z, float("nan")) for z in range(N_ZONES)]
        ax.bar(x_zones + (i - 1.5) * width, vals, width,
               color=colors[i], label=name, edgecolor="white", alpha=0.9)
    ax.set_xticks(x_zones); ax.set_xticklabels([ZONE_LABELS[z] for z in range(N_ZONES)])
    ax.set_ylabel("RMSE (dB)"); ax.legend(fontsize=8, ncol=2)
    ax.set_title("(h) Per-zone RMSE\n(which zones benefit most from sharing?)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # (i) Cold-start recovery curve
    ax = fig2.add_subplot(gs2[0, 1])
    max_dt = 25
    dt_ax  = list(range(max_dt + 1))
    # FSPL as flat reference
    fspl_avg = np.nanmean([r["abs_err"] for r in res_fspl["eval_records"]]) if res_fspl["eval_records"] else 0
    ax.axhline(fspl_avg, color=colors[0], linestyle="--", linewidth=1.5,
               alpha=0.7, label="FSPL (reference)")
    for idx, (r_name, curve) in enumerate(coldstart_curves.items(), start=1):
        ax.plot(dt_ax, curve[:max_dt + 1],
                lstyles[idx], color=colors[idx], linewidth=2.2, label=r_name.replace("\n", " "))
    ax.set_xlabel("Steps since zone entry"); ax.set_ylabel("Mean absolute error (dB)")
    ax.set_title("(i) Cold-start recovery\n(how fast does the model converge after zone entry?)",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.annotate("← warm start\n   advantage", xy=(1, coldstart_curves.get(
        res_greedy["name"], [float("nan")])[1] if coldstart_curves else 0),
        xytext=(5, fspl_avg * 0.95), fontsize=8, color=colors[2], alpha=0.8)

    # (j) RMSE vs distance bin
    ax = fig2.add_subplot(gs2[0, 2])
    x_d = np.arange(len(DIST_LABELS))
    for i, (r, name) in enumerate(zip(results, short_names)):
        vals = dist_break[r["name"]]
        ax.bar(x_d + (i - 1.5) * width, vals, width,
               color=colors[i], label=name, edgecolor="white", alpha=0.9)
    ax.set_xticks(x_d); ax.set_xticklabels(DIST_LABELS, fontsize=9)
    ax.set_ylabel("Mean abs error (dB)"); ax.legend(fontsize=8, ncol=2)
    ax.set_title("(j) Error vs distance\n(sharing helps more at longer ranges?)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # (k) LOS vs NLOS breakdown
    ax = fig2.add_subplot(gs2[1, 0])
    x_ln = np.array([0, 1])
    for i, (r, name) in enumerate(zip(results, short_names)):
        los_err, nlos_err = nlos_break[r["name"]]
        ax.bar(x_ln + (i - 1.5) * width, [los_err, nlos_err], width,
               color=colors[i], label=name, edgecolor="white", alpha=0.9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LOS (clear path)", "NLOS (buildings)"])
    ax.set_ylabel("Mean abs error (dB)"); ax.legend(fontsize=8, ncol=2)
    ax.set_title("(k) LOS vs NLOS breakdown\n(sharing should mainly help NLOS links)",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # (l) Error CDF
    ax = fig2.add_subplot(gs2[1, 1])
    for i, (r, name) in enumerate(zip(results, short_names)):
        if r["eval_records"]:
            errs, cdf_vals = cdfs[r["name"]]
            ax.plot(errs, cdf_vals * 100, lstyles[i], color=colors[i],
                    linewidth=2.2, label=name)
    ax.axhline(90, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.text(0.01, 91, "90th percentile", fontsize=8, alpha=0.6, transform=ax.get_yaxis_transform())
    ax.set_xlabel("Absolute error (dB)"); ax.set_ylabel("Fraction of links ≤ error (%)")
    ax.set_title("(l) Error CDF\n(90th-percentile error tells reliability story)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (m) Per-zone bytes vs RMSE scatter (Greedy only, by zone)
    ax = fig2.add_subplot(gs2[1, 2])
    zone_colors_local = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]
    for z in range(N_ZONES):
        mb  = per_zone_bytes_greedy[z][0]
        rmse_z_greedy = per_zone_rmse[res_greedy["name"]].get(z, float("nan"))
        rmse_z_local  = per_zone_rmse[res_local["name"]].get(z, float("nan"))
        if not math.isnan(rmse_z_greedy):
            ax.scatter(mb, rmse_z_greedy, s=180, color=zone_colors_local[z],
                       marker="o", zorder=5, edgecolors="white", linewidths=1.2)
            ax.scatter(mb, rmse_z_local, s=180, color=zone_colors_local[z],
                       marker="s", zorder=4, edgecolors="white", linewidths=1.2, alpha=0.5)
            ax.annotate(ZONE_LABELS[z],
                        (mb, rmse_z_greedy),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=8, color=zone_colors_local[z], fontweight="bold")
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="gray", label="Greedy LoRA", markersize=8,
               linestyle="none"),
        Line2D([0], [0], marker="s", color="gray", label="Local LoRA (ref)", markersize=8,
               linestyle="none", alpha=0.5),
    ]
    ax.legend(handles=legend_elems, fontsize=9)
    ax.set_xlabel("Zone comm. cost (MB)"); ax.set_ylabel("Eval RMSE (dB)")
    ax.set_title("(m) Per-zone cost vs accuracy\n(Greedy LoRA — each point = one zone)",
                 fontsize=11)
    ax.grid(alpha=0.3)

    fig2.suptitle(
        "Zone-based Benchmark — Deep Analysis\n"
        f"Munich 3D  |  {N_ZONES}×50 m zones  |  {total_zone_transitions} zone transitions",
        fontsize=13, fontweight="bold",
    )
    p2 = results_dir / "benchmark_zones_analysis.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig2)
    print(f"Saved → {p2}", flush=True)

    # ── Summary text ───────────────────────────────────────────────────────
    W = 95
    summary_path = results_dir / "benchmark_zones_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Zone-based Benchmark — Munich 3D  |  {N_ZONES} zones  |  "
                f"{num_steps} steps  |  {total_zone_transitions} zone transitions\n")
        f.write(f"Sharing gate: same zone + ≤{share_range}m + RSSI > {rssi_share_thr} dBm\n")
        f.write("=" * W + "\n")
        f.write(f"{'Strategy':<30} {'Eval RMSE':>9} {'Eval MAE':>9} "
                f"{'Size (KB)':>10} {'Comm (MB)':>10} {'Shareable':>10}\n")
        f.write("-" * W + "\n")
        for r, n in zip(results, short_names):
            f.write(f"{n:<30} {r['eval_rmse']:>9.2f}  {r['eval_mae']:>8.2f}  "
                    f"{r['adapter_kb']:>9.1f}  {r['total_bytes']/1024**2:>9.3f}  "
                    f"{'Yes' if r['shareable'] else 'No':>9}\n")
        f.write("=" * W + "\n")

        f.write("\n── Per-zone RMSE ──\n")
        f.write(f"{'Strategy':<30}")
        for z in range(N_ZONES): f.write(f"  {ZONE_LABELS[z]:>8}")
        f.write("\n")
        for r, n in zip(results, short_names):
            f.write(f"{n:<30}")
            for z in range(N_ZONES):
                v = per_zone_rmse[r["name"]].get(z, float("nan"))
                f.write(f"  {v:>8.2f}")
            f.write("\n")

        f.write("\n── Error CDF percentiles (abs error in dB) ──\n")
        f.write(f"{'Strategy':<30}  {'P50':>6}  {'P75':>6}  {'P90':>6}  {'P95':>6}\n")
        for r, n in zip(results, short_names):
            if r["eval_records"]:
                errs = np.array([rec["abs_err"] for rec in r["eval_records"]])
                p50, p75, p90, p95 = np.percentile(errs, [50, 75, 90, 95])
                f.write(f"{n:<30}  {p50:>6.2f}  {p75:>6.2f}  {p90:>6.2f}  {p95:>6.2f}\n")

        f.write("\n── LOS vs NLOS mean abs error ──\n")
        f.write(f"{'Strategy':<30}  {'LOS':>8}  {'NLOS':>8}\n")
        for r, n in zip(results, short_names):
            los_e, nlos_e = nlos_break[r["name"]]
            f.write(f"{n:<30}  {los_e:>8.2f}  {nlos_e:>8.2f}\n")

        f.write("\n── Gain of Greedy LoRA ──\n")
        f.write(f"  vs FSPL:       {res_fspl['eval_rmse']  - res_greedy['eval_rmse']:+.2f} dB\n")
        f.write(f"  vs Local LoRA: {res_local['eval_rmse'] - res_greedy['eval_rmse']:+.2f} dB"
                f"  ← benefit of sharing\n")
        f.write(f"  vs Centralized:{res_greedy['eval_rmse'] - res_central['eval_rmse']:+.2f} dB"
                f"  ← gap to centralized baseline\n")
        g_mb = res_greedy["total_bytes"] / 1024**2
        c_mb = res_central["total_bytes"] / 1024**2
        f.write(f"\n── Communication ──\n")
        f.write(f"  Greedy adapter swaps: {g_mb:.3f} MB\n")
        f.write(f"  Centralized uploads:  {c_mb:.3f} MB\n")
        if c_mb > 0 and g_mb > 0:
            f.write(f"  Ratio: {c_mb/g_mb:.2f}× more for centralized\n")
        f.write(f"  Zone transitions: {total_zone_transitions}\n")

    print(f"Saved → {summary_path}", flush=True)


def main() -> None:
    run_benchmarks_zones(Path("benchmark_results"))


if __name__ == "__main__":
    main()
