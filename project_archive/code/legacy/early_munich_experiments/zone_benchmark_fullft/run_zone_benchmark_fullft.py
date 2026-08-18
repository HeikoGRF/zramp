"""
run_zone_benchmark_fullft.py — Self-contained zone benchmark (FULL fine-tuning, no adapters).

All outputs are written into ./results/ within this folder.
"""

import math
import random
import time
from collections import defaultdict
from pathlib import Path

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

from sionna_baseline.mobility import RandomWaypoint
from sionna_baseline.radio_mlp import MapConfig, RadioFeatureEncoder, RadioMLP
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sionna_baseline.backbone_utils import (  # type: ignore
    sample_node_tx_powers,
    compute_rssi,
    load_or_pretrain_full_backbone,
)


# ---------------- Zones (2x2) ----------------
N_COL, N_ROW = 2, 2
N_ZONES = N_COL * N_ROW
ZONE_LABELS = {0: "Z0 SW", 1: "Z1 SE", 2: "Z2 NW", 3: "Z3 NE"}


def get_zone_id(x: float, y: float, bounds_x: tuple, bounds_y: tuple) -> int:
    zone_w = (bounds_x[1] - bounds_x[0]) / N_COL
    zone_h = (bounds_y[1] - bounds_y[0]) / N_ROW
    col = min(int((x - bounds_x[0]) / zone_w), N_COL - 1)
    row = min(int((y - bounds_y[0]) / zone_h), N_ROW - 1)
    return row * N_COL + col


# ---------------- Building grid helpers ----------------
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
            faces = np.array(mesh.faces_buffer()).reshape(-1, 3)
        except Exception:
            continue
        verts2d = vertices[:, :2]
        for face in faces:
            poly = verts2d[face]
            p = MplPath(poly)
            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            if xmax < bounds_x[0] or xmin > bounds_x[1] or ymax < bounds_y[0] or ymin > bounds_y[1]:
                continue
            in_bbox = (
                (points[:, 0] >= xmin - res)
                & (points[:, 0] <= xmax + res)
                & (points[:, 1] >= ymin - res)
                & (points[:, 1] <= ymax + res)
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
        (idx_x >= 0)
        & (idx_x < grid_mask.shape[1])
        & (idx_y >= 0)
        & (idx_y < grid_mask.shape[0])
    )
    unique = set(zip(idx_y[valid], idx_x[valid]))
    return sum(1 for y, x in unique if grid_mask[y, x])


def compute_fspl(d: float, fspl_const: float,
                 tx_power_dbm: float = 0.0, eps: float = 1e-6) -> float:
    """FSPL prediction: P_tx + path-loss (dBm)."""
    return tx_power_dbm - 20.0 * math.log10(d + eps) + fspl_const


def _cum_rmse(sse: float, n: int) -> float:
    return math.sqrt(sse / n) if n > 0 else 0.0


def _metrics(preds, truths):
    p, t = np.array(preds), np.array(truths)
    return float(np.sqrt(np.mean((p - t) ** 2))), float(np.mean(np.abs(p - t)))


def _model_exchange_bytes(model: nn.Module) -> int:
    """Both nodes send their full model → ×2."""
    return sum(p.numel() for p in model.parameters()) * 4 * 2


_OBS_UPLOAD_BYTES = (8 + 1) * 4  # 8 features + 1 RSSI


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ───────────────────────────────────────────────────────────
    num_nodes = 20
    num_steps = 100
    eval_steps = 20

    dt = 1.0
    obs_range = 250.0
    share_range = 30.0
    rssi_share_thr = -100.0
    bounds_x = (0, 99)
    bounds_y = (-199, -100)
    freq_hz = 3.5e9
    c_light = 3e8
    fspl_const = -20.0 * math.log10(4.0 * math.pi * freq_hz / c_light)
    noise_floor = -150.0

    lr_local = 1e-3
    lr_central = 1e-3
    grad_steps_local = 25
    grad_steps_central = 50
    min_samples = 8
    max_buf = 500
    max_train = 500
    share_every = 5

    # Step 5: data sparsity (collect fewer observations)
    obs_every_k_steps = 1
    obs_keep_prob = 1.0

    device = torch.device("cpu")
    map_cfg = MapConfig(x_min=bounds_x[0], x_max=bounds_x[1], y_min=bounds_y[0], y_max=bounds_y[1])
    encoder = RadioFeatureEncoder(map_cfg)

    print(f"Zone benchmark FULL-FT (no adapters). Steps={num_steps}, eval_last={eval_steps}", flush=True)
    print(f"Data sparsity: obs_every_k_steps={obs_every_k_steps}, obs_keep_prob={obs_keep_prob}", flush=True)

    # ── Scene ────────────────────────────────────────────────────────────
    scene = rt.load_scene(rt.scene.munich)
    scene.frequency = freq_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    nodes: dict[int, np.ndarray] = {}
    for i in range(num_nodes):
        nodes[i] = np.array([np.random.uniform(*bounds_x), np.random.uniform(*bounds_y), 1.5])
        scene.add(Transmitter(name=f"tx_{i}", position=nodes[i]))
        scene.add(Receiver(name=f"rx_{i}", position=nodes[i]))

    # Each node has a fixed TX power for the full run (heterogeneous hardware)
    node_tx_power_dbm: dict[int, float] = sample_node_tx_powers(num_nodes)

    mobility = RandomWaypoint(bounds_x=bounds_x, bounds_y=bounds_y, velocity=(2.0, 5.0), pause_time=1.0)

    gpus = tf.config.list_physical_devices("GPU")
    compute_dev = "/GPU:0" if gpus else "/CPU:0"
    print(f"Compute device: {gpus[0].name if gpus else 'CPU'}", flush=True)

    grid_mask = _build_grid_safe(scene, bounds_x, bounds_y, res=1.0)
    print(f"Building grid occupied cells: {int(np.sum(grid_mask))}", flush=True)

    # ── Collect simulation data ──────────────────────────────────────────
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
                scene,
                max_depth=3,
                samples_per_src=int(1e5),
                specular_reflection=True,
                diffraction=True,
                edge_diffraction=False,
            )
        try:
            a_cplx = np.array(paths.a[0]) + 1j * np.array(paths.a[1])
            a_cplx = np.squeeze(a_cplx, axis=(1, 3))
            power = np.sum(np.abs(a_cplx) ** 2, axis=-1)
        except Exception:
            power = np.zeros((num_nodes, num_nodes))

        node_zones = {i: get_zone_id(nodes[i][0], nodes[i][1], bounds_x, bounds_y) for i in range(num_nodes)}
        step_transitions = sum(
            1 for i in range(num_nodes) if i in prev_node_zones and prev_node_zones[i] != node_zones[i]
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
                if collect_obs and (np.random.rand() <= obs_keep_prob):
                    obs.append(
                        {
                            "tx_i": tx_i,
                            "rx_i": rx_i,
                            "tx_zone": tx_zone,
                            "rx_zone": rx_zone,
                            "rssi": rssi,
                            "d": d,
                            "nc": nc,
                            "fspl": compute_fspl(d, fspl_const,
                                                 tx_power_dbm=node_tx_power_dbm[tx_i]),
                            "feat": feat,
                        }
                    )
                key = (min(tx_i, rx_i), max(tx_i, rx_i))
                if key not in link_rssi or rssi < link_rssi[key]:
                    link_rssi[key] = rssi

        share_contacts: set[tuple[int, int]] = set()
        for a in range(num_nodes):
            for b in range(a + 1, num_nodes):
                if node_zones[a] != node_zones[b]:
                    continue
                d2d = float(np.linalg.norm(nodes[a][:2] - nodes[b][:2]))
                if d2d > share_range:
                    continue
                key = (a, b)
                if link_rssi.get(key, noise_floor) > rssi_share_thr:
                    share_contacts.add(key)

        simulation_data.append(
            {
                "step": step,
                "obs": obs,
                "node_zones": node_zones,
                "share_contacts": share_contacts,
                "n_transitions": step_transitions,
            }
        )
        if (step + 1) % 20 == 0:
            n_total = sum(len(s["obs"]) for s in simulation_data)
            n_intra = sum(sum(1 for o in s["obs"] if o["tx_zone"] == o["rx_zone"]) for s in simulation_data)
            print(
                f"  Step {step+1:3d}/{num_steps}  obs={n_total:5d} (intra={n_intra}) "
                f"trans={total_zone_transitions}  elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    # ── Load or pretrain base model ───────────────────────────────────────
    # Tries backbone_urban.pt in parent dir; falls back to FSPL pretraining.
    base_state, eff_fspl = load_or_pretrain_full_backbone(
        encoder, fspl_const,
        backbone_path=str(Path(__file__).resolve().parents[1] / "backbone_etoile.pt"),
        hidden_sizes=(128, 128),
        n_samples=50_000, epochs=200, lr=1e-3,
    )

    loss_fn = nn.MSELoss()

    def _fit_full(model, opt, buf):
        unc = [(f, r) for f, r in buf if r != noise_floor]
        if len(unc) < min_samples:
            return
        samp = unc if len(unc) <= max_train else random.sample(unc, max_train)
        x = torch.stack([f for f, _ in samp]).to(device)
        y = torch.tensor([r for _, r in samp], dtype=torch.float32).to(device)
        model.train()
        for _ in range(grad_steps_local):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        model.eval()

    # ====================================================================
    # Approach 1 — FSPL baseline
    # ====================================================================
    def run_fspl(sim_data):
        cum_sse = cum_n = 0.0
        curve, step_rmse = [], []
        eval_p, eval_t = [], []
        for sd in sim_data:
            sp, st = [], []
            for o in sd["obs"]:
                if o["tx_zone"] != o["rx_zone"]:
                    continue
                sp.append(o["fspl"])
                st.append(o["rssi"])
                if sd["step"] >= num_steps - eval_steps:
                    eval_p.append(o["fspl"])
                    eval_t.append(o["rssi"])
            if st:
                arr_p, arr_t = np.array(sp), np.array(st)
                sse = float(np.sum((arr_p - arr_t) ** 2))
                cum_sse += sse
                cum_n += len(st)
                step_rmse.append(math.sqrt(sse / len(st)))
            else:
                step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
            curve.append(_cum_rmse(cum_sse, int(cum_n)))
        rmse, mae = _metrics(eval_p, eval_t)
        return dict(name="FSPL", cum_rmse=curve, step_rmse=step_rmse, eval_rmse=rmse, eval_mae=mae,
                    total_bytes=0.0, bytes_curve=[0.0]*len(curve))

    # ====================================================================
    # Approach 2 — Local Full FT (no sharing, cold start)
    # ====================================================================
    def run_local_fullft(sim_data):
        models, opts, bufs = {}, {}, {}
        node_zone_rt, zone_entry = {}, {}

        def cold_start(i, z, step):
            m = RadioMLP(input_dim=8, hidden_sizes=(128, 128), fspl_const=eff_fspl)
            m.load_state(base_state)
            m.to(device)
            m.train()
            models[i] = m
            opts[i] = optim.Adam(models[i].parameters(), lr=lr_local, weight_decay=1e-4)
            bufs[i] = []
            node_zone_rt[i] = z
            zone_entry[i] = step

        for i in range(num_nodes):
            cold_start(i, sim_data[0]["node_zones"][i], 0)

        cum_sse = cum_n = 0.0
        curve, step_rmse = [], []
        eval_p, eval_t = [], []

        for sd in sim_data:
            step = sd["step"]
            for i in range(num_nodes):
                z = sd["node_zones"][i]
                if z != node_zone_rt[i]:
                    cold_start(i, z, step)

            sp, st = [], []
            for o in sd["obs"]:
                rx = o["rx_i"]
                if o["tx_zone"] != node_zone_rt[rx]:
                    continue
                pred = float(models[rx](o["feat"].unsqueeze(0)).item())
                sp.append(pred)
                st.append(o["rssi"])
                if step >= num_steps - eval_steps:
                    eval_p.append(pred)
                    eval_t.append(o["rssi"])
                bufs[rx].append((o["feat"].detach().clone(), o["rssi"]))
                bufs[rx] = bufs[rx][-max_buf:]

            for i in range(num_nodes):
                _fit_full(models[i], opts[i], bufs[i])

            if st:
                arr_p, arr_t = np.array(sp), np.array(st)
                sse = float(np.sum((arr_p - arr_t) ** 2))
                cum_sse += sse
                cum_n += len(st)
                step_rmse.append(math.sqrt(sse / len(st)))
            else:
                step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
            curve.append(_cum_rmse(cum_sse, int(cum_n)))

        rmse, mae = _metrics(eval_p, eval_t)
        return dict(name="Local FullFT", cum_rmse=curve, step_rmse=step_rmse, eval_rmse=rmse, eval_mae=mae,
                    total_bytes=0.0, bytes_curve=[0.0]*len(curve))

    # ====================================================================
    # Approach 3 — Greedy Full Model Sharing (zone cache + RSSI gate)
    # ====================================================================
    def run_greedy_fullshare(sim_data):
        models, opts, bufs = {}, {}, {}
        node_zone_rt, zone_entry = {}, {}
        zone_cache: dict[int, dict] = {}
        cum_bytes = 0.0
        bytes_curve = []

        def enter_zone(i, z, step):
            m = RadioMLP(input_dim=8, hidden_sizes=(128, 128), fspl_const=eff_fspl)
            m.load_state(base_state)
            if z in zone_cache:
                m.load_state(zone_cache[z])
            m.to(device)
            models[i] = m
            opts[i] = optim.Adam(models[i].parameters(), lr=lr_local, weight_decay=1e-4)
            bufs[i] = []
            node_zone_rt[i] = z
            zone_entry[i] = step

        def save_cache(i, z):
            if len(bufs.get(i, [])) < min_samples:
                return
            zone_cache[z] = models[i].get_state()

        for i in range(num_nodes):
            enter_zone(i, sim_data[0]["node_zones"][i], 0)

        cum_sse = cum_n = 0.0
        curve, step_rmse = [], []
        eval_p, eval_t = [], []

        for sd in sim_data:
            step = sd["step"]
            # transitions
            for i in range(num_nodes):
                z = sd["node_zones"][i]
                if z != node_zone_rt[i]:
                    save_cache(i, node_zone_rt[i])
                    enter_zone(i, z, step)

            sp, st = [], []
            for o in sd["obs"]:
                rx = o["rx_i"]
                if o["tx_zone"] != node_zone_rt[rx]:
                    continue
                pred = float(models[rx](o["feat"].unsqueeze(0)).item())
                sp.append(pred)
                st.append(o["rssi"])
                if step >= num_steps - eval_steps:
                    eval_p.append(pred)
                    eval_t.append(o["rssi"])
                bufs[rx].append((o["feat"].detach().clone(), o["rssi"]))
                bufs[rx] = bufs[rx][-max_buf:]

            for i in range(num_nodes):
                _fit_full(models[i], opts[i], bufs[i])

            # share full model every share_every steps
            if step % share_every == 0:
                for (a, b) in sd["share_contacts"]:
                    za = node_zone_rt[a]
                    zb = node_zone_rt[b]
                    if za != zb:
                        continue
                    sa = models[a].get_state()
                    sb = models[b].get_state()
                    # obs-count weights
                    n_a = max(1, len(bufs[a]))
                    n_b = max(1, len(bufs[b]))
                    w_a = n_a / (n_a + n_b)
                    w_b = 1.0 - w_a
                    avg = {k: w_a * sa[k] + w_b * sb[k] for k in sa}
                    models[a].load_state(avg)
                    models[b].load_state(avg)
                    zone_cache[za] = avg
                    cum_bytes += _model_exchange_bytes(models[a])

            bytes_curve.append(float(cum_bytes))

            if st:
                arr_p, arr_t = np.array(sp), np.array(st)
                sse = float(np.sum((arr_p - arr_t) ** 2))
                cum_sse += sse
                cum_n += len(st)
                step_rmse.append(math.sqrt(sse / len(st)))
            else:
                step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
            curve.append(_cum_rmse(cum_sse, int(cum_n)))

        rmse, mae = _metrics(eval_p, eval_t)
        return dict(name="Greedy FullShare", cum_rmse=curve, step_rmse=step_rmse, eval_rmse=rmse, eval_mae=mae,
                    total_bytes=float(cum_bytes), bytes_curve=bytes_curve)

    # ====================================================================
    # Approach 4 — Centralized Full FT (same architecture as devices)
    # ====================================================================
    def run_centralized_fullft(sim_data):
        model = RadioMLP(input_dim=8, hidden_sizes=(128, 128), fspl_const=eff_fspl)
        model.load_state(base_state)
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=lr_central, weight_decay=1e-4)
        buf = []
        cum_bytes = 0.0
        bytes_curve = []

        cum_sse = cum_n = 0.0
        curve, step_rmse = [], []
        eval_p, eval_t = [], []

        for sd in sim_data:
            step = sd["step"]
            sp, st, step_bytes = [], [], 0
            for o in sd["obs"]:
                pred = float(model(o["feat"].unsqueeze(0)).item())
                sp.append(pred)
                st.append(o["rssi"])
                buf.append((o["feat"].detach().clone(), o["rssi"]))
                if len(buf) > max_buf * num_nodes:
                    buf = random.sample(buf, max_buf * num_nodes)
                step_bytes += _OBS_UPLOAD_BYTES
                if step >= num_steps - eval_steps and o["tx_zone"] == o["rx_zone"]:
                    eval_p.append(pred)
                    eval_t.append(o["rssi"])

            # train more steps centrally
            unc = [(f, r) for f, r in buf if r != noise_floor]
            if len(unc) >= min_samples:
                samp = unc if len(unc) <= max_train * 4 else random.sample(unc, max_train * 4)
                x = torch.stack([f for f, _ in samp]).to(device)
                y = torch.tensor([r for _, r in samp], dtype=torch.float32).to(device)
                model.train()
                for _ in range(grad_steps_central):
                    opt.zero_grad()
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    opt.step()
                model.eval()

            cum_bytes += step_bytes
            bytes_curve.append(float(cum_bytes))

            intra = [(p, t) for p, t, o in zip(sp, st, sd["obs"]) if o["tx_zone"] == o["rx_zone"]]
            if intra:
                arr_p = np.array([x[0] for x in intra])
                arr_t = np.array([x[1] for x in intra])
                sse = float(np.sum((arr_p - arr_t) ** 2))
                cum_sse += sse
                cum_n += len(intra)
                step_rmse.append(math.sqrt(sse / len(intra)))
            else:
                step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
            curve.append(_cum_rmse(cum_sse, int(cum_n)))

        rmse, mae = _metrics(eval_p, eval_t)
        return dict(name="Centralized FullFT", cum_rmse=curve, step_rmse=step_rmse, eval_rmse=rmse, eval_mae=mae,
                    total_bytes=float(cum_bytes), bytes_curve=bytes_curve)

    # ── Run ───────────────────────────────────────────────────────────────
    res_fspl = run_fspl(simulation_data)
    res_local = run_local_fullft(simulation_data)
    res_greedy = run_greedy_fullshare(simulation_data)
    res_central = run_centralized_fullft(simulation_data)
    results = [res_fspl, res_local, res_greedy, res_central]

    # ── Plots (overview) ─────────────────────────────────────────────────
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#F44336"]
    lstyles = ["--", "-.", "-", ":"]
    markers = ["s", "^", "o", "D"]
    steps_x = list(range(1, num_steps + 1))
    names = [r["name"] for r in results]

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, :2])
    for i, r in enumerate(results):
        ax.plot(steps_x, r["cum_rmse"], lstyles[i], color=colors[i], linewidth=2.2, label=names[i])
    ax.axvspan(num_steps - eval_steps + 0.5, num_steps + 0.5, alpha=0.08, color="orange")
    ax.set_title("(a) Cumulative RMSE (intra-zone)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative RMSE (dB)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)

    ax = fig.add_subplot(gs[0, 2])
    for i, r in enumerate(results):
        ax.scatter(r["total_bytes"] / 1024**2, r["eval_rmse"], s=180, color=colors[i], marker=markers[i],
                   edgecolors="white", linewidths=1.2)
        ax.annotate(names[i], (r["total_bytes"] / 1024**2, r["eval_rmse"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, fontweight="bold")
    ax.set_title("(b) Accuracy–communication tradeoff")
    ax.set_xlabel("Total data exchanged (MB)")
    ax.set_ylabel("Eval RMSE (dB)")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, :2])
    for i, r in enumerate(results):
        ax.plot(steps_x, [b / 1024**2 for b in r["bytes_curve"]], lstyles[i], color=colors[i], linewidth=2.2,
                label=f"{names[i]} ({r['total_bytes']/1024**2:.2f} MB)")
    ax.set_title("(c) Cumulative communication")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative MB")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    rmses = [r["eval_rmse"] for r in results]
    bars = ax.bar(names, rmses, color=colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(f"(d) Eval RMSE (last {eval_steps} steps)")
    ax.set_ylabel("RMSE (dB)")
    ax.set_ylim(0, max(rmses) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Zone benchmark (FULL fine-tuning, no adapters)\n"
        f"{num_nodes} nodes | {num_steps} steps | {total_zone_transitions} zone transitions",
        fontsize=13,
        fontweight="bold",
    )

    p = out_dir / "benchmark_fullft.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {p}", flush=True)

    # ── Summary text ─────────────────────────────────────────────────────
    summary = out_dir / "benchmark_fullft_summary.txt"
    with summary.open("w") as f:
        f.write(f"Zone benchmark FULL-FT (no adapters) | steps={num_steps} | eval_last={eval_steps}\n")
        f.write(f"share_gate: same zone + <= {share_range}m + RSSI > {rssi_share_thr} dBm\n")
        f.write(f"zone_transitions: {total_zone_transitions}\n\n")
        f.write(f"{'Strategy':<22} {'EvalRMSE':>8} {'EvalMAE':>8} {'Comm(MB)':>10}\n")
        f.write("-" * 55 + "\n")
        for r in results:
            f.write(f"{r['name']:<22} {r['eval_rmse']:>8.2f} {r['eval_mae']:>8.2f} {r['total_bytes']/1024**2:>10.3f}\n")
    print(f"Saved → {summary}", flush=True)


if __name__ == "__main__":
    main()

