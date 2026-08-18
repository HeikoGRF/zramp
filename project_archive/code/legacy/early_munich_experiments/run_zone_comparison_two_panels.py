"""
run_zone_comparison_two_panels.py

Collect the Sionna zone simulation ONCE (same node walks), then replay it under:
  (A) Adapter strategy (LoRA adapters only)
  (B) Full fine-tuning strategy (no adapters)

Outputs:
  - benchmark_results/zone_compare_*.png
  - benchmark_results/zone_compare_summary.txt
"""

import math
import random
import time
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf
import sionna.rt as rt
from sionna.rt import PlanarArray, Transmitter, Receiver, PathSolver

from sionna_baseline.radio_mlp_lora import MapConfig, RadioFeatureEncoder, RadioMLPWithLoRA
from sionna_baseline.mobility import RandomWaypoint
from sionna_baseline.backbone_utils import (
    sample_node_tx_powers,
    compute_rssi,
    load_or_pretrain_lora_backbone,
)

# Import full-FT backbone from the self-contained folder
_FULLFT_DIR = Path(__file__).resolve().parent / "zone_benchmark_fullft"
import importlib.util

_RADIO_MLP_PATH = _FULLFT_DIR / "sionna_baseline" / "radio_mlp.py"
_spec = importlib.util.spec_from_file_location("zone_benchmark_fullft_radio_mlp", _RADIO_MLP_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Failed to load module spec from {_RADIO_MLP_PATH}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
RadioMLP = _mod.RadioMLP  # type: ignore[attr-defined]


N_COL, N_ROW = 2, 2
N_ZONES = N_COL * N_ROW


def get_zone_id(x: float, y: float, bounds_x: tuple, bounds_y: tuple) -> int:
    zone_w = (bounds_x[1] - bounds_x[0]) / N_COL
    zone_h = (bounds_y[1] - bounds_y[0]) / N_ROW
    col = min(int((x - bounds_x[0]) / zone_w), N_COL - 1)
    row = min(int((y - bounds_y[0]) / zone_h), N_ROW - 1)
    return row * N_COL + col


def _build_grid_safe(scene, bounds_x, bounds_y, res=1.0):
    from matplotlib.path import Path as MplPath

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


def _ema(values: list[float], alpha: float = 0.2) -> list[float]:
    if not values:
        return []
    out = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def _fit_lora(model, opt, buf, loss_fn, noise_floor, min_samples, max_train, grad_steps, device):
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


def _fit_full(model, opt, buf, loss_fn, noise_floor, min_samples, max_train, grad_steps, device):
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


def _adapter_exchange_bytes(model: RadioMLPWithLoRA) -> int:
    return sum(p.numel() for p in model.adapter_parameters()) * 4 * 2


def _model_exchange_bytes(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters()) * 4 * 2


def main() -> None:
    # ---------------- config ----------------
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_nodes = 20
    num_steps = 200
    eval_steps = 20

    dt = 1.0
    obs_range = 250.0
    share_range = 30.0
    rssi_share_thr = -100.0
    bounds_x = (0, 99)
    bounds_y = (-199, -100)
    zone_size = 50  # 2x2 grid of 50m zones

    # Data sparsity knobs (Step 5) — keep identical for both replays
    obs_every_k_steps = 1
    obs_keep_prob = 1.0

    lr_lora = 1e-2
    grad_steps_lora = 20
    lr_full = 1e-3
    grad_steps_full = 25

    min_samples = 8
    max_buf = 500
    max_train = 500
    share_every = 5

    freq_hz = 3.5e9
    c_light = 3e8
    fspl_const = -20.0 * math.log10(4.0 * math.pi * freq_hz / c_light)
    noise_floor = -150.0

    device = torch.device("cpu")

    out_dir = Path("benchmark_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- collect simulation once ----------------
    print("Collecting Sionna simulation data (once) …", flush=True)
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

    map_cfg = MapConfig(x_min=bounds_x[0], x_max=bounds_x[1], y_min=bounds_y[0], y_max=bounds_y[1])
    encoder = RadioFeatureEncoder(map_cfg)

    solver = PathSolver()
    sim_data: list[dict] = []

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
        obs = []
        link_rssi = {}
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
                key  = (min(tx_i, rx_i), max(tx_i, rx_i))
                if key not in link_rssi or rssi < link_rssi[key]:
                    link_rssi[key] = rssi

                if collect_obs and (np.random.rand() <= obs_keep_prob):
                    obs.append(
                        dict(
                            tx_i=tx_i,
                            rx_i=rx_i,
                            tx_zone=node_zones[tx_i],
                            rx_zone=node_zones[rx_i],
                            rssi=rssi,
                            d=d,
                            nc=nc,
                            fspl=compute_fspl(d, fspl_const,
                                              tx_power_dbm=node_tx_power_dbm[tx_i]),
                            feat=feat,
                        )
                    )

        share_contacts = set()
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

        sim_data.append(dict(step=step, obs=obs, node_zones=node_zones, share_contacts=share_contacts))
        if (step + 1) % 20 == 0:
            print(f"  step {step+1:3d}/{num_steps}  obs={sum(len(s['obs']) for s in sim_data):6d}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    # ---------------- pretrain / load shared base (same for both) ----------------
    # LoRA backbone
    base_state, eff_fspl = load_or_pretrain_lora_backbone(
        encoder, fspl_const,
        backbone_path="backbone_etoile.pt",
        hidden_sizes=(128, 128), rank=16, alpha=1.0,
        n_samples=50_000, epochs=200, lr=1e-3,
    )
    # Full-FT backbone (shares the same urban checkpoint when available)
    from sionna_baseline.backbone_utils import load_or_pretrain_full_backbone
    base_full_state, eff_fspl_full = load_or_pretrain_full_backbone(
        encoder, fspl_const,
        backbone_path="backbone_etoile.pt",
        hidden_sizes=(128, 128),
        n_samples=50_000, epochs=200, lr=1e-3,
    )

    loss_fn = nn.MSELoss()

    # ---------------- replay: adapter strategy ----------------
    def replay_adapter():
        def new_adapter():
            m = RadioMLPWithLoRA(input_dim=8, hidden_sizes=(128, 128), rank=16, alpha=1.0, fspl_const=eff_fspl)
            m.load_base_state(base_state)
            m.to(device)
            return m

        # FSPL
        def run_fspl():
            sse = n = 0.0
            cum_curve: list[float] = []
            step_rmse: list[float] = []
            for sd in sim_data:
                step_sse = 0.0
                step_n = 0
                for o in sd["obs"]:
                    if o["tx_zone"] != o["rx_zone"]:
                        continue
                    e = o["fspl"] - o["rssi"]
                    step_sse += e * e
                    step_n += 1
                if step_n > 0:
                    sse += step_sse
                    n += step_n
                    step_rmse.append(math.sqrt(step_sse / step_n))
                else:
                    step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
                cum_curve.append(_cum_rmse(sse, int(n)))
            return dict(cum_rmse=cum_curve, step_rmse=step_rmse, bytes_curve=[0.0] * num_steps, total_bytes=0.0)

        # Local LoRA (cold start per zone)
        def run_local():
            adapters, opts, bufs, node_zone = {}, {}, {}, {}
            def reset(i, z):
                adapters[i] = new_adapter()
                opts[i] = optim.Adam(adapters[i].adapter_parameters(), lr=lr_lora, weight_decay=1e-4)
                bufs[i] = []
                node_zone[i] = z
            for i in range(num_nodes):
                reset(i, sim_data[0]["node_zones"][i])

            sse = n = 0.0
            cum_curve: list[float] = []
            step_rmse: list[float] = []
            for sd in sim_data:
                step_sse = 0.0
                step_n = 0
                for i in range(num_nodes):
                    z = sd["node_zones"][i]
                    if z != node_zone[i]:
                        reset(i, z)
                for o in sd["obs"]:
                    rx = o["rx_i"]
                    if o["tx_zone"] != node_zone[rx]:
                        continue
                    pred = float(adapters[rx](o["feat"].unsqueeze(0)).item())
                    e = pred - o["rssi"]
                    step_sse += e * e
                    step_n += 1
                    bufs[rx].append((o["feat"].detach().clone(), o["rssi"]))
                    bufs[rx] = bufs[rx][-max_buf:]
                for i in adapters:
                    _fit_lora(adapters[i], opts[i], bufs[i], loss_fn, noise_floor, min_samples, max_train,
                              grad_steps_lora, device)
                if step_n > 0:
                    sse += step_sse
                    n += step_n
                    step_rmse.append(math.sqrt(step_sse / step_n))
                else:
                    step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
                cum_curve.append(_cum_rmse(sse, int(n)))
            return dict(cum_rmse=cum_curve, step_rmse=step_rmse, bytes_curve=[0.0] * num_steps, total_bytes=0.0)

        # Greedy LoRA (zone cache)
        def run_greedy():
            adapters, opts, bufs, node_zone = {}, {}, {}, {}
            zone_cache = {}
            comm = 0.0

            def enter(i, z):
                m = new_adapter()
                if z in zone_cache:
                    m.load_adapter_state(zone_cache[z])
                adapters[i] = m
                opts[i] = optim.Adam(adapters[i].adapter_parameters(), lr=lr_lora, weight_decay=1e-4)
                bufs[i] = []
                node_zone[i] = z

            for i in range(num_nodes):
                enter(i, sim_data[0]["node_zones"][i])

            sse = n = 0.0
            cum_curve: list[float] = []
            step_rmse: list[float] = []
            bytes_curve: list[float] = []
            for sd in sim_data:
                step = sd["step"]
                step_sse = 0.0
                step_n = 0
                for i in range(num_nodes):
                    z = sd["node_zones"][i]
                    if z != node_zone[i]:
                        enter(i, z)

                for o in sd["obs"]:
                    rx = o["rx_i"]
                    if o["tx_zone"] != node_zone[rx]:
                        continue
                    pred = float(adapters[rx](o["feat"].unsqueeze(0)).item())
                    e = pred - o["rssi"]
                    step_sse += e * e
                    step_n += 1
                    bufs[rx].append((o["feat"].detach().clone(), o["rssi"]))
                    bufs[rx] = bufs[rx][-max_buf:]

                for i in adapters:
                    _fit_lora(adapters[i], opts[i], bufs[i], loss_fn, noise_floor, min_samples, max_train,
                              grad_steps_lora, device)

                if step % share_every == 0:
                    for (a, b) in sd["share_contacts"]:
                        if node_zone[a] != node_zone[b]:
                            continue
                        sa = adapters[a].get_adapter_state()
                        sb = adapters[b].get_adapter_state()
                        n_a = max(1, len(bufs[a]))
                        n_b = max(1, len(bufs[b]))
                        w_a = n_a / (n_a + n_b)
                        w_b = 1.0 - w_a
                        avg = {k: w_a * sa[k] + w_b * sb[k] for k in sa}
                        adapters[a].load_adapter_state(avg)
                        adapters[b].load_adapter_state(avg)
                        zone_cache[node_zone[a]] = avg
                        comm += float(_adapter_exchange_bytes(adapters[a]))

                if step_n > 0:
                    sse += step_sse
                    n += step_n
                    step_rmse.append(math.sqrt(step_sse / step_n))
                else:
                    step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
                cum_curve.append(_cum_rmse(sse, int(n)))
                bytes_curve.append(comm)
            return dict(cum_rmse=cum_curve, step_rmse=step_rmse, bytes_curve=bytes_curve, total_bytes=comm)

        # Centralized LoRA (adapters-only)
        def run_central():
            model = new_adapter()
            opt = optim.Adam(model.adapter_parameters(), lr=lr_lora, weight_decay=1e-4)
            buf = []
            comm = 0.0
            sse = n = 0.0
            cum_curve: list[float] = []
            step_rmse: list[float] = []
            bytes_curve: list[float] = []
            for sd in sim_data:
                step_sse = 0.0
                step_n = 0
                for o in sd["obs"]:
                    pred = float(model(o["feat"].unsqueeze(0)).item())
                    e = pred - o["rssi"]
                    if o["tx_zone"] == o["rx_zone"]:
                        step_sse += e * e
                        step_n += 1
                    buf.append((o["feat"].detach().clone(), o["rssi"]))
                    comm += (8 + 1) * 4
                _fit_lora(model, opt, buf, loss_fn, noise_floor, min_samples, max_train * 4, grad_steps_lora, device)
                if step_n > 0:
                    sse += step_sse
                    n += step_n
                    step_rmse.append(math.sqrt(step_sse / step_n))
                else:
                    step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
                cum_curve.append(_cum_rmse(sse, int(n)))
                bytes_curve.append(comm)
            return dict(cum_rmse=cum_curve, step_rmse=step_rmse, bytes_curve=bytes_curve, total_bytes=comm)

        fspl = run_fspl()
        local = run_local()
        greedy = run_greedy()
        cent = run_central()

        def _eval_rmse_last(step_rmse: list[float]) -> float:
            tail = step_rmse[-eval_steps:]
            return float(np.mean(tail)) if tail else float("nan")

        return dict(
            title="Adapter strategy (LoRA)",
            series={
                "FSPL": fspl,
                "Local LoRA": local,
                "Greedy LoRA": greedy,
                "Centralized LoRA": cent,
            },
            eval_rmse={
                "FSPL": _eval_rmse_last(fspl["step_rmse"]),
                "Local LoRA": _eval_rmse_last(local["step_rmse"]),
                "Greedy LoRA": _eval_rmse_last(greedy["step_rmse"]),
                "Centralized LoRA": _eval_rmse_last(cent["step_rmse"]),
            },
        )

    # ---------------- replay: full fine-tuning strategy ----------------
    def replay_fullft():
        def new_full():
            m = RadioMLP(input_dim=8, hidden_sizes=(128, 128), fspl_const=eff_fspl_full)
            m.load_state(base_full_state)
            m.to(device)
            return m

        def run_fspl():
            sse = n = 0.0
            cum_curve: list[float] = []
            step_rmse: list[float] = []
            for sd in sim_data:
                step_sse = 0.0
                step_n = 0
                for o in sd["obs"]:
                    if o["tx_zone"] != o["rx_zone"]:
                        continue
                    e = o["fspl"] - o["rssi"]
                    step_sse += e * e
                    step_n += 1
                if step_n > 0:
                    sse += step_sse
                    n += step_n
                    step_rmse.append(math.sqrt(step_sse / step_n))
                else:
                    step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
                cum_curve.append(_cum_rmse(sse, int(n)))
            return dict(cum_rmse=cum_curve, step_rmse=step_rmse, bytes_curve=[0.0] * num_steps, total_bytes=0.0)

        def run_local():
            models, opts, bufs, node_zone = {}, {}, {}, {}
            def reset(i, z):
                models[i] = new_full()
                opts[i] = optim.Adam(models[i].parameters(), lr=lr_full, weight_decay=1e-4)
                bufs[i] = []
                node_zone[i] = z
            for i in range(num_nodes):
                reset(i, sim_data[0]["node_zones"][i])
            sse = n = 0.0
            cum_curve: list[float] = []
            step_rmse: list[float] = []
            for sd in sim_data:
                step_sse = 0.0
                step_n = 0
                for i in range(num_nodes):
                    z = sd["node_zones"][i]
                    if z != node_zone[i]:
                        reset(i, z)
                for o in sd["obs"]:
                    rx = o["rx_i"]
                    if o["tx_zone"] != node_zone[rx]:
                        continue
                    pred = float(models[rx](o["feat"].unsqueeze(0)).item())
                    e = pred - o["rssi"]
                    step_sse += e * e
                    step_n += 1
                    bufs[rx].append((o["feat"].detach().clone(), o["rssi"]))
                    bufs[rx] = bufs[rx][-max_buf:]
                for i in models:
                    _fit_full(models[i], opts[i], bufs[i], loss_fn, noise_floor, min_samples, max_train,
                              grad_steps_full, device)
                if step_n > 0:
                    sse += step_sse
                    n += step_n
                    step_rmse.append(math.sqrt(step_sse / step_n))
                else:
                    step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
                cum_curve.append(_cum_rmse(sse, int(n)))
            return dict(cum_rmse=cum_curve, step_rmse=step_rmse, bytes_curve=[0.0] * num_steps, total_bytes=0.0)

        def run_greedy():
            models, opts, bufs, node_zone = {}, {}, {}, {}
            zone_cache = {}
            comm = 0.0
            def enter(i, z):
                m = new_full()
                if z in zone_cache:
                    m.load_state(zone_cache[z])
                models[i] = m
                opts[i] = optim.Adam(models[i].parameters(), lr=lr_full, weight_decay=1e-4)
                bufs[i] = []
                node_zone[i] = z
            for i in range(num_nodes):
                enter(i, sim_data[0]["node_zones"][i])
            sse = n = 0.0
            cum_curve: list[float] = []
            step_rmse: list[float] = []
            bytes_curve: list[float] = []
            for sd in sim_data:
                step = sd["step"]
                step_sse = 0.0
                step_n = 0
                for i in range(num_nodes):
                    z = sd["node_zones"][i]
                    if z != node_zone[i]:
                        enter(i, z)
                for o in sd["obs"]:
                    rx = o["rx_i"]
                    if o["tx_zone"] != node_zone[rx]:
                        continue
                    pred = float(models[rx](o["feat"].unsqueeze(0)).item())
                    e = pred - o["rssi"]
                    step_sse += e * e
                    step_n += 1
                    bufs[rx].append((o["feat"].detach().clone(), o["rssi"]))
                    bufs[rx] = bufs[rx][-max_buf:]
                for i in models:
                    _fit_full(models[i], opts[i], bufs[i], loss_fn, noise_floor, min_samples, max_train,
                              grad_steps_full, device)
                if step % share_every == 0:
                    for (a, b) in sd["share_contacts"]:
                        if node_zone[a] != node_zone[b]:
                            continue
                        sa = models[a].get_state()
                        sb = models[b].get_state()
                        n_a = max(1, len(bufs[a]))
                        n_b = max(1, len(bufs[b]))
                        w_a = n_a / (n_a + n_b)
                        w_b = 1.0 - w_a
                        avg = {k: w_a * sa[k] + w_b * sb[k] for k in sa}
                        models[a].load_state(avg)
                        models[b].load_state(avg)
                        zone_cache[node_zone[a]] = avg
                        comm += float(_model_exchange_bytes(models[a]))
                if step_n > 0:
                    sse += step_sse
                    n += step_n
                    step_rmse.append(math.sqrt(step_sse / step_n))
                else:
                    step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
                cum_curve.append(_cum_rmse(sse, int(n)))
                bytes_curve.append(comm)
            return dict(cum_rmse=cum_curve, step_rmse=step_rmse, bytes_curve=bytes_curve, total_bytes=comm)

        def run_central():
            model = new_full()
            opt = optim.Adam(model.parameters(), lr=lr_full, weight_decay=1e-4)
            buf = []
            comm = 0.0
            sse = n = 0.0
            cum_curve: list[float] = []
            step_rmse: list[float] = []
            bytes_curve: list[float] = []
            for sd in sim_data:
                step_sse = 0.0
                step_n = 0
                for o in sd["obs"]:
                    pred = float(model(o["feat"].unsqueeze(0)).item())
                    e = pred - o["rssi"]
                    if o["tx_zone"] == o["rx_zone"]:
                        step_sse += e * e
                        step_n += 1
                    buf.append((o["feat"].detach().clone(), o["rssi"]))
                    comm += (8 + 1) * 4
                _fit_full(model, opt, buf, loss_fn, noise_floor, min_samples, max_train * 4, grad_steps_full, device)
                if step_n > 0:
                    sse += step_sse
                    n += step_n
                    step_rmse.append(math.sqrt(step_sse / step_n))
                else:
                    step_rmse.append(step_rmse[-1] if step_rmse else 0.0)
                cum_curve.append(_cum_rmse(sse, int(n)))
                bytes_curve.append(comm)
            return dict(cum_rmse=cum_curve, step_rmse=step_rmse, bytes_curve=bytes_curve, total_bytes=comm)

        fspl = run_fspl()
        local = run_local()
        greedy = run_greedy()
        cent = run_central()

        def _eval_rmse_last(step_rmse: list[float]) -> float:
            tail = step_rmse[-eval_steps:]
            return float(np.mean(tail)) if tail else float("nan")

        return dict(
            title="Full fine-tuning strategy (no adapters)",
            series={
                "FSPL": fspl,
                "Local FullFT": local,
                "Greedy FullShare": greedy,
                "Centralized FullFT": cent,
            },
            eval_rmse={
                "FSPL": _eval_rmse_last(fspl["step_rmse"]),
                "Local FullFT": _eval_rmse_last(local["step_rmse"]),
                "Greedy FullShare": _eval_rmse_last(greedy["step_rmse"]),
                "Centralized FullFT": _eval_rmse_last(cent["step_rmse"]),
            },
        )

    res_adapter = replay_adapter()
    res_full = replay_fullft()

    # ---------------- plot: one PNG per metric (two panels each) ----------------
    colors = {
        "FSPL": "#9E9E9E",
        "Local LoRA": "#2196F3",
        "Greedy LoRA": "#4CAF50",
        "Centralized LoRA": "#F44336",
        "Local FullFT": "#2196F3",
        "Greedy FullShare": "#4CAF50",
        "Centralized FullFT": "#F44336",
    }
    styles = {"FSPL": "--"}

    def _save_metric_png(filename: str, plot_metric_fn) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        plot_metric_fn(axes[0], res_adapter)
        plot_metric_fn(axes[1], res_full)
        axes[0].set_title(res_adapter["title"])
        axes[1].set_title(res_full["title"])
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle(
            "Same simulation data, two training strategies\n"
            f"{num_nodes} nodes | {num_steps} steps | obs_every_k_steps={obs_every_k_steps}, obs_keep_prob={obs_keep_prob}",
            fontsize=12,
            fontweight="bold",
        )
        p = out_dir / filename
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return p

    def _plot_cum_rmse(ax, res):
        for name, s in res["series"].items():
            ax.plot(range(1, num_steps + 1), s["cum_rmse"], styles.get(name, "-"), color=colors[name], linewidth=2.2, label=name)
        ax.axvspan(num_steps - eval_steps + 0.5, num_steps + 0.5, alpha=0.08, color="orange")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative RMSE (dB) — intra-zone")
        ax.legend(fontsize=8)

    def _plot_step_rmse(ax, res):
        for name, s in res["series"].items():
            ax.plot(range(1, num_steps + 1), _ema(s["step_rmse"]), styles.get(name, "-"), color=colors[name], linewidth=2.0, label=name)
        ax.axvspan(num_steps - eval_steps + 0.5, num_steps + 0.5, alpha=0.08, color="orange")
        ax.set_xlabel("Step")
        ax.set_ylabel("Per-step RMSE (dB, EMA) — intra-zone")
        ax.legend(fontsize=8)

    def _plot_comm(ax, res):
        for name, s in res["series"].items():
            mb_curve = [b / 1024**2 for b in s["bytes_curve"]]
            ax.plot(
                range(1, num_steps + 1),
                mb_curve,
                styles.get(name, "-"),
                color=colors[name],
                linewidth=2.2,
                label=f"{name} ({s['total_bytes']/1024**2:.2f} MB)",
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative communication (MB)")
        ax.legend(fontsize=8)

    def _plot_eval_bars(ax, res):
        labels = list(res["eval_rmse"].keys())
        vals = [res["eval_rmse"][k] for k in labels]
        cols = [colors[k] for k in labels]
        bars = ax.bar(labels, vals, color=cols, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.1,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax.set_ylabel(f"Eval RMSE (dB) — last {eval_steps} steps")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    def _plot_tradeoff(ax, res):
        for name, s in res["series"].items():
            x = s["total_bytes"] / 1024**2
            y = res["eval_rmse"][name]
            ax.scatter(x, y, s=160, color=colors[name], edgecolors="white", linewidths=1.2)
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, fontweight="bold")
        ax.set_xlabel("Total communication (MB)")
        ax.set_ylabel(f"Eval RMSE (dB) — last {eval_steps} steps")
        ax.grid(alpha=0.3)

    p_cum = _save_metric_png("zone_compare_cum_rmse.png", _plot_cum_rmse)
    p_step = _save_metric_png("zone_compare_step_rmse.png", _plot_step_rmse)
    p_comm = _save_metric_png("zone_compare_comm_curve.png", _plot_comm)
    p_eval = _save_metric_png("zone_compare_eval_rmse.png", _plot_eval_bars)
    p_trade = _save_metric_png("zone_compare_tradeoff.png", _plot_tradeoff)

    # ---------------- write summary ----------------
    out_txt = out_dir / "zone_compare_summary.txt"
    with out_txt.open("w") as f:
        f.write("Zone comparison (same simulation data; multiple metrics)\n")
        f.write(f"steps={num_steps}, eval_last={eval_steps}\n")
        f.write(f"obs_every_k_steps={obs_every_k_steps}, obs_keep_prob={obs_keep_prob}\n\n")
        f.write("Generated figures:\n")
        for p in [p_cum, p_step, p_comm, p_eval, p_trade]:
            f.write(f"  - {p.name}\n")
        f.write("\nEval RMSE (mean per-step RMSE over last window):\n")
        f.write("Adapter strategy:\n")
        for k, v in res_adapter["eval_rmse"].items():
            f.write(f"  {k:<16} {v:>8.3f}\n")
        f.write("Full fine-tuning strategy:\n")
        for k, v in res_full["eval_rmse"].items():
            f.write(f"  {k:<16} {v:>8.3f}\n")

    print("Saved metric PNGs:")
    for p in [p_cum, p_step, p_comm, p_eval, p_trade]:
        print(f"  {p}")
    print(f"Saved → {out_txt}")


if __name__ == "__main__":
    main()

