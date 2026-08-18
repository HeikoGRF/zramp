"""
Greedy flooding baseline using the Neural Radio Map + LoRA adapters (Munich).

The base MLP is pre-trained on synthetic FSPL data so hidden layers learn
meaningful feature projections.  During simulation, only LoRA adapters are
trained to learn local deviations (building shadows, multipath) and are
exchanged via weighted federated averaging when nodes meet.
"""

import csv
import math
import time
from typing import Dict, Tuple
import random

import numpy as np
from matplotlib.path import Path
import tensorflow as tf
import sionna.rt as rt
from sionna.rt import PlanarArray, Transmitter, Receiver, PathSolver

from sionna_baseline.mobility import RandomWaypoint, get_anchor_zone
from sionna_baseline.radio_mlp_lora import MapConfig, RadioFeatureEncoder, RadioMLPWithLoRA

import torch
import torch.nn as nn
import torch.optim as optim


def check_in_range(p1: np.ndarray, p2: np.ndarray, max_range: float = 150.0) -> bool:
    """Check if two nodes are within max_range of each other in the XY plane."""
    return np.linalg.norm(p1[:2] - p2[:2]) <= max_range


def create_building_grid(scene, bounds_x, bounds_y, res=1.0):
    xs = np.arange(bounds_x[0], bounds_x[1] + res, res)
    ys = np.arange(bounds_y[0], bounds_y[1] + res, res)
    xx, yy = np.meshgrid(xs, ys)
    points = np.c_[xx.ravel(), yy.ravel()]
    mask = np.zeros(len(points), dtype=bool)

    for name, obj in scene.objects.items():
        mesh = obj.mi_mesh
        try:
            vertices = np.array(mesh.vertex_positions_buffer()).reshape(-1, 3)
            faces = np.array(mesh.faces_buffer()).reshape(-1, 3)
        except Exception:
            continue
            
        verts2d = vertices[:, :2]
        for face in faces:
            poly = verts2d[face]
            p = Path(poly)
            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            
            in_bbox = (points[:, 0] >= xmin - res) & (points[:, 0] <= xmax + res) & \
                      (points[:, 1] >= ymin - res) & (points[:, 1] <= ymax + res)
            
            if np.any(in_bbox):
                idx = np.where(in_bbox)[0]
                inside = p.contains_points(points[idx])
                mask[idx[inside]] = True
                
    return mask.reshape(xx.shape)


def count_building_cells(tx_xy, rx_xy, grid_mask, bounds_x, bounds_y, res=1.0):
    """Samples points along the line segment between tx and rx and counts building hits."""
    dist = np.linalg.norm(rx_xy - tx_xy)
    if dist < 1e-6:
        return 0
    
    # Sample every half-resolution to ensure we don't skip cells
    num_samples = int(np.ceil(dist / (res / 2.0)))
    if num_samples < 2:
        num_samples = 2
        
    ts = np.linspace(0, 1, num_samples)
    sampled_pts = tx_xy[None, :] + ts[:, None] * (rx_xy - tx_xy)[None, :]
    
    idx_x = np.round((sampled_pts[:, 0] - bounds_x[0]) / res).astype(int)
    idx_y = np.round((sampled_pts[:, 1] - bounds_y[0]) / res).astype(int)
    
    valid = (idx_x >= 0) & (idx_x < grid_mask.shape[1]) & (idx_y >= 0) & (idx_y < grid_mask.shape[0])
    idx_x = idx_x[valid]
    idx_y = idx_y[valid]
    
    # Unique grid cells visited
    unique_cells = set(zip(idx_y, idx_x))
    
    hits = 0
    for y, x in unique_cells:
        if grid_mask[y, x]:
            hits += 1
            
    return hits


class LoRAZoneAdapter:
    """
    Per-zone adapter built on top of RadioMLPWithLoRA.

    - Uses RadioFeatureEncoder to featurize (tx_xy, rx_xy)
    - Trains only LoRA parameters via MSE on RSSI (dB)
    - Tracks residual std as sigma
    """

    def __init__(self, map_cfg: MapConfig, noise_floor_db: float = -150.0, lr: float = 1e-2,
                 fspl_const: float | None = None, base_state: dict | None = None):
        self.encoder = RadioFeatureEncoder(map_cfg)
        self.model = RadioMLPWithLoRA(input_dim=8, hidden_sizes=(128, 128), rank=16, alpha=1.0,
                                      fspl_const=fspl_const)
        if base_state is not None:
            self.model.load_base_state(base_state)

        self.noise_floor_db = float(noise_floor_db)
        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.adapter_parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss()

        self.sigma = 25.0
        self.buffer = []
        self.min_samples = 8
        self.max_train_samples = 5000
        self.max_buffer_size = 10000

    @property
    def n_uncensored(self) -> int:
        return sum(1 for _, r in self.buffer if r != self.noise_floor_db)

    def add(self, tx_xy: np.ndarray, rx_xy: np.ndarray, rssi: float, num_building_cells: float) -> None:
        # Current setup uses a fixed TX power; pass 0.0 dBm as placeholder.
        feats = self.encoder.encode(tuple(tx_xy), tuple(rx_xy), tx_power_dbm=0.0, has_obstacle=(num_building_cells > 0))
        self.buffer.append((feats.detach().clone(), float(rssi)))
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = random.sample(self.buffer, self.max_buffer_size)

    def merge_buffer(self, other_buffer) -> None:
        """Merge another node's buffer into ours."""
        self.buffer.extend((f.detach(), float(r)) for f, r in other_buffer)
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = random.sample(self.buffer, self.max_buffer_size)

    def fit(self) -> None:
        """Train LoRA adapters on buffered samples; update sigma."""
        if len(self.buffer) < self.min_samples:
            return

        if len(self.buffer) > self.max_train_samples:
            samples = random.sample(self.buffer, self.max_train_samples)
        else:
            samples = self.buffer

        feats_stack = torch.stack([f for f, _ in samples])
        rssi_arr = torch.tensor([r for _, r in samples], dtype=torch.float32)

        unc_mask = rssi_arr != self.noise_floor_db
        if unc_mask.sum().item() < self.min_samples:
            return

        x = feats_stack[unc_mask].to(self.device)
        y = rssi_arr[unc_mask].to(self.device)

        self.model.train()
        for _ in range(20):
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            y_hat = self.model(x).cpu()
        resid = y.cpu() - y_hat
        self.sigma = float(max(3.0, resid.std().item()))

    def predict(self, tx_xy: np.ndarray, rx_xy: np.ndarray, num_building_cells: float) -> Tuple[float, float]:
        feats = self.encoder.encode(
            tuple(tx_xy), tuple(rx_xy), tx_power_dbm=0.0, has_obstacle=(num_building_cells > 0)
        ).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            mean = float(self.model(feats).cpu().item())

        n_unc = self.n_uncensored
        inflate = 1.0 + (1.0 / max(1.0, math.sqrt(max(1.0, float(n_unc)))))
        sigma = float(max(3.0, self.sigma * inflate))
        return mean, sigma

    def copy_from(self, other: "LoRAZoneAdapter") -> None:
        """Copy buffer and LoRA weights from another adapter."""
        self.buffer = list(other.buffer)
        self.model.load_adapter_state(other.model.get_adapter_state())
        self.sigma = other.sigma


def compute_fspl(d: float, fspl_const: float, eps: float = 1e-6) -> float:
    """Pure FSPL prediction (no MLP) for baseline comparison."""
    return -20.0 * math.log10(d + eps) + fspl_const


def main():
    print("Loading Munich scene …", flush=True)
    scene = rt.load_scene(rt.scene.munich)
    freq_hz: float = 3.5e9
    scene.frequency = freq_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    num_nodes = 20
    # Step 7: train continuously; compute evaluation on last N steps
    num_steps = 120
    eval_steps = 20
    dt = 1.0
    max_range = 250.0

    zone_size = 100
    bounds_x = (0, 99)
    bounds_y = (-199, -100)

    map_cfg = MapConfig(x_min=bounds_x[0], x_max=bounds_x[1], y_min=bounds_y[0], y_max=bounds_y[1])

    c_light = 3e8
    fspl_const = -20.0 * math.log10(4.0 * math.pi * freq_hz / c_light)
    print(f"FSPL constant at {freq_hz/1e9:.2f} GHz: {fspl_const:.2f} dB", flush=True)

    # --- Pre-train a shared base model on synthetic FSPL data ----------------
    print("Pre-training base MLP on synthetic FSPL data …", flush=True)
    encoder = RadioFeatureEncoder(map_cfg)
    base_model = RadioMLPWithLoRA(input_dim=8, hidden_sizes=(128, 128), rank=16, alpha=1.0,
                                  fspl_const=fspl_const)
    base_model.pretrain_on_fspl(encoder, fspl_const, n_samples=50_000, epochs=200, lr=1e-3)
    base_state = base_model.get_base_state()
    print("Base MLP pre-trained. Distributing to all nodes.", flush=True)

    # --- Initialize nodes ----------------------------------------------------
    nodes: Dict[int, np.ndarray] = {}
    for i in range(num_nodes):
        x = np.random.uniform(*bounds_x)
        y = np.random.uniform(*bounds_y)
        nodes[i] = np.array([x, y, 1.5])

    for i in range(num_nodes):
        scene.add(Transmitter(name=f"tx_{i}", position=nodes[i]))
        scene.add(Receiver(name=f"rx_{i}", position=nodes[i]))

    mobility = RandomWaypoint(bounds_x=bounds_x, bounds_y=bounds_y, velocity=(1.0, 3.0), pause_time=2.0)

    print("Generating 2D building grid representation...", flush=True)
    grid_mask = create_building_grid(scene, bounds_x, bounds_y, res=1.0)
    print(f"Building grid generated. ({np.sum(grid_mask)} building cells)", flush=True)

    gpus = tf.config.list_physical_devices("GPU")
    compute_device = "/GPU:0" if gpus else "/CPU:0"
    print(f"Using device: {gpus[0].name if gpus else 'CPU'}", flush=True)
    print(f"Single anchor zone: Zone_0_0 (bounds {bounds_x}, {bounds_y})", flush=True)

    solver = PathSolver()

    node_adapters: Dict[int, Dict[str, LoRAZoneAdapter]] = {
        i: {"Zone_0_0": LoRAZoneAdapter(map_cfg, fspl_const=fspl_const, base_state=base_state)}
        for i in range(num_nodes)
    }

    csv_data = []
    share_events = []
    total_shares = 0

    step_rmse_log = []
    eval_preds = []
    eval_truths = []
    eval_sigmas = []

    # Cumulative RMSE accumulators (running sums over ALL steps so far)
    cum_adapter_sse = 0.0
    cum_fspl_sse = 0.0
    cum_count = 0

    # Exponential moving average (alpha=0.1 gives ~10-step smoothing window)
    ema_alpha = 0.1
    ema_adapter = None
    ema_fspl = None

    start_time = time.time()

    for step in range(num_steps):
        elapsed = time.time() - start_time
        progress = (step + 1) / num_steps
        eta = elapsed / max(progress, 1e-6) * (1.0 - progress)
        # Always train; evaluation is computed on the last eval_steps only
        in_eval_window = step >= (num_steps - eval_steps)

        mobility.step(nodes, dt)

        for i in range(num_nodes):
            scene.get(f"tx_{i}").position = nodes[i]
            scene.get(f"rx_{i}").position = nodes[i]

        with tf.device(compute_device):
            paths = solver(
                scene,
                max_depth=3,
                samples_per_src=int(1e5),
                specular_reflection=True,
                diffraction=True,
                edge_diffraction=False,
            )

        try:
            a_real = np.array(paths.a[0])
            a_imag = np.array(paths.a[1])
            a_cplx = a_real + 1j * a_imag
            a_cplx = np.squeeze(a_cplx, axis=(1, 3))
            power = np.sum(np.abs(a_cplx) ** 2, axis=-1)
        except Exception:
            power = np.zeros((num_nodes, num_nodes))

        timestamp = step * dt
        contacts_this_step = set()

        step_preds = []
        step_truths = []
        step_fspl_preds = []

        for tx_idx in range(num_nodes):
            pos_tx = nodes[tx_idx]
            for rx_idx in range(num_nodes):
                if tx_idx == rx_idx:
                    continue

                pos_rx = nodes[rx_idx]
                if not check_in_range(pos_tx, pos_rx, max_range):
                    continue

                tx_xy = pos_tx[:2]
                rx_xy = pos_rx[:2]
                pwr = float(power[rx_idx, tx_idx])
                rssi = 10 * np.log10(pwr) if pwr > 0 else -150.0
                zone_rx = get_anchor_zone(pos_rx[0], pos_rx[1], square_size=zone_size)
                zone_tx = get_anchor_zone(pos_tx[0], pos_tx[1], square_size=zone_size)

                if zone_rx != zone_tx:
                    continue

                zone = zone_rx

                if zone not in node_adapters[rx_idx]:
                    node_adapters[rx_idx][zone] = LoRAZoneAdapter(
                        map_cfg, fspl_const=fspl_const, base_state=base_state
                    )
                adapter = node_adapters[rx_idx][zone]

                num_building_cells = count_building_cells(tx_xy, rx_xy, grid_mask, bounds_x, bounds_y, res=1.0)

                pred_mean, pred_sigma = adapter.predict(tx_xy, rx_xy, num_building_cells)

                d = float(np.linalg.norm(rx_xy - tx_xy))
                fspl_pred = compute_fspl(d, fspl_const)

                step_preds.append(float(pred_mean))
                step_truths.append(rssi)
                step_fspl_preds.append(fspl_pred)

                csv_data.append(
                    {
                        "timestamp": timestamp,
                        "node_id": rx_idx,
                        "x": pos_rx[0],
                        "y": pos_rx[1],
                        "in_anchor_zone": zone,
                        "measured_rssi": round(rssi, 2),
                        "predicted_rssi": round(float(pred_mean), 2),
                        "predicted_sigma": round(float(pred_sigma), 2),
                        "fspl_predicted_rssi": round(fspl_pred, 2),
                        "num_building_cells": num_building_cells,
                        "pinged_node": tx_idx,
                        "phase": "eval" if in_eval_window else "train",
                    }
                )

                adapter.add(tx_xy, rx_xy, rssi, num_building_cells)
                if in_eval_window:
                    eval_preds.append(float(pred_mean))
                    eval_truths.append(rssi)
                    eval_sigmas.append(float(pred_sigma))

                pair = (min(rx_idx, tx_idx), max(rx_idx, tx_idx))
                contacts_this_step.add(pair)

        # --- Per-step + cumulative + EMA RMSE tracking -----------------------
        if step_truths:
            arr_t = np.array(step_truths)
            arr_p = np.array(step_preds)
            arr_f = np.array(step_fspl_preds)

            step_adapter_rmse = float(np.sqrt(np.mean((arr_p - arr_t) ** 2)))
            step_fspl_rmse = float(np.sqrt(np.mean((arr_f - arr_t) ** 2)))

            # Cumulative RMSE over all pings seen so far
            cum_adapter_sse += float(np.sum((arr_p - arr_t) ** 2))
            cum_fspl_sse += float(np.sum((arr_f - arr_t) ** 2))
            cum_count += len(arr_t)
            cum_adapter_rmse = math.sqrt(cum_adapter_sse / cum_count)
            cum_fspl_rmse = math.sqrt(cum_fspl_sse / cum_count)

            # EMA RMSE
            if ema_adapter is None:
                ema_adapter = step_adapter_rmse
                ema_fspl = step_fspl_rmse
            else:
                ema_adapter = ema_alpha * step_adapter_rmse + (1 - ema_alpha) * ema_adapter
                ema_fspl = ema_alpha * step_fspl_rmse + (1 - ema_alpha) * ema_fspl

            step_rmse_log.append({
                "step": step,
                "step_adapter_rmse": round(step_adapter_rmse, 4),
                "step_fspl_rmse": round(step_fspl_rmse, 4),
                "cum_adapter_rmse": round(cum_adapter_rmse, 4),
                "cum_fspl_rmse": round(cum_fspl_rmse, 4),
                "ema_adapter_rmse": round(ema_adapter, 4),
                "ema_fspl_rmse": round(ema_fspl, 4),
                "n_pings": len(step_truths),
            })
            print(
                f"  Step {step+1:3d}/{num_steps}  "
                f"cum_adapter={cum_adapter_rmse:6.2f}  "
                f"cum_fspl={cum_fspl_rmse:6.2f}  "
                f"ema_adapter={ema_adapter:6.2f}  "
                f"ema_fspl={ema_fspl:6.2f}  "
                f"(elapsed {elapsed:5.1f}s, ETA {eta:5.1f}s)",
                flush=True,
            )

        # --- Local training + weighted federated averaging -------------------
        for node, zones in node_adapters.items():
            for zone, adapter in zones.items():
                adapter.fit()

        for (a, b) in contacts_this_step:
            shared_zones = node_adapters[a].keys() & node_adapters[b].keys()
            for zone in shared_zones:
                ad_a = node_adapters[a][zone]
                ad_b = node_adapters[b][zone]
                n_a = max(1, ad_a.n_uncensored)
                n_b = max(1, ad_b.n_uncensored)
                w_a = n_a / (n_a + n_b)
                w_b = n_b / (n_a + n_b)
                state_a = ad_a.model.get_adapter_state()
                state_b = ad_b.model.get_adapter_state()
                avg_state = {k: w_a * state_a[k] + w_b * state_b[k] for k in state_a}
                ad_a.model.load_adapter_state(avg_state)
                ad_b.model.load_adapter_state(avg_state)
                avg_sigma = w_a * ad_a.sigma + w_b * ad_b.sigma
                ad_a.sigma = avg_sigma
                ad_b.sigma = avg_sigma
            if shared_zones:
                total_shares += 1
                share_events.append({"timestamp": timestamp, "node_a": a, "node_b": b})

    # --- Final metrics -------------------------------------------------------
    eval_preds = np.array(eval_preds)
    eval_truths = np.array(eval_truths)
    eval_sigmas = np.array(eval_sigmas)
    rmse = float(np.sqrt(np.mean((eval_preds - eval_truths) ** 2)))
    mae = float(np.mean(np.abs(eval_preds - eval_truths)))

    sigma2 = np.maximum(1e-6, eval_sigmas ** 2)
    nll = float(
        np.mean(
            0.5 * np.log(2.0 * np.pi * sigma2)
            + 0.5 * ((eval_truths - eval_preds) ** 2) / sigma2
        )
    )

    log_file = "simulation_log_greedy_lora.csv"
    share_file = "greedy_flooding_shares_lora.csv"
    summary_file = "greedy_flooding_summary_lora.txt"
    rmse_file = "step_rmse_lora.csv"

    if csv_data:
        with open(log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)

    if share_events:
        with open(share_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=share_events[0].keys())
            writer.writeheader()
            writer.writerows(share_events)

    if step_rmse_log:
        with open(rmse_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=step_rmse_log[0].keys())
            writer.writeheader()
            writer.writerows(step_rmse_log)

    with open(summary_file, "w") as f:
        f.write("Greedy Flooding Baseline (MLP + LoRA Adapters, Munich) Summary\n")
        f.write("=" * 65 + "\n")
        f.write(f"Single anchor zone: Zone_0_0\n")
        f.write(f"Total steps: {num_steps}\n")
        f.write(f"Evaluation window: last {eval_steps} steps\n")
        f.write(f"Total pings logged: {len(csv_data)}\n")
        f.write(f"Total share events: {total_shares}\n")
        f.write(f"Eval samples: {len(eval_preds)}\n")
        f.write(f"\n--- Evaluation Accuracy (last {eval_steps} steps) ---\n")
        f.write(f"RMSE: {rmse:.4f} dB\n")
        f.write(f"MAE:  {mae:.4f} dB\n")
        f.write(f"NLL (Gaussian): {nll:.4f}\n")
        if step_rmse_log:
            f.write(f"\n--- Learning curve (cumulative RMSE) ---\n")
            f.write(f"Step {step_rmse_log[0]['step']+1}: "
                    f"adapter={step_rmse_log[0]['cum_adapter_rmse']:.2f} dB, "
                    f"fspl={step_rmse_log[0]['cum_fspl_rmse']:.2f} dB\n")
            f.write(f"Step {step_rmse_log[-1]['step']+1}: "
                    f"adapter={step_rmse_log[-1]['cum_adapter_rmse']:.2f} dB, "
                    f"fspl={step_rmse_log[-1]['cum_fspl_rmse']:.2f} dB\n")

    print("\n✓ Greedy flooding with MLP+LoRA adapters completed.")
    print(f"  Pings: {len(csv_data)}, Share events: {total_shares}")
    print(f"  Eval accuracy over last {eval_steps} steps (RMSE vs Sionna): {rmse:.4f} dB")
    print(f"  Uncertainty score (Gaussian NLL): {nll:.4f}")
    print(f"  Logs: {log_file}, {share_file}, {summary_file}, {rmse_file}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total execution time: {time.time() - start:.2f} seconds")

