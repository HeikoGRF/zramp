"""
backbone_utils.py — Shared utilities for backbone loading and TX power.

Provides:
  - TX_POWER_MIN_DBM / TX_POWER_MAX_DBM  – constants used by encoder & dataset gen.
  - sample_node_tx_powers()              – assign fixed TX powers to nodes at init.
  - compute_rssi()                       – RSSI from Sionna linear gain + TX power.
  - load_or_pretrain_lora_backbone()     – load urban backbone or fall back to FSPL.
  - load_or_pretrain_full_backbone()     – same for full-FT RadioMLP.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# TX power constants (shared with encoder normalisation in radio_mlp_lora.py)
# ---------------------------------------------------------------------------
TX_POWER_MIN_DBM: float = 0.0    # minimum node TX power (dBm)
TX_POWER_MAX_DBM: float = 23.0   # maximum node TX power (dBm)  ≈ 200 mW


# ---------------------------------------------------------------------------
# Per-node TX power helpers
# ---------------------------------------------------------------------------

def sample_node_tx_powers(
    num_nodes: int,
    rng: Optional[np.random.Generator] = None,
    low:  float = TX_POWER_MIN_DBM,
    high: float = TX_POWER_MAX_DBM,
) -> Dict[int, float]:
    """
    Assign a fixed TX power (dBm) to each node, drawn once from Uniform[low, high].

    Each device keeps its power for the whole simulation run, mimicking the
    real-world scenario where devices have different hardware capabilities
    (e.g., IoT sensors vs smartphones vs vehicular units).
    """
    gen = rng if rng is not None else np.random.default_rng()
    return {i: float(gen.uniform(low, high)) for i in range(num_nodes)}


# ---------------------------------------------------------------------------
# RSSI helper
# ---------------------------------------------------------------------------

def compute_rssi(
    gain_lin: float,
    tx_power_dbm: float,
    noise_floor_dbm: float = -150.0,
) -> float:
    """
    Compute received RSSI (dBm) from Sionna linear channel gain and TX power.

    Sionna's ``paths.a`` coefficients represent the complex channel impulse
    response normalised to unit TX power, so:

        P_rx_dBm = 10·log10(|a|²) + P_tx_dBm

    Links below *noise_floor_dbm* are clamped to that value.
    """
    if gain_lin <= 0.0:
        return noise_floor_dbm
    rssi = 10.0 * math.log10(gain_lin) + tx_power_dbm
    return max(rssi, noise_floor_dbm)


# ---------------------------------------------------------------------------
# Backbone loading (LoRA model)
# ---------------------------------------------------------------------------

def load_or_pretrain_lora_backbone(
    encoder,
    fspl_const: float,
    backbone_path: str = "backbone_etoile.pt",
    hidden_sizes: Tuple[int, int] = (128, 128),
    rank: int = 16,
    alpha: float = 1.0,
    n_samples: int = 50_000,
    epochs: int = 200,
    lr: float = 1e-3,
) -> Tuple[dict, Optional[float]]:
    """
    Try to load a pre-trained urban backbone; fall back to FSPL pretraining.

    Returns
    -------
    base_state : dict
        Frozen base-layer weights ready for ``model.load_base_state()``.
    eff_fspl_const : float or None
        ``None`` when urban backbone is used (no physics prior needed).
        ``fspl_const`` when the FSPL fallback is active.
    """
    from .radio_mlp_lora import RadioMLPWithLoRA  # late import – avoids circularity

    path = Path(backbone_path)
    if path.exists():
        print(f"Loading urban backbone from {path} …", flush=True)
        model = RadioMLPWithLoRA.load_backbone_from_file(
            str(path),
            input_dim=8,
            hidden_sizes=hidden_sizes,
            rank=rank,
            alpha=alpha,
            fspl_const=None,   # urban backbone: no physics prior
        )
        return model.get_base_state(), None

    print(f"Urban backbone not found at '{backbone_path}' — "
          "falling back to FSPL pretraining …", flush=True)
    model = RadioMLPWithLoRA(
        input_dim=8, hidden_sizes=hidden_sizes,
        rank=rank, alpha=alpha, fspl_const=fspl_const,
    )
    model.pretrain_on_fspl(encoder, fspl_const,
                           n_samples=n_samples, epochs=epochs, lr=lr)
    return model.get_base_state(), fspl_const


# ---------------------------------------------------------------------------
# Backbone loading (full fine-tuning RadioMLP)
# ---------------------------------------------------------------------------

def load_or_pretrain_full_backbone(
    encoder,
    fspl_const: float,
    backbone_path: str = "backbone_etoile.pt",
    hidden_sizes: Tuple[int, int] = (128, 128),
    n_samples: int = 50_000,
    epochs: int = 200,
    lr: float = 1e-3,
) -> Tuple[dict, Optional[float]]:
    """Same as ``load_or_pretrain_lora_backbone`` but for the adapter-free RadioMLP."""
    import importlib, sys
    from pathlib import Path as _P
    _mod_path = _P(__file__).resolve().parents[1] / "zone_benchmark_fullft" / "sionna_baseline" / "radio_mlp.py"
    import importlib.util
    _spec = importlib.util.spec_from_file_location("_radio_mlp_fullft", str(_mod_path))
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
    RadioMLP = _mod.RadioMLP

    path = Path(backbone_path)
    if path.exists():
        print(f"Loading urban backbone (full-FT) from {path} …", flush=True)
        model = RadioMLP.load_from_file(
            str(path), input_dim=8, hidden_sizes=hidden_sizes, fspl_const=None,
        )
        return model.get_state(), None

    print(f"Urban backbone not found — FSPL fallback (full-FT) …", flush=True)
    model = RadioMLP(input_dim=8, hidden_sizes=hidden_sizes, fspl_const=fspl_const)
    model.pretrain_on_fspl(encoder, fspl_const,
                           n_samples=n_samples, epochs=epochs, lr=lr)
    return model.get_state(), fspl_const
