"""
radio_mlp.py — device-sized radio MLP WITHOUT adapters (full fine-tuning baseline).

This module is intentionally minimal for the self-contained benchmark in
`zone_benchmark_fullft/`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.optim as optim

# TX power range — kept in sync with sionna_baseline/backbone_utils.py
TX_POWER_MIN_DBM: float = 0.0
TX_POWER_MAX_DBM: float = 23.0
_TX_POWER_MID:    float = (TX_POWER_MIN_DBM + TX_POWER_MAX_DBM) / 2.0
_TX_POWER_SCALE:  float = (TX_POWER_MAX_DBM - TX_POWER_MIN_DBM) / 2.0


@dataclass
class MapConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


class RadioFeatureEncoder:
    """
    Features (8D):
      0  xtx_n, 1 ytx_n, 2 xrx_n, 3 yrx_n  – normalised coords [0, 1]
      4  d                                    – distance in metres
      5  log10(d + eps)                       – used by physics prior
      6  angle/π in [-1, 1]
      7  tx_power_n                           – TX power normalised to [-1, 1]
    """

    def __init__(self, map_cfg: MapConfig, eps: float = 1e-6):
        self.map_cfg = map_cfg
        self.eps = eps

    def encode(self, tx_xy, rx_xy, tx_power_dbm: float,
               has_obstacle: bool = False) -> torch.Tensor:
        tx_x, tx_y = tx_xy
        rx_x, rx_y = rx_xy

        xtx_n = (tx_x - self.map_cfg.x_min) / max(self.map_cfg.width,  self.eps)
        ytx_n = (tx_y - self.map_cfg.y_min) / max(self.map_cfg.height, self.eps)
        xrx_n = (rx_x - self.map_cfg.x_min) / max(self.map_cfg.width,  self.eps)
        yrx_n = (rx_y - self.map_cfg.y_min) / max(self.map_cfg.height, self.eps)

        dx    = rx_x - tx_x
        dy    = rx_y - tx_y
        d     = math.sqrt(dx * dx + dy * dy)
        log_d = math.log10(d + self.eps)
        ang_n = math.atan2(dy, dx) / math.pi
        pw_n  = (float(tx_power_dbm) - _TX_POWER_MID) / _TX_POWER_SCALE

        return torch.tensor(
            [xtx_n, ytx_n, xrx_n, yrx_n, d, log_d, ang_n, pw_n],
            dtype=torch.float32,
        )


class RadioMLP(nn.Module):
    """Device-sized backbone MLP: 8 → h1 → h2 → 1."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_sizes: Tuple[int, int] = (128, 128),
        fspl_const: Optional[float] = None,
    ):
        super().__init__()
        h1, h2 = hidden_sizes

        self.use_physics_base = fspl_const is not None
        if self.use_physics_base:
            self.register_buffer("fspl_const",
                                 torch.tensor(float(fspl_const), dtype=torch.float32))

        self.fc1  = nn.Linear(input_dim, h1)
        self.act1 = nn.SiLU()
        self.fc2  = nn.Linear(h1, h2)
        self.act2 = nn.SiLU()
        self.fc3  = nn.Linear(h2, 1)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        physics_out = (-20.0 * x[..., 5] + self.fspl_const
                       if self.use_physics_base else None)
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        h = self.fc3(h).squeeze(-1)
        return (physics_out + h) if physics_out is not None else h

    def get_state(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def load_state(self, state: dict) -> None:
        self.load_state_dict(state, strict=True)

    def pretrain_on_fspl(
        self,
        encoder: RadioFeatureEncoder,
        fspl_const: float,
        n_samples: int = 50_000,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 2048,
    ) -> None:
        """Vectorised FSPL pretraining with random TX powers."""
        map_cfg = encoder.map_cfg
        eps     = encoder.eps

        tx_x  = torch.empty(n_samples).uniform_(map_cfg.x_min, map_cfg.x_max)
        tx_y  = torch.empty(n_samples).uniform_(map_cfg.y_min, map_cfg.y_max)
        rx_x  = torch.empty(n_samples).uniform_(map_cfg.x_min, map_cfg.x_max)
        rx_y  = torch.empty(n_samples).uniform_(map_cfg.y_min, map_cfg.y_max)
        tx_pw = torch.empty(n_samples).uniform_(TX_POWER_MIN_DBM, TX_POWER_MAX_DBM)

        w = max(map_cfg.width, eps);  h = max(map_cfg.height, eps)
        dx    = rx_x - tx_x;          dy = rx_y - tx_y
        d     = (dx.pow(2) + dy.pow(2)).sqrt().clamp(min=eps)
        log_d = torch.log10(d + eps)
        pw_n  = (tx_pw - _TX_POWER_MID) / _TX_POWER_SCALE

        X = torch.stack([
            (tx_x - map_cfg.x_min) / w, (tx_y - map_cfg.y_min) / h,
            (rx_x - map_cfg.x_min) / w, (rx_y - map_cfg.y_min) / h,
            d, log_d, torch.atan2(dy, dx) / math.pi, pw_n,
        ], dim=1)
        Y = tx_pw - 20.0 * log_d + fspl_const

        opt     = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            el, nb = 0.0, 0
            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                opt.zero_grad()
                loss = loss_fn(self(X[idx]), Y[idx])
                loss.backward()
                opt.step()
                el += float(loss.item()); nb += 1
            if (epoch + 1) % 50 == 0:
                print(f"  pretrain epoch {epoch+1}/{epochs}  MSE={el/nb:.4f}",
                      flush=True)
        self.eval()

    def pretrain_on_dataset(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 4096,
    ) -> None:
        """Pre-train on urban ray-tracing (features, RSSI) dataset."""
        n   = len(X)
        opt = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.HuberLoss(delta=5.0)
        self.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            el, nb = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                opt.zero_grad()
                loss = loss_fn(self(X[idx]), Y[idx])
                loss.backward()
                opt.step()
                el += float(loss.item()); nb += 1
            if (epoch + 1) % 50 == 0:
                print(f"  pretrain epoch {epoch+1}/{epochs}  Huber={el/nb:.4f}",
                      flush=True)
        self.eval()

    @classmethod
    def load_from_file(cls, path: str, **kwargs) -> "RadioMLP":
        """Load a backbone checkpoint into a new RadioMLP instance."""
        model = cls(**kwargs)
        ckpt  = torch.load(path, map_location="cpu", weights_only=True)
        src   = ckpt.get("model_state", ckpt)
        own   = model.state_dict()
        for k, v in src.items():
            if k in own:
                own[k] = v
        model.load_state_dict(own)
        model.eval()
        return model

