"""
Neural Radio Map with LoRA adapters.

This module defines:

- MapConfig: basic (x, y) bounds of the map.
- RadioFeatureEncoder: converts (tx_x, tx_y, rx_x, rx_y, tx_power_dbm) into an
  8-D feature vector [xtx, ytx, xrx, yrx, d, log10(d), angle, tx_power_n].
- RadioMLPWithLoRA: a small MLP with LoRA adapters on hidden layers that predicts
  RSSI (dBm) from features.

Intended usage:
1. Either call pretrain_on_fspl() (synthetic fallback) or pretrain_on_dataset()
   (urban ray-tracing data) to warm-start the backbone.
2. Freeze base weights and attach LoRA adapters.
3. During simulation, train only LoRA parameters to learn local deviations
   (building shadows, multipath) from the pretrained base.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.optim as optim

# TX power range used for normalisation and synthetic FSPL pretraining
TX_POWER_MIN_DBM: float = 0.0
TX_POWER_MAX_DBM: float = 23.0
_TX_POWER_MID:    float = (TX_POWER_MIN_DBM + TX_POWER_MAX_DBM) / 2.0   # 11.5
_TX_POWER_SCALE:  float = (TX_POWER_MAX_DBM - TX_POWER_MIN_DBM) / 2.0   # 11.5


@dataclass
class MapConfig:
    """Configuration of the (x, y) coordinate system."""

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
    Encode TX/RX coordinates + TX power into an 8-D feature vector.

    Features:
      0  xtx_n       – TX x in [0, 1]
      1  ytx_n       – TX y in [0, 1]
      2  xrx_n       – RX x in [0, 1]
      3  yrx_n       – RX y in [0, 1]
      4  d           – Euclidean distance (metres)
      5  log_d       – log10(d + eps)   ← used by physics prior
      6  angle_n     – atan2(Δy, Δx)/π in [-1, 1]
      7  tx_power_n  – TX power normalised to [-1, 1] over [0, 23] dBm
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
        ang_n = math.atan2(dy, dx) / math.pi                            # [-1, 1]
        pw_n  = (float(tx_power_dbm) - _TX_POWER_MID) / _TX_POWER_SCALE  # [-1, 1]

        return torch.tensor(
            [xtx_n, ytx_n, xrx_n, yrx_n, d, log_d, ang_n, pw_n],
            dtype=torch.float32,
        )


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a single linear layer: W x + B(Ax) * scaling.

    - W (base weight/bias) is frozen in the wrapped linear layer.
    - A and B are low-rank trainable matrices (rank r).
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA parameters: A (r x in), B (out x r)
        self.A = nn.Parameter(torch.zeros(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        Ax = torch.matmul(x, self.A.t())
        delta = torch.matmul(Ax, self.B.t()) * self.scaling
        return base_out + delta


class RadioMLPWithLoRA(nn.Module):
    """
    Radio MLP with LoRA adapters on the hidden layers.

    - Two hidden linear layers are wrapped with LoRALinear.
    - Only LoRA parameters (A, B for each layer) are trainable and should be
      exchanged between nodes.
    - If ``fspl_const`` is provided the model uses a physics-informed base:

          output = (-20 * log10(d) + fspl_const) + MLP_delta(features)

      where feature index 5 carries log10(d + eps). This means LoRA only
      needs to learn residuals over Free Space Path Loss, which dramatically
      improves cold-start accuracy compared to a random frozen backbone.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_sizes: Tuple[int, int] = (128, 64),
        rank: int = 16,
        alpha: float = 1.0,
        fspl_const: float | None = None,
    ):
        super().__init__()
        h1, h2 = hidden_sizes

        # Physics-base flag -------------------------------------------------
        # When enabled the MLP learns residuals over FSPL; the constant C is
        # stored as a non-trainable buffer so it travels with the model state.
        self.use_physics_base = fspl_const is not None
        if self.use_physics_base:
            self.register_buffer(
                "fspl_const", torch.tensor(float(fspl_const), dtype=torch.float32)
            )

        # Base layers (frozen – only LoRA adapters are trained) -------------
        self.fc1 = nn.Linear(input_dim, h1)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(h1, h2)
        self.act2 = nn.SiLU()
        self.fc3 = nn.Linear(h2, 1)

        nn.init.zeros_(self.fc3.bias)  # start residual output near 0

        for p in self.fc1.parameters():
            p.requires_grad = False
        for p in self.fc2.parameters():
            p.requires_grad = False
        for p in self.fc3.parameters():
            p.requires_grad = False

        # LoRA adapters on hidden layers
        self.lora1 = LoRALinear(self.fc1, rank=rank, alpha=alpha)
        self.lora2 = LoRALinear(self.fc2, rank=rank, alpha=alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_physics_base:
            # feature[:, 5] = log10(d + eps)  →  physics prior
            physics_out = -20.0 * x[..., 5] + self.fspl_const
        else:
            physics_out = None

        # MLP delta (LoRA adapters learn the residual)
        h = self.lora1(x)
        h = self.act1(h)
        h = self.lora2(h)
        h = self.act2(h)
        h = self.fc3(h).squeeze(-1)

        if physics_out is not None:
            return physics_out + h
        return h

    def adapter_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Iterator over LoRA (adapter) parameters only."""
        # IMPORTANT: Do not return `self.lora*.parameters()` here, because that
        # would also include the frozen base-layer parameters of `LoRALinear.base`.
        # We only want to optimize/exchange the low-rank matrices A and B.
        return [self.lora1.A, self.lora1.B, self.lora2.A, self.lora2.B]

    def get_adapter_state(self) -> dict:
        """Return a CPU state dict containing only adapter (A,B) parameters."""
        return {
            "lora1.A": self.lora1.A.detach().cpu().clone(),
            "lora1.B": self.lora1.B.detach().cpu().clone(),
            "lora2.A": self.lora2.A.detach().cpu().clone(),
            "lora2.B": self.lora2.B.detach().cpu().clone(),
        }

    def load_adapter_state(self, state: dict) -> None:
        """Load adapter parameters from a state dict (e.g., received from a peer)."""
        with torch.no_grad():
            if "lora1.A" in state:
                self.lora1.A.copy_(state["lora1.A"])
            if "lora1.B" in state:
                self.lora1.B.copy_(state["lora1.B"])
            if "lora2.A" in state:
                self.lora2.A.copy_(state["lora2.A"])
            if "lora2.B" in state:
                self.lora2.B.copy_(state["lora2.B"])

    def get_base_state(self) -> dict:
        """Return frozen base-layer weights (fc1, fc2, fc3) as a CPU state dict."""
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith(("fc1.", "fc2.", "fc3.", "fspl_const"))
        }

    def load_base_state(self, state: dict) -> None:
        """Load pre-trained base weights (fc1, fc2, fc3) and re-init LoRA adapters."""
        own = self.state_dict()
        for k, v in state.items():
            if k in own:
                own[k] = v
        self.load_state_dict(own)
        self.lora1.reset_parameters()
        self.lora2.reset_parameters()

    # ------------------------------------------------------------------
    # Pre-training helpers
    # ------------------------------------------------------------------

    def _unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def _refreeze_base_reset_lora(self) -> None:
        for p in self.fc1.parameters(): p.requires_grad = False
        for p in self.fc2.parameters(): p.requires_grad = False
        for p in self.fc3.parameters(): p.requires_grad = False
        self.lora1.reset_parameters()
        self.lora2.reset_parameters()
        for p in self.lora1.parameters(): p.requires_grad = True
        for p in self.lora2.parameters(): p.requires_grad = True

    def _train_loop(self, X: torch.Tensor, Y: torch.Tensor,
                    epochs: int, lr: float, batch_size: int,
                    loss_fn: nn.Module) -> None:
        """Generic mini-batch training loop (model must already be in the
        desired requires_grad state before calling this)."""
        n = len(X)
        optimizer = optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=lr
        )
        self.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            el, nb = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb, yb = X[idx], Y[idx]
                optimizer.zero_grad()
                loss = loss_fn(self(xb), yb)
                loss.backward()
                optimizer.step()
                el += float(loss.item())
                nb += 1
            if (epoch + 1) % 50 == 0:
                print(f"  pretrain epoch {epoch+1}/{epochs}  "
                      f"loss={el/nb:.4f}", flush=True)
        self.eval()

    def pretrain_on_fspl(
        self,
        encoder: "RadioFeatureEncoder",
        fspl_const: float,
        n_samples: int = 50_000,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 2048,
    ) -> None:
        """
        Pre-train backbone on synthetic FSPL data (vectorised, no Python loop).

        Target: RSSI = tx_power_dbm + (-20·log10(d) + fspl_const)
        TX powers are sampled uniformly from [TX_POWER_MIN_DBM, TX_POWER_MAX_DBM]
        so the backbone learns the full RSSI range, not just the path-loss offset.
        """
        map_cfg = encoder.map_cfg
        eps     = encoder.eps

        tx_x  = torch.empty(n_samples).uniform_(map_cfg.x_min, map_cfg.x_max)
        tx_y  = torch.empty(n_samples).uniform_(map_cfg.y_min, map_cfg.y_max)
        rx_x  = torch.empty(n_samples).uniform_(map_cfg.x_min, map_cfg.x_max)
        rx_y  = torch.empty(n_samples).uniform_(map_cfg.y_min, map_cfg.y_max)
        tx_pw = torch.empty(n_samples).uniform_(TX_POWER_MIN_DBM, TX_POWER_MAX_DBM)

        # Vectorised feature construction (avoids a 50 k-iteration Python loop)
        w = max(map_cfg.width,  eps)
        h = max(map_cfg.height, eps)
        xtx_n = (tx_x - map_cfg.x_min) / w
        ytx_n = (tx_y - map_cfg.y_min) / h
        xrx_n = (rx_x - map_cfg.x_min) / w
        yrx_n = (rx_y - map_cfg.y_min) / h
        dx    = rx_x - tx_x
        dy    = rx_y - tx_y
        d     = (dx.pow(2) + dy.pow(2)).sqrt().clamp(min=eps)
        log_d = torch.log10(d + eps)
        ang_n = torch.atan2(dy, dx) / math.pi
        pw_n  = (tx_pw - _TX_POWER_MID) / _TX_POWER_SCALE

        X = torch.stack([xtx_n, ytx_n, xrx_n, yrx_n, d, log_d, ang_n, pw_n], dim=1)
        Y = tx_pw - 20.0 * log_d + fspl_const   # RSSI = P_tx + path-loss

        self._unfreeze_all()
        self._train_loop(X, Y, epochs, lr, batch_size, nn.MSELoss())
        self._refreeze_base_reset_lora()

    def pretrain_on_dataset(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 4096,
    ) -> None:
        """
        Pre-train backbone on a pre-computed (features, RSSI) dataset from
        urban ray-tracing.  Uses Huber loss to tolerate deep-fade outliers.
        """
        self._unfreeze_all()
        self._train_loop(X, Y, epochs, lr, batch_size, nn.HuberLoss(delta=5.0))
        self._refreeze_base_reset_lora()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_backbone(self, path: str) -> None:
        """Persist frozen base-layer weights to *path* (PyTorch checkpoint)."""
        torch.save({"model_state": self.get_base_state()}, path)
        print(f"Backbone saved → {path}", flush=True)

    @classmethod
    def load_backbone_from_file(
        cls,
        path: str,
        input_dim: int = 8,
        hidden_sizes: Tuple[int, int] = (128, 128),
        rank: int = 16,
        alpha: float = 1.0,
        fspl_const: Optional[float] = None,
    ) -> "RadioMLPWithLoRA":
        """Load a backbone checkpoint and return a model ready for LoRA fine-tuning."""
        model = cls(input_dim=input_dim, hidden_sizes=hidden_sizes,
                    rank=rank, alpha=alpha, fspl_const=fspl_const)
        ckpt  = torch.load(path, map_location="cpu", weights_only=True)
        src   = ckpt.get("model_state", ckpt)
        own   = model.state_dict()
        for k, v in src.items():
            if k in own:
                own[k] = v
        model.load_state_dict(own)
        model._refreeze_base_reset_lora()
        model.eval()
        return model

