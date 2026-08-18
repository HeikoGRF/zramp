import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sionna_baseline.radio_mlp_lora import MapConfig, RadioFeatureEncoder, RadioMLPWithLoRA


def main() -> None:
    map_cfg = MapConfig(x_min=0.0, x_max=100.0, y_min=-200.0, y_max=-100.0)
    enc = RadioFeatureEncoder(map_cfg)

    f1 = enc.encode((10.0, -150.0), (20.0, -140.0), tx_power_dbm=0.0, has_obstacle=False)
    f2 = enc.encode((10.0, -150.0), (20.0, -140.0), tx_power_dbm=0.0, has_obstacle=True)

    assert f1.shape == (8,), f"Expected 8D features, got {tuple(f1.shape)}"
    assert f2.shape == (8,), f"Expected 8D features, got {tuple(f2.shape)}"

    model = RadioMLPWithLoRA()  # default input_dim must match encoder (8)
    x = torch.stack([f1, f2], dim=0)
    y = model(x)
    assert y.shape == (2,), f"Expected output shape (2,), got {tuple(y.shape)}"

    print("OK: encoder=8D (with tx_power_n), model forward pass works.")


if __name__ == "__main__":
    main()

