import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sionna_baseline.radio_mlp import MapConfig, RadioFeatureEncoder, RadioMLP


def main() -> None:
    map_cfg = MapConfig(x_min=0.0, x_max=100.0, y_min=-200.0, y_max=-100.0)
    enc = RadioFeatureEncoder(map_cfg)
    f = enc.encode((0.0, -150.0), (10.0, -140.0), tx_power_dbm=0.0, has_obstacle=True)
    assert f.shape == (8,)

    m = RadioMLP(fspl_const=-40.0)
    y = m(f.unsqueeze(0))
    assert y.shape == (1,)
    print("OK: fullft folder smoke test passed.")


if __name__ == "__main__":
    main()

