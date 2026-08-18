#!/usr/bin/env python3
"""Keep a source-pretrained encoder while deterministically resetting its head."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from online_policy_learning.online_local_validation_policy import ExactModelTrajectoryPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "cross_map_pretrained_exact_policy_v1":
        raise ValueError("unsupported source policy checkpoint")
    architecture = dict(checkpoint["architecture"])
    with torch.random.fork_rng():
        torch.manual_seed(int(args.seed))
        policy = ExactModelTrajectoryPolicy(
            group_widths=tuple(int(value) for value in architecture["group_widths"]),
            trajectory_dim=int(architecture["trajectory_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            embedding_dim=int(architecture["embedding_dim"]),
            gain_hidden_dim=int(architecture["gain_hidden_dim"]),
            pair_feature_mode=str(architecture["pair_feature_mode"]),
        )
    state = policy.state_dict()
    source_state = checkpoint["policy_state_dict"]
    for name in state:
        if not name.startswith("gain_head."):
            state[name] = source_state[name].detach().cpu().clone()
    output = dict(checkpoint)
    output["policy_state_dict"] = state
    output["head_initialization"] = {
        "kind": "fresh_deterministic",
        "seed": int(args.seed),
        "source_head_discarded": True,
    }
    output["encoder_pretraining_preserved"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
