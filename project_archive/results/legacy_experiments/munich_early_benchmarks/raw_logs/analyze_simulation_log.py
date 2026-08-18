import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualitative and quantitative analysis of simulation_log.csv."
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="simulation_log.csv",
        help="Path to simulation_log.csv (default: simulation_log.csv in repo root).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Directory to store plots and summary tables.",
    )
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find log file at '{path}'")
    df = pd.read_csv(path)
    expected_cols = {
        "timestamp",
        "node_id",
        "x",
        "y",
        "in_anchor_zone",
        "measured_rssi",
        "pinged_node",
    }
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Log file is missing expected columns: {missing}")
    return df


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def add_anchor_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Parse in_anchor_zone like 'Zone_-1_0' into integer indices (zone_x, zone_y)."""

    def _parse_zone(zone: str) -> Tuple[int, int]:
        try:
            _, zx, zy = zone.split("_")
            return int(zx), int(zy)
        except Exception:
            return 0, 0

    zx, zy = zip(*df["in_anchor_zone"].map(_parse_zone))
    df = df.copy()
    df["zone_x"] = zx
    df["zone_y"] = zy
    return df


def save_summary_stats(df: pd.DataFrame, out_dir: str) -> None:
    """Compute and save quantitative summary statistics for later comparison."""

    summary_global = df["measured_rssi"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    summary_global.to_csv(os.path.join(out_dir, "summary_rssi_global.csv"))

    summary_by_node = (
        df.groupby("node_id")["measured_rssi"]
        .agg(["count", "mean", "std", "min", "max", "median"])
        .reset_index()
    )
    summary_by_node.to_csv(os.path.join(out_dir, "summary_rssi_by_node.csv"), index=False)

    summary_by_zone = (
        df.groupby("in_anchor_zone")["measured_rssi"]
        .agg(["count", "mean", "std", "min", "max", "median"])
        .reset_index()
    )
    summary_by_zone.to_csv(os.path.join(out_dir, "summary_rssi_by_zone.csv"), index=False)


def plot_rssi_histogram(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(df["measured_rssi"], bins=40, kde=True)
    plt.xlabel("Measured RSSI [dB]")
    plt.ylabel("Count")
    plt.title("RSSI Distribution (All Nodes)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rssi_histogram.png"), dpi=200)
    plt.close()


def plot_rssi_by_node(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="node_id", y="measured_rssi")
    plt.xlabel("Receiver node_id")
    plt.ylabel("Measured RSSI [dB]")
    plt.title("RSSI per Receiver Node")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rssi_by_node_boxplot.png"), dpi=200)
    plt.close()


def plot_rssi_time_series(df: pd.DataFrame, out_dir: str, max_nodes: int = 3) -> None:
    """Plot RSSI over time for up to max_nodes distinct receiver nodes."""
    nodes = sorted(df["node_id"].unique())[:max_nodes]
    plt.figure(figsize=(7, 4))
    for nid in nodes:
        sub = df[df["node_id"] == nid]
        plt.scatter(sub["timestamp"], sub["measured_rssi"], s=6, alpha=0.6, label=f"node {nid}")
    plt.xlabel("Time [s]")
    plt.ylabel("Measured RSSI [dB]")
    plt.title(f"RSSI over Time for First {len(nodes)} Nodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rssi_time_series_sample_nodes.png"), dpi=200)
    plt.close()


def plot_rssi_vs_zone_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """Heatmap of mean RSSI per anchor zone."""
    df_z = add_anchor_indices(df)
    pivot = df_z.pivot_table(
        index="zone_y",
        columns="zone_x",
        values="measured_rssi",
        aggfunc=np.mean,
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, cmap="viridis", annot=False, cbar_kws={"label": "Mean RSSI [dB]"})
    plt.xlabel("zone_x")
    plt.ylabel("zone_y")
    plt.title("Mean RSSI per Anchor Zone")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rssi_anchor_zone_heatmap.png"), dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    df = load_data(args.log_path)

    # Basic type cleaning
    df["timestamp"] = df["timestamp"].astype(float)
    df["measured_rssi"] = df["measured_rssi"].astype(float)

    save_summary_stats(df, args.output_dir)
    plot_rssi_histogram(df, args.output_dir)
    plot_rssi_by_node(df, args.output_dir)
    plot_rssi_time_series(df, args.output_dir)
    plot_rssi_vs_zone_heatmap(df, args.output_dir)


if __name__ == "__main__":
    main()

