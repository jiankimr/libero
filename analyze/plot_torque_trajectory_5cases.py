"""
Plot torque trajectories for a single common success episode across 5 cases:
0 (clean baseline), 25, 50, 75, 100 (rollout ratios).

Output: one PNG with 7 subplots (joint_0..6), overlaying 5 lines.
"""

from __future__ import annotations

import argparse
import glob
import pathlib
from typing import Dict, List

import numpy as np


def find_common_success_file(dirs: Dict[int, pathlib.Path], prefer_substr: str | None = None) -> str:
    sets = {}
    for k, d in dirs.items():
        files = set(pathlib.Path(p).name for p in glob.glob(str(d / "torque_current_*_success.npy")))
        sets[k] = files
        if not files:
            raise FileNotFoundError(f"No torque_current_*_success.npy in {d}")

    common = set.intersection(*sets.values())
    if not common:
        raise FileNotFoundError("No common success npy file across all cases.")

    if prefer_substr:
        prefer = [f for f in sorted(common) if prefer_substr in f]
        if prefer:
            return prefer[0]

    return sorted(common)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_dir", type=str, default="./plots", help="Output dir for png")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional exact filename (e.g., torque_current_..._success.npy). If omitted, auto-picks a common file.",
    )
    parser.add_argument(
        "--prefer_substr",
        type=str,
        default="pick_up_the_book",
        help="When auto-picking, prefer files containing this substring.",
    )
    parser.add_argument(
        "--truncate_to_min_len",
        action="store_true",
        default=True,
        help="Truncate all trajectories to min length for fair overlay (default: True).",
    )
    args = parser.parse_args()

    libero_dir = pathlib.Path(__file__).resolve().parents[1]
    analysis_root = libero_dir / "analysis"

    dirs: Dict[int, pathlib.Path] = {
        0: analysis_root / "analysis_libero_10_20251202_170035_noise_00000_clean",
        25: analysis_root / "analysis_libero_10_20260130_120729_noise_00000_rollout_0.3x_25",
        50: analysis_root / "analysis_libero_10_20260130_120729_noise_00000_rollout_0.3x_50",
        75: analysis_root / "analysis_libero_10_20260130_120742_noise_00000_rollout_0.3x_75",
        100: analysis_root / "analysis_libero_10_20260130_032712_noise_00000_rollout_0.3x_100",
    }

    plots_dir = pathlib.Path(args.plots_dir)
    if not plots_dir.is_absolute():
        plots_dir = (libero_dir / plots_dir).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    filename = args.file or find_common_success_file(dirs, args.prefer_substr)

    # load
    series: Dict[int, np.ndarray] = {}
    lengths: Dict[int, int] = {}
    for k, d in dirs.items():
        p = d / filename
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        x = np.load(p)
        if x.ndim != 2 or x.shape[1] != 7:
            raise ValueError(f"Unexpected shape in {p}: {x.shape} (expected (T,7))")
        series[k] = x.astype(float)
        lengths[k] = x.shape[0]

    min_len = min(lengths.values())
    if args.truncate_to_min_len:
        for k in series:
            series[k] = series[k][:min_len]

    # plot
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    colors = {0: "#2c3e50", 25: "#1f77b4", 50: "#ff7f0e", 75: "#2ca02c", 100: "#d62728"}
    order = [0, 25, 50, 75, 100]

    for j in range(7):
        ax = axes[j]
        for k in order:
            y = series[k][:, j]
            ax.plot(np.arange(len(y)), y, linewidth=1.2, color=colors[k], label=f"{k}%")
        ax.set_ylabel(f"joint_{j}\n(Nm)")
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(ncol=5, frameon=False, loc="upper right")

    axes[-1].set_xlabel("timestep")

    title = f"Torque trajectory overlay (5 cases)\\n{filename}"
    if args.truncate_to_min_len:
        title += f"  | truncated to min_len={min_len}"
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out = plots_dir / "torque_trajectory_5cases_pick_up_the_book.png"
    fig.savefig(out)
    plt.close(fig)

    print(str(out))


if __name__ == "__main__":
    main()



