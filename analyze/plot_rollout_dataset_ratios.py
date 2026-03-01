"""
Make 4 plots for rollout dataset ratios (0,25,50,75,100):
1) Success rate (includes baseline=0)
2) Joint-wise life ratio (rainflow) + annotate worst joint per case
3) Joint-wise life ratio (fatpack)  + annotate worst joint per case
4) Energy sums: draw.sum, regen.sum (downward), net.sum
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def find_analysis_summary_csv(analysis_dir: pathlib.Path) -> pathlib.Path:
    analysis_dir = pathlib.Path(analysis_dir)
    parent = analysis_dir.parent
    target = analysis_dir.name
    if target.startswith("analysis_"):
        target = target[len("analysis_") :]

    candidates = sorted(parent.glob("analysis_summary_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No analysis_summary_*.csv in {parent}")

    for c in candidates:
        if target in c.name:
            return c

    parts = target.split("_")
    if len(parts) > 1:
        core = "_".join(parts[:-1])
        for c in candidates:
            if core in c.name:
                return c

    raise FileNotFoundError(f"Could not match analysis_summary csv for {analysis_dir}")


def read_success_rate(summary_csv: pathlib.Path) -> float:
    df = pd.read_csv(summary_csv)
    if "success" not in df.columns:
        raise ValueError(f"'success' column not found in {summary_csv}")
    return float(df["success"].sum()) / float(len(df)) * 100.0


def read_energy_means(summary_csv: pathlib.Path) -> Dict[str, float]:
    df = pd.read_csv(summary_csv)
    if "success" in df.columns:
        df = df[df["success"] == 1].copy()
    cols = ["energy_draw.sum", "energy_regen.sum", "energy_net.sum"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing energy columns in {summary_csv}: {missing}")
    return {c: float(pd.to_numeric(df[c], errors="coerce").mean()) for c in cols}


def _finite_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    return x


def read_joint_life_ratio_means(npy_csv: pathlib.Path, method: str) -> Tuple[np.ndarray, Dict[str, int]]:
    df = pd.read_csv(npy_csv)
    reliable_col = f"{method}_life_ratio_reliable"
    if reliable_col in df.columns:
        df = df[df[reliable_col] == True].copy()  # noqa: E712

    rows_total = len(df)
    joint_means: List[float] = []
    for j in range(7):
        col = f"{method}_joint_{j}_damage_ratio"
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {npy_csv}")
        vals = _finite_series(df[col])
        joint_means.append(float(vals.mean()) if len(vals) else float("nan"))

    lr_col = f"{method}_life_ratio"
    rows_used = len(_finite_series(df[lr_col])) if lr_col in df.columns else rows_total
    return np.array(joint_means, dtype=float), {"rows_total": rows_total, "rows_used": rows_used}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_dir", type=str, default="./plots", help="Output dir for png files")
    args = parser.parse_args()

    # LIBERO/...
    libero_dir = pathlib.Path(__file__).resolve().parents[1]
    analysis_root = libero_dir / "analysis"
    results_root = libero_dir / "results"

    plots_dir = pathlib.Path(args.plots_dir)
    if not plots_dir.is_absolute():
        plots_dir = (libero_dir / plots_dir).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    before_dir = analysis_root / "analysis_libero_10_20251202_170035_noise_00000_clean"

    after_dirs = {
        25: analysis_root / "analysis_libero_10_20260130_120729_noise_00000_rollout_0.3x_25",
        50: analysis_root / "analysis_libero_10_20260130_120729_noise_00000_rollout_0.3x_50",
        75: analysis_root / "analysis_libero_10_20260130_120742_noise_00000_rollout_0.3x_75",
        100: analysis_root / "analysis_libero_10_20260130_032712_noise_00000_rollout_0.3x_100",
    }

    # NPY compare CSVs created by compare_metrics.py --output_npy_csv
    ratio_csvs = {
        25: results_root / "rollout_0.3x_25_vs_clean_npy.csv",
        50: results_root / "rollout_0.3x_50_vs_clean_npy.csv",
        75: results_root / "rollout_0.3x_75_vs_clean_npy.csv",
        100: results_root / "rollout_0.3x_100_vs_clean_npy.csv",
    }

    # ---- load summary CSVs (success/energy) ----
    summary_csvs: Dict[int, pathlib.Path] = {0: find_analysis_summary_csv(before_dir)}
    for k, d in after_dirs.items():
        summary_csvs[k] = find_analysis_summary_csv(d)

    x = [0, 25, 50, 75, 100]

    # 1) success rate
    success_rates = [read_success_rate(summary_csvs[p]) for p in x]

    # 4) energy
    energy_draw = []
    energy_regen_down = []  # plot downward
    energy_net = []
    for p in x:
        em = read_energy_means(summary_csvs[p])
        energy_draw.append(em["energy_draw.sum"])
        energy_regen_down.append(-abs(em["energy_regen.sum"]))
        energy_net.append(em["energy_net.sum"])

    # 2/3) life ratio (joint-wise)
    methods = ["rainflow", "fatpack"]
    joint_ratios: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in methods}
    worst_ann: Dict[str, Dict[int, Tuple[int, float]]] = {m: {} for m in methods}

    for m in methods:
        # baseline (0): ratio = 1.0 for all joints
        joint_ratios[m][0] = np.ones(7, dtype=float)
        worst_ann[m][0] = (0, 1.0)
        for p in [25, 50, 75, 100]:
            r, _counts = read_joint_life_ratio_means(ratio_csvs[p], m)
            joint_ratios[m][p] = r
            j_worst = int(np.nanargmax(r))
            worst_ann[m][p] = (j_worst, float(r[j_worst]))

    # ---- plotting ----
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
        }
    )

    # (1) success rate plot
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(x, success_rates, marker="o", linewidth=2)
    ax.set_title("Success Rate vs Rollout Dataset Ratio (libero_10)")
    ax.set_xlabel("Rollout ratio (%)  [0=clean baseline]")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    for xp, yp in zip(x, success_rates):
        ax.text(xp, yp + 0.6, f"{yp:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "rollout_ratio_success_rate.png")
    plt.close(fig)

    def plot_life_ratio(method: str) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for j in range(7):
            ys = [joint_ratios[method][p][j] for p in x]
            ax.plot(x, ys, marker="o", linewidth=1.8, label=f"joint_{j}")

        ax.set_title(f"Joint-wise Life Ratio vs Rollout Dataset Ratio ({method})")
        ax.set_xlabel("Rollout ratio (%)  [0=baseline → ratio=1.0]")
        ax.set_ylabel("Life ratio (after/before)  [>1 worse]")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)

        # annotate worst joint per case (skip baseline text clutter)
        for p in [25, 50, 75, 100]:
            j_w, v = worst_ann[method][p]
            if np.isnan(v):
                continue
            ax.text(p, v * 1.02, f"worst: j{j_w} {v:.2f}x", ha="center", va="bottom", fontsize=9)

        ax.legend(ncol=2, fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(plots_dir / f"rollout_ratio_life_ratio_{method}.png")
        plt.close(fig)

    plot_life_ratio("rainflow")
    plot_life_ratio("fatpack")

    # (4) energy bar plot
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    idx = np.arange(len(x), dtype=float)
    w = 0.25
    ax.bar(idx - w, energy_draw, width=w, label="energy_draw.sum", color="#1f77b4")
    ax.bar(idx, energy_regen_down, width=w, label="energy_regen.sum (down)", color="#2ca02c")
    ax.bar(idx + w, energy_net, width=w, label="energy_net.sum", color="#ff7f0e")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Energy Consumption (mean over successful episodes)")
    ax.set_xlabel("Rollout ratio (%)  [0=clean baseline]")
    ax.set_ylabel("Energy sum (J)")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(v) for v in x])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots_dir / "rollout_ratio_energy_sums.png")
    plt.close(fig)


if __name__ == "__main__":
    main()



