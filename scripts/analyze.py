"""
analyze.py — Standalone snapshot analysis script.

Reads the live training metrics CSV (read-only, safe to run alongside
an active training process) and writes a timestamped snapshot to
artifacts/snapshots/<timestamp>/ with:
  - metrics_snapshot.csv   (clean copy of what was available)
  - training_metrics.png   (full multi-panel plot)
  - fitness_curve.png      (zoomed fitness only)
  - progress_summary.png   (floor / bosses / win-rate)
  - summary.txt            (human-readable stats)

Usage:
  py -3 scripts/analyze.py
  py -3 scripts/analyze.py --csv artifacts/metrics.csv --out artifacts/snapshots
  py -3 scripts/analyze.py --watch          # refresh every 60 s
  py -3 scripts/analyze.py --watch --interval 30
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display required — runs headlessly
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot training analytics without touching the live run.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "artifacts" / "metrics.csv",
        help="Path to the live metrics CSV (default: artifacts/metrics.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "snapshots",
        help="Root directory for snapshot output (default: artifacts/snapshots)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep running and refresh the snapshot periodically.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between refreshes in --watch mode (default: 60).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CSV reader — robust against partial last line (training may be mid-write)
# ---------------------------------------------------------------------------

def load_metrics(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        print(f"[analyze] CSV not found: {csv_path}", file=sys.stderr)
        return []
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                row = {(key.lstrip("\ufeff") if key else key): value for key, value in row.items()}
                parsed = {
                    "generation":           int(row["generation"]),
                    "best_fitness":         float(row["best_fitness"]),
                    "mean_fitness":         float(row["mean_fitness"]),
                    "mean_reward":          float(row["mean_reward"]),
                    "mean_survival":        float(row["mean_survival"]),
                    "mean_floor_reached":   float(row["mean_floor_reached"]),
                    "mean_bosses_defeated": float(row["mean_bosses_defeated"]),
                    "mean_chests_opened":   float(row["mean_chests_opened"]),
                    "mean_powerups_picked": float(row.get("mean_powerups_picked", row.get("mean_chests_opened", 0.0)) or 0.0),
                    "mean_damage":          float(row.get("mean_damage", 0.0) or 0.0),
                    "mean_gate_distance":   float(row.get("mean_gate_distance", 0.0) or 0.0),
                    "mean_best_gate_distance": float(row.get("mean_best_gate_distance", row.get("mean_gate_distance", 0.0)) or 0.0),
                    "mean_gate_tile_visits": float(row.get("mean_gate_tile_visits", 0.0) or 0.0),
                    "mean_use_gate_attempts": float(row.get("mean_use_gate_attempts", 0.0) or 0.0),
                    "mean_invalid_use_gate_attempts": float(row.get("mean_invalid_use_gate_attempts", 0.0) or 0.0),
                    "success_rate":         float(row.get("success_rate", 0.0) or 0.0),
                    "champion_id":          int(row["champion_id"]),
                }
                rows.append(parsed)
            except (ValueError, KeyError):
                # skip partial / malformed last row that may be mid-write
                continue
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_snapshot_csv(rows: list[dict], dest: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_txt(rows: list[dict], dest: Path, source_csv: Path) -> None:
    if not rows:
        return
    best_idx = int(np.argmax([r["best_fitness"] for r in rows]))
    last = rows[-1]
    best = rows[best_idx]

    with dest.open("w", encoding="utf-8") as fh:
        fh.write("=" * 56 + "\n")
        fh.write("  Neuroevolution Arena — Training Snapshot\n")
        fh.write("=" * 56 + "\n")
        fh.write(f"  Snapshot time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"  Source CSV    : {source_csv}\n")
        fh.write(f"  Generations   : {len(rows)}  (0 → {last['generation']})\n")
        fh.write("\n")
        fh.write("  — Last generation —\n")
        fh.write(f"    Best fitness   : {last['best_fitness']:.2f}\n")
        fh.write(f"    Mean fitness   : {last['mean_fitness']:.2f}\n")
        fh.write(f"    Mean floor     : {last['mean_floor_reached']:.2f}\n")
        fh.write(f"    Mean survival  : {last['mean_survival']:.0f} steps\n")
        fh.write(f"    Mean bosses    : {last['mean_bosses_defeated']:.2f}\n")
        fh.write(f"    Mean chests    : {last['mean_chests_opened']:.2f}\n")
        fh.write(f"    Mean powerups  : {last['mean_powerups_picked']:.2f}\n")
        fh.write(f"    Best gate dist : {last['mean_best_gate_distance']:.2f}\n")
        fh.write(f"    Gate visits    : {last['mean_gate_tile_visits']:.2f}\n")
        fh.write(f"    Bad gate acts  : {last['mean_invalid_use_gate_attempts']:.2f}\n")
        fh.write(f"    Win rate       : {last['success_rate'] * 100:.1f}%\n")
        fh.write(f"    Champion agent : P{last['champion_id']}\n")
        fh.write("\n")
        fh.write(f"  — All-time best (gen {best['generation']}) —\n")
        fh.write(f"    Best fitness   : {best['best_fitness']:.2f}\n")
        fh.write(f"    Mean fitness   : {best['mean_fitness']:.2f}\n")
        fh.write(f"    Floor reached  : {best['mean_floor_reached']:.2f}\n")
        fh.write(f"    Win rate       : {best['success_rate'] * 100:.1f}%\n")
        fh.write("=" * 56 + "\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

STYLE = {
    "figure.facecolor": "#0e1218",
    "axes.facecolor":   "#131820",
    "axes.edgecolor":   "#2a3244",
    "axes.labelcolor":  "#c8d0e0",
    "xtick.color":      "#8899aa",
    "ytick.color":      "#8899aa",
    "text.color":       "#c8d0e0",
    "grid.color":       "#1e2840",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "legend.facecolor": "#131820",
    "legend.edgecolor": "#2a3244",
    "legend.labelcolor":"#c8d0e0",
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _smooth(values: list[float], window: int = 15) -> list[float]:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return list(np.convolve(padded, kernel, mode="valid")[: len(values)])


def plot_full(rows: list[dict], dest: Path) -> None:
    _apply_style()
    gens            = [r["generation"] for r in rows]
    best_fitness    = [r["best_fitness"] for r in rows]
    mean_fitness    = [r["mean_fitness"] for r in rows]
    floor_reached   = [r["mean_floor_reached"] for r in rows]
    bosses          = [r["mean_bosses_defeated"] for r in rows]
    chests          = [r["mean_chests_opened"] for r in rows]
    gate_dist       = [r["mean_gate_distance"] for r in rows]
    best_gate_dist  = [r["mean_best_gate_distance"] for r in rows]
    gate_visits     = [r["mean_gate_tile_visits"] for r in rows]
    success_rate    = [r["success_rate"] for r in rows]
    mean_survival   = [r["mean_survival"] for r in rows]

    fig, axes = plt.subplots(5, 1, figsize=(12, 22), sharex=True)
    fig.suptitle(
        f"Training Snapshot — {len(rows)} generations",
        color="#e8edf5", fontsize=15, fontweight="bold", y=0.995,
    )

    # Fitness
    axes[0].plot(gens, best_fitness, color="#5ec4ff", linewidth=2.2, label="best fitness")
    axes[0].plot(gens, _smooth(mean_fitness), color="#ff9f5e", linewidth=1.6, alpha=0.9, label="mean fitness (smoothed)")
    axes[0].plot(gens, mean_fitness, color="#ff9f5e", linewidth=0.7, alpha=0.35)
    axes[0].set_ylabel("Fitness")
    axes[0].legend(loc="upper left")
    axes[0].grid(True)

    # Floor
    axes[1].fill_between(gens, floor_reached, alpha=0.25, color="#50c8a0")
    axes[1].plot(gens, _smooth(floor_reached), color="#50c8a0", linewidth=2.0)
    axes[1].set_ylabel("Avg Floor Reached")
    axes[1].grid(True)

    # Combat
    axes[2].plot(gens, _smooth(bosses), color="#e05c5c", linewidth=2.0, label="bosses")
    axes[2].plot(gens, _smooth(chests), color="#6dc96d", linewidth=1.8, label="chests", alpha=0.85)
    axes[2].set_ylabel("Bosses / Chests")
    axes[2].legend(loc="upper left")
    axes[2].grid(True)

    # Gate distance
    axes[3].plot(gens, _smooth(gate_dist), color="#c49bff", linewidth=2.0, label="end gate")
    axes[3].plot(gens, _smooth(best_gate_dist), color="#bcbd22", linewidth=1.5, label="best gate")
    axes[3].set_ylabel("Gate Distance")
    axes[3].legend(loc="upper right")
    axes[3].grid(True)

    # Win rate + survival
    ax3b = axes[4].twinx()
    axes[4].plot(gens, _smooth(success_rate), color="#ffd966", linewidth=2.2, label="win rate")
    axes[4].plot(gens, _smooth(gate_visits), color="#17becf", linewidth=1.4, alpha=0.85, label="gate visits")
    ax3b.plot(gens, _smooth(mean_survival), color="#7bafd4", linewidth=1.5, alpha=0.7, label="survival steps")
    axes[4].set_ylabel("Win Rate", color="#ffd966")
    ax3b.set_ylabel("Survival Steps", color="#7bafd4")
    axes[4].tick_params(axis="y", labelcolor="#ffd966")
    ax3b.tick_params(axis="y", labelcolor="#7bafd4")
    axes[4].set_xlabel("Generation")
    axes[4].grid(True)
    lines_a, labels_a = axes[4].get_legend_handles_labels()
    lines_b, labels_b = ax3b.get_legend_handles_labels()
    axes[4].legend(lines_a + lines_b, labels_a + labels_b, loc="upper left")

    fig.tight_layout()
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_fitness_curve(rows: list[dict], dest: Path) -> None:
    _apply_style()
    gens         = [r["generation"] for r in rows]
    best_fitness = [r["best_fitness"] for r in rows]
    mean_fitness = [r["mean_fitness"] for r in rows]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(gens, best_fitness, color="#5ec4ff", linewidth=2.4, label="best fitness")
    ax.fill_between(gens, mean_fitness, best_fitness, alpha=0.12, color="#5ec4ff")
    ax.plot(gens, _smooth(mean_fitness, window=20), color="#ff9f5e", linewidth=1.8,
            linestyle="--", label="mean fitness (smoothed)")

    all_time_best_gen = gens[int(np.argmax(best_fitness))]
    all_time_best_val = max(best_fitness)
    ax.axvline(all_time_best_gen, color="#ffd966", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.annotate(
        f"★ {all_time_best_val:.1f}",
        xy=(all_time_best_gen, all_time_best_val),
        xytext=(10, -20),
        textcoords="offset points",
        color="#ffd966",
        fontsize=10,
    )

    ax.set_title("Fitness Curve", color="#e8edf5", fontsize=13, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_progress_summary(rows: list[dict], dest: Path) -> None:
    _apply_style()
    gens          = [r["generation"] for r in rows]
    floor_reached = [r["mean_floor_reached"] for r in rows]
    bosses        = [r["mean_bosses_defeated"] for r in rows]
    success_rate  = [r["success_rate"] for r in rows]

    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(gens, _smooth(floor_reached), color="#50c8a0", linewidth=2.2)
    ax0.fill_between(gens, floor_reached, alpha=0.18, color="#50c8a0")
    ax0.set_title("Avg Floor Reached", color="#e8edf5")
    ax0.set_xlabel("Generation")
    ax0.grid(True)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(gens, _smooth(bosses), color="#e05c5c", linewidth=2.2)
    ax1.fill_between(gens, bosses, alpha=0.18, color="#e05c5c")
    ax1.set_title("Avg Bosses Defeated", color="#e8edf5")
    ax1.set_xlabel("Generation")
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(gens, _smooth(success_rate), color="#ffd966", linewidth=2.2)
    ax2.fill_between(gens, success_rate, alpha=0.18, color="#ffd966")
    ax2.set_title("Win Rate", color="#e8edf5")
    ax2.set_xlabel("Generation")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax2.grid(True)

    fig.suptitle("Progress Summary", color="#e8edf5", fontsize=13, fontweight="bold")
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main snapshot logic
# ---------------------------------------------------------------------------

def run_snapshot(source_csv: Path, out_root: Path) -> Path:
    rows = load_metrics(source_csv)
    if not rows:
        print("[analyze] No valid rows found — nothing to plot.")
        return out_root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir  = out_root / timestamp
    snap_dir.mkdir(parents=True, exist_ok=True)

    # 1. Snapshot CSV
    write_snapshot_csv(rows, snap_dir / "metrics_snapshot.csv")

    # 2. Summary text
    write_summary_txt(rows, snap_dir / "summary.txt", source_csv)

    # 3. Plots
    plot_full(rows,             snap_dir / "training_metrics.png")
    plot_fitness_curve(rows,    snap_dir / "fitness_curve.png")
    plot_progress_summary(rows, snap_dir / "progress_summary.png")

    # 4. Also write a "latest" symlink-equivalent (just copy) for quick access
    latest_dir = out_root / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(snap_dir, latest_dir)

    gen_count = len(rows)
    best      = max(r["best_fitness"] for r in rows)
    last_gen  = rows[-1]["generation"]
    print(
        f"[analyze] Snapshot saved -> {snap_dir}\n"
        f"          Generations: {gen_count}  |  Last gen: {last_gen}  |  Best fitness: {best:.2f}"
    )
    return snap_dir


def main() -> None:
    args = _parse_args()

    if args.watch:
        print(f"[analyze] Watch mode — refreshing every {args.interval}s. Ctrl+C to stop.")
        try:
            while True:
                run_snapshot(args.csv, args.out)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[analyze] Stopped.")
    else:
        run_snapshot(args.csv, args.out)


if __name__ == "__main__":
    main()
