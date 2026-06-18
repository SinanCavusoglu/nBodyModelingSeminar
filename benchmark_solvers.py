"""Benchmark direct vs optimized Barnes-Hut force solvers.

Examples:
  python benchmark_solvers.py --particles 100 300 500 1000 --theta 0.3 0.5 0.8 1.0
  python benchmark_solvers.py --particles 100 300 500 1000 3000 --max-direct-particles 1000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from scripts.generate_connection_velocity import generate
from src.data_loader import load_bodies
from src.physics import compute_direct_softened_accelerations
from src.barnes_hut_fast import NUMBA_AVAILABLE, compute_barnes_hut_fast_accelerations


def ensure_input() -> Path:
    csv_path = Path(config.RADIAL_CSV_PATH)
    if not csv_path.exists():
        generate(
            input_path=config.RAW_CSV_PATH,
            output_dir=csv_path.parent,
            mode="radial",
            scale=float(config.CONNECTION_VELOCITY_SCALE),
            use_mass_strength=bool(config.USE_CONNECTION_MASS_STRENGTH),
        )
    return csv_path


def time_call(fn, repeat: int = 3) -> tuple[float, Any]:
    best = float("inf")
    last = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        last = fn()
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best, last


def relative_acc_error(reference: np.ndarray, approx: np.ndarray) -> tuple[float, float]:
    diff_norm = np.linalg.norm(reference - approx, axis=1)
    ref_norm = np.linalg.norm(reference, axis=1)
    denom = np.maximum(ref_norm, 1.0e-12)
    rel = diff_norm / denom
    return float(np.mean(rel)), float(np.max(rel))


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    csv_path = ensure_input()
    max_n = max(args.particles)
    loaded = load_bodies(csv_path, max_particles=max_n)
    all_pos = loaded["positions"]
    all_mass = loaded["masses"]

    rows: list[dict[str, Any]] = []

    # Warm up Numba once on a small subset so first benchmark is not dominated by compilation.
    if NUMBA_AVAILABLE and not args.no_numba:
        warm_n = min(32, len(all_pos))
        if warm_n >= 2:
            _ = compute_barnes_hut_fast_accelerations(
                all_pos[:warm_n], all_mass[:warm_n],
                G=args.G, softening=args.softening, theta=0.5,
                max_particles_per_leaf=args.leaf_size,
                max_depth=args.max_depth,
                use_numba=True,
                direct_fallback_n=0,
            )

    for n in args.particles:
        pos = all_pos[:n]
        mass = all_mass[:n]
        print(f"[benchmark] N={n}")

        direct_time = None
        direct_acc = None
        if n <= args.max_direct_particles:
            direct_time, direct_acc = time_call(
                lambda: compute_direct_softened_accelerations(pos, mass, args.G, args.softening),
                repeat=args.repeat,
            )
            print(f"  direct: {direct_time:.4f}s")
        else:
            print("  direct: skipped")

        for theta in args.theta:
            bh_time, bh_acc = time_call(
                lambda th=theta: compute_barnes_hut_fast_accelerations(
                    pos,
                    mass,
                    G=args.G,
                    softening=args.softening,
                    theta=th,
                    max_particles_per_leaf=args.leaf_size,
                    max_depth=args.max_depth,
                    use_numba=not args.no_numba,
                    direct_fallback_n=0,  # force true BH for benchmark
                ),
                repeat=args.repeat,
            )
            mean_err = None
            max_err = None
            speedup = None
            if direct_acc is not None and direct_time is not None:
                mean_err, max_err = relative_acc_error(direct_acc, bh_acc)
                speedup = direct_time / bh_time if bh_time > 0 else None
            print(f"  BH theta={theta}: {bh_time:.4f}s speedup={speedup} mean_err={mean_err}")

            rows.append({
                "particles": n,
                "theta": theta,
                "direct_time_s": direct_time,
                "barnes_hut_time_s": bh_time,
                "speedup_direct_over_bh": speedup,
                "mean_relative_acc_error": mean_err,
                "max_relative_acc_error": max_err,
                "leaf_size": args.leaf_size,
                "max_depth": args.max_depth,
                "numba_available": NUMBA_AVAILABLE,
                "numba_used": NUMBA_AVAILABLE and not args.no_numba,
            })

    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[benchmark] Plotting skipped: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for theta, group in df.groupby("theta"):
        ax.plot(group["particles"], group["barnes_hut_time_s"], marker="o", label=f"BH θ={theta}")
    direct = df.dropna(subset=["direct_time_s"]).drop_duplicates("particles")
    if len(direct):
        ax.plot(direct["particles"], direct["direct_time_s"], marker="o", linestyle="--", label="direct")
    ax.set_xlabel("Particles")
    ax.set_ylabel("Best force evaluation time (s)")
    ax.set_title("Direct vs optimized Barnes-Hut force time")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "barnes_hut_time.png", dpi=160)
    plt.close(fig)

    if "mean_relative_acc_error" in df.columns and df["mean_relative_acc_error"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        for theta, group in df.dropna(subset=["mean_relative_acc_error"]).groupby("theta"):
            ax.plot(group["particles"], group["mean_relative_acc_error"], marker="o", label=f"θ={theta}")
        ax.set_xlabel("Particles")
        ax.set_ylabel("Mean relative acceleration error")
        ax.set_title("Barnes-Hut approximation error vs direct")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "barnes_hut_error.png", dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark direct vs optimized Barnes-Hut force solvers.")
    parser.add_argument("--particles", nargs="+", type=int, default=[100, 300, 500, 1000])
    parser.add_argument("--theta", nargs="+", type=float, default=[0.3, 0.5, 0.8, 1.0])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-direct-particles", type=int, default=1500)
    parser.add_argument("--leaf-size", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--G", type=float, default=config.G)
    parser.add_argument("--softening", type=float, default=config.SOFTENING)
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--output-dir", default=str(config.OUTPUT_ROOT / "benchmarks"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = run_benchmark(args)
    csv_path = output_dir / "barnes_hut_benchmark.csv"
    df.to_csv(csv_path, index=False)
    save_plots(df, output_dir)
    print(f"[benchmark] Saved: {csv_path}")
    print(f"[benchmark] Numba available: {NUMBA_AVAILABLE}; used: {NUMBA_AVAILABLE and not args.no_numba}")


if __name__ == "__main__":
    main()
