"""Run anti-collapse / ring-motion experiment candidates.

Purpose
-------
This script runs a second benchmark after the oscillation benchmark.
The goal is not only to reduce oscillation, but also to reduce inward collapse
and preserve a smoother, wider, more orbital/ring-like motion.

It intentionally keeps the best working setup fixed:
  - solver: Barnes-Hut
  - theta: 0.5
  - GIF: disabled by default
  - HTML: disabled by default
  - default particles: 3426
  - default steps: 1000
  - default save_every: 2

It varies:
  - initial position scale
  - softening
  - angular velocity scale
  - mass model: current/raw, sqrt, log, logclip
  - H0 expansion control

Example
-------
python run_anti_collapse_experiments.py --run-id anti_collapse_full_bh_no_visual_01 --max-particles 3426 --steps 1000 --save-every 2
python run_anti_collapse_experiments.py --only osc_anti_log_pos10_soft20_ang0p02_H0_0p027 --run-id best_anti_collapse_html_01 --save-every 5 --with-html
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from main import run_single_experiment
from scripts.generate_connection_velocity import default_output_dir, generate
from src.analysis import analyze_run_outputs
from src.run_context import (
    base_run_manifest,
    make_run_id,
    now_iso,
    rows_to_csv,
    write_json,
    write_run_readme,
)


# -----------------------------------------------------------------------------
# Baseline from the best detailed oscillation benchmark:
# osc_soft_eps10p0
# -----------------------------------------------------------------------------
BASE_VELOCITY_MODE = "angular"
BASE_H0 = 0.01
BASE_DT = 0.02
BASE_THETA = 0.5
BASE_SOLVER = "barnes_hut"
BASE_SOFTENING = 10.0
BASE_ANGULAR_SCALE = 0.01

POSITION_COLUMNS = ["x", "y", "z"]
VELOCITY_COLUMNS = ["vx", "vy", "vz"]
MASS_COLUMN_CANDIDATES = [
    "NetWorth_Billions",
    "networth_billions",
    "mass",
    "Mass",
    "NetWorth",
    "net_worth",
]


ANTI_COLLAPSE_EXPERIMENTS: list[dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # A) Position scale sweep.
    # Everything else stays fixed at the current best configuration.
    # -------------------------------------------------------------------------
    {
        "name": "osc_anti_pos1_soft10",
        "position_scale": 1.0,
        "mass_model": "current",
        "softening": 10.0,
        "velocity_scale": 0.01,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_pos5_soft10",
        "position_scale": 5.0,
        "mass_model": "current",
        "softening": 10.0,
        "velocity_scale": 0.01,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_pos10_soft10",
        "position_scale": 10.0,
        "mass_model": "current",
        "softening": 10.0,
        "velocity_scale": 0.01,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_pos20_soft10",
        "position_scale": 20.0,
        "mass_model": "current",
        "softening": 10.0,
        "velocity_scale": 0.01,
        "H0": 0.01,
        "dt": 0.02,
    },

    # -------------------------------------------------------------------------
    # B) Softening + angular velocity sweep.
    # Position scale is kept at 10 as the first practical wide-space candidate.
    # -------------------------------------------------------------------------
    {
        "name": "osc_anti_pos10_soft15_ang0p01",
        "position_scale": 10.0,
        "mass_model": "current",
        "softening": 15.0,
        "velocity_scale": 0.01,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_pos10_soft20_ang0p01",
        "position_scale": 10.0,
        "mass_model": "current",
        "softening": 20.0,
        "velocity_scale": 0.01,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_pos10_soft30_ang0p01",
        "position_scale": 10.0,
        "mass_model": "current",
        "softening": 30.0,
        "velocity_scale": 0.01,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_pos10_soft10_ang0p02",
        "position_scale": 10.0,
        "mass_model": "current",
        "softening": 10.0,
        "velocity_scale": 0.02,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_pos10_soft20_ang0p02",
        "position_scale": 10.0,
        "mass_model": "current",
        "softening": 20.0,
        "velocity_scale": 0.02,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_pos10_soft30_ang0p02",
        "position_scale": 10.0,
        "mass_model": "current",
        "softening": 30.0,
        "velocity_scale": 0.02,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_pos10_soft20_ang0p03",
        "position_scale": 10.0,
        "mass_model": "current",
        "softening": 20.0,
        "velocity_scale": 0.03,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_pos20_soft20_ang0p02",
        "position_scale": 20.0,
        "mass_model": "current",
        "softening": 20.0,
        "velocity_scale": 0.02,
        "H0": 0.01,
        "dt": 0.02,
    },

    # -------------------------------------------------------------------------
    # C) Mass normalization sweep.
    # These are the most important anti-collapse candidates because raw wealth
    # values may create excessive attraction toward the most massive bodies.
    # -------------------------------------------------------------------------
    {
        "name": "osc_anti_mass_raw_pos10_soft20_ang0p02",
        "position_scale": 10.0,
        "mass_model": "current",
        "softening": 20.0,
        "velocity_scale": 0.02,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_mass_sqrt_pos10_soft20_ang0p02",
        "position_scale": 10.0,
        "mass_model": "sqrt",
        "softening": 20.0,
        "velocity_scale": 0.02,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_mass_log_pos10_soft20_ang0p02",
        "position_scale": 10.0,
        "mass_model": "log",
        "softening": 20.0,
        "velocity_scale": 0.02,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_mass_logclip_pos10_soft20_ang0p02",
        "position_scale": 10.0,
        "mass_model": "logclip",
        "softening": 20.0,
        "velocity_scale": 0.02,
        "H0": 0.01,
        "dt": 0.02,
    },

    # -------------------------------------------------------------------------
    # D) Expansion control.
    # H0=0.05 was bad in the earlier benchmark, so we only test no expansion,
    # the current low expansion, and Sebastian's 2.7% growth value.
    # -------------------------------------------------------------------------
    {
        "name": "osc_anti_log_pos10_soft20_ang0p02_H0_0p0",
        "position_scale": 10.0,
        "mass_model": "log",
        "softening": 20.0,
        "velocity_scale": 0.02,
        "H0": 0.0,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_log_pos10_soft20_ang0p02_H0_0p01",
        "position_scale": 10.0,
        "mass_model": "log",
        "softening": 20.0,
        "velocity_scale": 0.02,
        "H0": 0.01,
        "dt": 0.02,
    },
    {
        "name": "osc_anti_log_pos10_soft20_ang0p02_H0_0p027",
        "position_scale": 10.0,
        "mass_model": "log",
        "softening": 20.0,
        "velocity_scale": 0.02,
        "H0": 0.027,
        "dt": 0.02,
    },
]


def make_run_overrides(run_id: str | None, notes: str) -> dict[str, Any]:
    run_id = run_id or make_run_id("anti_collapse")
    run_root = config.RUNS_DIR / run_id
    return {
        "RUN_ID": run_id,
        "RUN_STARTED_AT": now_iso(),
        "RUN_OUTPUT_ROOT": run_root,
        "RUN_NOTES": notes or "",
        "USE_TIMESTAMPED_RUN_DIRS": True,
    }


def safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def find_mass_column(df: pd.DataFrame) -> str | None:
    for col in MASS_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def transform_mass_values(values: np.ndarray, model: str) -> np.ndarray:
    """Transform mass values while keeping the same approximate mean mass.

    This reduces extreme mass dominance without making total gravity vanish.
    """
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)

    if model == "current":
        return values

    original_mean = float(np.mean(values[values > 0])) if np.any(values > 0) else 1.0

    if model == "sqrt":
        transformed = np.sqrt(values)
    elif model == "log":
        transformed = np.log1p(values)
    elif model == "logclip":
        transformed = np.log1p(values)
    else:
        raise ValueError(f"Unknown mass model: {model}")

    positive_mean = float(np.mean(transformed[transformed > 0])) if np.any(transformed > 0) else 1.0
    transformed = transformed / positive_mean * original_mean

    if model == "logclip":
        # Clip after normalizing to keep extreme masses from dominating.
        # These values are intentionally conservative for visual stability.
        transformed = np.clip(transformed, 0.2, 10.0)
        clipped_mean = float(np.mean(transformed[transformed > 0])) if np.any(transformed > 0) else 1.0
        transformed = transformed / clipped_mean * original_mean

    return transformed


def ensure_generated_velocity_input(velocity_scale: float, raw_csv: Path | None = None) -> tuple[Path, Path]:
    """Ensure angular velocity input exists for a given scale."""
    raw_csv = raw_csv or config.RAW_CSV_PATH
    out_dir = default_output_dir(BASE_VELOCITY_MODE, float(velocity_scale), config.GENERATED_DATA_DIR)
    minimal_csv = out_dir / "minimal.csv"
    edges_csv = out_dir / "edges.csv"

    if not minimal_csv.exists() or not edges_csv.exists():
        print(f"[anti] Generating angular velocity input scale={velocity_scale}")
        generate(
            input_path=raw_csv,
            output_dir=out_dir,
            mode=BASE_VELOCITY_MODE,
            scale=float(velocity_scale),
            use_mass_strength=bool(config.USE_CONNECTION_MASS_STRENGTH),
        )

    return minimal_csv, edges_csv


def prepare_experiment_csv(exp: dict[str, Any]) -> tuple[Path, Path]:
    """Create a transformed CSV for one anti-collapse experiment."""
    velocity_scale = float(exp["velocity_scale"])
    source_csv, source_edges = ensure_generated_velocity_input(velocity_scale)

    experiment_dir = config.GENERATED_DATA_DIR / "anti_collapse" / str(exp["name"])
    experiment_dir.mkdir(parents=True, exist_ok=True)

    out_csv = experiment_dir / "minimal.csv"
    out_edges = experiment_dir / "edges.csv"

    df = pd.read_csv(source_csv)

    missing_pos = [col for col in POSITION_COLUMNS if col not in df.columns]
    if missing_pos:
        raise ValueError(f"Missing position columns in {source_csv}: {missing_pos}")

    # Store originals for easier debugging/reproducibility.
    for col in POSITION_COLUMNS:
        original_col = f"original_{col}"
        if original_col not in df.columns:
            df[original_col] = df[col]

    # Scale positions around their centroid, not around absolute origin.
    pos = df[POSITION_COLUMNS].apply(safe_float_series).to_numpy(dtype=float)
    center = np.nanmean(pos, axis=0)
    position_scale = float(exp["position_scale"])
    scaled_pos = center + (pos - center) * position_scale
    df[POSITION_COLUMNS] = scaled_pos

    # Transform mass if a known mass column is available.
    mass_col = find_mass_column(df)
    mass_model = str(exp["mass_model"])
    if mass_col is not None and mass_model != "current":
        original_mass_col = f"original_{mass_col}"
        if original_mass_col not in df.columns:
            df[original_mass_col] = df[mass_col]
        mass_values = safe_float_series(df[mass_col]).to_numpy(dtype=float)
        df[mass_col] = transform_mass_values(mass_values, mass_model)
    elif mass_col is None:
        print(f"[anti] Warning: no mass column found in {source_csv}; mass_model={mass_model} skipped")

    df["anti_position_scale"] = position_scale
    df["anti_mass_model"] = mass_model
    df["anti_softening"] = float(exp["softening"])
    df["anti_angular_velocity_scale"] = velocity_scale

    df.to_csv(out_csv, index=False)
    shutil.copyfile(source_edges, out_edges)

    return out_csv, out_edges


def experiment_overrides(exp: dict[str, Any]) -> dict[str, Any]:
    csv_path, edge_path = prepare_experiment_csv(exp)
    H0 = float(exp["H0"])
    use_expansion = H0 > 0.0

    return {
        "SIMULATION_MODE": str(exp["name"]),
        "EXPERIMENT_NAME": str(exp["name"]),
        "FORCE_SOLVER": BASE_SOLVER,
        "BARNES_HUT_IMPLEMENTATION": "fast",
        "BARNES_HUT_USE_NUMBA": True,
        "BARNES_HUT_THETA": BASE_THETA,
        "USE_EXPANSION": use_expansion,
        "EXPANSION_MODEL": "linear" if use_expansion else "none",
        "H0": H0,
        "SOFTENING": float(exp["softening"]),
        "DT": float(exp["dt"]),
        "CONNECTION_VELOCITY_MODE": BASE_VELOCITY_MODE,
        "CONNECTION_VELOCITY_SCALE": float(exp["velocity_scale"]),
        "ANTI_POSITION_SCALE": float(exp["position_scale"]),
        "ANTI_MASS_MODEL": str(exp["mass_model"]),
        "CSV_PATH": csv_path,
        "EDGE_CSV_PATH": edge_path,
        # This experiment should be analysis-only; no rendering here.
        "SAVE_GIF": False,
        "SAVE_INTERACTIVE_HTML": False,
        "ANALYZE_OSCILLATION": True,
    }


def output_rows(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for out in outputs:
        cfg = out["config"]
        rows.append(
            {
                "run_id": cfg.get("RUN_ID"),
                "run_started_at": cfg.get("RUN_STARTED_AT"),
                "experiment_started_at": cfg.get("experiment_started_at"),
                "experiment_finished_at": cfg.get("experiment_finished_at"),
                "experiment": cfg.get("EXPERIMENT_NAME"),
                "simulation_mode": cfg.get("SIMULATION_MODE"),
                "force_solver": cfg.get("FORCE_SOLVER"),
                "particles": int(cfg.get("MAX_PARTICLES")),
                "steps": int(cfg.get("STEPS")),
                "dt": float(cfg.get("DT")),
                "save_every": int(cfg.get("SAVE_EVERY")),
                "runtime_seconds": out.get("runtime_seconds"),
                "softening": float(cfg.get("SOFTENING")),
                "H0": float(cfg.get("H0")),
                "theta": float(cfg.get("BARNES_HUT_THETA")),
                "connection_velocity_mode": cfg.get("CONNECTION_VELOCITY_MODE"),
                "connection_velocity_scale": float(cfg.get("CONNECTION_VELOCITY_SCALE", 0.0)),
                "anti_position_scale": float(cfg.get("ANTI_POSITION_SCALE", 1.0)),
                "anti_mass_model": cfg.get("ANTI_MASS_MODEL", "current"),
                "use_virial_velocity_scaling": bool(cfg.get("USE_VIRIAL_VELOCITY_SCALING", False)),
                "target_virial_ratio": float(cfg.get("TARGET_VIRIAL_RATIO", 1.0)),
                "use_adaptive_softening": bool(cfg.get("USE_ADAPTIVE_SOFTENING", False)),
                "adaptive_softening_mode": cfg.get("ADAPTIVE_SOFTENING_MODE", "density_boost"),
                "adaptive_softening_k": float(cfg.get("ADAPTIVE_SOFTENING_K", 1.0)),
                "adaptive_softening_min": float(cfg.get("ADAPTIVE_SOFTENING_MIN", 0.5)),
                "adaptive_softening_max": float(cfg.get("ADAPTIVE_SOFTENING_MAX", 30.0)),
                "adaptive_softening_update_every": int(cfg.get("ADAPTIVE_SOFTENING_UPDATE_EVERY", 10)),
                "csv_path": str(cfg.get("CSV_PATH")),
                "edge_csv_path": str(cfg.get("EDGE_CSV_PATH")),
                "output_dir": str(out["output_dir"]),
                "save_gif": bool(cfg.get("SAVE_GIF")),
                "save_interactive_html": bool(cfg.get("SAVE_INTERACTIVE_HTML")),
            }
        )
    return rows


def save_run_outputs(outputs: list[dict[str, Any]], run_root: Path, run_manifest: dict[str, Any]) -> None:
    rows = output_rows(outputs)
    run_root.mkdir(parents=True, exist_ok=True)

    run_index_json = run_root / "run_index.json"
    write_json(run_index_json, rows)
    rows_to_csv(run_root / "run_index.csv", rows)

    analysis_result = None
    try:
        analysis_result = analyze_run_outputs(rows, run_root)
        print(f"[anti] Analysis summary: {analysis_result.get('oscillation_summary_csv')}")
    except Exception as exc:
        print(f"[anti] Analysis skipped: {exc}")

    # Create anti-collapse aliases so the folder reads naturally.
    osc_summary = run_root / "oscillation_summary.csv"
    osc_json = run_root / "oscillation_summary.json"
    osc_md = run_root / "OSCILLATION_ANALYSIS.md"
    if osc_summary.exists():
        shutil.copyfile(osc_summary, run_root / "anti_collapse_summary.csv")
    if osc_json.exists():
        shutil.copyfile(osc_json, run_root / "anti_collapse_summary.json")
    if osc_md.exists():
        shutil.copyfile(osc_md, run_root / "ANTI_COLLAPSE_ANALYSIS.md")

    manifest = dict(run_manifest)
    manifest["finished_at"] = now_iso()
    manifest["experiment_count"] = len(rows)
    manifest["experiments"] = [row.get("experiment") for row in rows]
    manifest["analysis_goal"] = "anti-collapse and ring-like motion benchmark"
    if analysis_result is not None:
        manifest["analysis"] = analysis_result

    write_json(run_root / "run_manifest.json", manifest)
    write_run_readme(run_root / "RUN_README.md", manifest, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anti-collapse N-body experiments.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-particles", type=int, default=3426)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--notes", default="Full participant anti-collapse benchmark with optimized Barnes-Hut, no GIF or HTML")
    parser.add_argument(
        "--with-gif",
        action="store_true",
        help="Enable GIF rendering for selected experiment(s). Default is off.",
    )
    parser.add_argument(
        "--with-html",
        action="store_true",
        help="Enable interactive HTML rendering for selected experiment(s). Default is off.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional list of experiment names to run.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiments = list(ANTI_COLLAPSE_EXPERIMENTS)
    if args.list:
        print("Available anti-collapse experiments:")
        for exp in experiments:
            print(f"  {exp['name']}")
        return

    if args.only:
        by_name = {str(exp["name"]): exp for exp in experiments}
        missing = [name for name in args.only if name not in by_name]
        if missing:
            available = ", ".join(sorted(by_name.keys()))
            raise ValueError(f"Unknown experiment(s): {missing}\nAvailable: {available}")
        experiments = [by_name[name] for name in args.only]
        print(f"[anti] Selected experiments: {', '.join(args.only)}")

    common = make_run_overrides(args.run_id, args.notes)
    common["MAX_PARTICLES"] = int(args.max_particles)
    common["STEPS"] = int(args.steps)
    common["SAVE_EVERY"] = int(args.save_every)
    common["SAVE_GIF"] = bool(args.with_gif)
    common["SAVE_INTERACTIVE_HTML"] = bool(args.with_html)

    run_root = Path(common["RUN_OUTPUT_ROOT"])
    run_manifest = base_run_manifest(
        run_id=str(common["RUN_ID"]),
        run_output_root=run_root,
        project_root=config.PROJECT_ROOT,
        preset="anti-collapse",
        raw_csv=config.RAW_CSV_PATH,
        command=sys.argv,
        common_overrides=common,
        notes=args.notes,
    )
    write_json(run_root / "run_manifest.json", run_manifest)

    print("[anti] Starting anti-collapse benchmark")
    print(f"[anti] Experiments: {len(experiments)}")
    print(f"[anti] Output: {run_root}")
    print(f"[anti] Particles={args.max_particles}, steps={args.steps}, save_every={args.save_every}")
    print(
        f"[anti] GIF={common['SAVE_GIF']}, "
        f"HTML={common['SAVE_INTERACTIVE_HTML']}, "
        "solver=barnes_hut, theta=0.5"
    )

    outputs: list[dict[str, Any]] = []
    for exp in experiments:
        print("\n" + "=" * 80)
        print(f"[anti] Running {exp['name']}")
        print(
            "[anti] "
            f"pos_scale={exp['position_scale']} | "
            f"mass={exp['mass_model']} | "
            f"softening={exp['softening']} | "
            f"angular_scale={exp['velocity_scale']} | "
            f"H0={exp['H0']} | dt={exp['dt']}"
        )
        overrides = experiment_overrides(exp)
        overrides.update(common)
        outputs.append(run_single_experiment(overrides))

    save_run_outputs(outputs, run_root, run_manifest)

    print("\n[anti] Completed.")
    print(f"[anti] Run folder: {run_root}")
    print(f"[anti] Run index: {run_root / 'run_index.csv'}")
    print(f"[anti] Summary: {run_root / 'anti_collapse_summary.csv'}")
    print(f"[anti] Analysis: {run_root / 'ANTI_COLLAPSE_ANALYSIS.md'}")
    print(f"[anti] Plots: {run_root / 'plots'}")


if __name__ == "__main__":
    main()
