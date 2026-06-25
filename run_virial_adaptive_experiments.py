"""Run focused virial/adaptive-softening follow-up experiments.

This runner is built on top of the anti-collapse pipeline and keeps the best
anti-collapse candidate fixed:
  - log mass
  - position scale 10
  - softening 20
  - angular velocity scale 0.02
  - H0 0.027
  - Barnes-Hut theta 0.5

It then tests the two future-work ideas that were added to the code:
  1) virialized initial velocity scaling
  2) adaptive gravitational softening

Examples
--------
python run_virial_adaptive_experiments.py --run-id virial_adaptive_no_visual_01 --max-particles 3426 --steps 1000 --save-every 2
python run_virial_adaptive_experiments.py --only osc_va_virial_adaptive_k1 --run-id virial_adaptive_html_01 --steps 3000 --with-html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import config
from main import run_single_experiment
from run_anti_collapse_experiments import experiment_overrides as anti_experiment_overrides
from run_anti_collapse_experiments import output_rows as anti_output_rows
from src.analysis import analyze_run_outputs
from src.run_context import base_run_manifest, make_run_id, now_iso, rows_to_csv, write_json, write_run_readme


BASE_FINAL: dict[str, Any] = {
    "position_scale": 10.0,
    "mass_model": "log",
    "softening": 20.0,
    "velocity_scale": 0.02,
    "H0": 0.027,
    "dt": 0.02,
}


VIRIAL_ADAPTIVE_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "osc_va_base_final",
        **BASE_FINAL,
        "virial": False,
        "adaptive": False,
    },
    {
        "name": "osc_va_virial_q1",
        **BASE_FINAL,
        "virial": True,
        "target_virial_ratio": 1.0,
        "adaptive": False,
    },
    {
        "name": "osc_va_adaptive_k1",
        **BASE_FINAL,
        "virial": False,
        "adaptive": True,
        "adaptive_mode": "density_boost",
        "adaptive_k": 1.0,
    },
    {
        "name": "osc_va_adaptive_k2",
        **BASE_FINAL,
        "virial": False,
        "adaptive": True,
        "adaptive_mode": "density_boost",
        "adaptive_k": 2.0,
    },
    {
        "name": "osc_va_virial_adaptive_k1",
        **BASE_FINAL,
        "virial": True,
        "target_virial_ratio": 1.0,
        "adaptive": True,
        "adaptive_mode": "density_boost",
        "adaptive_k": 1.0,
    },
    {
        "name": "osc_va_virial_adaptive_k2",
        **BASE_FINAL,
        "virial": True,
        "target_virial_ratio": 1.0,
        "adaptive": True,
        "adaptive_mode": "density_boost",
        "adaptive_k": 2.0,
    },
]


def make_run_overrides(run_id: str | None, notes: str) -> dict[str, Any]:
    run_id = run_id or make_run_id("virial_adaptive")
    run_root = config.RUNS_DIR / run_id
    return {
        "RUN_ID": run_id,
        "RUN_STARTED_AT": now_iso(),
        "RUN_OUTPUT_ROOT": run_root,
        "RUN_NOTES": notes or "",
        "USE_TIMESTAMPED_RUN_DIRS": True,
    }


def experiment_overrides(exp: dict[str, Any]) -> dict[str, Any]:
    overrides = anti_experiment_overrides(exp)
    overrides["SIMULATION_MODE"] = str(exp["name"])
    overrides["EXPERIMENT_NAME"] = str(exp["name"])
    overrides["USE_VIRIAL_VELOCITY_SCALING"] = bool(exp.get("virial", False))
    overrides["TARGET_VIRIAL_RATIO"] = float(exp.get("target_virial_ratio", 1.0))
    overrides["USE_ADAPTIVE_SOFTENING"] = bool(exp.get("adaptive", False))
    overrides["ADAPTIVE_SOFTENING_MODE"] = str(exp.get("adaptive_mode", "density_boost"))
    overrides["ADAPTIVE_SOFTENING_K"] = float(exp.get("adaptive_k", 1.0))
    overrides["ADAPTIVE_SOFTENING_MIN"] = float(exp.get("adaptive_min", 0.5))
    overrides["ADAPTIVE_SOFTENING_MAX"] = float(exp.get("adaptive_max", 30.0))
    overrides["ADAPTIVE_SOFTENING_UPDATE_EVERY"] = int(exp.get("adaptive_update_every", 10))
    overrides["VIRIAL_VELOCITY_SCALE_MIN"] = float(exp.get("virial_scale_min", 0.1))
    overrides["VIRIAL_VELOCITY_SCALE_MAX"] = float(exp.get("virial_scale_max", 10.0))
    overrides["ANALYZE_OSCILLATION"] = True
    return overrides


def output_rows(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = anti_output_rows(outputs)
    for row, out in zip(rows, outputs):
        cfg = out["config"]
        row["benchmark_family"] = "virial_adaptive"
        row["use_virial_velocity_scaling"] = bool(cfg.get("USE_VIRIAL_VELOCITY_SCALING", False))
        row["target_virial_ratio"] = float(cfg.get("TARGET_VIRIAL_RATIO", 1.0))
        row["use_adaptive_softening"] = bool(cfg.get("USE_ADAPTIVE_SOFTENING", False))
        row["adaptive_softening_mode"] = cfg.get("ADAPTIVE_SOFTENING_MODE", "density_boost")
        row["adaptive_softening_k"] = float(cfg.get("ADAPTIVE_SOFTENING_K", 1.0))
        row["adaptive_softening_min"] = float(cfg.get("ADAPTIVE_SOFTENING_MIN", 0.5))
        row["adaptive_softening_max"] = float(cfg.get("ADAPTIVE_SOFTENING_MAX", 30.0))
        row["adaptive_softening_update_every"] = int(cfg.get("ADAPTIVE_SOFTENING_UPDATE_EVERY", 10))
    return rows


def save_run_outputs(outputs: list[dict[str, Any]], run_root: Path, run_manifest: dict[str, Any]) -> None:
    rows = output_rows(outputs)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_index.json", rows)
    rows_to_csv(run_root / "run_index.csv", rows)

    analysis_result = None
    try:
        analysis_result = analyze_run_outputs(rows, run_root)
        print(f"[va] Analysis summary: {analysis_result.get('oscillation_summary_csv')}")
    except Exception as exc:
        print(f"[va] Analysis skipped: {exc}")

    manifest = dict(run_manifest)
    manifest["finished_at"] = now_iso()
    manifest["experiment_count"] = len(rows)
    manifest["experiments"] = [row.get("experiment") for row in rows]
    manifest["analysis_goal"] = "virialized velocity scaling and adaptive softening follow-up"
    if analysis_result is not None:
        manifest["analysis"] = analysis_result
    write_json(run_root / "run_manifest.json", manifest)
    write_run_readme(run_root / "RUN_README.md", manifest, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run virial/adaptive softening follow-up experiments.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-particles", type=int, default=3426)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--notes", default="Virial/adaptive-softening follow-up benchmark, no GIF or HTML")
    parser.add_argument("--only", nargs="+", default=None, help="Optional list of experiment names to run.")
    parser.add_argument("--list", action="store_true", help="List available experiments and exit.")
    parser.add_argument("--with-gif", action="store_true", help="Enable GIF rendering.")
    parser.add_argument("--with-html", action="store_true", help="Enable interactive HTML rendering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = list(VIRIAL_ADAPTIVE_EXPERIMENTS)

    if args.list:
        print("Available virial/adaptive experiments:")
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
        print(f"[va] Selected experiments: {', '.join(args.only)}")

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
        preset="virial-adaptive",
        raw_csv=config.RAW_CSV_PATH,
        command=sys.argv,
        common_overrides=common,
        notes=args.notes,
    )
    write_json(run_root / "run_manifest.json", run_manifest)

    print("[va] Starting virial/adaptive benchmark")
    print(f"[va] Experiments: {len(experiments)}")
    print(f"[va] Output: {run_root}")
    print(f"[va] Particles={args.max_particles}, steps={args.steps}, save_every={args.save_every}")
    print(f"[va] GIF={common['SAVE_GIF']}, HTML={common['SAVE_INTERACTIVE_HTML']}, solver=barnes_hut, theta=0.5")

    outputs: list[dict[str, Any]] = []
    for exp in experiments:
        print("\n" + "=" * 80)
        print(f"[va] Running {exp['name']}")
        print(
            "[va] "
            f"virial={exp.get('virial', False)} | "
            f"adaptive={exp.get('adaptive', False)} | "
            f"adaptive_k={exp.get('adaptive_k', '-')} | "
            f"pos_scale={exp['position_scale']} | mass={exp['mass_model']} | "
            f"softening={exp['softening']} | H0={exp['H0']}"
        )
        overrides = experiment_overrides(exp)
        overrides.update(common)
        outputs.append(run_single_experiment(overrides))

    save_run_outputs(outputs, run_root, run_manifest)
    print("\n[va] Completed.")
    print(f"[va] Run folder: {run_root}")
    print(f"[va] Run index: {run_root / 'run_index.csv'}")
    print(f"[va] Summary: {run_root / 'oscillation_summary.csv'}")
    print(f"[va] Plots: {run_root / 'plots'}")


if __name__ == "__main__":
    main()
