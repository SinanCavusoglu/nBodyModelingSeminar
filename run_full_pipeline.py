"""One-command full pipeline for the Forbes N-body simulation upgrade.

This script is the easiest way to run the project from code:

1. Read the raw Forbes CSV.
2. Generate radial and angular vx/vy/vz CSV files.
3. Run the selected experiment set.
4. Save positions, metrics, summaries, edges, and optionally GIFs.

Output policy:
  Every execution creates a timestamped folder under:
      output/experiments/runs/<RUN_ID>/

  This prevents each new run from overwriting previous outputs.

Examples:
  python run_full_pipeline.py --preset quick
  python run_full_pipeline.py --preset full
  python run_full_pipeline.py --preset full --include-sweep --include-theta --no-gif
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import config
from scripts.generate_connection_velocity import generate
from run_experiments import (
    BARNES_HUT_EXPERIMENTS,
    SEBASTIAN_EXPERIMENTS,
    run_named_experiments,
    run_parameter_sweep,
    run_theta_sweep,
    save_run_index,
)
from src.animation import save_comparison_gif
from src.run_context import base_run_manifest, make_run_id, now_iso, write_json


def ensure_raw_csv(raw_csv: Path) -> Path:
    """Ensure the raw CSV exists in config.DATA_DIR and return its project path."""
    raw_csv = Path(raw_csv)
    target = config.RAW_CSV_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    if raw_csv.exists() and raw_csv.resolve() != target.resolve():
        shutil.copy2(raw_csv, target)
        print(f"[pipeline] Copied raw CSV to: {target}")
    elif target.exists():
        print(f"[pipeline] Using raw CSV: {target}")
    elif raw_csv.exists():
        print(f"[pipeline] Using raw CSV: {raw_csv}")
        return raw_csv
    else:
        raise FileNotFoundError(
            "Raw CSV not found. Put it at "
            f"{target} or pass --raw-csv /path/to/forbes_billionaires_WorthPosConnectionsColor_20260610.csv"
        )
    return target


def generate_velocity_inputs(raw_csv: Path) -> None:
    """Create generated radial/angular simulation input CSVs."""
    print("[pipeline] Generating radial velocity input...")
    generate(
        input_path=raw_csv,
        output_dir=config.GENERATED_DATA_DIR / "radial",
        mode="radial",
        scale=float(config.CONNECTION_VELOCITY_SCALE),
        use_mass_strength=bool(config.USE_CONNECTION_MASS_STRENGTH),
    )

    print("[pipeline] Generating angular velocity input...")
    generate(
        input_path=raw_csv,
        output_dir=config.GENERATED_DATA_DIR / "angular",
        mode="angular",
        scale=float(config.CONNECTION_VELOCITY_SCALE),
        use_mass_strength=bool(config.USE_CONNECTION_MASS_STRENGTH),
    )


def common_overrides_from_args(args: argparse.Namespace, run_id: str, run_root: Path) -> dict[str, Any]:
    common: dict[str, Any] = {
        "RUN_ID": run_id,
        "RUN_STARTED_AT": now_iso(),
        "RUN_OUTPUT_ROOT": run_root,
        "RUN_NOTES": args.notes or "",
        "USE_TIMESTAMPED_RUN_DIRS": True,
    }
    if args.max_particles is not None:
        common["MAX_PARTICLES"] = args.max_particles
    if args.steps is not None:
        common["STEPS"] = args.steps
    if args.save_every is not None:
        common["SAVE_EVERY"] = args.save_every
    if args.no_gif:
        common["SAVE_GIF"] = False
    if args.H0 is not None:
        common["H0"] = args.H0
    if args.softening is not None:
        common["SOFTENING"] = args.softening
    if args.theta is not None:
        common["BARNES_HUT_THETA"] = args.theta
    return common


def apply_preset_defaults(args: argparse.Namespace) -> None:
    """Set safe defaults for quick/full presets unless explicitly provided."""
    if args.preset == "quick":
        if args.max_particles is None:
            args.max_particles = 50
        if args.steps is None:
            args.steps = 200
        if args.save_every is None:
            args.save_every = 5
        # Quick runs should not spend time rendering GIFs unless requested.
        if not args.with_gif:
            args.no_gif = True
    elif args.preset == "full":
        if args.max_particles is None:
            args.max_particles = config.MAX_PARTICLES
        if args.steps is None:
            args.steps = config.STEPS
        if args.save_every is None:
            args.save_every = config.SAVE_EVERY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full N-body roadmap pipeline from code.")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--raw-csv", type=Path, default=config.RAW_CSV_PATH)
    parser.add_argument("--max-particles", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--H0", type=float, default=None)
    parser.add_argument("--softening", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--with-gif", action="store_true", help="Enable GIF rendering even for quick preset.")
    parser.add_argument("--include-barnes-hut", action="store_true", help="Run Barnes-Hut experiments too.")
    parser.add_argument("--include-sweep", action="store_true", help="Run epsilon/H0 parameter sweep.")
    parser.add_argument("--include-theta", action="store_true", help="Run Barnes-Hut theta sweep.")
    parser.add_argument("--comparison-gif", action="store_true", help="Create a four-panel comparison GIF.")
    parser.add_argument("--run-id", default=None, help="Optional custom run id. Defaults to timestamped id.")
    parser.add_argument("--notes", default="", help="Optional note stored in run_manifest.json and RUN_README.md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_preset_defaults(args)

    run_id = args.run_id or make_run_id(args.preset)
    run_root = config.RUNS_DIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    raw_csv = ensure_raw_csv(args.raw_csv)
    generate_velocity_inputs(raw_csv)

    common = common_overrides_from_args(args, run_id, run_root)
    run_manifest = base_run_manifest(
        run_id=run_id,
        run_output_root=run_root,
        project_root=config.PROJECT_ROOT,
        preset=args.preset,
        raw_csv=raw_csv,
        command=sys.argv,
        common_overrides=common,
        notes=args.notes,
    )
    run_manifest.update(
        {
            "include_barnes_hut": bool(args.include_barnes_hut or args.preset == "full"),
            "include_sweep": bool(args.include_sweep),
            "include_theta": bool(args.include_theta),
            "comparison_gif": bool(args.comparison_gif),
            "generated_radial_csv": str(config.RADIAL_CSV_PATH),
            "generated_angular_csv": str(config.ANGULAR_CSV_PATH),
        }
    )
    write_json(run_root / "run_manifest.json", run_manifest)

    outputs: list[dict[str, Any]] = []

    print("[pipeline] Running Sebastian comparison experiments...")
    outputs.extend(run_named_experiments(SEBASTIAN_EXPERIMENTS, common))

    if args.include_barnes_hut or args.preset == "full":
        print("[pipeline] Running Barnes-Hut experiments...")
        outputs.extend(run_named_experiments(BARNES_HUT_EXPERIMENTS, common))

    if args.include_sweep:
        print("[pipeline] Running epsilon/H0 sweep...")
        outputs.extend(run_parameter_sweep(common))

    if args.include_theta:
        print("[pipeline] Running Barnes-Hut theta sweep...")
        outputs.extend(run_theta_sweep(common))

    save_run_index(outputs, run_root / "run_index.json", run_manifest)

    if args.comparison_gif and outputs:
        first_four = outputs[:4]
        masses = first_four[0]["masses"]
        colors = first_four[0].get("colors")
        histories = {out["config"]["EXPERIMENT_NAME"]: out["result"]["positions"] for out in first_four}
        try:
            save_comparison_gif(histories, masses, run_root / "comparison.gif", colors=colors)
        except Exception as exc:
            print(f"[pipeline] Comparison GIF skipped: {exc}")

    print(f"[pipeline] DONE. Experiments completed: {len(outputs)}")
    print(f"[pipeline] Run folder: {run_root}")
    print(f"[pipeline] Run manifest: {run_root / 'run_manifest.json'}")
    print(f"[pipeline] Index JSON: {run_root / 'run_index.json'}")
    print(f"[pipeline] Index CSV: {run_root / 'run_index.csv'}")
    print(f"[pipeline] Human-readable summary: {run_root / 'RUN_README.md'}")


if __name__ == "__main__":
    main()
