"""Run the full experiment roadmap.

Examples:
  python run_experiments.py --set sebastian --no-gif
  python run_experiments.py --set barnes-hut --max-particles 500 --no-gif
  python run_experiments.py --set sweep --no-gif

Output policy:
  Every execution creates a timestamped run folder under:
      output/experiments/runs/<RUN_ID>/

  This prevents new runs from overwriting old experiment outputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import config
from main import run_single_experiment
from src.animation import save_comparison_gif
from src.run_context import (
    base_run_manifest,
    make_run_id,
    now_iso,
    rows_to_csv,
    write_json,
    write_run_readme,
)


SEBASTIAN_EXPERIMENTS = [
    "current_direct",
    "softened_no_expansion",
    "softened_expansion",
    "softened_expansion_angular",
]

BARNES_HUT_EXPERIMENTS = [
    "barnes_hut_softened",
    "barnes_hut_expansion",
    "barnes_hut_expansion_angular",
]


def make_run_overrides(run_id: str | None = None, notes: str | None = None) -> dict[str, Any]:
    """Create common overrides that put all experiments into one run folder."""
    run_id = run_id or make_run_id("experiments")
    run_root = config.RUNS_DIR / run_id
    return {
        "RUN_ID": run_id,
        "RUN_STARTED_AT": now_iso(),
        "RUN_OUTPUT_ROOT": run_root,
        "RUN_NOTES": notes or "",
        "USE_TIMESTAMPED_RUN_DIRS": True,
    }


def run_named_experiments(names: list[str], common_overrides: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = []
    for name in names:
        overrides = config.experiment_overrides(name)
        overrides.update(common_overrides)
        # Preserve the experiment-specific name unless the sweep changes it.
        result = run_single_experiment(overrides)
        outputs.append(result)
    return outputs


def run_parameter_sweep(common_overrides: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = []
    epsilons = [1.0, 2.0, 5.0]
    H0_values = [0.0, 0.005, 0.01]
    for eps in epsilons:
        for H0 in H0_values:
            base = config.experiment_overrides("softened_expansion")
            name = f"sweep_epsilon_{str(eps).replace('.', 'p')}_H0_{str(H0).replace('.', 'p')}"
            base.update(
                {
                    "EXPERIMENT_NAME": name,
                    "SIMULATION_MODE": "softened_expansion",
                    "SOFTENING": eps,
                    "H0": H0,
                    "USE_EXPANSION": H0 > 0,
                }
            )
            base.update(common_overrides)
            outputs.append(run_single_experiment(base))
    return outputs


def run_theta_sweep(common_overrides: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = []
    theta_values = [0.3, 0.5, 0.8, 1.0]
    for theta in theta_values:
        base = config.experiment_overrides("barnes_hut_expansion")
        name = f"theta_sweep_{str(theta).replace('.', 'p')}"
        base.update({"EXPERIMENT_NAME": name, "BARNES_HUT_THETA": theta})
        base.update(common_overrides)
        outputs.append(run_single_experiment(base))
    return outputs


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
                "use_expansion": bool(cfg.get("USE_EXPANSION")),
                "expansion_model": cfg.get("EXPANSION_MODEL"),
                "connection_velocity_mode": cfg.get("CONNECTION_VELOCITY_MODE"),
                "output_dir": str(out["output_dir"]),
                "runtime_seconds": out.get("runtime_seconds"),
                "particles": int(cfg.get("MAX_PARTICLES")),
                "steps": int(cfg.get("STEPS")),
                "save_every": int(cfg.get("SAVE_EVERY")),
                "softening": float(cfg.get("SOFTENING")),
                "H0": float(cfg.get("H0")),
                "theta": float(cfg.get("BARNES_HUT_THETA")),
                "csv_path": str(cfg.get("CSV_PATH")),
                "edge_csv_path": str(cfg.get("EDGE_CSV_PATH")),
                "save_gif": bool(cfg.get("SAVE_GIF")),
                "save_interactive_html": bool(cfg.get("SAVE_INTERACTIVE_HTML")),
            }
        )
    return rows


def save_run_index(outputs: list[dict[str, Any]], output_path: Path, run_manifest: dict[str, Any] | None = None) -> None:
    """Save run index as JSON, CSV, and a human-readable Markdown README."""
    rows = output_rows(outputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, rows)
    rows_to_csv(output_path.with_suffix(".csv"), rows)

    if run_manifest is not None:
        manifest = dict(run_manifest)
        manifest["finished_at"] = now_iso()
        manifest["experiment_count"] = len(rows)
        manifest["experiments"] = [row.get("experiment") for row in rows]
        write_json(output_path.parent / "run_manifest.json", manifest)
        write_run_readme(output_path.parent / "RUN_README.md", manifest, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run N-body experiment sets.")
    parser.add_argument(
        "--set",
        choices=["sebastian", "barnes-hut", "all", "sweep", "theta"],
        default="sebastian",
    )
    parser.add_argument("--max-particles", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-html", action="store_true", help="Disable interactive_3d.html export.")
    parser.add_argument("--html-max-frames", type=int, default=None)
    parser.add_argument("--html-max-particles", type=int, default=None)
    parser.add_argument("--comparison-gif", action="store_true")
    parser.add_argument("--run-id", default=None, help="Optional custom run id. Defaults to timestamped id.")
    parser.add_argument("--notes", default="", help="Optional note stored in run_manifest.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common: dict[str, Any] = make_run_overrides(args.run_id, args.notes)
    if args.max_particles is not None:
        common["MAX_PARTICLES"] = args.max_particles
    if args.steps is not None:
        common["STEPS"] = args.steps
    if args.save_every is not None:
        common["SAVE_EVERY"] = args.save_every
    if args.no_gif:
        common["SAVE_GIF"] = False
    if args.no_html:
        common["SAVE_INTERACTIVE_HTML"] = False
    if args.html_max_frames is not None:
        common["INTERACTIVE_HTML_MAX_FRAMES"] = args.html_max_frames
    if args.html_max_particles is not None:
        common["INTERACTIVE_HTML_MAX_PARTICLES"] = args.html_max_particles

    run_root = Path(common["RUN_OUTPUT_ROOT"])
    run_manifest = base_run_manifest(
        run_id=str(common["RUN_ID"]),
        run_output_root=run_root,
        project_root=config.PROJECT_ROOT,
        preset=args.set,
        raw_csv=config.RAW_CSV_PATH,
        command=sys.argv,
        common_overrides=common,
        notes=args.notes,
    )
    write_json(run_root / "run_manifest.json", run_manifest)

    outputs: list[dict[str, Any]] = []
    if args.set in {"sebastian", "all"}:
        outputs.extend(run_named_experiments(SEBASTIAN_EXPERIMENTS, common))
    if args.set in {"barnes-hut", "all"}:
        outputs.extend(run_named_experiments(BARNES_HUT_EXPERIMENTS, common))
    if args.set == "sweep":
        outputs.extend(run_parameter_sweep(common))
    if args.set == "theta":
        outputs.extend(run_theta_sweep(common))

    save_run_index(outputs, run_root / "run_index.json", run_manifest)

    if args.comparison_gif and outputs:
        # Make comparison GIF from first four outputs, if their particle counts match.
        first_four = outputs[:4]
        masses = first_four[0]["masses"]
        colors = first_four[0].get("colors")
        histories = {out["config"]["EXPERIMENT_NAME"]: out["result"]["positions"] for out in first_four}
        try:
            save_comparison_gif(histories, masses, run_root / "comparison.gif", colors=colors)
        except Exception as exc:
            print(f"[runner] Comparison GIF skipped: {exc}")

    print(f"[runner] Completed {len(outputs)} experiment(s).")
    print(f"[runner] Run folder: {run_root}")
    print(f"[runner] Index JSON: {run_root / 'run_index.json'}")
    print(f"[runner] Index CSV: {run_root / 'run_index.csv'}")
    print(f"[runner] Human-readable summary: {run_root / 'RUN_README.md'}")


if __name__ == "__main__":
    main()
