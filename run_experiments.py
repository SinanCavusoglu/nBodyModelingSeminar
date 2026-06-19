"""Run the full experiment roadmap.

Examples:
  python run_experiments.py --set sebastian --no-gif
  python run_experiments.py --set barnes-hut --max-particles 500 --no-gif
  python run_experiments.py --set sweep --no-gif
  python run_experiments.py --set oscillation --no-gif

Single selected experiment example:
  python run_experiments.py --set oscillation --only osc_soft_eps10p0 --max-particles 3426 --save-every 2

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
from scripts.generate_connection_velocity import default_output_dir, generate
from src.analysis import analyze_run_outputs
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

# Experiments specifically designed to diagnose and reduce the strong initial
# oscillation effect Sebastian mentioned. They vary initial velocity mode/scale,
# softening, economic-growth expansion rate, and timestep.
OSCILLATION_VELOCITY_SCALES = [0.05, 0.01]
OSCILLATION_GENERATED_INPUTS = [
    ("radial", 0.05),
    ("angular", 0.05),
    ("radial", 0.01),
    ("angular", 0.01),
    ("none", 0.0),
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


def _scale_label(scale: float) -> str:
    return str(float(scale)).replace(".", "p").replace("-", "m")


def generated_paths(mode: str, scale: float) -> tuple[Path, Path]:
    out_dir = default_output_dir(mode, float(scale), config.GENERATED_DATA_DIR)
    return out_dir / "minimal.csv", out_dir / "edges.csv"


def ensure_velocity_input(mode: str, scale: float, raw_csv: Path | None = None) -> tuple[Path, Path]:
    """Ensure a generated input CSV exists for a velocity mode/scale pair."""
    raw_csv = raw_csv or config.RAW_CSV_PATH
    csv_path, edge_path = generated_paths(mode, scale)

    if not csv_path.exists() or not edge_path.exists():
        print(f"[runner] Generating input for mode={mode}, scale={scale}")
        generate(
            input_path=raw_csv,
            output_dir=csv_path.parent,
            mode=mode,
            scale=float(scale),
            use_mass_strength=bool(config.USE_CONNECTION_MASS_STRENGTH),
        )

    return csv_path, edge_path


def ensure_oscillation_velocity_inputs(raw_csv: Path | None = None) -> None:
    for mode, scale in OSCILLATION_GENERATED_INPUTS:
        ensure_velocity_input(mode, scale, raw_csv=raw_csv)


def base_expansion_experiment(
    *,
    name: str,
    velocity_mode: str,
    velocity_scale: float,
    softening: float = 2.0,
    H0: float = 0.01,
    dt: float | None = None,
    solver: str = "barnes_hut",
) -> dict[str, Any]:
    """Create one oscillation experiment override.

    Important:
      solver default is barnes_hut because the full-participant oscillation
      experiments should use the optimized Barnes-Hut solver.
    """
    csv_path, edge_path = generated_paths(velocity_mode, velocity_scale)
    use_expansion = float(H0) > 0.0

    overrides: dict[str, Any] = {
        "SIMULATION_MODE": name,
        "EXPERIMENT_NAME": name,
        "FORCE_SOLVER": solver,
        "USE_EXPANSION": use_expansion,
        "EXPANSION_MODEL": "linear" if use_expansion else "none",
        "H0": float(H0),
        "SOFTENING": float(softening),
        "CONNECTION_VELOCITY_MODE": velocity_mode,
        "CONNECTION_VELOCITY_SCALE": float(velocity_scale),
        "CSV_PATH": csv_path,
        "EDGE_CSV_PATH": edge_path,
    }

    if dt is not None:
        overrides["DT"] = float(dt)

    return overrides


def oscillation_experiment_overrides() -> list[dict[str, Any]]:
    """Return the full oscillation reduction experiment plan."""
    exps: list[dict[str, Any]] = []

    # Set 1: Is oscillation caused mainly by the initial velocity field?
    exps.extend(
        [
            base_expansion_experiment(
                name="osc_radial_0p05",
                velocity_mode="radial",
                velocity_scale=0.05,
                softening=2.0,
                H0=0.01,
            ),
            base_expansion_experiment(
                name="osc_angular_0p05",
                velocity_mode="angular",
                velocity_scale=0.05,
                softening=2.0,
                H0=0.01,
            ),
            base_expansion_experiment(
                name="osc_radial_0p01",
                velocity_mode="radial",
                velocity_scale=0.01,
                softening=2.0,
                H0=0.01,
            ),
            base_expansion_experiment(
                name="osc_angular_0p01",
                velocity_mode="angular",
                velocity_scale=0.01,
                softening=2.0,
                H0=0.01,
            ),
            base_expansion_experiment(
                name="osc_none",
                velocity_mode="none",
                velocity_scale=0.0,
                softening=2.0,
                H0=0.01,
            ),
        ]
    )

    # Set 2: Does stronger softening reduce close-range rebound/oscillation?
    for eps in [2.0, 5.0, 8.0, 10.0]:
        exps.append(
            base_expansion_experiment(
                name=f"osc_soft_eps{_scale_label(eps)}",
                velocity_mode="angular",
                velocity_scale=0.01,
                softening=eps,
                H0=0.01,
            )
        )

    # Set 3: Does expansion/damping reduce oscillation?
    # Include Sebastian's 2.7% economic growth idea.
    for H0 in [0.0, 0.01, 0.027, 0.05]:
        exps.append(
            base_expansion_experiment(
                name=f"osc_exp_H0_{_scale_label(H0)}",
                velocity_mode="angular",
                velocity_scale=0.01,
                softening=5.0,
                H0=H0,
            )
        )

    # Set 4: Is any remaining oscillation a timestep/numerical artifact?
    for dt in [0.02, 0.01, 0.005]:
        exps.append(
            base_expansion_experiment(
                name=f"osc_dt_{_scale_label(dt)}",
                velocity_mode="angular",
                velocity_scale=0.01,
                softening=5.0,
                H0=0.027,
                dt=dt,
            )
        )

    # Final Barnes-Hut candidate.
    # This is intentionally similar to osc_dt_0p01, but kept as a named candidate.
    exps.append(
        base_expansion_experiment(
            name="osc_barnes_hut_candidate",
            velocity_mode="angular",
            velocity_scale=0.01,
            softening=5.0,
            H0=0.027,
            dt=0.01,
            solver="barnes_hut",
        )
    )

    return exps


def filter_experiments_by_name(
    experiments: list[dict[str, Any]],
    only: list[str] | None,
) -> list[dict[str, Any]]:
    """Keep only selected experiments by EXPERIMENT_NAME."""
    if not only:
        return experiments

    by_name = {
        str(exp.get("EXPERIMENT_NAME")): exp
        for exp in experiments
    }

    missing = [name for name in only if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name.keys()))
        raise ValueError(
            f"Unknown experiment name(s): {missing}\n"
            f"Available experiments: {available}"
        )

    print(f"[runner] Selected experiments: {', '.join(only)}")
    return [by_name[name] for name in only]


def run_named_experiments(names: list[str], common_overrides: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = []

    for name in names:
        overrides = config.experiment_overrides(name)
        overrides.update(common_overrides)
        result = run_single_experiment(overrides)
        outputs.append(result)

    return outputs


def run_custom_experiments(
    experiments: list[dict[str, Any]],
    common_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    outputs = []

    for overrides in experiments:
        merged = dict(overrides)
        merged.update(common_overrides)
        result = run_single_experiment(merged)
        outputs.append(result)

    return outputs


def run_oscillation_experiments(
    common_overrides: dict[str, Any],
    only: list[str] | None = None,
) -> list[dict[str, Any]]:
    ensure_oscillation_velocity_inputs()

    common = dict(common_overrides)
    common["ANALYZE_OSCILLATION"] = True

    experiments = oscillation_experiment_overrides()
    experiments = filter_experiments_by_name(experiments, only)

    return run_custom_experiments(experiments, common)


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
        base.update(
            {
                "EXPERIMENT_NAME": name,
                "BARNES_HUT_THETA": theta,
            }
        )
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
                "connection_velocity_scale": float(cfg.get("CONNECTION_VELOCITY_SCALE", 0.0)),
                "output_dir": str(out["output_dir"]),
                "runtime_seconds": out.get("runtime_seconds"),
                "particles": int(cfg.get("MAX_PARTICLES")),
                "steps": int(cfg.get("STEPS")),
                "dt": float(cfg.get("DT")),
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


def save_run_index(
    outputs: list[dict[str, Any]],
    output_path: Path,
    run_manifest: dict[str, Any] | None = None,
) -> None:
    """Save run index as JSON, CSV, Markdown, and optional oscillation analysis."""
    rows = output_rows(outputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_json(output_path, rows)
    rows_to_csv(output_path.with_suffix(".csv"), rows)

    should_analyze = any(
        str(row.get("experiment", "")).startswith("osc_")
        for row in rows
    )

    analysis_result = None
    if should_analyze:
        try:
            analysis_result = analyze_run_outputs(rows, output_path.parent)
            print(f"[runner] Oscillation analysis: {analysis_result.get('oscillation_summary_csv')}")
        except Exception as exc:
            print(f"[runner] Oscillation analysis skipped: {exc}")

    if run_manifest is not None:
        manifest = dict(run_manifest)
        manifest["finished_at"] = now_iso()
        manifest["experiment_count"] = len(rows)
        manifest["experiments"] = [row.get("experiment") for row in rows]

        if analysis_result is not None:
            manifest["oscillation_analysis"] = analysis_result

        write_json(output_path.parent / "run_manifest.json", manifest)
        write_run_readme(output_path.parent / "RUN_README.md", manifest, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run N-body experiment sets.")

    parser.add_argument(
        "--set",
        choices=["sebastian", "barnes-hut", "all", "sweep", "theta", "oscillation"],
        default="sebastian",
    )

    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only selected oscillation experiment names, e.g. --only osc_soft_eps10p0",
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

    if args.only and args.set != "oscillation":
        raise ValueError("--only is currently supported only with --set oscillation")

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

    if args.set == "oscillation":
        outputs.extend(run_oscillation_experiments(common, args.only))

    save_run_index(outputs, run_root / "run_index.json", run_manifest)

    if args.comparison_gif and outputs:
        first_four = outputs[:4]
        masses = first_four[0]["masses"]
        colors = first_four[0].get("colors")
        histories = {
            out["config"]["EXPERIMENT_NAME"]: out["result"]["positions"]
            for out in first_four
        }

        try:
            save_comparison_gif(
                histories,
                masses,
                run_root / "comparison.gif",
                colors=colors,
            )
        except Exception as exc:
            print(f"[runner] Comparison GIF skipped: {exc}")

    print(f"[runner] Completed {len(outputs)} experiment(s).")
    print(f"[runner] Run folder: {run_root}")
    print(f"[runner] Index JSON: {run_root / 'run_index.json'}")
    print(f"[runner] Index CSV: {run_root / 'run_index.csv'}")
    print(f"[runner] Human-readable summary: {run_root / 'RUN_README.md'}")

    if any(str(out["config"].get("EXPERIMENT_NAME", "")).startswith("osc_") for out in outputs):
        print(f"[runner] Oscillation summary: {run_root / 'oscillation_summary.csv'}")
        print(f"[runner] Oscillation readme: {run_root / 'OSCILLATION_ANALYSIS.md'}")


if __name__ == "__main__":
    main()