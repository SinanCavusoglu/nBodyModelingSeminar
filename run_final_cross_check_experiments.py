"""Run a final coarse cross-check benchmark with all stabilization families.

This runner is meant for a strong PC and analysis-only execution. It does not
render GIF/HTML, but this version saves the full positions.csv by default
for each experiment so the final motion can be inspected or rendered later.
It also keeps metrics, summaries, manifests, and ranked comparison CSV files.

The benchmark cross-checks the main anti-collapse parameters together with the
new stabilization extensions:

  - position scale
  - mass transformation
  - gravitational softening
  - connection-based angular velocity scale
  - fixed expansion rate H0=0.027
  - virialized initial velocity scaling
  - adaptive gravitational softening

Recommended final run:

    python run_final_cross_check_experiments.py \
        --profile final \
        --run-id final_cross_check_01 \
        --max-particles 3426 \
        --steps 3000 \
        --save-every 5

Quick smoke test:

    python run_final_cross_check_experiments.py --profile smoke --max-particles 100 --steps 20

List experiments without running:

    python run_final_cross_check_experiments.py --profile final --list
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from main import build_config, run_single_experiment
from run_anti_collapse_experiments import experiment_overrides as anti_experiment_overrides
from run_anti_collapse_experiments import output_rows as anti_output_rows
from src.analysis import analyze_run_outputs
from src.run_context import base_run_manifest, make_run_id, now_iso, rows_to_csv, write_json, write_run_readme


# -----------------------------------------------------------------------------
# Profiles
# -----------------------------------------------------------------------------
# "final" is the intended big-step cross-check. It covers every important
# parameter family without exploding into thousands of runs.
#
# total final experiments = 2 * 4 * 3 * 3 * 1 * 4 = 288
#
# Expansion is kept ON at H0=0.027 for every experiment because earlier
# benchmarks showed it was useful. It is no longer a cross-check factor.
#
# For even heavier checking, "full" adds adaptive_k2 states.
# total full experiments = 2 * 4 * 3 * 3 * 1 * 6 = 432
# -----------------------------------------------------------------------------
PROFILE_LEVELS: dict[str, dict[str, Any]] = {
    "smoke": {
        "position_scales": [10.0],
        "mass_models": ["log"],
        "softenings": [20.0],
        "velocity_scales": [0.02],
        "H0_values": [0.027],
        "states": ["base", "virial_adaptive_k1"],
    },
    "focused": {
        "position_scales": [10.0],
        "mass_models": ["current", "log"],
        "softenings": [10.0, 20.0, 30.0],
        "velocity_scales": [0.01, 0.02],
        "H0_values": [0.027],
        "states": ["base", "virial", "adaptive_k1", "virial_adaptive_k1"],
    },
    "final": {
        "position_scales": [10.0, 20.0],
        "mass_models": ["current", "sqrt", "log", "logclip"],
        "softenings": [10.0, 20.0, 30.0],
        "velocity_scales": [0.01, 0.02, 0.03],
        "H0_values": [0.027],
        "states": ["base", "virial", "adaptive_k1", "virial_adaptive_k1"],
    },
    "full": {
        "position_scales": [10.0, 20.0],
        "mass_models": ["current", "sqrt", "log", "logclip"],
        "softenings": [10.0, 20.0, 30.0],
        "velocity_scales": [0.01, 0.02, 0.03],
        "H0_values": [0.027],
        "states": ["base", "virial", "adaptive_k1", "adaptive_k2", "virial_adaptive_k1", "virial_adaptive_k2"],
    },
}

STATE_OVERRIDES: dict[str, dict[str, Any]] = {
    "base": {
        "virial": False,
        "adaptive": False,
    },
    "virial": {
        "virial": True,
        "target_virial_ratio": 1.0,
        "adaptive": False,
    },
    "adaptive_k1": {
        "virial": False,
        "adaptive": True,
        "adaptive_mode": "density_boost",
        "adaptive_k": 1.0,
    },
    "adaptive_k2": {
        "virial": False,
        "adaptive": True,
        "adaptive_mode": "density_boost",
        "adaptive_k": 2.0,
    },
    "virial_adaptive_k1": {
        "virial": True,
        "target_virial_ratio": 1.0,
        "adaptive": True,
        "adaptive_mode": "density_boost",
        "adaptive_k": 1.0,
    },
    "virial_adaptive_k2": {
        "virial": True,
        "target_virial_ratio": 1.0,
        "adaptive": True,
        "adaptive_mode": "density_boost",
        "adaptive_k": 2.0,
    },
}


def value_token(value: Any) -> str:
    """Make compact, filename-safe tokens such as 0p027, 10, logclip."""
    if isinstance(value, float):
        if float(value).is_integer():
            return str(int(value))
        return (f"{value:.6g}").replace("-", "m").replace(".", "p")
    return str(value).replace(".", "p").replace("-", "m")


def make_experiment_name(exp: dict[str, Any]) -> str:
    return (
        "fc_"
        f"pos{value_token(exp['position_scale'])}_"
        f"mass{value_token(exp['mass_model'])}_"
        f"soft{value_token(exp['softening'])}_"
        f"ang{value_token(exp['velocity_scale'])}_"
        f"H0_{value_token(exp['H0'])}_"
        f"{exp['stabilization_state']}"
    )


def build_experiments(profile: str) -> list[dict[str, Any]]:
    if profile not in PROFILE_LEVELS:
        valid = ", ".join(sorted(PROFILE_LEVELS))
        raise ValueError(f"Unknown profile '{profile}'. Valid profiles: {valid}")

    levels = PROFILE_LEVELS[profile]
    experiments: list[dict[str, Any]] = []

    for position_scale, mass_model, softening, velocity_scale, H0, state in itertools.product(
        levels["position_scales"],
        levels["mass_models"],
        levels["softenings"],
        levels["velocity_scales"],
        levels["H0_values"],
        levels["states"],
    ):
        exp: dict[str, Any] = {
            "position_scale": float(position_scale),
            "mass_model": str(mass_model),
            "softening": float(softening),
            "velocity_scale": float(velocity_scale),
            "H0": float(H0),
            "dt": 0.02,
            "stabilization_state": str(state),
        }
        exp.update(STATE_OVERRIDES[str(state)])
        exp["name"] = make_experiment_name(exp)
        experiments.append(exp)

    return experiments


def make_run_overrides(run_id: str | None, notes: str) -> dict[str, Any]:
    run_id = run_id or make_run_id("final_cross_check")
    run_root = config.RUNS_DIR / run_id
    return {
        "RUN_ID": run_id,
        "RUN_STARTED_AT": now_iso(),
        "RUN_OUTPUT_ROOT": run_root,
        "RUN_NOTES": notes or "",
        "USE_TIMESTAMPED_RUN_DIRS": True,
    }


def experiment_overrides(exp: dict[str, Any]) -> dict[str, Any]:
    """Build runtime overrides for one final cross-check experiment."""
    overrides = anti_experiment_overrides(exp)
    overrides["SIMULATION_MODE"] = str(exp["name"])
    overrides["EXPERIMENT_NAME"] = str(exp["name"])
    overrides["FINAL_CROSS_CHECK_PROFILE"] = str(exp.get("profile", ""))
    overrides["FINAL_STABILIZATION_STATE"] = str(exp.get("stabilization_state", "base"))

    overrides["USE_VIRIAL_VELOCITY_SCALING"] = bool(exp.get("virial", False))
    overrides["TARGET_VIRIAL_RATIO"] = float(exp.get("target_virial_ratio", 1.0))
    overrides["VIRIAL_VELOCITY_SCALE_MIN"] = float(exp.get("virial_scale_min", 0.1))
    overrides["VIRIAL_VELOCITY_SCALE_MAX"] = float(exp.get("virial_scale_max", 10.0))

    overrides["USE_ADAPTIVE_SOFTENING"] = bool(exp.get("adaptive", False))
    overrides["ADAPTIVE_SOFTENING_MODE"] = str(exp.get("adaptive_mode", "density_boost"))
    overrides["ADAPTIVE_SOFTENING_K"] = float(exp.get("adaptive_k", 1.0))
    overrides["ADAPTIVE_SOFTENING_MIN"] = float(exp.get("adaptive_min", 0.5))
    overrides["ADAPTIVE_SOFTENING_MAX"] = float(exp.get("adaptive_max", 30.0))
    overrides["ADAPTIVE_SOFTENING_UPDATE_EVERY"] = int(exp.get("adaptive_update_every", 10))

    # Analysis-only by default. Metrics are enough for ranking; positions and
    # edges are the largest outputs and are not needed for this final cross-check.
    overrides["SAVE_GIF"] = False
    overrides["SAVE_INTERACTIVE_HTML"] = False
    overrides["SAVE_VVVV_CSV"] = False
    overrides["SAVE_EDGE_CSV"] = False
    overrides["SAVE_METRICS"] = True
    overrides["SAVE_SUMMARY"] = True
    overrides["ANALYZE_OSCILLATION"] = True
    return overrides


def output_rows(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = anti_output_rows(outputs)
    for row, out in zip(rows, outputs):
        cfg = out["config"]
        row["benchmark_family"] = "final_cross_check"
        row["profile"] = cfg.get("FINAL_CROSS_CHECK_PROFILE")
        row["stabilization_state"] = cfg.get("FINAL_STABILIZATION_STATE")
        row["use_expansion"] = bool(cfg.get("USE_EXPANSION", False))
        row["use_virial_velocity_scaling"] = bool(cfg.get("USE_VIRIAL_VELOCITY_SCALING", False))
        row["target_virial_ratio"] = float(cfg.get("TARGET_VIRIAL_RATIO", 1.0))
        row["use_adaptive_softening"] = bool(cfg.get("USE_ADAPTIVE_SOFTENING", False))
        row["adaptive_softening_mode"] = cfg.get("ADAPTIVE_SOFTENING_MODE", "density_boost")
        row["adaptive_softening_k"] = float(cfg.get("ADAPTIVE_SOFTENING_K", 1.0))
        row["adaptive_softening_min"] = float(cfg.get("ADAPTIVE_SOFTENING_MIN", 0.5))
        row["adaptive_softening_max"] = float(cfg.get("ADAPTIVE_SOFTENING_MAX", 30.0))
        row["adaptive_softening_update_every"] = int(cfg.get("ADAPTIVE_SOFTENING_UPDATE_EVERY", 10))
        row["save_positions_csv"] = bool(cfg.get("SAVE_VVVV_CSV", False))
        row["skipped_existing"] = bool(out.get("skipped_existing", False))
    return rows


def write_final_cross_summary(run_root: Path, rows: list[dict[str, Any]]) -> Path | None:
    """Merge analysis scores with experiment parameters into a final table."""
    summary_path = run_root / "oscillation_summary.csv"
    if not summary_path.exists():
        return None

    summary = pd.read_csv(summary_path)
    if summary.empty or "experiment" not in summary.columns:
        return None

    index_df = pd.DataFrame(rows)
    if index_df.empty or "experiment" not in index_df.columns:
        return None

    param_cols = [
        "experiment",
        "profile",
        "stabilization_state",
        "anti_position_scale",
        "anti_mass_model",
        "softening",
        "connection_velocity_scale",
        "H0",
        "use_expansion",
        "use_virial_velocity_scaling",
        "target_virial_ratio",
        "use_adaptive_softening",
        "adaptive_softening_k",
        "adaptive_softening_min",
        "adaptive_softening_max",
        "adaptive_softening_update_every",
        "particles",
        "steps",
        "dt",
        "save_every",
        "runtime_seconds",
        "skipped_existing",
        "output_dir",
    ]
    param_cols = [c for c in param_cols if c in index_df.columns]
    extra_cols = [c for c in param_cols if c == "experiment" or c not in summary.columns]
    merged = summary.merge(index_df[extra_cols], on="experiment", how="left")

    # Put the most useful columns first when available.
    preferred = [
        "rank",
        "experiment",
        "stabilization_state",
        "anti_position_scale",
        "anti_mass_model",
        "softening",
        "connection_velocity_scale",
        "H0",
        "use_virial_velocity_scaling",
        "use_adaptive_softening",
        "adaptive_softening_k",
        "collapse_time",
        "stability_score",
        "radius_oscillation_score",
        "kinetic_spike_score",
        "virial_spike_score",
        "nearest_neighbor_oscillation_score",
        "final_mean_radius",
        "final_virial_ratio",
        "runtime_seconds",
        "output_dir",
    ]
    ordered = [c for c in preferred if c in merged.columns] + [c for c in merged.columns if c not in preferred]
    merged = merged[ordered]

    out_path = run_root / "final_cross_check_summary.csv"
    merged.to_csv(out_path, index=False)

    # Also write a compact markdown top-20 report.
    md_path = run_root / "FINAL_CROSS_CHECK_ANALYSIS.md"
    lines: list[str] = []
    lines.append("# Final Cross-Check Analysis")
    lines.append("")
    lines.append("This benchmark compares the main anti-collapse parameters together with virialized velocity scaling and adaptive softening.")
    lines.append("")
    lines.append("Lower `stability_score` means smoother behavior according to the existing analysis metrics.")
    lines.append("")
    top = merged.head(20).copy()
    cols = [
        "rank",
        "experiment",
        "stabilization_state",
        "anti_position_scale",
        "anti_mass_model",
        "softening",
        "connection_velocity_scale",
        "H0",
        "collapse_time",
        "stability_score",
        "runtime_seconds",
    ]
    cols = [c for c in cols if c in top.columns]
    lines.append("## Top 20")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in top[cols].iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append("nan" if not np.isfinite(v) else f"{float(v):.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `final_cross_check_summary.csv`: ranked summary with parameters")
    lines.append("- `oscillation_summary.csv`: raw stability ranking from the analysis module")
    lines.append("- `run_index.csv`: all experiments and output folders")
    lines.append("- `plots/`: comparison plots")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def save_run_outputs(outputs: list[dict[str, Any]], run_root: Path, run_manifest: dict[str, Any]) -> None:
    rows = output_rows(outputs)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_index.json", rows)
    rows_to_csv(run_root / "run_index.csv", rows)

    analysis_result = None
    try:
        analysis_result = analyze_run_outputs(rows, run_root)
        print(f"[final-cross] Analysis summary: {analysis_result.get('oscillation_summary_csv')}")
    except Exception as exc:
        print(f"[final-cross] Analysis skipped: {exc}")

    final_summary = write_final_cross_summary(run_root, rows)
    if final_summary is not None:
        print(f"[final-cross] Final merged summary: {final_summary}")

    manifest = dict(run_manifest)
    manifest["finished_at"] = now_iso()
    manifest["experiment_count"] = len(rows)
    manifest["experiments"] = [row.get("experiment") for row in rows]
    manifest["analysis_goal"] = "final coarse cross-check of anti-collapse parameters, virial scaling, and adaptive softening"
    if analysis_result is not None:
        manifest["analysis"] = analysis_result
    if final_summary is not None:
        manifest["final_cross_check_summary_csv"] = str(final_summary)
    write_json(run_root / "run_manifest.json", manifest)
    write_run_readme(run_root / "RUN_README.md", manifest, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final coarse cross-check N-body experiments.")
    parser.add_argument("--profile", choices=sorted(PROFILE_LEVELS), default="final")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-particles", type=int, default=3426)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--notes", default="Final cross-check with expansion fixed at H0=0.027; positions.csv enabled; no GIF/HTML rendering")
    parser.add_argument("--only", nargs="+", default=None, help="Optional list of experiment names to run.")
    parser.add_argument("--list", action="store_true", help="List experiments and exit.")
    parser.add_argument("--max-experiments", type=int, default=None, help="Run only the first N experiments after filtering.")
    parser.add_argument("--resume", action="store_true", help="Skip experiments that already have metrics.csv in the run folder.")
    parser.add_argument("--save-positions", dest="save_positions", action="store_true", default=True, help="Save positions.csv for each experiment. Default: ON in this final version.")
    parser.add_argument("--no-save-positions", dest="save_positions", action="store_false", help="Disable positions.csv export for faster/lighter analysis-only runs.")
    parser.add_argument("--save-edges", action="store_true", help="Also save edges.csv for each experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Print run plan and exit without simulation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = build_experiments(args.profile)
    for exp in experiments:
        exp["profile"] = args.profile

    if args.only:
        by_name = {str(exp["name"]): exp for exp in experiments}
        missing = [name for name in args.only if name not in by_name]
        if missing:
            available = "\n".join(sorted(by_name.keys()))
            raise ValueError(f"Unknown experiment(s): {missing}\nAvailable:\n{available}")
        experiments = [by_name[name] for name in args.only]

    if args.max_experiments is not None:
        experiments = experiments[: int(args.max_experiments)]

    if args.list or args.dry_run:
        print(f"Profile: {args.profile}")
        print(f"Experiment count: {len(experiments)}")
        print("Experiments:")
        for exp in experiments:
            print(f"  {exp['name']}")
        if args.dry_run:
            print("\nDry run only; no simulation executed.")
        return

    common = make_run_overrides(args.run_id, args.notes)
    common["MAX_PARTICLES"] = int(args.max_particles)
    common["STEPS"] = int(args.steps)
    common["SAVE_EVERY"] = int(args.save_every)
    common["SAVE_GIF"] = False
    common["SAVE_INTERACTIVE_HTML"] = False
    common["SAVE_VVVV_CSV"] = bool(args.save_positions)
    common["SAVE_EDGE_CSV"] = bool(args.save_edges)
    common["SAVE_METRICS"] = True
    common["SAVE_SUMMARY"] = True

    run_root = Path(common["RUN_OUTPUT_ROOT"])
    run_manifest = base_run_manifest(
        run_id=str(common["RUN_ID"]),
        run_output_root=run_root,
        project_root=config.PROJECT_ROOT,
        preset=f"final-cross-check-{args.profile}",
        raw_csv=config.RAW_CSV_PATH,
        command=sys.argv,
        common_overrides=common,
        notes=args.notes,
    )
    run_manifest["profile"] = args.profile
    run_manifest["profile_levels"] = PROFILE_LEVELS[args.profile]
    run_manifest["expansion_policy"] = "fixed_on_H0_0p027"
    write_json(run_root / "run_manifest.json", run_manifest)

    print("[final-cross] Starting final cross-check benchmark")
    print(f"[final-cross] Profile: {args.profile}")
    print(f"[final-cross] Experiments: {len(experiments)}")
    print(f"[final-cross] Output: {run_root}")
    print(f"[final-cross] Particles={args.max_particles}, steps={args.steps}, save_every={args.save_every}")
    print("[final-cross] GIF=False, HTML=False, positions_csv=" + str(bool(args.save_positions)))
    if args.save_positions:
        print("[final-cross] WARNING: positions.csv is ON; output folders will be large.")
    print("[final-cross] solver=barnes_hut, theta=0.5, dt=0.02")
    print("[final-cross] expansion=fixed ON, H0=0.027")

    outputs: list[dict[str, Any]] = []
    for idx, exp in enumerate(experiments, start=1):
        print("\n" + "=" * 80)
        print(f"[final-cross] {idx}/{len(experiments)} Running {exp['name']}")
        print(
            "[final-cross] "
            f"pos={exp['position_scale']} | mass={exp['mass_model']} | "
            f"soft={exp['softening']} | ang={exp['velocity_scale']} | H0={exp['H0']} | "
            f"state={exp['stabilization_state']}"
        )
        overrides = experiment_overrides(exp)
        overrides.update(common)
        cfg = build_config(overrides)
        output_dir = Path(cfg["OUTPUT_DIR"])
        metrics_path = output_dir / "metrics.csv"
        if args.resume and metrics_path.exists():
            print(f"[final-cross] Skipping existing metrics: {metrics_path}")
            cfg = dict(cfg)
            cfg["runtime_seconds"] = None
            outputs.append({
                "config": cfg,
                "output_dir": output_dir,
                "runtime_seconds": None,
                "skipped_existing": True,
                "result": {},
                "ids": [],
                "names": [],
                "masses": [],
                "colors": None,
            })
            continue
        outputs.append(run_single_experiment(overrides))

    save_run_outputs(outputs, run_root, run_manifest)

    print("\n[final-cross] Completed.")
    print(f"[final-cross] Run folder: {run_root}")
    print(f"[final-cross] Run index: {run_root / 'run_index.csv'}")
    print(f"[final-cross] Stability summary: {run_root / 'oscillation_summary.csv'}")
    print(f"[final-cross] Final merged summary: {run_root / 'final_cross_check_summary.csv'}")
    print(f"[final-cross] Analysis markdown: {run_root / 'FINAL_CROSS_CHECK_ANALYSIS.md'}")
    print(f"[final-cross] Plots: {run_root / 'plots'}")


if __name__ == "__main__":
    main()
