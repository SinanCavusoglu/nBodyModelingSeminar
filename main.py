"""Main entry point for one Forbes N-body experiment."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import config as default_config
from src.animation import save_gif
from src.data_loader import load_bodies
from src.export import (
    ensure_dir,
    export_edges_csv,
    export_metrics_csv,
    export_positions_csv,
    export_summary_json,
)
from src.run_context import make_run_id, now_iso, write_json
from src.simulation import run_simulation
from src.interactive import export_interactive_3d_html


def _normalize_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def build_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = default_config.as_dict()
    if overrides:
        cfg.update(overrides)

    experiment_name = str(cfg.get("EXPERIMENT_NAME") or cfg.get("SIMULATION_MODE") or "experiment")
    cfg["EXPERIMENT_NAME"] = experiment_name

    # Output policy:
    # - run_full_pipeline.py and run_experiments.py pass one shared RUN_ID so
    #   all experiments from the same run go into the same timestamped folder.
    # - direct calls to main.py get a manual timestamped RUN_ID by default, so
    #   old outputs are not overwritten accidentally.
    if cfg.get("USE_TIMESTAMPED_RUN_DIRS", True):
        if not cfg.get("RUN_ID"):
            cfg["RUN_ID"] = make_run_id("manual")
        if not cfg.get("RUN_STARTED_AT"):
            cfg["RUN_STARTED_AT"] = now_iso()
        if not cfg.get("RUN_OUTPUT_ROOT"):
            cfg["RUN_OUTPUT_ROOT"] = default_config.RUNS_DIR / str(cfg["RUN_ID"])
        cfg["OUTPUT_DIR"] = default_config.output_dir_for(
            experiment_name,
            run_id=str(cfg.get("RUN_ID")),
            run_output_root=_normalize_path(cfg["RUN_OUTPUT_ROOT"]),
        )
    else:
        cfg["OUTPUT_DIR"] = default_config.output_dir_for(experiment_name)

    # Angular experiments prefer generated angular CSV if it exists. If it does
    # not exist yet, fall back to the default current CSV so the run does not die.
    csv_path = _normalize_path(cfg["CSV_PATH"])
    if not csv_path.exists() and "angular" in str(cfg.get("CONNECTION_VELOCITY_MODE", "")):
        fallback = _normalize_path(default_config.CSV_PATH)
        print(f"[main] Angular CSV not found: {csv_path}")
        print(f"[main] Falling back to current CSV: {fallback}")
        cfg["CSV_PATH"] = fallback
        cfg["EDGE_CSV_PATH"] = default_config.EDGE_CSV_PATH

    return cfg


def run_single_experiment(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = build_config(overrides)
    output_dir = ensure_dir(cfg["OUTPUT_DIR"])

    experiment_started_at = now_iso()
    print(f"[main] Experiment: {cfg['EXPERIMENT_NAME']}")
    print(f"[main] Run ID: {cfg.get('RUN_ID')}")
    print(f"[main] Solver: {cfg['FORCE_SOLVER']} | expansion={cfg['USE_EXPANSION']} | H0={cfg['H0']}")
    print(f"[main] CSV: {cfg['CSV_PATH']}")
    print(f"[main] Output: {output_dir}")

    loaded = load_bodies(cfg["CSV_PATH"], max_particles=int(cfg["MAX_PARTICLES"]))
    ids = loaded["ids"]
    names = loaded["names"]
    positions = loaded["positions"]
    velocities = loaded["velocities"]
    masses = loaded["masses"]
    colors = loaded.get("colors")

    t0 = time.perf_counter()
    result = run_simulation(positions, velocities, masses, cfg)
    runtime_seconds = time.perf_counter() - t0
    print(f"[main] Simulation runtime: {runtime_seconds:.3f}s")

    positions_csv_path = output_dir / "positions.csv"
    if cfg.get("SAVE_VVVV_CSV", True):
        export_positions_csv(
            positions_csv_path,
            result["positions"],
            result["velocities"],
            result["times"],
            ids,
            names,
            masses,
            colors=colors,
        )

    if cfg.get("SAVE_EDGE_CSV", True):
        export_edges_csv(cfg["EDGE_CSV_PATH"], output_dir / "edges.csv", ids)

    if cfg.get("SAVE_METRICS", True):
        export_metrics_csv(output_dir / "metrics.csv", result["metrics"])

    # Put runtime and timestamps into config before summary export.
    cfg = dict(cfg)
    cfg["runtime_seconds"] = runtime_seconds
    cfg["experiment_started_at"] = experiment_started_at
    cfg["experiment_finished_at"] = now_iso()
    cfg["command"] = sys.argv
    if cfg.get("SAVE_SUMMARY", True):
        export_summary_json(output_dir / "summary.json", cfg, result["metrics"], len(ids))
        write_json(output_dir / "experiment_manifest.json", {
            "run_id": cfg.get("RUN_ID"),
            "run_started_at": cfg.get("RUN_STARTED_AT"),
            "experiment": cfg.get("EXPERIMENT_NAME"),
            "simulation_mode": cfg.get("SIMULATION_MODE"),
            "started_at": cfg.get("experiment_started_at"),
            "finished_at": cfg.get("experiment_finished_at"),
            "runtime_seconds": cfg.get("runtime_seconds"),
            "output_dir": str(output_dir),
            "csv_path": str(cfg.get("CSV_PATH")),
            "edge_csv_path": str(cfg.get("EDGE_CSV_PATH")),
            "force_solver": cfg.get("FORCE_SOLVER"),
            "use_expansion": cfg.get("USE_EXPANSION"),
            "expansion_model": cfg.get("EXPANSION_MODEL"),
            "H0": cfg.get("H0"),
            "softening": cfg.get("SOFTENING"),
            "connection_velocity_mode": cfg.get("CONNECTION_VELOCITY_MODE"),
            "barnes_hut_theta": cfg.get("BARNES_HUT_THETA"),
            "particles": len(ids),
            "steps": cfg.get("STEPS"),
            "save_every": cfg.get("SAVE_EVERY"),
            "save_gif": cfg.get("SAVE_GIF"),
            "save_interactive_html": cfg.get("SAVE_INTERACTIVE_HTML"),
            "interactive_html_max_frames": cfg.get("INTERACTIVE_HTML_MAX_FRAMES"),
            "interactive_html_max_particles": cfg.get("INTERACTIVE_HTML_MAX_PARTICLES"),
            "notes": cfg.get("RUN_NOTES"),
        })

    if cfg.get("SAVE_INTERACTIVE_HTML", True):
        try:
            if not positions_csv_path.exists():
                export_positions_csv(
                    positions_csv_path,
                    result["positions"],
                    result["velocities"],
                    result["times"],
                    ids,
                    names,
                    masses,
                    colors=colors,
                )
            html_path = output_dir / "interactive_3d.html"
            export_interactive_3d_html(
                positions_csv=positions_csv_path,
                output_html=html_path,
                title=str(cfg["EXPERIMENT_NAME"]),
                max_frames=int(cfg.get("INTERACTIVE_HTML_MAX_FRAMES", 120)),
                max_particles=cfg.get("INTERACTIVE_HTML_MAX_PARTICLES", 300),
                include_plotlyjs=cfg.get("INTERACTIVE_HTML_INCLUDE_PLOTLYJS", True),
            )
            print(f"[main] Interactive HTML: {html_path}")
        except Exception as exc:
            print(f"[main] Interactive HTML skipped: {exc}")

    if cfg.get("SAVE_GIF", True):
        save_gif(
            result["positions"],
            masses,
            output_dir / "animation.gif",
            colors=colors,
            fps=int(cfg.get("GIF_FPS", 20)),
            dpi=int(cfg.get("ANIMATION_DPI", 120)),
            title=str(cfg["EXPERIMENT_NAME"]),
            point_size_min=float(cfg.get("ANIMATION_POINT_SIZE_MIN", 8)),
            point_size_max=float(cfg.get("ANIMATION_POINT_SIZE_MAX", 120)),
        )

    return {
        "config": cfg,
        "output_dir": output_dir,
        "runtime_seconds": runtime_seconds,
        "result": result,
        "ids": ids,
        "names": names,
        "masses": masses,
        "colors": colors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one N-body simulation experiment.")
    parser.add_argument("--experiment", default=None, help="Named experiment from config.EXPERIMENTS")
    parser.add_argument("--max-particles", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--solver", choices=["direct", "barnes_hut"], default=None)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-html", action="store_true", help="Disable interactive_3d.html export.")
    parser.add_argument("--html-max-frames", type=int, default=None)
    parser.add_argument("--html-max-particles", type=int, default=None)
    parser.add_argument("--H0", type=float, default=None)
    parser.add_argument("--softening", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None, help="Barnes-Hut theta")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict[str, Any] = {}
    if args.experiment:
        overrides.update(default_config.experiment_overrides(args.experiment))
    if args.max_particles is not None:
        overrides["MAX_PARTICLES"] = args.max_particles
    if args.steps is not None:
        overrides["STEPS"] = args.steps
    if args.save_every is not None:
        overrides["SAVE_EVERY"] = args.save_every
    if args.solver is not None:
        overrides["FORCE_SOLVER"] = args.solver
    if args.no_gif:
        overrides["SAVE_GIF"] = False
    if args.no_html:
        overrides["SAVE_INTERACTIVE_HTML"] = False
    if args.html_max_frames is not None:
        overrides["INTERACTIVE_HTML_MAX_FRAMES"] = args.html_max_frames
    if args.html_max_particles is not None:
        overrides["INTERACTIVE_HTML_MAX_PARTICLES"] = args.html_max_particles
    if args.H0 is not None:
        overrides["H0"] = args.H0
    if args.softening is not None:
        overrides["SOFTENING"] = args.softening
    if args.theta is not None:
        overrides["BARNES_HUT_THETA"] = args.theta

    run_single_experiment(overrides)


if __name__ == "__main__":
    main()
