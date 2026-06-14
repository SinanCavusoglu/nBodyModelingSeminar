"""Export helpers for simulation outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import first_collapse_time


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_positions_csv(
    path: str | Path,
    positions: np.ndarray,
    velocities: np.ndarray,
    times: np.ndarray,
    ids: np.ndarray,
    names: np.ndarray,
    masses: np.ndarray,
    colors: np.ndarray | None = None,
) -> Path:
    """Export frame-wise particle positions in a vvvv-friendly long CSV format."""
    path = Path(path)
    ensure_dir(path.parent)

    rows = []
    n_frames = len(positions)
    n_particles = len(ids)
    has_colors = colors is not None and len(colors) == n_particles
    for f in range(n_frames):
        pos_f = positions[f]
        vel_f = velocities[f]
        speed = np.linalg.norm(vel_f, axis=1)
        for i in range(n_particles):
            row = {
                    "frame": f,
                    "time": float(times[f]),
                    "id": ids[i],
                    "name": names[i],
                    "x": float(pos_f[i, 0]),
                    "y": float(pos_f[i, 1]),
                    "z": float(pos_f[i, 2]),
                    "vx": float(vel_f[i, 0]),
                    "vy": float(vel_f[i, 1]),
                    "vz": float(vel_f[i, 2]),
                    "mass": float(masses[i]),
                    "speed": float(speed[i]),
                }
            if has_colors:
                value = colors[i]
                try:
                    row["color"] = float(value)
                except (TypeError, ValueError):
                    row["color"] = str(value)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_metrics_csv(path: str | Path, metrics_rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(metrics_rows).to_csv(path, index=False)
    return path


def export_summary_json(
    path: str | Path,
    config: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    n_particles: int,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)

    final_metrics = metrics_rows[-1] if metrics_rows else {}
    summary = {
        "run_id": config.get("RUN_ID"),
        "run_started_at": config.get("RUN_STARTED_AT"),
        "experiment_started_at": config.get("experiment_started_at"),
        "experiment_finished_at": config.get("experiment_finished_at"),
        "experiment": str(config.get("EXPERIMENT_NAME")),
        "simulation_mode": str(config.get("SIMULATION_MODE")),
        "force_solver": str(config.get("FORCE_SOLVER")),
        "output_dir": str(config.get("OUTPUT_DIR")),
        "csv_path": str(config.get("CSV_PATH")),
        "edge_csv_path": str(config.get("EDGE_CSV_PATH")),
        "runtime_seconds": config.get("runtime_seconds"),
        "particles": int(n_particles),
        "G": float(config.get("G")),
        "dt": float(config.get("DT")),
        "steps": int(config.get("STEPS")),
        "save_every": int(config.get("SAVE_EVERY")),
        "softening": float(config.get("SOFTENING")),
        "use_expansion": bool(config.get("USE_EXPANSION")),
        "expansion_model": str(config.get("EXPANSION_MODEL")),
        "H0": float(config.get("H0")),
        "connection_velocity_mode": str(config.get("CONNECTION_VELOCITY_MODE")),
        "barnes_hut_theta": float(config.get("BARNES_HUT_THETA")),
        "collapse_time": first_collapse_time(metrics_rows),
        "final_metrics": final_metrics,
    }

    # Convert Path values to strings.
    for key, value in list(summary.items()):
        if isinstance(value, Path):
            summary[key] = str(value)

    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


def export_edges_csv(
    source_edge_csv: str | Path,
    output_path: str | Path,
    selected_ids: np.ndarray,
) -> Path | None:
    """Filter an edge CSV to the selected particle IDs.

    Supports common edge column names. If the source edge file does not exist,
    this function returns None instead of failing the whole experiment.
    """
    source_edge_csv = Path(source_edge_csv)
    output_path = Path(output_path)
    if not source_edge_csv.exists():
        print(f"[export] Edge CSV not found, skipping: {source_edge_csv}")
        return None

    df = pd.read_csv(source_edge_csv)
    selected = set(map(str, selected_ids))

    candidate_pairs = [
        ("source", "target"),
        ("source_id", "target_id"),
        ("from", "to"),
        ("id1", "id2"),
        ("ID1", "ID2"),
    ]
    pair = None
    for a, b in candidate_pairs:
        if a in df.columns and b in df.columns:
            pair = (a, b)
            break

    if pair is None:
        print("[export] Could not identify edge columns; copying edge file unchanged.")
        ensure_dir(output_path.parent)
        df.to_csv(output_path, index=False)
        return output_path

    a, b = pair
    mask = df[a].astype(str).isin(selected) & df[b].astype(str).isin(selected)
    ensure_dir(output_path.parent)
    df.loc[mask].to_csv(output_path, index=False)
    return output_path
