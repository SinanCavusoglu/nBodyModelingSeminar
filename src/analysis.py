"""Post-run analysis helpers for oscillation/stability experiments.

The goal is to compare experiments quantitatively, not only visually.  This
module reads each experiment's metrics.csv and summary.json, computes simple
oscillation/stability scores, and writes run-level summary tables and plots.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _finite_series(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _normalized_diff_std(values: np.ndarray) -> float:
    values = _finite_series(values)
    if values.size < 3:
        return float("nan")
    diffs = np.diff(values)
    scale = float(np.nanmean(np.abs(values))) + 1e-12
    return float(np.nanstd(diffs) / scale)


def _spike_score(values: np.ndarray) -> float:
    values = _finite_series(values)
    if values.size == 0:
        return float("nan")
    median = float(np.nanmedian(np.abs(values))) + 1e-12
    peak = float(np.nanmax(np.abs(values)))
    return float(max(0.0, (peak - median) / median))


def _range_score(values: np.ndarray) -> float:
    values = _finite_series(values)
    if values.size == 0:
        return float("nan")
    denom = float(np.nanmean(np.abs(values))) + 1e-12
    return float((float(np.nanmax(values)) - float(np.nanmin(values))) / denom)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summarize_experiment_output(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return one oscillation-analysis row for an experiment output row."""
    output_dir = Path(row.get("output_dir", ""))
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"

    if not metrics_path.exists():
        return None

    metrics = pd.read_csv(metrics_path)
    summary = load_json(summary_path)

    mean_radius = metrics.get("mean_radius", pd.Series(dtype=float))
    kinetic = metrics.get("kinetic_energy", pd.Series(dtype=float))
    virial = metrics.get("virial_ratio", pd.Series(dtype=float))
    nearest = metrics.get("nearest_neighbor_mean", pd.Series(dtype=float))

    radius_oscillation_score = _normalized_diff_std(mean_radius)
    radius_range_score = _range_score(mean_radius)
    kinetic_spike_score = _spike_score(kinetic)
    virial_spike_score = _spike_score(virial)
    nearest_oscillation_score = _normalized_diff_std(nearest)

    collapse_time = summary.get("collapse_time")
    if collapse_time is None:
        collapse_time = None
        collapse_penalty = 0.0
    else:
        collapse_time = _safe_float(collapse_time, default=float("nan"))
        max_time = _safe_float(metrics["time"].max() if "time" in metrics else np.nan)
        if np.isfinite(collapse_time) and np.isfinite(max_time) and max_time > 0:
            # Early collapse is penalized more strongly than late collapse.
            collapse_penalty = max(0.0, 1.0 - collapse_time / max_time)
        else:
            collapse_penalty = 1.0

    # Lower is better. The weights are intentionally simple and interpretable.
    parts = [
        radius_oscillation_score,
        0.25 * radius_range_score,
        0.35 * kinetic_spike_score,
        0.35 * virial_spike_score,
        0.20 * nearest_oscillation_score,
        collapse_penalty,
    ]
    finite_parts = [p for p in parts if np.isfinite(p)]
    stability_score = float(np.sum(finite_parts)) if finite_parts else float("nan")

    final = metrics.iloc[-1].to_dict() if len(metrics) else {}

    return {
        "experiment": row.get("experiment"),
        "output_dir": str(output_dir),
        "force_solver": row.get("force_solver"),
        "use_expansion": row.get("use_expansion"),
        "connection_velocity_mode": row.get("connection_velocity_mode"),
        "connection_velocity_scale": row.get("connection_velocity_scale"),
        "particles": row.get("particles"),
        "steps": row.get("steps"),
        "dt": row.get("dt"),
        "save_every": row.get("save_every"),
        "softening": row.get("softening"),
        "H0": row.get("H0"),
        "theta": row.get("theta"),
        "runtime_seconds": row.get("runtime_seconds"),
        "collapse_time": collapse_time,
        "radius_oscillation_score": radius_oscillation_score,
        "radius_range_score": radius_range_score,
        "kinetic_spike_score": kinetic_spike_score,
        "virial_spike_score": virial_spike_score,
        "nearest_neighbor_oscillation_score": nearest_oscillation_score,
        "collapse_penalty": collapse_penalty,
        "stability_score": stability_score,
        "final_mean_radius": _safe_float(final.get("mean_radius")),
        "final_median_radius": _safe_float(final.get("median_radius")),
        "final_kinetic_energy": _safe_float(final.get("kinetic_energy")),
        "final_virial_ratio": _safe_float(final.get("virial_ratio")),
        "final_nearest_neighbor_mean": _safe_float(final.get("nearest_neighbor_mean")),
    }


def write_oscillation_summary(rows: list[dict[str, Any]], run_root: str | Path) -> pd.DataFrame:
    """Write oscillation summary CSV/JSON and return the dataframe."""
    run_root = Path(run_root)
    summaries = []
    for row in rows:
        summary = summarize_experiment_output(row)
        if summary is not None:
            summaries.append(summary)

    df = pd.DataFrame(summaries)
    if not df.empty and "stability_score" in df.columns:
        df = df.sort_values("stability_score", ascending=True, na_position="last")
        df.insert(0, "rank", range(1, len(df) + 1))

    csv_path = run_root / "oscillation_summary.csv"
    json_path = run_root / "oscillation_summary.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return df


def write_oscillation_readme(summary_df: pd.DataFrame, run_root: str | Path) -> Path:
    run_root = Path(run_root)
    path = run_root / "OSCILLATION_ANALYSIS.md"
    lines: list[str] = []
    lines.append("# Oscillation Analysis")
    lines.append("")
    lines.append("This file summarizes the automatic oscillation/stability analysis for this run.")
    lines.append("")
    lines.append("Lower `stability_score` means smoother behavior according to these simple diagnostics:")
    lines.append("")
    lines.append("- `radius_oscillation_score`: how much `mean_radius` zigzags frame-to-frame")
    lines.append("- `kinetic_spike_score`: how strong kinetic-energy spikes are")
    lines.append("- `virial_spike_score`: how strongly the virial ratio spikes")
    lines.append("- `collapse_penalty`: penalty for early collapse")
    lines.append("")
    if summary_df.empty:
        lines.append("No oscillation summary rows were created.")
    else:
        cols = [
            "rank", "experiment", "connection_velocity_mode", "connection_velocity_scale",
            "softening", "H0", "dt", "collapse_time", "stability_score",
            "radius_oscillation_score", "kinetic_spike_score", "virial_spike_score",
        ]
        available = [c for c in cols if c in summary_df.columns]
        lines.append("## Ranked experiments")
        lines.append("")
        lines.append("| " + " | ".join(available) + " |")
        lines.append("|" + "|".join(["---"] * len(available)) + "|")
        for _, row in summary_df[available].iterrows():
            vals = []
            for c in available:
                v = row[c]
                if isinstance(v, float):
                    vals.append("nan" if not np.isfinite(v) else f"{v:.6g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
        lines.append("## Main files")
        lines.append("")
        lines.append("- `oscillation_summary.csv`: spreadsheet-friendly ranking")
        lines.append("- `oscillation_summary.json`: machine-readable ranking")
        lines.append("- `plots/oscillation_mean_radius.png`: mean-radius comparison")
        lines.append("- `plots/oscillation_kinetic_energy.png`: kinetic-energy comparison")
        lines.append("- `plots/oscillation_virial_ratio.png`: virial-ratio comparison")
        lines.append("- `plots/oscillation_nearest_neighbor.png`: nearest-neighbor comparison")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_metric_comparison(
    rows: list[dict[str, Any]],
    run_root: str | Path,
    metric: str,
    filename: str,
    ylabel: str,
    max_experiments: int = 16,
) -> Path | None:
    """Write one comparison plot for a metric across experiments."""
    # matplotlib import is kept local so non-plot analysis can still run without a display.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_root = Path(run_root)
    plot_dir = run_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    usable = []
    for row in rows:
        metrics_path = Path(row.get("output_dir", "")) / "metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            if "time" in df.columns and metric in df.columns:
                usable.append((row, df))
    if not usable:
        return None

    # Prefer all rows for small runs; if too many, keep first max_experiments.
    usable = usable[:max_experiments]

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    for row, df in usable:
        label = str(row.get("experiment", "experiment"))
        ax.plot(df["time"], df[metric], label=label, linewidth=1.5)
    ax.set_title(ylabel)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = plot_dir / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def write_oscillation_plots(rows: list[dict[str, Any]], run_root: str | Path) -> list[Path]:
    specs = [
        ("mean_radius", "oscillation_mean_radius.png", "Mean radius over time"),
        ("kinetic_energy", "oscillation_kinetic_energy.png", "Kinetic energy over time"),
        ("virial_ratio", "oscillation_virial_ratio.png", "Virial ratio over time"),
        ("nearest_neighbor_mean", "oscillation_nearest_neighbor.png", "Mean nearest-neighbor distance over time"),
    ]
    paths: list[Path] = []
    for metric, filename, ylabel in specs:
        path = plot_metric_comparison(rows, run_root, metric, filename, ylabel)
        if path is not None:
            paths.append(path)
    return paths


def analyze_run_outputs(rows: list[dict[str, Any]], run_root: str | Path) -> dict[str, Any]:
    """Create all run-level oscillation analysis outputs."""
    summary_df = write_oscillation_summary(rows, run_root)
    readme_path = write_oscillation_readme(summary_df, run_root)
    plot_paths = write_oscillation_plots(rows, run_root)
    return {
        "oscillation_summary_csv": str(Path(run_root) / "oscillation_summary.csv"),
        "oscillation_summary_json": str(Path(run_root) / "oscillation_summary.json"),
        "oscillation_readme": str(readme_path),
        "plots": [str(p) for p in plot_paths],
        "rows": int(len(summary_df)),
    }
