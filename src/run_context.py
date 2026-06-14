"""Helpers for timestamped run folders and human-readable run metadata."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def now_iso() -> str:
    """Return local time with timezone in ISO format."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_run_id(prefix: str = "run") -> str:
    """Create a filesystem-safe timestamped run id."""
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix).strip("_")
    return f"{stamp}_{safe_prefix}" if safe_prefix else stamp


def current_git_commit(project_root: str | Path) -> str | None:
    """Return the current git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def current_git_branch(project_root: str | Path) -> str | None:
    """Return the current git branch when available."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def json_safe(value: Any) -> Any:
    """Convert common non-JSON values such as Path objects into JSON-safe forms."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return value


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, default=str), encoding="utf-8")
    return path


def write_run_readme(path: str | Path, run_manifest: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    """Write a small Markdown overview so users can inspect a run folder quickly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# N-body Simulation Run: `{run_manifest.get('run_id', 'unknown')}`")
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"- Started at: `{run_manifest.get('started_at', '')}`")
    if run_manifest.get("finished_at"):
        lines.append(f"- Finished at: `{run_manifest.get('finished_at')}`")
    lines.append(f"- Preset: `{run_manifest.get('preset', '')}`")
    lines.append(f"- Output root: `{run_manifest.get('run_output_root', '')}`")
    lines.append(f"- Raw CSV: `{run_manifest.get('raw_csv', '')}`")
    if run_manifest.get("git_branch"):
        lines.append(f"- Git branch: `{run_manifest.get('git_branch')}`")
    if run_manifest.get("git_commit"):
        lines.append(f"- Git commit: `{run_manifest.get('git_commit')}`")
    lines.append(f"- Python: `{run_manifest.get('python', '')}`")
    lines.append(f"- Platform: `{run_manifest.get('platform', '')}`")
    lines.append("")

    if run_manifest.get("common_overrides"):
        lines.append("## Common overrides")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(json_safe(run_manifest.get("common_overrides")), indent=2, default=str))
        lines.append("```")
        lines.append("")

    lines.append("## Experiments")
    lines.append("")
    if rows:
        lines.append("| Experiment | Solver | Expansion | Velocity | Particles | Steps | Softening | H0 | Theta | Runtime (s) | Output |")
        lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
        for row in rows:
            output_dir = str(row.get("output_dir", ""))
            rel_output = output_dir
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("experiment", "")),
                        str(row.get("force_solver", "")),
                        str(row.get("use_expansion", "")),
                        str(row.get("connection_velocity_mode", "")),
                        str(row.get("particles", "")),
                        str(row.get("steps", "")),
                        str(row.get("softening", "")),
                        str(row.get("H0", "")),
                        str(row.get("theta", "")),
                        f"{float(row.get('runtime_seconds') or 0):.3f}",
                        f"`{rel_output}`",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No experiments recorded yet.")
    lines.append("")
    lines.append("## Important files")
    lines.append("")
    lines.append("- `run_manifest.json`: metadata for the whole run")
    lines.append("- `run_index.json`: machine-readable experiment index")
    lines.append("- `run_index.csv`: spreadsheet-friendly experiment index")
    lines.append("- each experiment folder contains `summary.json`, `metrics.csv`, `positions.csv`, and optionally `animation.gif`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def rows_to_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([json_safe(row) for row in rows]).to_csv(path, index=False)
    return path


def base_run_manifest(
    *,
    run_id: str,
    run_output_root: str | Path,
    project_root: str | Path,
    preset: str | None = None,
    raw_csv: str | Path | None = None,
    command: list[str] | None = None,
    common_overrides: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    return {
        "run_id": run_id,
        "started_at": now_iso(),
        "finished_at": None,
        "preset": preset,
        "raw_csv": str(raw_csv) if raw_csv is not None else None,
        "run_output_root": str(run_output_root),
        "project_root": str(project_root),
        "command": command if command is not None else sys.argv,
        "common_overrides": json_safe(common_overrides or {}),
        "notes": notes,
        "git_branch": current_git_branch(project_root),
        "git_commit": current_git_commit(project_root),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
