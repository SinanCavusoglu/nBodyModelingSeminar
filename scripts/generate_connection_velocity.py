"""Generate radial or angular connection-based initial velocities.

This script keeps connections as initial-condition information only. It does not
create spring forces for the later simulation.

Examples:
  python scripts/generate_connection_velocity.py --mode radial
  python scripts/generate_connection_velocity.py --mode angular
  python scripts/generate_connection_velocity.py --mode none
  python scripts/generate_connection_velocity.py --mode angular --velocity-scale 0.01
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


ID_CANDIDATES = ["ID", "id"]
NAME_CANDIDATES = ["Name", "name"]
MASS_CANDIDATES = ["NetWorth_Billions", "NetWorth", "net_worth", "networth", "mass"]
CONNECTION_CANDIDATES = ["connections", "Connections", "connection", "connected_ids"]
INDUSTRY_CANDIDATES = ["industries", "Industries", "industry"]
COLOR_CANDIDATES = ["hue color value", "hue_color_value", "hue", "hue_value", "color", "Color", "colour"]


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Missing required column. Expected one of: {candidates}")
    return None



def scale_label(scale: float) -> str:
    """Filesystem-safe label for velocity scale values."""
    return str(float(scale)).replace(".", "p").replace("-", "m")


def default_output_dir(mode: str, scale: float, generated_root: Path) -> Path:
    """Return the default output directory for a velocity mode/scale pair.

    Backwards compatibility:
      radial scale 0.05  -> data/generated/radial/
      angular scale 0.05 -> data/generated/angular/
      none              -> data/generated/none/

    Oscillation experiments get scale-specific folders:
      radial scale 0.01  -> data/generated/radial_scale_0p01/
      angular scale 0.005 -> data/generated/angular_scale_0p005/
    """
    mode = (mode or "radial").lower()
    if mode == "none" or abs(float(scale)) <= 1e-15:
        return generated_root / "none"
    if mode in {"radial", "angular"} and abs(float(scale) - float(config.CONNECTION_VELOCITY_SCALE)) <= 1e-15:
        return generated_root / mode
    return generated_root / f"{mode}_scale_{scale_label(scale)}"

def normalize_id(value: object) -> str:
    """Normalize IDs so CSV values like 4.0 match connection values like 4."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def parse_connections(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    # Original data used '/', but this also accepts comma, semicolon, and pipes.
    parts = re.split(r"[/,;|]+", text)
    return [normalize_id(p) for p in parts if str(p).strip()]


def normalize_mass_values(values: np.ndarray) -> np.ndarray:
    """Convert raw net worth dollars to billions when necessary.

    The uploaded 20260610 CSV stores values like 701141591413.  For simulation
    stability and comparability with the old code, convert such values to
    billions. If the data already appears to be in billions, leave it unchanged.
    """
    masses = np.asarray(values, dtype=float)
    masses = np.nan_to_num(masses, nan=1.0, posinf=1.0, neginf=1.0)
    masses[masses <= 0] = 1.0
    finite = masses[np.isfinite(masses)]
    if len(finite) and float(np.nanmedian(finite)) > 1.0e6:
        masses = masses / 1.0e9
    masses[masses <= 0] = 1.0
    return masses


def normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return np.zeros_like(v, dtype=float)
    return v / norm


def tangential_direction(r: np.ndarray) -> np.ndarray:
    """Return a stable 3D tangential direction around a global axis."""
    r = np.asarray(r, dtype=float)
    axis = np.array([0.0, 0.0, 1.0])
    tang = np.cross(axis, r)
    if np.linalg.norm(tang) <= 1.0e-12:
        axis = np.array([0.0, 1.0, 0.0])
        tang = np.cross(axis, r)
    if np.linalg.norm(tang) <= 1.0e-12:
        axis = np.array([1.0, 0.0, 0.0])
        tang = np.cross(axis, r)
    return normalize(tang)


def compute_connected_barycenters(
    ids: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    connection_lists: list[list[str]],
) -> tuple[list[np.ndarray | None], np.ndarray]:
    id_to_index = {str(id_): i for i, id_ in enumerate(ids)}
    barycenters: list[np.ndarray | None] = []
    connected_mass_sums = np.zeros(len(ids), dtype=float)

    for i, connected_ids in enumerate(connection_lists):
        indices = [id_to_index[c] for c in connected_ids if c in id_to_index and id_to_index[c] != i]
        if not indices:
            barycenters.append(None)
            continue
        idx = np.asarray(indices, dtype=int)
        m = masses[idx]
        total_m = float(np.sum(m))
        connected_mass_sums[i] = total_m
        if total_m <= 0:
            barycenters.append(np.mean(positions[idx], axis=0))
        else:
            barycenters.append(np.sum(positions[idx] * m[:, None], axis=0) / total_m)
    return barycenters, connected_mass_sums


def compute_connection_velocities(
    ids: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    connection_lists: list[list[str]],
    mode: str,
    scale: float,
    use_mass_strength: bool = True,
) -> np.ndarray:
    mode = (mode or "radial").lower()
    velocities = np.zeros_like(positions, dtype=float)
    barycenters, connected_mass_sums = compute_connected_barycenters(ids, positions, masses, connection_lists)

    max_connected_mass = float(np.max(connected_mass_sums)) if len(connected_mass_sums) else 0.0

    for i, bary in enumerate(barycenters):
        if bary is None or mode == "none":
            continue

        if mode == "radial":
            direction = normalize(bary - positions[i])
        elif mode == "angular":
            r = positions[i] - bary
            direction = tangential_direction(r)
        else:
            raise ValueError("mode must be one of: radial, angular, none")

        strength = 1.0
        if use_mass_strength and max_connected_mass > 0:
            strength = np.log1p(connected_mass_sums[i]) / np.log1p(max_connected_mass)
        velocities[i] = direction * scale * strength

    return velocities


def build_edges(ids: np.ndarray, connection_lists: list[list[str]]) -> pd.DataFrame:
    valid = set(map(str, ids))
    rows = []
    for source, conns in zip(ids, connection_lists):
        s = str(source)
        for target in conns:
            t = str(target)
            if t in valid and t != s:
                rows.append({"source": s, "target": t})
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["source", "target"])


def generate(
    input_path: Path,
    output_dir: Path,
    mode: str,
    scale: float,
    use_mass_strength: bool,
) -> dict[str, Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    id_col = find_column(df, ID_CANDIDATES)
    name_col = find_column(df, NAME_CANDIDATES, required=False)
    mass_col = find_column(df, MASS_CANDIDATES)
    conn_col = find_column(df, CONNECTION_CANDIDATES, required=False)
    industry_col = find_column(df, INDUSTRY_CANDIDATES, required=False)
    color_col = find_column(df, COLOR_CANDIDATES, required=False)

    for col in ["x", "y", "z"]:
        if col not in df.columns:
            raise ValueError(f"Missing required position column: {col}")

    working = df.copy()
    working[id_col] = working[id_col].apply(normalize_id)
    if name_col is None:
        working["Name"] = working[id_col]
        name_col = "Name"
    if conn_col is None:
        working["connections"] = ""
        conn_col = "connections"
    if industry_col is None:
        working["industries"] = ""
        industry_col = "industries"

    positions = working[["x", "y", "z"]].to_numpy(dtype=float)
    positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
    mass_raw = pd.to_numeric(working[mass_col], errors="coerce").to_numpy(dtype=float)
    masses = normalize_mass_values(mass_raw)
    ids = working[id_col].apply(normalize_id).to_numpy()
    connection_lists = [parse_connections(v) for v in working[conn_col]]

    velocities = compute_connection_velocities(
        ids=ids,
        positions=positions,
        masses=masses,
        connection_lists=connection_lists,
        mode=mode,
        scale=scale,
        use_mass_strength=use_mass_strength,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    minimal_path = output_dir / "minimal.csv"
    rich_path = output_dir / "rich.csv"
    edges_path = output_dir / "edges.csv"

    out = pd.DataFrame(
        {
            "ID": ids,
            "Name": working[name_col].fillna("").astype(str),
            "NetWorth_Billions": masses,
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "vx": velocities[:, 0],
            "vy": velocities[:, 1],
            "vz": velocities[:, 2],
            "industries": working[industry_col].fillna("").astype(str),
            "connections": working[conn_col].fillna("").astype(str),
            "velocity_mode": mode,
            "velocity_scale": float(scale),
        }
    )
    if color_col is not None:
        out["hue color value"] = working[color_col]
    out.to_csv(minimal_path, index=False)

    rich = working.copy()
    rich["vx"] = velocities[:, 0]
    rich["vy"] = velocities[:, 1]
    rich["vz"] = velocities[:, 2]
    rich["velocity_mode"] = mode
    rich["velocity_scale"] = float(scale)
    rich.to_csv(rich_path, index=False)

    edges = build_edges(ids, connection_lists)
    edges.to_csv(edges_path, index=False)

    speeds = np.linalg.norm(velocities, axis=1)
    print(f"[generate] mode={mode} scale={scale}")
    print(f"[generate] speed mean={float(np.mean(speeds)):.6g} max={float(np.max(speeds)):.6g}")
    print(f"[generate] minimal: {minimal_path}")
    print(f"[generate] rich:    {rich_path}")
    print(f"[generate] edges:   {edges_path}")
    return {"minimal": minimal_path, "rich": rich_path, "edges": edges_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate connection-based initial velocity CSVs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=config.RAW_CSV_PATH,
    )
    parser.add_argument("--mode", choices=["radial", "angular", "none"], default=config.CONNECTION_VELOCITY_MODE)
    parser.add_argument("--scale", type=float, default=None, help="Initial velocity scale multiplier.")
    parser.add_argument("--velocity-scale", type=float, default=None, help="Alias for --scale.")
    parser.add_argument("--no-mass-strength", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scale = args.velocity_scale if args.velocity_scale is not None else args.scale
    if scale is None:
        scale = config.CONNECTION_VELOCITY_SCALE
    output_dir = args.output_dir or default_output_dir(args.mode, float(scale), config.GENERATED_DATA_DIR)
    generate(
        input_path=args.input,
        output_dir=output_dir,
        mode=args.mode,
        scale=float(scale),
        use_mass_strength=not args.no_mass_strength,
    )


if __name__ == "__main__":
    main()
