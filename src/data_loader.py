"""CSV loading utilities for the Forbes N-body simulation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MASS_COLUMN_CANDIDATES = [
    "NetWorth_Billions",
    "NetWorth",
    "net_worth",
    "networth",
    "mass",
]

COLOR_COLUMN_CANDIDATES = [
    "hue color value",
    "hue_color_value",
    "hue",
    "hue_value",
    "color",
    "Color",
    "colour",
    "colour_value",
]


def _find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Missing required column. Expected one of: {candidates}")
    return None


def _normalize_id(value: object) -> str:
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


def _normalize_mass(values: np.ndarray) -> np.ndarray:
    mass = np.asarray(values, dtype=float)
    mass = np.nan_to_num(mass, nan=1.0, posinf=1.0, neginf=1.0)
    mass[mass <= 0] = 1.0
    finite = mass[np.isfinite(mass)]
    if len(finite) and float(np.nanmedian(finite)) > 1.0e6:
        mass = mass / 1.0e9
    mass[mass <= 0] = 1.0
    return mass


def load_bodies(csv_path: str | Path, max_particles: int | None = None) -> dict[str, Any]:
    """Load particle data from CSV.

    Required columns:
      ID, x, y, z, and a mass column.
      vx/vy/vz are preferred; if missing, zero velocities are used.
    Supported mass columns:
      NetWorth_Billions, NetWorth, net_worth, networth, mass
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if max_particles is not None and max_particles > 0:
        df = df.head(int(max_particles)).copy()

    id_col = _find_column(df, ["ID", "id"])
    name_col = _find_column(df, ["Name", "name"], required=False)
    mass_col = _find_column(df, MASS_COLUMN_CANDIDATES)
    color_col = _find_column(df, COLOR_COLUMN_CANDIDATES, required=False)

    for col in ["x", "y", "z"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")
    for col in ["vx", "vy", "vz"]:
        if col not in df.columns:
            print(f"[data_loader] Column '{col}' missing in {csv_path}; using zeros.")
            df[col] = 0.0

    ids = df[id_col].apply(_normalize_id).to_numpy()
    if name_col:
        names = df[name_col].fillna("").astype(str).to_numpy()
    else:
        names = ids.copy()

    pos = df[["x", "y", "z"]].to_numpy(dtype=float)
    vel = df[["vx", "vy", "vz"]].to_numpy(dtype=float)
    mass_raw = pd.to_numeric(df[mass_col], errors="coerce").to_numpy(dtype=float)

    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
    mass = _normalize_mass(mass_raw)

    colors = None
    if color_col is not None:
        # Keep visualization colors separate from the physics state. Numeric hue
        # values are used by animation.py with an HSV colormap. Non-numeric
        # values, such as hex strings, are preserved as strings.
        numeric_colors = pd.to_numeric(df[color_col], errors="coerce")
        if numeric_colors.notna().any():
            colors = numeric_colors.to_numpy(dtype=float)
        else:
            colors = df[color_col].fillna("").astype(str).to_numpy()

    return {
        "ids": ids,
        "names": names,
        "positions": pos,
        "velocities": vel,
        "masses": mass,
        "colors": colors,
        "color_column": color_col,
        "dataframe": df,
        "mass_column": mass_col,
    }
