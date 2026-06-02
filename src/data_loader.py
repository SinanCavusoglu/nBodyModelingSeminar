import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "ID",
    "Name",
    "NetWorth_Billions",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(
            "The CSV file is missing the following required columns: "
            + ", ".join(missing)
        )


def load_particles_from_csv(
    csv_path: str,
    velocity_scale: float = 1.0,
    max_particles: int | None = None,
):
    df = pd.read_csv(csv_path)
    validate_columns(df)

    if max_particles is not None and len(df) > max_particles:
        print(f"Particle count is {len(df)}. Using the first {max_particles} particles.")
        df = df.head(max_particles).copy()

    ids = df["ID"].to_numpy(dtype=int)
    names = df["Name"].astype(str).to_numpy()

    pos = df[["x", "y", "z"]].to_numpy(dtype=float)
    vel = df[["vx", "vy", "vz"]].to_numpy(dtype=float) * velocity_scale
    mass = df["NetWorth_Billions"].to_numpy(dtype=float)

    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
    mass = np.nan_to_num(mass, nan=1.0, posinf=1.0, neginf=1.0)

    mass[mass <= 0] = 1.0

    return ids, names, pos, vel, mass