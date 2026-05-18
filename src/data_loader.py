import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Name",
    "NetWorth_Billions",
    "X",
    "Y",
    "Z",
    "ForceBasedOnAge",
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
    inward_bias: float = 0.75,
    velocity_scale: float = 1.0,
    max_particles: int | None = None,
    random_seed: int | None = None,
):
   
    rng = np.random.default_rng(random_seed)

    df = pd.read_csv(csv_path)
    validate_columns(df)

    if max_particles is not None and len(df) > max_particles:
        print(f"Particle count is {len(df)}. Using the first {max_particles} particles.")
        df = df.head(max_particles).copy()

    names = df["Name"].astype(str).to_numpy()
    pos = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    mass = df["NetWorth_Billions"].to_numpy(dtype=float)
    speed = df["ForceBasedOnAge"].to_numpy(dtype=float) * velocity_scale

    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    mass = np.nan_to_num(mass, nan=1.0, posinf=1.0, neginf=1.0)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    mass[mass <= 0] = 1.0

    particle_count = len(pos)

    inward_direction = -pos.copy()
    inward_norm = np.linalg.norm(inward_direction, axis=1)
    inward_norm[inward_norm == 0] = 1e-12
    inward_direction = inward_direction / inward_norm[:, None]

    random_direction = rng.normal(size=(particle_count, 3))
    random_norm = np.linalg.norm(random_direction, axis=1)
    random_norm[random_norm == 0] = 1e-12
    random_direction = random_direction / random_norm[:, None]

    direction = inward_bias * inward_direction + (1.0 - inward_bias) * random_direction
    direction_norm = np.linalg.norm(direction, axis=1)
    direction_norm[direction_norm == 0] = 1e-12
    direction = direction / direction_norm[:, None]

    vel = direction * speed[:, None]

    return names, pos, vel, mass
