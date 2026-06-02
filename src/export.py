from pathlib import Path

import numpy as np
import pandas as pd


def export_positions_for_vvvv(
    positions_history: np.ndarray,
    velocity_history: np.ndarray,
    ids: np.ndarray,
    names: np.ndarray,
    mass: np.ndarray,
    output_path: str | Path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if positions_history.shape != velocity_history.shape:
        raise ValueError(
            "positions_history and velocity_history must have the same shape. "
            f"Got {positions_history.shape} and {velocity_history.shape}."
        )

    frame_count = positions_history.shape[0]
    particle_count = positions_history.shape[1]

    if len(ids) != particle_count:
        raise ValueError(
            f"ids length must match particle count. "
            f"Got {len(ids)} ids and {particle_count} particles."
        )

    if len(names) != particle_count:
        raise ValueError(
            f"names length must match particle count. "
            f"Got {len(names)} names and {particle_count} particles."
        )

    if len(mass) != particle_count:
        raise ValueError(
            f"mass length must match particle count. "
            f"Got {len(mass)} mass values and {particle_count} particles."
        )

    rows = []

    for frame_idx in range(frame_count):
        for particle_idx in range(particle_count):
            velocity = velocity_history[frame_idx, particle_idx]

            rows.append(
                {
                    "frame": frame_idx,
                    "id": int(ids[particle_idx]),
                    "name": names[particle_idx],
                    "x": positions_history[frame_idx, particle_idx, 0],
                    "y": positions_history[frame_idx, particle_idx, 1],
                    "z": positions_history[frame_idx, particle_idx, 2],
                    "mass": mass[particle_idx],
                    "speed": float(np.linalg.norm(velocity)),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return output_path


def export_edges_for_vvvv(
    edges_csv_path: str | Path,
    selected_ids: np.ndarray,
    output_path: str | Path,
):
    edges_csv_path = Path(edges_csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not edges_csv_path.exists():
        raise FileNotFoundError(
            f"Edge CSV file not found: {edges_csv_path}\n"
            "Run scripts/generate_connection_velocity.py first."
        )

    edges_df = pd.read_csv(edges_csv_path)

    required_columns = ["source_id", "target_id"]
    missing = [column for column in required_columns if column not in edges_df.columns]

    if missing:
        raise ValueError(
            "The edge CSV file is missing the following required columns: "
            + ", ".join(missing)
        )

    selected_id_set = set(int(x) for x in selected_ids)

    edges_df["source_id"] = pd.to_numeric(edges_df["source_id"], errors="coerce")
    edges_df["target_id"] = pd.to_numeric(edges_df["target_id"], errors="coerce")

    edges_df = edges_df.dropna(subset=["source_id", "target_id"]).copy()
    edges_df["source_id"] = edges_df["source_id"].astype(int)
    edges_df["target_id"] = edges_df["target_id"].astype(int)

    filtered_edges_df = edges_df[
        edges_df["source_id"].isin(selected_id_set)
        & edges_df["target_id"].isin(selected_id_set)
    ].copy()

    filtered_edges_df = filtered_edges_df.drop_duplicates(
        subset=["source_id", "target_id"]
    )

    filtered_edges_df.to_csv(output_path, index=False)

    return output_path, len(filtered_edges_df)