from pathlib import Path

import numpy as np
import pandas as pd


def export_positions_for_vvvv(
    positions_history: np.ndarray,
    velocity_history: np.ndarray,
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

    rows = []
    for frame_idx in range(frame_count):
        for particle_id in range(particle_count):
            velocity = velocity_history[frame_idx, particle_id]
            rows.append(
                {
                    "frame": frame_idx,
                    "id": particle_id,
                    "name": names[particle_id],
                    "x": positions_history[frame_idx, particle_id, 0],
                    "y": positions_history[frame_idx, particle_id, 1],
                    "z": positions_history[frame_idx, particle_id, 2],
                    "mass": mass[particle_id],
                    "speed": float(np.linalg.norm(velocity)),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path
