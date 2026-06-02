import pathlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def create_animation(
    positions_history: np.ndarray,
    mass: np.ndarray,
    title_prefix: str = "3D Billionaires N-body Simulation",
    interval_ms: int = 30,
    point_min_size: float = 10,
    point_max_extra_size: float = 200,
    camera_elevation: float = 25,
    camera_azimuth: float = 45,
    camera_rotation_speed: float = 0.0,
):
  
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    x_all = positions_history[:, :, 0]
    y_all = positions_history[:, :, 1]
    z_all = positions_history[:, :, 2]

    limit = max(abs(x_all).max(), abs(y_all).max(), abs(z_all).max())
    if limit == 0:
        limit = 1

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=camera_elevation, azim=camera_azimuth)

    size = mass / mass.max()
    size = point_min_size + size * point_max_extra_size

    scatter = ax.scatter(
        positions_history[0, :, 0],
        positions_history[0, :, 1],
        positions_history[0, :, 2],
        s=size,
        alpha=0.75,
    )

    def init():
        scatter._offsets3d = (
            positions_history[0, :, 0],
            positions_history[0, :, 1],
            positions_history[0, :, 2],
        )
        ax.set_title(f"{title_prefix} | Frame = 0")
        return (scatter,)

    def update(frame: int):
        x = positions_history[frame, :, 0]
        y = positions_history[frame, :, 1]
        z = positions_history[frame, :, 2]

        scatter._offsets3d = (x, y, z)
        ax.set_title(f"{title_prefix} | Frame = {frame}")

        if camera_rotation_speed != 0:
            ax.view_init(
                elev=camera_elevation,
                azim=camera_azimuth + frame * camera_rotation_speed,
            )

        return (scatter,)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(positions_history),
        init_func=init,
        interval=interval_ms,
        blit=False,
    )

    return fig, animation


def save_animation_gif(
    animation: FuncAnimation,
    output_path: str | pathlib.Path,
    fps: int,
    dpi: int,
):
    
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    animation.save(output_path, writer=writer, dpi=dpi)
