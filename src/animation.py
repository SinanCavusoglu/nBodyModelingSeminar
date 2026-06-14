"""Matplotlib animation helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _point_sizes(masses: np.ndarray, min_size: float, max_size: float) -> np.ndarray:
    masses = np.asarray(masses, dtype=float)
    if len(masses) == 0:
        return masses
    m = np.log1p(np.maximum(masses, 0))
    if float(np.max(m) - np.min(m)) < 1e-12:
        return np.full_like(m, 0.5 * (min_size + max_size), dtype=float)
    scaled = (m - np.min(m)) / (np.max(m) - np.min(m))
    return min_size + scaled * (max_size - min_size)




def _normalize_visual_colors(colors: np.ndarray | None, n_particles: int):
    """Return a Matplotlib-compatible color payload for fixed particle colors.

    Numeric arrays are interpreted as hue/color values and normalized for a
    stable HSV mapping. String arrays are passed through, which supports hex
    colors if the input data provides them. Returning ``None`` lets Matplotlib
    use its default color.
    """
    if colors is None:
        return None, {}
    arr = np.asarray(colors)
    if len(arr) != n_particles:
        return None, {}

    # Numeric hue values, e.g. the uploaded CSV column "hue color value".
    try:
        numeric = arr.astype(float)
        numeric = np.nan_to_num(numeric, nan=float(np.nanmedian(numeric)) if np.isfinite(numeric).any() else 0.0)
        cmin = float(np.min(numeric))
        cmax = float(np.max(numeric))
        if abs(cmax - cmin) < 1.0e-12:
            normalized = np.zeros_like(numeric, dtype=float)
        else:
            normalized = (numeric - cmin) / (cmax - cmin)
        return normalized, {"cmap": "hsv", "vmin": 0.0, "vmax": 1.0}
    except (TypeError, ValueError):
        pass

    # String colors such as #RRGGBB. Empty values are ignored by falling back.
    string_colors = arr.astype(str)
    if any(c.strip() for c in string_colors):
        return string_colors, {}
    return None, {}


def save_gif(
    positions: np.ndarray,
    masses: np.ndarray,
    output_path: str | Path,
    colors: np.ndarray | None = None,
    fps: int = 20,
    dpi: int = 120,
    title: str | None = None,
    point_size_min: float = 8,
    point_size_max: float = 120,
) -> Path:
    """Save a 3D scatter GIF for one simulation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions = np.asarray(positions, dtype=float)
    sizes = _point_sizes(masses, point_size_min, point_size_max)
    color_values, color_kwargs = _normalize_visual_colors(colors, positions.shape[1])

    mins = np.nanmin(positions.reshape(-1, 3), axis=0)
    maxs = np.nanmax(positions.reshape(-1, 3), axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    radius *= 1.05

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    initial = positions[0]
    if len(sizes) != len(initial):
        sizes_for_plot = float(np.nanmean(sizes)) if len(sizes) else point_size_min
    else:
        sizes_for_plot = sizes
    scatter_kwargs = dict(s=sizes_for_plot, alpha=0.75)
    if color_values is not None:
        scatter_kwargs.update({"c": color_values})
        scatter_kwargs.update(color_kwargs)
    scat = ax.scatter(initial[:, 0], initial[:, 1], initial[:, 2], **scatter_kwargs)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    def update(frame: int):
        p = positions[frame]
        scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        ax.set_title(f"{title or 'N-body simulation'} | frame {frame}")
        return (scat,)

    anim = FuncAnimation(fig, update, frames=len(positions), interval=1000 / max(1, fps), blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output_path


def save_comparison_gif(
    histories: dict[str, np.ndarray],
    masses: np.ndarray,
    output_path: str | Path,
    colors: np.ndarray | None = None,
    fps: int = 20,
    dpi: int = 120,
) -> Path:
    """Save a simple 2x2 comparison GIF for up to four experiments."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(histories.keys())[:4]
    arrays = [np.asarray(histories[name], dtype=float) for name in names]
    frame_count = min(len(a) for a in arrays)
    all_positions = np.concatenate([a[:frame_count].reshape(-1, 3) for a in arrays], axis=0)
    mins = np.nanmin(all_positions, axis=0)
    maxs = np.nanmax(all_positions, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    radius *= 1.05

    sizes = _point_sizes(masses, 8, 80)
    first_n = arrays[0].shape[1] if arrays else 0
    color_values, color_kwargs = _normalize_visual_colors(colors, first_n)
    fig = plt.figure(figsize=(10, 9))
    axes = []
    scatters = []
    for idx, name in enumerate(names):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        initial = arrays[idx][0]
        if len(sizes) != len(initial):
            sizes_for_plot = float(np.nanmean(sizes)) if len(sizes) else 20.0
        else:
            sizes_for_plot = sizes
        scatter_kwargs = dict(s=sizes_for_plot, alpha=0.75)
        if color_values is not None and len(color_values) == len(initial):
            scatter_kwargs.update({"c": color_values})
            scatter_kwargs.update(color_kwargs)
        scat = ax.scatter(initial[:, 0], initial[:, 1], initial[:, 2], **scatter_kwargs)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_title(name)
        axes.append(ax)
        scatters.append(scat)

    def update(frame: int):
        artists = []
        for ax, scat, arr, name in zip(axes, scatters, arrays, names):
            p = arr[frame]
            scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
            ax.set_title(f"{name} | frame {frame}")
            artists.append(scat)
        return artists

    anim = FuncAnimation(fig, update, frames=frame_count, interval=1000 / max(1, fps), blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output_path
