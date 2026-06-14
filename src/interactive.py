"""Interactive Plotly HTML export for experiment outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _normalize_sizes(mass: np.ndarray, min_size: float = 4.0, max_size: float = 24.0) -> np.ndarray:
    """Return marker sizes scaled from particle mass using log scaling."""
    mass = np.asarray(mass, dtype=float)
    if mass.size == 0:
        return mass
    values = np.log1p(np.maximum(mass, 0.0))
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.full_like(values, (min_size + max_size) / 2.0, dtype=float)
    return min_size + (values - vmin) / (vmax - vmin) * (max_size - min_size)


def _color_values(frame_df: pd.DataFrame) -> Any:
    """Return marker colors from positions.csv.

    The pipeline stores `hue color value` as the `color` column in positions.csv.
    Plotly can accept numeric values or CSS color strings. Numeric values are
    rendered with a colorscale.
    """
    if "color" not in frame_df.columns:
        return "#1f77b4"

    color_series = frame_df["color"]
    numeric = pd.to_numeric(color_series, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=float)
    return color_series.astype(str).to_list()


def export_interactive_3d_html(
    positions_csv: str | Path,
    output_html: str | Path,
    title: str | None = None,
    max_frames: int = 120,
    max_particles: int | None = 300,
    include_plotlyjs: bool | str = True,
) -> Path:
    """Create a standalone interactive 3D Plotly animation.

    Parameters
    ----------
    positions_csv:
        Path to an experiment `positions.csv`.
    output_html:
        Destination HTML path. Usually next to `animation.gif`.
    title:
        Plot title / experiment name.
    max_frames:
        Maximum number of animation frames embedded in the HTML. This keeps the
        browser and file size manageable for long simulations.
    max_particles:
        Maximum number of particles embedded in the HTML. The most massive
        particles from the first frame are kept. Use None to include all.
    include_plotlyjs:
        True creates a self-contained offline HTML file. Use "cdn" for a smaller
        file that needs internet access.
    """
    positions_csv = Path(positions_csv)
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    if not positions_csv.exists():
        raise FileNotFoundError(f"positions.csv not found: {positions_csv}")

    df = pd.read_csv(positions_csv)
    required = {"frame", "time", "id", "name", "x", "y", "z", "mass", "speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{positions_csv} is missing required columns: {sorted(missing)}")

    df["id"] = df["id"].astype(str)

    if max_particles is not None and max_particles > 0:
        first_frame = df["frame"].min()
        first = df[df["frame"] == first_frame].copy()
        top_ids = (
            first.sort_values("mass", ascending=False)
            .head(int(max_particles))["id"]
            .astype(str)
            .tolist()
        )
        df = df[df["id"].isin(set(top_ids))].copy()

    frames_sorted = sorted(df["frame"].unique().tolist())
    if not frames_sorted:
        raise ValueError(f"No frames found in {positions_csv}")

    if max_frames is not None and max_frames > 0 and len(frames_sorted) > max_frames:
        selected_indices = np.linspace(0, len(frames_sorted) - 1, int(max_frames)).round().astype(int)
        selected_frames = {frames_sorted[i] for i in selected_indices}
        df = df[df["frame"].isin(selected_frames)].copy()
        frames_sorted = sorted(df["frame"].unique().tolist())

    # Stable global cube range so the scene does not rescale while playing.
    xmin, xmax = float(df["x"].min()), float(df["x"].max())
    ymin, ymax = float(df["y"].min()), float(df["y"].max())
    zmin, zmax = float(df["z"].min()), float(df["z"].max())
    cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    radius = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2.0
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    radius *= 1.15

    x_range = [cx - radius, cx + radius]
    y_range = [cy - radius, cy + radius]
    z_range = [cz - radius, cz + radius]

    experiment_title = title or output_html.parent.name

    def make_trace(frame_df: pd.DataFrame) -> go.Scatter3d:
        sizes = _normalize_sizes(frame_df["mass"].to_numpy(dtype=float))
        color_values = _color_values(frame_df)
        marker: dict[str, Any] = {
            "size": sizes,
            "color": color_values,
            "opacity": 0.85,
            "line": {"width": 0},
        }
        # If colors are numeric, use a continuous colorscale and keep it visible.
        if not isinstance(color_values, str):
            arr = np.asarray(color_values)
            if np.issubdtype(arr.dtype, np.number):
                marker.update({"colorscale": "Turbo", "showscale": True, "colorbar": {"title": "color"}})

        hover_text = [
            (
                f"<b>{name}</b><br>"
                f"ID: {pid}<br>"
                f"Mass: {mass:.4f}<br>"
                f"Speed: {speed:.6f}<br>"
                f"Time: {time:.4f}"
            )
            for name, pid, mass, speed, time in zip(
                frame_df["name"].astype(str),
                frame_df["id"].astype(str),
                frame_df["mass"].astype(float),
                frame_df["speed"].astype(float),
                frame_df["time"].astype(float),
            )
        ]

        return go.Scatter3d(
            x=frame_df["x"],
            y=frame_df["y"],
            z=frame_df["z"],
            mode="markers",
            marker=marker,
            text=hover_text,
            hoverinfo="text",
        )

    first_frame = frames_sorted[0]
    first_df = df[df["frame"] == first_frame].copy()

    plotly_frames: list[go.Frame] = []
    for frame_id in frames_sorted:
        frame_df = df[df["frame"] == frame_id].copy()
        time_value = float(frame_df["time"].iloc[0]) if len(frame_df) else 0.0
        plotly_frames.append(
            go.Frame(
                data=[make_trace(frame_df)],
                name=str(frame_id),
                layout=go.Layout(
                    title=f"{experiment_title} — frame {frame_id}, t={time_value:.3f}"
                ),
            )
        )

    fig = go.Figure(data=[make_trace(first_df)], frames=plotly_frames)

    slider_steps = [
        {
            "method": "animate",
            "args": [
                [str(frame_id)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
            "label": str(frame_id),
        }
        for frame_id in frames_sorted
    ]

    fig.update_layout(
        title=experiment_title,
        scene={
            "xaxis": {"title": "x", "range": x_range},
            "yaxis": {"title": "y", "range": y_range},
            "zaxis": {"title": "z", "range": z_range},
            "aspectmode": "cube",
        },
        showlegend=False,
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 80, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": slider_steps,
                "x": 0.1,
                "y": 0,
                "len": 0.85,
                "currentvalue": {"prefix": "Frame: "},
            }
        ],
    )

    fig.write_html(str(output_html), include_plotlyjs=include_plotlyjs, full_html=True)
    return output_html
