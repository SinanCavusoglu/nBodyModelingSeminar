"""Create interactive_3d.html for an existing experiment folder."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.interactive import export_interactive_3d_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interactive_3d.html next to an experiment GIF.")
    parser.add_argument("experiment_dir", type=Path, help="Folder containing positions.csv")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--max-particles", type=int, default=300)
    parser.add_argument("--cdn", action="store_true", help="Use Plotly CDN instead of embedding Plotly JS. Smaller file, needs internet.")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    positions_csv = experiment_dir / "positions.csv"
    output_html = experiment_dir / "interactive_3d.html"

    path = export_interactive_3d_html(
        positions_csv=positions_csv,
        output_html=output_html,
        title=experiment_dir.name,
        max_frames=args.max_frames,
        max_particles=args.max_particles,
        include_plotlyjs="cdn" if args.cdn else True,
    )
    print(f"Created: {path}")


if __name__ == "__main__":
    main()
