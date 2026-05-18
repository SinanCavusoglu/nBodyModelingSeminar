from pathlib import Path

import matplotlib.pyplot as plt

import config
from src.animation import create_animation, save_animation_gif
from src.data_loader import load_particles_from_csv
from src.export import export_positions_for_vvvv
from src.simulation import run_simulation


def main() -> None:
    csv_path = Path(config.CSV_PATH)
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            "Place your CSV file in the data folder or update CSV_PATH in config.py."
        )

    print("Loading particles...")
    names, initial_pos, initial_vel, mass = load_particles_from_csv(
        csv_path=str(csv_path),
        inward_bias=config.INWARD_BIAS,
        velocity_scale=config.VELOCITY_SCALE,
        max_particles=config.MAX_PARTICLES,
        random_seed=config.RANDOM_SEED,
    )

    print(f"Loaded particles: {len(names)}")
    print(f"Position shape: {initial_pos.shape}")
    print(f"Velocity shape: {initial_vel.shape}")
    print(f"Mass shape: {mass.shape}")

    print("Running simulation...")
    positions_history, velocity_history = run_simulation(
        pos=initial_pos,
        vel=initial_vel,
        mass=mass,
        gravitational_constant=config.G,
        dt=config.DT,
        steps=config.STEPS,
        softening=config.SOFTENING,
        save_every=config.SAVE_EVERY,
    )

    print("Simulation complete.")
    print(f"Saved frames: {len(positions_history)}")
    print(f"Position history shape: {positions_history.shape}")
    print(f"Velocity history shape: {velocity_history.shape}")

    if config.SAVE_VVVV_CSV:
        vvvv_path = output_dir / config.OUTPUT_VVVV_CSV_NAME
        export_positions_for_vvvv(
            positions_history=positions_history,
            velocity_history=velocity_history,
            names=names,
            mass=mass,
            output_path=vvvv_path,
        )
        print(f"vvvv CSV saved to: {vvvv_path}")

    if config.SAVE_GIF:
        print("Creating animation...")
        fig, animation = create_animation(
            positions_history=positions_history,
            mass=mass,
            title_prefix="3D Billionaires N-body Simulation",
            interval_ms=config.INTERVAL_MS,
            point_min_size=config.POINT_MIN_SIZE,
            point_max_extra_size=config.POINT_MAX_EXTRA_SIZE,
            camera_elevation=config.CAMERA_ELEVATION,
            camera_azimuth=config.CAMERA_AZIMUTH,
            camera_rotation_speed=config.CAMERA_ROTATION_SPEED,
        )

        if config.SAVE_GIF:
            gif_path = output_dir / config.OUTPUT_GIF_NAME
            print(f"Saving GIF to: {gif_path}")
            save_animation_gif(animation, gif_path, fps=config.FPS, dpi=config.DPI)
            print(f"GIF saved to: {gif_path}")

        
    print("Done.")


if __name__ == "__main__":
    main()
