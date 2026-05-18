

import numpy as np

from src.physics import compute_acceleration


def run_simulation(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    gravitational_constant: float,
    dt: float,
    steps: int,
    softening: float,
    save_every: int,
):
    
    pos = pos.copy()
    vel = vel.copy()

    positions_history = []
    velocity_history = []

    acceleration = compute_acceleration(
        pos=pos,
        mass=mass,
        gravitational_constant=gravitational_constant,
        softening=softening,
    )

    for step in range(steps):
        vel += 0.5 * acceleration * dt
        pos += vel * dt
        acceleration = compute_acceleration(
            pos=pos,
            mass=mass,
            gravitational_constant=gravitational_constant,
            softening=softening,
        )
        vel += 0.5 * acceleration * dt

        if step % save_every == 0:
            positions_history.append(pos.copy())
            velocity_history.append(vel.copy())

        if step % 50 == 0:
            print(f"Step {step}/{steps}")

    return np.array(positions_history), np.array(velocity_history)
