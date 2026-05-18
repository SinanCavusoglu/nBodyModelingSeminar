import numpy as np


def compute_acceleration(
    pos: np.ndarray,
    mass: np.ndarray,
    gravitational_constant: float = 1.0,
    softening: float = 0.5,
) -> np.ndarray:
  
    acceleration = np.zeros_like(pos)
    particle_count = len(pos)

    for i in range(particle_count):
        difference = pos - pos[i]
        distance_squared = np.sum(difference**2, axis=1) + softening**2
        inverse_distance_cubed = distance_squared ** (-1.5)

        # A particle should not accelerate itself.
        inverse_distance_cubed[i] = 0.0

        acceleration[i] = gravitational_constant * np.sum(
            difference * mass[:, None] * inverse_distance_cubed[:, None],
            axis=0,
        )

    return acceleration