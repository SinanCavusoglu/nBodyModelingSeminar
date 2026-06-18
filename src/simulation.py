"""Time integration for the Forbes N-body simulation."""
from __future__ import annotations

from typing import Any

import numpy as np

from .cosmology import expansion_rate as compute_H
from .cosmology import scale_factor as compute_a
from .metrics import compute_metrics, radius_stats
from .physics import compute_accelerations


def _cfg(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config.get(key, default)


def run_simulation(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the selected simulation and return histories + metrics."""
    pos = np.asarray(positions, dtype=float).copy()
    vel = np.asarray(velocities, dtype=float).copy()
    mass = np.asarray(masses, dtype=float).copy()

    dt = float(_cfg(config, "DT", 0.02))
    steps = int(_cfg(config, "STEPS", 3000))
    save_every = max(1, int(_cfg(config, "SAVE_EVERY", 3)))

    G = float(_cfg(config, "G", 0.005))
    softening = float(_cfg(config, "SOFTENING", 2.0))
    force_solver = str(_cfg(config, "FORCE_SOLVER", "direct"))
    use_expansion = bool(_cfg(config, "USE_EXPANSION", False))
    expansion_model = str(_cfg(config, "EXPANSION_MODEL", "linear"))
    H0 = float(_cfg(config, "H0", 0.01))

    bh_theta = float(_cfg(config, "BARNES_HUT_THETA", 0.5))
    bh_leaf = int(_cfg(config, "BARNES_HUT_MAX_PARTICLES_PER_LEAF", 8))
    bh_depth = int(_cfg(config, "BARNES_HUT_MAX_DEPTH", 32))
    bh_impl = str(_cfg(config, "BARNES_HUT_IMPLEMENTATION", "fast"))
    bh_use_numba = bool(_cfg(config, "BARNES_HUT_USE_NUMBA", True))
    bh_fallback_n = int(_cfg(config, "BARNES_HUT_DIRECT_FALLBACK_N", 512))

    save_metrics = bool(_cfg(config, "SAVE_METRICS", True))
    exact_potential_max_n = int(_cfg(config, "METRICS_EXACT_POTENTIAL_MAX_N", 700))
    nearest_neighbor_max_n = int(_cfg(config, "METRICS_NEAREST_NEIGHBOR_MAX_N", 1500))
    potential_scaling = str(_cfg(config, "METRICS_POTENTIAL_SCALING", "a3"))
    collapse_fraction = float(_cfg(config, "COLLAPSE_RADIUS_FRACTION", 0.3))

    pos_history: list[np.ndarray] = []
    vel_history: list[np.ndarray] = []
    time_history: list[float] = []
    metrics_history: list[dict[str, Any]] = []

    initial_mean_radius = radius_stats(pos, mass)["mean_radius"]

    def save_frame(step: int, t: float) -> None:
        pos_history.append(pos.copy())
        vel_history.append(vel.copy())
        time_history.append(float(t))
        if save_metrics:
            a_t = compute_a(t, H0, expansion_model) if use_expansion else 1.0
            H_t = compute_H(t, H0, expansion_model) if use_expansion else 0.0
            metrics_history.append(
                compute_metrics(
                    frame=len(pos_history) - 1,
                    time=t,
                    pos=pos,
                    vel=vel,
                    mass=mass,
                    G=G,
                    softening=softening,
                    scale_factor=a_t,
                    expansion_rate=H_t,
                    exact_potential_max_n=exact_potential_max_n,
                    nearest_neighbor_max_n=nearest_neighbor_max_n,
                    potential_scaling=potential_scaling,
                    initial_mean_radius=initial_mean_radius,
                    collapse_radius_fraction=collapse_fraction,
                )
            )

    t = 0.0
    save_frame(step=0, t=t)

    for step in range(1, steps + 1):
        acc = compute_accelerations(
            pos=pos,
            vel=vel,
            mass=mass,
            t=t,
            G=G,
            softening=softening,
            force_solver=force_solver,
            use_expansion=use_expansion,
            H0=H0,
            expansion_model=expansion_model,
            barnes_hut_theta=bh_theta,
            barnes_hut_max_particles_per_leaf=bh_leaf,
            barnes_hut_max_depth=bh_depth,
            barnes_hut_implementation=bh_impl,
            barnes_hut_use_numba=bh_use_numba,
            barnes_hut_direct_fallback_n=bh_fallback_n,
        )

        # Velocity-Verlet style update. Because expansion damping depends on
        # velocity, this is an approximation but keeps the existing structure.
        vel_half = vel + 0.5 * dt * acc
        pos = pos + dt * vel_half
        t = step * dt

        acc_new = compute_accelerations(
            pos=pos,
            vel=vel_half,
            mass=mass,
            t=t,
            G=G,
            softening=softening,
            force_solver=force_solver,
            use_expansion=use_expansion,
            H0=H0,
            expansion_model=expansion_model,
            barnes_hut_theta=bh_theta,
            barnes_hut_max_particles_per_leaf=bh_leaf,
            barnes_hut_max_depth=bh_depth,
            barnes_hut_implementation=bh_impl,
            barnes_hut_use_numba=bh_use_numba,
            barnes_hut_direct_fallback_n=bh_fallback_n,
        )
        vel = vel_half + 0.5 * dt * acc_new

        if step % save_every == 0 or step == steps:
            save_frame(step=step, t=t)

    return {
        "positions": np.asarray(pos_history),
        "velocities": np.asarray(vel_history),
        "times": np.asarray(time_history),
        "metrics": metrics_history,
        "final_positions": pos,
        "final_velocities": vel,
    }
