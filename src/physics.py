"""Acceleration models for the Forbes N-body simulation."""
from __future__ import annotations

import numpy as np

from .cosmology import expansion_rate, scale_factor


def compute_direct_softened_accelerations(
    pos: np.ndarray,
    mass: np.ndarray,
    G: float,
    softening: float,
) -> np.ndarray:
    """Exact O(N^2) softened gravitational acceleration.

    acc_i = sum_j G*m_j*(x_j-x_i)/(||x_j-x_i||^2 + eps^2)^(3/2)
    """
    pos = np.asarray(pos, dtype=float)
    mass = np.asarray(mass, dtype=float)
    n = len(pos)
    acc = np.zeros_like(pos, dtype=float)
    eps2 = float(softening) ** 2

    for i in range(n):
        diff = pos - pos[i]
        dist2 = np.einsum("ij,ij->i", diff, diff) + eps2
        inv_dist3 = np.zeros(n, dtype=float)
        valid = dist2 > 0
        inv_dist3[valid] = dist2[valid] ** -1.5
        inv_dist3[i] = 0.0
        weights = mass * inv_dist3
        acc[i] = G * np.sum(diff * weights[:, None], axis=0)
    return acc


def compute_gravity_accelerations(
    pos: np.ndarray,
    mass: np.ndarray,
    G: float,
    softening: float,
    force_solver: str = "direct",
    barnes_hut_theta: float = 0.5,
    barnes_hut_max_particles_per_leaf: int = 1,
    barnes_hut_max_depth: int = 32,
) -> np.ndarray:
    """Compute gravity acceleration using either direct or Barnes-Hut solver."""
    solver = (force_solver or "direct").lower()
    if solver == "direct":
        return compute_direct_softened_accelerations(pos, mass, G, softening)
    if solver in {"barnes_hut", "barnes-hut", "bh"}:
        from .barnes_hut import compute_barnes_hut_accelerations

        return compute_barnes_hut_accelerations(
            pos,
            mass,
            G=G,
            softening=softening,
            theta=barnes_hut_theta,
            max_particles_per_leaf=barnes_hut_max_particles_per_leaf,
            max_depth=barnes_hut_max_depth,
        )
    raise ValueError(f"Unknown FORCE_SOLVER: {force_solver}")


def apply_expansion_terms(
    gravity_acc: np.ndarray,
    vel: np.ndarray,
    t: float,
    use_expansion: bool,
    H0: float,
    expansion_model: str,
) -> tuple[np.ndarray, float, float]:
    """Apply comoving expansion scaling and damping.

    Sebastian's proposed model:
        x_i'' + 2H(t)x_i' = gravity/a(t)^3

    Therefore:
        x_i'' = gravity/a(t)^3 - 2H(t)x_i'
    """
    if not use_expansion:
        return gravity_acc, 1.0, 0.0

    a_t = scale_factor(t, H0, expansion_model)
    H_t = expansion_rate(t, H0, expansion_model)
    acc = gravity_acc / (a_t ** 3) - 2.0 * H_t * vel
    return acc, a_t, H_t


def compute_accelerations(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    t: float,
    G: float,
    softening: float,
    force_solver: str = "direct",
    use_expansion: bool = False,
    H0: float = 0.01,
    expansion_model: str = "linear",
    barnes_hut_theta: float = 0.5,
    barnes_hut_max_particles_per_leaf: int = 1,
    barnes_hut_max_depth: int = 32,
) -> np.ndarray:
    """Compute total acceleration for the selected model."""
    gravity_acc = compute_gravity_accelerations(
        pos=pos,
        mass=mass,
        G=G,
        softening=softening,
        force_solver=force_solver,
        barnes_hut_theta=barnes_hut_theta,
        barnes_hut_max_particles_per_leaf=barnes_hut_max_particles_per_leaf,
        barnes_hut_max_depth=barnes_hut_max_depth,
    )
    acc, _, _ = apply_expansion_terms(
        gravity_acc,
        vel,
        t=t,
        use_expansion=use_expansion,
        H0=H0,
        expansion_model=expansion_model,
    )
    return acc
