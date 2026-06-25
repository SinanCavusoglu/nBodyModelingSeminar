"""Acceleration models for the Forbes N-body simulation."""
from __future__ import annotations

import numpy as np

from .cosmology import expansion_rate, scale_factor


def nearest_neighbor_distances(pos: np.ndarray, block_size: int = 512) -> np.ndarray:
    """Return each particle's nearest-neighbor distance.

    This is a block-wise NumPy implementation so it avoids allocating the full
    N x N x 3 distance tensor at once. It is still O(N^2), so adaptive softening
    should usually be updated every few integration steps rather than every step.
    """
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.full(1, np.inf, dtype=float)

    block_size = max(1, int(block_size))
    nearest2 = np.full(n, np.inf, dtype=float)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = pos[start:end]
        diff = block[:, None, :] - pos[None, :, :]
        dist2 = np.einsum("bij,bij->bi", diff, diff)
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        dist2[rows, cols] = np.inf
        nearest2[start:end] = np.min(dist2, axis=1)

    nearest = np.sqrt(nearest2)
    nearest[~np.isfinite(nearest)] = 0.0
    return nearest


def adaptive_softening_lengths(
    pos: np.ndarray,
    base_softening: float,
    mode: str = "density_boost",
    k: float = 1.0,
    min_softening: float = 0.5,
    max_softening: float = 30.0,
    block_size: int = 512,
) -> np.ndarray:
    """Compute per-particle adaptive softening lengths.

    Modes
    -----
    density_boost:
        Keeps the global base softening as the minimum behavior and increases
        epsilon in locally dense regions using nearest-neighbor distance:
            eps_i = base * (1 + k * max(0, median(d_NN)/d_NN_i - 1))
        This is the most useful mode for collapse-prevention visuals.

    nearest_neighbor:
        Direct report-style formula:
            eps_i = k * d_NN_i
        Values are still clamped between min_softening and max_softening.
    """
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    if n == 0:
        return np.zeros(0, dtype=float)

    base = float(base_softening)
    min_s = float(min_softening)
    max_s = float(max_softening)
    if max_s < min_s:
        min_s, max_s = max_s, min_s

    d_nn = nearest_neighbor_distances(pos, block_size=block_size)
    finite = d_nn[np.isfinite(d_nn) & (d_nn > 1.0e-12)]
    if len(finite) == 0:
        return np.full(n, np.clip(base, min_s, max_s), dtype=float)

    mode = (mode or "density_boost").lower()
    k = float(k)

    if mode in {"nearest_neighbor", "nn", "distance"}:
        eps = k * d_nn
    elif mode in {"density_boost", "density", "boost"}:
        median_nn = float(np.median(finite))
        safe_d = np.maximum(d_nn, 1.0e-12)
        crowding = np.maximum(0.0, (median_nn / safe_d) - 1.0)
        eps = base * (1.0 + k * crowding)
    else:
        raise ValueError(
            "ADAPTIVE_SOFTENING_MODE must be 'density_boost' or 'nearest_neighbor'"
        )

    eps = np.nan_to_num(eps, nan=base, posinf=max_s, neginf=min_s)
    return np.clip(eps, min_s, max_s).astype(float)


def _softening_values_or_none(softening_values: np.ndarray | None) -> np.ndarray | None:
    if softening_values is None:
        return None
    values = np.asarray(softening_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("softening_values must be a 1D array")
    return values


def compute_direct_softened_accelerations(
    pos: np.ndarray,
    mass: np.ndarray,
    G: float,
    softening: float,
    softening_values: np.ndarray | None = None,
) -> np.ndarray:
    """Exact O(N^2) softened gravitational acceleration.

    Scalar softening:
        acc_i = sum_j G*m_j*(x_j-x_i)/(||x_j-x_i||^2 + eps^2)^(3/2)

    Adaptive softening:
        eps_ij^2 = 0.5 * (eps_i^2 + eps_j^2)
    """
    pos = np.asarray(pos, dtype=float)
    mass = np.asarray(mass, dtype=float)
    n = len(pos)
    acc = np.zeros_like(pos, dtype=float)

    adaptive_eps = _softening_values_or_none(softening_values)
    if adaptive_eps is not None and len(adaptive_eps) != n:
        raise ValueError("softening_values must have the same length as pos")

    scalar_eps2 = float(softening) ** 2
    adaptive_eps2 = adaptive_eps ** 2 if adaptive_eps is not None else None

    for i in range(n):
        diff = pos - pos[i]
        if adaptive_eps2 is None:
            dist2 = np.einsum("ij,ij->i", diff, diff) + scalar_eps2
        else:
            pair_eps2 = 0.5 * (adaptive_eps2[i] + adaptive_eps2)
            dist2 = np.einsum("ij,ij->i", diff, diff) + pair_eps2
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
    barnes_hut_max_particles_per_leaf: int = 8,
    barnes_hut_max_depth: int = 32,
    barnes_hut_implementation: str = "fast",
    barnes_hut_use_numba: bool = True,
    barnes_hut_direct_fallback_n: int = 512,
    softening_values: np.ndarray | None = None,
) -> np.ndarray:
    """Compute gravity acceleration using either direct or Barnes-Hut solver."""
    solver = (force_solver or "direct").lower()
    if solver == "direct":
        return compute_direct_softened_accelerations(
            pos, mass, G, softening, softening_values=softening_values
        )
    if solver in {"barnes_hut", "barnes-hut", "bh"}:
        implementation = (barnes_hut_implementation or "fast").lower()
        if implementation in {"fast", "flat", "numba", "auto"}:
            from .barnes_hut_fast import compute_barnes_hut_fast_accelerations

            return compute_barnes_hut_fast_accelerations(
                pos,
                mass,
                G=G,
                softening=softening,
                theta=barnes_hut_theta,
                max_particles_per_leaf=barnes_hut_max_particles_per_leaf,
                max_depth=barnes_hut_max_depth,
                use_numba=barnes_hut_use_numba,
                direct_fallback_n=barnes_hut_direct_fallback_n,
                softening_values=softening_values,
            )

        # Legacy implementation kept for correctness/debug comparisons.
        if softening_values is not None:
            # The legacy object-based tree only supports scalar softening. Use the
            # exact adaptive direct solver rather than silently dropping eps_i.
            return compute_direct_softened_accelerations(
                pos, mass, G, softening, softening_values=softening_values
            )

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
    barnes_hut_max_particles_per_leaf: int = 8,
    barnes_hut_max_depth: int = 32,
    barnes_hut_implementation: str = "fast",
    barnes_hut_use_numba: bool = True,
    barnes_hut_direct_fallback_n: int = 512,
    softening_values: np.ndarray | None = None,
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
        barnes_hut_implementation=barnes_hut_implementation,
        barnes_hut_use_numba=barnes_hut_use_numba,
        barnes_hut_direct_fallback_n=barnes_hut_direct_fallback_n,
        softening_values=softening_values,
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
