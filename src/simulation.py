"""Time integration for the Forbes N-body simulation."""
from __future__ import annotations

from typing import Any

import numpy as np

from .cosmology import expansion_rate as compute_H
from .cosmology import scale_factor as compute_a
from .metrics import compute_metrics, kinetic_energy, radius_stats, softened_potential_energy
from .physics import adaptive_softening_lengths, compute_accelerations


def _cfg(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config.get(key, default)


def _sampled_softened_potential_energy(
    pos: np.ndarray,
    mass: np.ndarray,
    G: float,
    softening: float,
    sample_pairs: int,
    seed: int,
) -> float:
    """Estimate softened potential energy from random particle pairs.

    This is only used when exact O(N^2) initial potential is too expensive for
    virial scaling. The estimator samples unordered pairs and scales the mean
    pair potential to the total number of pairs.
    """
    n = len(pos)
    if n < 2:
        return 0.0
    total_pairs = n * (n - 1) // 2
    sample_pairs = int(max(1, min(sample_pairs, total_pairs)))
    rng = np.random.default_rng(int(seed))

    i = rng.integers(0, n, size=sample_pairs, endpoint=False)
    j = rng.integers(0, n - 1, size=sample_pairs, endpoint=False)
    j = np.where(j >= i, j + 1, j)

    diff = pos[j] - pos[i]
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff) + float(softening) ** 2)
    pair_u = -G * mass[i] * mass[j] / np.maximum(dist, 1.0e-12)
    return float(np.mean(pair_u) * total_pairs)


def _apply_virial_velocity_scaling(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale initial velocities toward target virial ratio Q = 2K/|U|."""
    if not bool(_cfg(config, "USE_VIRIAL_VELOCITY_SCALING", False)):
        return vel, {"enabled": False}

    G = float(_cfg(config, "G", 0.005))
    softening = float(_cfg(config, "SOFTENING", 2.0))
    target_q = float(_cfg(config, "TARGET_VIRIAL_RATIO", 1.0))
    exact_max_n = int(_cfg(config, "VIRIAL_POTENTIAL_MAX_N", 5000))
    sample_pairs = int(_cfg(config, "VIRIAL_POTENTIAL_SAMPLE_PAIRS", 200000))
    seed = int(_cfg(config, "VIRIAL_RANDOM_SEED", 42))
    scale_min = float(_cfg(config, "VIRIAL_VELOCITY_SCALE_MIN", 0.1))
    scale_max = float(_cfg(config, "VIRIAL_VELOCITY_SCALE_MAX", 10.0))

    K0 = kinetic_energy(mass, vel)
    if len(pos) <= exact_max_n:
        U0 = softened_potential_energy(pos, mass, G, softening)
        potential_method = "exact"
    else:
        U0 = _sampled_softened_potential_energy(pos, mass, G, softening, sample_pairs, seed)
        potential_method = "sampled"

    info: dict[str, Any] = {
        "enabled": True,
        "target_virial_ratio": target_q,
        "initial_kinetic_energy_before": float(K0),
        "initial_potential_energy_estimate": float(U0),
        "potential_method": potential_method,
    }

    if K0 <= 0 or not np.isfinite(K0) or U0 == 0 or not np.isfinite(U0) or target_q <= 0:
        info.update({"applied": False, "reason": "invalid K, U, or target ratio", "velocity_scale": 1.0})
        return vel, info

    # Target: Q' = 2K'/|U| = target_q. Since K' = lambda^2 K,
    # lambda = sqrt(target_q * |U| / (2K)).
    lam = float(np.sqrt((target_q * abs(U0)) / (2.0 * K0)))
    lam_clipped = float(np.clip(lam, min(scale_min, scale_max), max(scale_min, scale_max)))
    vel_scaled = vel * lam_clipped
    K1 = kinetic_energy(mass, vel_scaled)
    Q0 = float(2.0 * K0 / abs(U0)) if abs(U0) > 0 else float("nan")
    Q1 = float(2.0 * K1 / abs(U0)) if abs(U0) > 0 else float("nan")

    info.update({
        "applied": True,
        "velocity_scale_unclipped": lam,
        "velocity_scale": lam_clipped,
        "initial_virial_ratio_before": Q0,
        "initial_kinetic_energy_after": float(K1),
        "initial_virial_ratio_after": Q1,
    })
    return vel_scaled, info


def _compute_adaptive_softening_if_needed(
    pos: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray | None:
    if not bool(_cfg(config, "USE_ADAPTIVE_SOFTENING", False)):
        return None
    return adaptive_softening_lengths(
        pos=pos,
        base_softening=float(_cfg(config, "SOFTENING", 2.0)),
        mode=str(_cfg(config, "ADAPTIVE_SOFTENING_MODE", "density_boost")),
        k=float(_cfg(config, "ADAPTIVE_SOFTENING_K", 1.0)),
        min_softening=float(_cfg(config, "ADAPTIVE_SOFTENING_MIN", 0.5)),
        max_softening=float(_cfg(config, "ADAPTIVE_SOFTENING_MAX", 30.0)),
        block_size=int(_cfg(config, "ADAPTIVE_SOFTENING_BLOCK_SIZE", 512)),
    )


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

    adaptive_enabled = bool(_cfg(config, "USE_ADAPTIVE_SOFTENING", False))
    adaptive_update_every = max(1, int(_cfg(config, "ADAPTIVE_SOFTENING_UPDATE_EVERY", 10)))
    adaptive_softening = _compute_adaptive_softening_if_needed(pos, config)

    vel, virial_info = _apply_virial_velocity_scaling(pos, vel, mass, config)

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
            row = compute_metrics(
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
            if adaptive_softening is not None and len(adaptive_softening):
                row["adaptive_softening_mean"] = float(np.mean(adaptive_softening))
                row["adaptive_softening_min"] = float(np.min(adaptive_softening))
                row["adaptive_softening_max"] = float(np.max(adaptive_softening))
            metrics_history.append(row)

    t = 0.0
    save_frame(step=0, t=t)

    for step in range(1, steps + 1):
        if adaptive_enabled and ((step - 1) % adaptive_update_every == 0):
            adaptive_softening = _compute_adaptive_softening_if_needed(pos, config)

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
            softening_values=adaptive_softening,
        )

        # Velocity-Verlet style update. Because expansion damping depends on
        # velocity, this is an approximation but keeps the existing structure.
        vel_half = vel + 0.5 * dt * acc
        pos = pos + dt * vel_half
        t = step * dt

        if adaptive_enabled and (step % adaptive_update_every == 0):
            adaptive_softening = _compute_adaptive_softening_if_needed(pos, config)

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
            softening_values=adaptive_softening,
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
        "initial_condition_adjustments": {
            "virial_velocity_scaling": virial_info,
            "adaptive_softening_enabled": adaptive_enabled,
            "adaptive_softening_update_every": adaptive_update_every if adaptive_enabled else None,
        },
    }
