"""Quantitative diagnostics for N-body experiments."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def kinetic_energy(mass: np.ndarray, vel: np.ndarray) -> float:
    speeds2 = np.einsum("ij,ij->i", vel, vel)
    return float(0.5 * np.sum(mass * speeds2))


def softened_potential_energy(
    pos: np.ndarray,
    mass: np.ndarray,
    G: float,
    softening: float,
) -> float:
    n = len(pos)
    if n < 2:
        return 0.0
    eps2 = softening * softening
    U = 0.0
    for i in range(n - 1):
        diff = pos[i + 1 :] - pos[i]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff) + eps2)
        pair_mass = mass[i] * mass[i + 1 :]
        valid = dist > 0
        U -= float(np.sum(G * pair_mass[valid] / dist[valid]))
    return U


def barycenter(pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
    total_mass = float(np.sum(mass))
    if total_mass <= 0:
        return np.mean(pos, axis=0)
    return np.sum(pos * mass[:, None], axis=0) / total_mass


def radius_stats(pos: np.ndarray, mass: np.ndarray) -> dict[str, float]:
    if len(pos) == 0:
        return {"mean_radius": 0.0, "median_radius": 0.0, "max_radius": 0.0}
    center = barycenter(pos, mass)
    radii = np.linalg.norm(pos - center, axis=1)
    return {
        "mean_radius": float(np.mean(radii)),
        "median_radius": float(np.median(radii)),
        "max_radius": float(np.max(radii)),
    }


def nearest_neighbor_mean(pos: np.ndarray, max_n: int = 1500) -> float:
    n = len(pos)
    if n < 2:
        return 0.0
    if n > max_n:
        return float("nan")

    nearest = np.full(n, np.inf, dtype=float)
    for i in range(n):
        diff = pos - pos[i]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        dist2[i] = np.inf
        nearest[i] = math.sqrt(float(np.min(dist2)))
    return float(np.mean(nearest))


def effective_potential(raw_potential: float, a_t: float, scaling: str) -> float:
    scaling = (scaling or "none").lower()
    if scaling == "none":
        return raw_potential
    if scaling == "a":
        return raw_potential / max(a_t, 1.0e-12)
    if scaling == "a3":
        return raw_potential / max(a_t ** 3, 1.0e-12)
    raise ValueError(f"Unknown METRICS_POTENTIAL_SCALING: {scaling}")


def compute_metrics(
    frame: int,
    time: float,
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    G: float,
    softening: float,
    scale_factor: float = 1.0,
    expansion_rate: float = 0.0,
    exact_potential_max_n: int = 700,
    nearest_neighbor_max_n: int = 1500,
    potential_scaling: str = "a3",
    initial_mean_radius: float | None = None,
    collapse_radius_fraction: float = 0.3,
) -> dict[str, Any]:
    K = kinetic_energy(mass, vel)

    if len(pos) <= exact_potential_max_n:
        U_raw = softened_potential_energy(pos, mass, G, softening)
        U_eff = effective_potential(U_raw, scale_factor, potential_scaling)
    else:
        U_raw = float("nan")
        U_eff = float("nan")

    if U_eff != 0 and np.isfinite(U_eff):
        virial = float(2.0 * K / abs(U_eff))
    else:
        virial = float("nan")

    radii = radius_stats(pos, mass)
    nn_mean = nearest_neighbor_mean(pos, max_n=nearest_neighbor_max_n)

    collapsed = False
    if initial_mean_radius and initial_mean_radius > 0:
        collapsed = radii["mean_radius"] < collapse_radius_fraction * initial_mean_radius

    return {
        "frame": int(frame),
        "time": float(time),
        "kinetic_energy": K,
        "potential_energy_raw": U_raw,
        "potential_energy_effective": U_eff,
        "virial_ratio": virial,
        "mean_radius": radii["mean_radius"],
        "median_radius": radii["median_radius"],
        "max_radius": radii["max_radius"],
        "nearest_neighbor_mean": nn_mean,
        "scale_factor": float(scale_factor),
        "expansion_rate": float(expansion_rate),
        "collapsed": bool(collapsed),
    }


def first_collapse_time(metrics_rows: list[dict[str, Any]]) -> float | None:
    for row in metrics_rows:
        if row.get("collapsed"):
            return float(row["time"])
    return None
