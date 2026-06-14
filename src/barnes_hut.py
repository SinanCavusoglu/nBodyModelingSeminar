"""3D Barnes-Hut octree force solver.

The implementation is intentionally dependency-light and works with NumPy arrays.
It approximates far-away particle groups by their center of mass and keeps exact
pairwise forces for nearby leaves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class OctreeNode:
    center: np.ndarray
    half_size: float
    indices: np.ndarray
    total_mass: float = 0.0
    center_of_mass: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    children: list["OctreeNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def size(self) -> float:
        return 2.0 * self.half_size


def _compute_mass_properties(node: OctreeNode, pos: np.ndarray, mass: np.ndarray) -> None:
    if len(node.indices) == 0:
        node.total_mass = 0.0
        node.center_of_mass = node.center.copy()
        return
    m = mass[node.indices]
    total = float(np.sum(m))
    node.total_mass = total
    if total > 0:
        node.center_of_mass = np.sum(pos[node.indices] * m[:, None], axis=0) / total
    else:
        node.center_of_mass = np.mean(pos[node.indices], axis=0)


def _child_code(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    return ((points[:, 0] >= center[0]).astype(np.int8) << 2) | \
           ((points[:, 1] >= center[1]).astype(np.int8) << 1) | \
           (points[:, 2] >= center[2]).astype(np.int8)


def build_octree(
    pos: np.ndarray,
    mass: np.ndarray,
    indices: Iterable[int] | None = None,
    max_particles_per_leaf: int = 1,
    max_depth: int = 32,
    _center: np.ndarray | None = None,
    _half_size: float | None = None,
    _depth: int = 0,
) -> OctreeNode:
    """Build and return a Barnes-Hut octree for the given particle positions."""
    pos = np.asarray(pos, dtype=float)
    mass = np.asarray(mass, dtype=float)
    if indices is None:
        indices_arr = np.arange(len(pos), dtype=int)
    else:
        indices_arr = np.asarray(list(indices), dtype=int)

    if _center is None or _half_size is None:
        if len(indices_arr) == 0:
            center = np.zeros(3, dtype=float)
            half_size = 1.0
        else:
            mins = np.min(pos[indices_arr], axis=0)
            maxs = np.max(pos[indices_arr], axis=0)
            center = 0.5 * (mins + maxs)
            half_size = 0.5 * float(np.max(maxs - mins))
            if not np.isfinite(half_size) or half_size <= 0:
                half_size = 1.0
            half_size *= 1.000001
    else:
        center = np.asarray(_center, dtype=float)
        half_size = float(_half_size)

    node = OctreeNode(center=center, half_size=half_size, indices=indices_arr)
    _compute_mass_properties(node, pos, mass)

    if (
        len(indices_arr) <= max_particles_per_leaf
        or _depth >= max_depth
        or half_size <= 1.0e-14
    ):
        return node

    codes = _child_code(pos[indices_arr], center)
    child_half = 0.5 * half_size
    for code in range(8):
        child_indices = indices_arr[codes == code]
        if len(child_indices) == 0:
            continue
        offset = np.array([
            1.0 if (code & 4) else -1.0,
            1.0 if (code & 2) else -1.0,
            1.0 if (code & 1) else -1.0,
        ]) * child_half
        child_center = center + offset
        child = build_octree(
            pos,
            mass,
            child_indices,
            max_particles_per_leaf=max_particles_per_leaf,
            max_depth=max_depth,
            _center=child_center,
            _half_size=child_half,
            _depth=_depth + 1,
        )
        node.children.append(child)

    return node


def _direct_leaf_acceleration(
    target_index: int,
    node: OctreeNode,
    pos: np.ndarray,
    mass: np.ndarray,
    G: float,
    softening: float,
) -> np.ndarray:
    acc = np.zeros(3, dtype=float)
    p = pos[target_index]
    eps2 = softening * softening
    for j in node.indices:
        if j == target_index:
            continue
        diff = pos[j] - p
        dist2 = float(np.dot(diff, diff) + eps2)
        if dist2 <= 0:
            continue
        acc += G * mass[j] * diff / (dist2 ** 1.5)
    return acc


def _node_as_mass_acceleration(
    target_index: int,
    node: OctreeNode,
    pos: np.ndarray,
    G: float,
    softening: float,
) -> np.ndarray:
    diff = node.center_of_mass - pos[target_index]
    dist2 = float(np.dot(diff, diff) + softening * softening)
    if dist2 <= 0 or node.total_mass <= 0:
        return np.zeros(3, dtype=float)
    return G * node.total_mass * diff / (dist2 ** 1.5)


def _contains_index(node: OctreeNode, index: int) -> bool:
    # `in` over a NumPy array is fine for the moderate leaf/node counts used here.
    return bool(np.any(node.indices == index))


def compute_particle_acceleration(
    target_index: int,
    node: OctreeNode,
    pos: np.ndarray,
    mass: np.ndarray,
    G: float,
    softening: float,
    theta: float,
) -> np.ndarray:
    """Compute Barnes-Hut acceleration for one particle."""
    if len(node.indices) == 0 or node.total_mass <= 0:
        return np.zeros(3, dtype=float)

    if node.is_leaf:
        return _direct_leaf_acceleration(target_index, node, pos, mass, G, softening)

    target_inside_node = _contains_index(node, target_index)
    d = float(np.linalg.norm(node.center_of_mass - pos[target_index]))

    # If the target is not inside this node and it is sufficiently far away,
    # approximate the whole node by its center of mass.
    if (not target_inside_node) and d > 0 and (node.size / d) < theta:
        return _node_as_mass_acceleration(target_index, node, pos, G, softening)

    acc = np.zeros(3, dtype=float)
    for child in node.children:
        acc += compute_particle_acceleration(
            target_index, child, pos, mass, G, softening, theta
        )
    return acc


def compute_barnes_hut_accelerations(
    pos: np.ndarray,
    mass: np.ndarray,
    G: float,
    softening: float,
    theta: float = 0.5,
    max_particles_per_leaf: int = 1,
    max_depth: int = 32,
) -> np.ndarray:
    """Compute accelerations for all particles using the Barnes-Hut octree."""
    pos = np.asarray(pos, dtype=float)
    mass = np.asarray(mass, dtype=float)
    n = len(pos)
    if n == 0:
        return np.zeros((0, 3), dtype=float)

    root = build_octree(
        pos,
        mass,
        max_particles_per_leaf=max_particles_per_leaf,
        max_depth=max_depth,
    )
    acc = np.zeros_like(pos, dtype=float)
    for i in range(n):
        acc[i] = compute_particle_acceleration(i, root, pos, mass, G, softening, theta)
    return acc
