"""Fast flat-array Barnes-Hut octree solver.

This module is a performance-oriented replacement for the first object/recursive
Barnes-Hut implementation.  The old solver used Python dataclass nodes and a
recursive traversal for every particle.  That was conceptually correct but slow,
especially for small N, because Python object/recursion overhead dominated.

This version builds a flat octree and traverses it iteratively.  If Numba is
installed, the traversal loop is JIT-compiled and parallelized.  If Numba is not
installed, it falls back to a pure-Python iterative traversal that is still much
lighter than the original recursive object tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # Optional acceleration. The code works without numba.
    from numba import njit, prange  # type: ignore

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on user's environment
    njit = None  # type: ignore
    prange = range  # type: ignore
    NUMBA_AVAILABLE = False


@dataclass
class FlatOctree:
    centers: np.ndarray           # (nodes, 3)
    half_sizes: np.ndarray        # (nodes,)
    total_masses: np.ndarray      # (nodes,)
    centers_of_mass: np.ndarray   # (nodes, 3)
    children: np.ndarray          # (nodes, 8), -1 for missing child
    leaf_starts: np.ndarray       # (nodes,)
    leaf_counts: np.ndarray       # (nodes,)
    leaf_indices: np.ndarray      # packed particle indices for leaves

    @property
    def node_count(self) -> int:
        return int(self.centers.shape[0])


class _FlatTreeBuilder:
    def __init__(
        self,
        pos: np.ndarray,
        mass: np.ndarray,
        max_particles_per_leaf: int,
        max_depth: int,
    ) -> None:
        self.pos = np.asarray(pos, dtype=np.float64)
        self.mass = np.asarray(mass, dtype=np.float64)
        self.max_particles_per_leaf = max(1, int(max_particles_per_leaf))
        self.max_depth = max(1, int(max_depth))

        self.centers: list[np.ndarray] = []
        self.half_sizes: list[float] = []
        self.total_masses: list[float] = []
        self.centers_of_mass: list[np.ndarray] = []
        self.children: list[list[int]] = []
        self.leaf_starts: list[int] = []
        self.leaf_counts: list[int] = []
        self.leaf_indices: list[int] = []

    def build(self) -> FlatOctree:
        n = len(self.pos)
        if n == 0:
            return FlatOctree(
                centers=np.zeros((0, 3), dtype=np.float64),
                half_sizes=np.zeros(0, dtype=np.float64),
                total_masses=np.zeros(0, dtype=np.float64),
                centers_of_mass=np.zeros((0, 3), dtype=np.float64),
                children=np.zeros((0, 8), dtype=np.int64),
                leaf_starts=np.zeros(0, dtype=np.int64),
                leaf_counts=np.zeros(0, dtype=np.int64),
                leaf_indices=np.zeros(0, dtype=np.int64),
            )

        indices = np.arange(n, dtype=np.int64)
        mins = np.nanmin(self.pos, axis=0)
        maxs = np.nanmax(self.pos, axis=0)
        center = 0.5 * (mins + maxs)
        half_size = 0.5 * float(np.max(maxs - mins))
        if not np.isfinite(half_size) or half_size <= 0:
            half_size = 1.0
        half_size *= 1.000001

        self._build_node(indices, center.astype(np.float64), half_size, depth=0)

        return FlatOctree(
            centers=np.asarray(self.centers, dtype=np.float64),
            half_sizes=np.asarray(self.half_sizes, dtype=np.float64),
            total_masses=np.asarray(self.total_masses, dtype=np.float64),
            centers_of_mass=np.asarray(self.centers_of_mass, dtype=np.float64),
            children=np.asarray(self.children, dtype=np.int64),
            leaf_starts=np.asarray(self.leaf_starts, dtype=np.int64),
            leaf_counts=np.asarray(self.leaf_counts, dtype=np.int64),
            leaf_indices=np.asarray(self.leaf_indices, dtype=np.int64),
        )

    def _append_node(self, center: np.ndarray, half_size: float, total_mass: float, com: np.ndarray) -> int:
        node_id = len(self.centers)
        self.centers.append(center.copy())
        self.half_sizes.append(float(half_size))
        self.total_masses.append(float(total_mass))
        self.centers_of_mass.append(com.copy())
        self.children.append([-1] * 8)
        self.leaf_starts.append(-1)
        self.leaf_counts.append(0)
        return node_id

    def _mass_properties(self, indices: np.ndarray, fallback_center: np.ndarray) -> tuple[float, np.ndarray]:
        if len(indices) == 0:
            return 0.0, fallback_center.copy()
        m = self.mass[indices]
        total = float(np.sum(m))
        if total > 0.0 and np.isfinite(total):
            com = np.sum(self.pos[indices] * m[:, None], axis=0) / total
        else:
            com = np.mean(self.pos[indices], axis=0)
        return total, np.asarray(com, dtype=np.float64)

    def _make_leaf(self, node_id: int, indices: np.ndarray) -> None:
        start = len(self.leaf_indices)
        self.leaf_indices.extend(int(i) for i in indices)
        self.leaf_starts[node_id] = start
        self.leaf_counts[node_id] = int(len(indices))

    def _build_node(self, indices: np.ndarray, center: np.ndarray, half_size: float, depth: int) -> int:
        total_mass, com = self._mass_properties(indices, center)
        node_id = self._append_node(center, half_size, total_mass, com)

        if (
            len(indices) <= self.max_particles_per_leaf
            or depth >= self.max_depth
            or half_size <= 1.0e-14
        ):
            self._make_leaf(node_id, indices)
            return node_id

        points = self.pos[indices]
        codes = ((points[:, 0] >= center[0]).astype(np.int8) << 2) | \
                ((points[:, 1] >= center[1]).astype(np.int8) << 1) | \
                (points[:, 2] >= center[2]).astype(np.int8)
        child_half = 0.5 * half_size

        any_child = False
        for code in range(8):
            child_indices = indices[codes == code]
            if len(child_indices) == 0:
                continue
            any_child = True
            offset = np.array([
                1.0 if (code & 4) else -1.0,
                1.0 if (code & 2) else -1.0,
                1.0 if (code & 1) else -1.0,
            ], dtype=np.float64) * child_half
            child_center = center + offset
            child_id = self._build_node(child_indices, child_center, child_half, depth + 1)
            self.children[node_id][code] = child_id

        # Degenerate case: all points identical and subdivision did not progress.
        if not any_child:
            self._make_leaf(node_id, indices)

        return node_id


def build_flat_octree(
    pos: np.ndarray,
    mass: np.ndarray,
    max_particles_per_leaf: int = 8,
    max_depth: int = 32,
) -> FlatOctree:
    """Build a flat-array octree."""
    builder = _FlatTreeBuilder(pos, mass, max_particles_per_leaf, max_depth)
    return builder.build()


if NUMBA_AVAILABLE:

    @njit(parallel=True, fastmath=True, cache=True)  # type: ignore[misc]
    def _accelerations_numba(
        pos: np.ndarray,
        mass: np.ndarray,
        centers: np.ndarray,
        half_sizes: np.ndarray,
        total_masses: np.ndarray,
        centers_of_mass: np.ndarray,
        children: np.ndarray,
        leaf_starts: np.ndarray,
        leaf_counts: np.ndarray,
        leaf_indices: np.ndarray,
        G: float,
        softening: float,
        theta: float,
        softening_values: np.ndarray,
        use_adaptive_softening: bool,
    ) -> np.ndarray:
        n = pos.shape[0]
        node_count = centers.shape[0]
        acc = np.zeros((n, 3), dtype=np.float64)
        scalar_eps2 = softening * softening
        theta = max(theta, 1.0e-12)

        for i in prange(n):
            px = pos[i, 0]
            py = pos[i, 1]
            pz = pos[i, 2]
            if use_adaptive_softening:
                soft_i2 = softening_values[i] * softening_values[i]
            else:
                soft_i2 = scalar_eps2

            stack = np.empty(max(node_count, 1), dtype=np.int64)
            top = 0
            if node_count > 0:
                stack[0] = 0
            ax = 0.0
            ay = 0.0
            az = 0.0

            while top >= 0:
                node_id = stack[top]
                top -= 1

                tm = total_masses[node_id]
                if tm <= 0.0:
                    continue

                leaf_count = leaf_counts[node_id]
                if leaf_count > 0:
                    start = leaf_starts[node_id]
                    for kk in range(start, start + leaf_count):
                        j = leaf_indices[kk]
                        if j == i:
                            continue
                        dx = pos[j, 0] - px
                        dy = pos[j, 1] - py
                        dz = pos[j, 2] - pz
                        if use_adaptive_softening:
                            soft_j = softening_values[j]
                            eps2 = 0.5 * (soft_i2 + soft_j * soft_j)
                        else:
                            eps2 = scalar_eps2
                        dist2 = dx * dx + dy * dy + dz * dz + eps2
                        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
                        weight = G * mass[j] * inv_dist3
                        ax += weight * dx
                        ay += weight * dy
                        az += weight * dz
                    continue

                cx = centers[node_id, 0]
                cy = centers[node_id, 1]
                cz = centers[node_id, 2]
                hs = half_sizes[node_id]
                inside = (
                    abs(px - cx) <= hs + 1.0e-12
                    and abs(py - cy) <= hs + 1.0e-12
                    and abs(pz - cz) <= hs + 1.0e-12
                )

                dx = centers_of_mass[node_id, 0] - px
                dy = centers_of_mass[node_id, 1] - py
                dz = centers_of_mass[node_id, 2] - pz
                dist2_noeps = dx * dx + dy * dy + dz * dz
                dist = np.sqrt(dist2_noeps)

                if (not inside) and dist > 0.0 and ((2.0 * hs) / dist) < theta:
                    if use_adaptive_softening:
                        eps2 = soft_i2
                    else:
                        eps2 = scalar_eps2
                    dist2 = dist2_noeps + eps2
                    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
                    weight = G * tm * inv_dist3
                    ax += weight * dx
                    ay += weight * dy
                    az += weight * dz
                else:
                    for c in range(8):
                        child = children[node_id, c]
                        if child >= 0:
                            top += 1
                            stack[top] = child

            acc[i, 0] = ax
            acc[i, 1] = ay
            acc[i, 2] = az

        return acc

else:
    _accelerations_numba = None


def _accelerations_python(
    pos: np.ndarray,
    mass: np.ndarray,
    tree: FlatOctree,
    G: float,
    softening: float,
    theta: float,
    softening_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = len(pos)
    acc = np.zeros_like(pos, dtype=np.float64)
    scalar_eps2 = softening * softening
    adaptive = softening_values is not None
    if adaptive:
        softening_values = np.asarray(softening_values, dtype=np.float64)
    if tree.node_count == 0:
        return acc

    for i in range(n):
        p = pos[i]
        soft_i2 = float(softening_values[i] ** 2) if adaptive else scalar_eps2
        stack = [0]
        while stack:
            node_id = stack.pop()
            tm = tree.total_masses[node_id]
            if tm <= 0.0:
                continue

            leaf_count = int(tree.leaf_counts[node_id])
            if leaf_count > 0:
                start = int(tree.leaf_starts[node_id])
                for j in tree.leaf_indices[start:start + leaf_count]:
                    jj = int(j)
                    if jj == i:
                        continue
                    diff = pos[jj] - p
                    if adaptive:
                        eps2 = 0.5 * (soft_i2 + float(softening_values[jj] ** 2))
                    else:
                        eps2 = scalar_eps2
                    dist2 = float(np.dot(diff, diff) + eps2)
                    if dist2 > 0.0:
                        acc[i] += G * mass[jj] * diff / (dist2 ** 1.5)
                continue

            center = tree.centers[node_id]
            hs = float(tree.half_sizes[node_id])
            inside = bool(np.all(np.abs(p - center) <= hs + 1.0e-12))
            diff = tree.centers_of_mass[node_id] - p
            dist = float(np.linalg.norm(diff))
            if (not inside) and dist > 0.0 and ((2.0 * hs) / dist) < theta:
                eps2 = soft_i2 if adaptive else scalar_eps2
                dist2 = float(np.dot(diff, diff) + eps2)
                acc[i] += G * tm * diff / (dist2 ** 1.5)
            else:
                for child in tree.children[node_id]:
                    if int(child) >= 0:
                        stack.append(int(child))

    return acc


def compute_barnes_hut_fast_accelerations(
    pos: np.ndarray,
    mass: np.ndarray,
    G: float,
    softening: float,
    theta: float = 0.5,
    max_particles_per_leaf: int = 8,
    max_depth: int = 32,
    use_numba: bool = True,
    direct_fallback_n: Optional[int] = None,
    softening_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Barnes-Hut accelerations with a flat octree.

    Parameters
    ----------
    direct_fallback_n:
        If provided and N <= this value, the function uses the vectorized direct
        solver.  This is intentional: for small particle counts the direct
        solver is usually faster and exact, while Barnes-Hut only becomes useful
        once tree approximation beats the vectorized O(N^2) overhead.
    """
    pos = np.asarray(pos, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    n = len(pos)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if softening_values is not None:
        softening_values = np.asarray(softening_values, dtype=np.float64)
        if len(softening_values) != n:
            raise ValueError("softening_values must have the same length as pos")
        use_adaptive_softening = True
    else:
        softening_values = np.full(n, float(softening), dtype=np.float64)
        use_adaptive_softening = False

    if direct_fallback_n is not None and direct_fallback_n >= 0 and n <= int(direct_fallback_n):
        from .physics import compute_direct_softened_accelerations

        return compute_direct_softened_accelerations(
            pos, mass, G, softening,
            softening_values=softening_values if use_adaptive_softening else None,
        )

    tree = build_flat_octree(
        pos,
        mass,
        max_particles_per_leaf=max_particles_per_leaf,
        max_depth=max_depth,
    )

    if use_numba and NUMBA_AVAILABLE and _accelerations_numba is not None:
        return _accelerations_numba(
            pos,
            mass,
            tree.centers,
            tree.half_sizes,
            tree.total_masses,
            tree.centers_of_mass,
            tree.children,
            tree.leaf_starts,
            tree.leaf_counts,
            tree.leaf_indices,
            float(G),
            float(softening),
            float(theta),
            softening_values,
            use_adaptive_softening,
        )

    return _accelerations_python(
        pos, mass, tree, float(G), float(softening), float(theta),
        softening_values=softening_values if use_adaptive_softening else None,
    )
