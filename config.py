"""
Central configuration for the Forbes N-body simulation upgrade.

This file is intentionally plain-Python so the existing project can keep using
`import config` style imports.  The experiment runner can override any value
through dictionaries without editing this file.
"""
from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DATA_DIR = DATA_DIR / "generated"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "experiments"
RUNS_DIR = OUTPUT_ROOT / "runs"
USE_TIMESTAMPED_RUN_DIRS = True
RUN_ID = None
RUN_OUTPUT_ROOT = None
RUN_STARTED_AT = None
RUN_NOTES = ""

# Raw CSV uploaded/provided by the project. This file has positions, net worth,
# industries, connections, and color information, but no vx/vy/vz.
RAW_CSV_PATH = DATA_DIR / "forbes_billionaires_WorthPosConnectionsColor_20260610.csv"

# Generated radial/angular files. The simulation reads these because they include
# vx/vy/vz initial velocities.
RADIAL_CSV_PATH = GENERATED_DATA_DIR / "radial" / "minimal.csv"
ANGULAR_CSV_PATH = GENERATED_DATA_DIR / "angular" / "minimal.csv"
RADIAL_EDGE_CSV_PATH = GENERATED_DATA_DIR / "radial" / "edges.csv"
ANGULAR_EDGE_CSV_PATH = GENERATED_DATA_DIR / "angular" / "edges.csv"

# Default runtime CSV uses radial generated velocities. run_full_pipeline.py creates
# this file automatically from RAW_CSV_PATH before experiments run.
CSV_PATH = RADIAL_CSV_PATH
EDGE_CSV_PATH = RADIAL_EDGE_CSV_PATH

# -----------------------------------------------------------------------------
# Data / run size
# -----------------------------------------------------------------------------
MAX_PARTICLES = 100
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Numerical parameters
# -----------------------------------------------------------------------------
G = 0.005
DT = 0.02
STEPS = 3000
SAVE_EVERY = 3
SOFTENING = 2.0

# -----------------------------------------------------------------------------
# Simulation mode
# -----------------------------------------------------------------------------
# Main modes:
#   current_direct
#   softened_no_expansion
#   softened_expansion
#   softened_expansion_angular
#   barnes_hut_softened
#   barnes_hut_expansion
#   barnes_hut_expansion_angular
SIMULATION_MODE = "current_direct"
EXPERIMENT_NAME = SIMULATION_MODE

# Force solver:
#   direct      -> O(N^2), exact pairwise direct summation
#   barnes_hut -> 3D octree approximation, useful for higher particle counts
FORCE_SOLVER = "barnes_hut"

# Expansion / comoving-coordinate model
USE_EXPANSION = False
EXPANSION_MODEL = "linear"  # linear, exponential, none
H0 = 0.01

# Connection velocity mode is primarily used by scripts/generate_connection_velocity.py.
# Runtime simulation reads vx/vy/vz from CSV.
CONNECTION_VELOCITY_MODE = "radial"  # radial, angular, none
CONNECTION_VELOCITY_SCALE = 0.05
USE_CONNECTION_MASS_STRENGTH = True

# Barnes-Hut parameters
# The optimized path uses a flat-array octree and optional Numba JIT traversal.
# For small N, the vectorized direct solver is often faster and exact, so Barnes-Hut
# uses a direct fallback below BARNES_HUT_DIRECT_FALLBACK_N by default.
BARNES_HUT_THETA = 0.5
BARNES_HUT_MAX_PARTICLES_PER_LEAF = 8
BARNES_HUT_MAX_DEPTH = 32
BARNES_HUT_IMPLEMENTATION = "fast"  # fast, legacy
BARNES_HUT_USE_NUMBA = True
BARNES_HUT_DIRECT_FALLBACK_N = 512

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
SAVE_GIF = True
SAVE_VVVV_CSV = True
SAVE_EDGE_CSV = True
SAVE_METRICS = True
SAVE_SUMMARY = True
SAVE_INTERACTIVE_HTML = True
INTERACTIVE_HTML_MAX_FRAMES = 120
INTERACTIVE_HTML_MAX_PARTICLES = 300
INTERACTIVE_HTML_INCLUDE_PLOTLYJS = True  # True = offline/self-contained, "cdn" = smaller but needs internet
GIF_FPS = 20

# Metrics
COLLAPSE_RADIUS_FRACTION = 0.30
METRICS_EXACT_POTENTIAL_MAX_N = 700
METRICS_NEAREST_NEIGHBOR_MAX_N = 1500
# effective potential scaling for expansion runs:
#   none -> raw softened potential
#   a    -> divide raw potential by a(t)
#   a3   -> divide raw potential by a(t)^3, matching acceleration scaling
METRICS_POTENTIAL_SCALING = "a3"

# Animation
ANIMATION_DPI = 120
ANIMATION_POINT_SIZE_MIN = 8
ANIMATION_POINT_SIZE_MAX = 120


EXPERIMENTS = {
    "current_direct": {
        "SIMULATION_MODE": "current_direct",
        "EXPERIMENT_NAME": "current_direct",
        "FORCE_SOLVER": "direct",
        "USE_EXPANSION": False,
        "CONNECTION_VELOCITY_MODE": "radial",
        "CSV_PATH": CSV_PATH,
        "EDGE_CSV_PATH": EDGE_CSV_PATH,
    },
    "softened_no_expansion": {
        "SIMULATION_MODE": "softened_no_expansion",
        "EXPERIMENT_NAME": "softened_no_expansion",
        "FORCE_SOLVER": "direct",
        "USE_EXPANSION": False,
        "CONNECTION_VELOCITY_MODE": "radial",
        "CSV_PATH": CSV_PATH,
        "EDGE_CSV_PATH": EDGE_CSV_PATH,
    },
    "softened_expansion": {
        "SIMULATION_MODE": "softened_expansion",
        "EXPERIMENT_NAME": "softened_expansion",
        "FORCE_SOLVER": "direct",
        "USE_EXPANSION": True,
        "EXPANSION_MODEL": "linear",
        "CONNECTION_VELOCITY_MODE": "radial",
        "CSV_PATH": CSV_PATH,
        "EDGE_CSV_PATH": EDGE_CSV_PATH,
    },
    "softened_expansion_angular": {
        "SIMULATION_MODE": "softened_expansion_angular",
        "EXPERIMENT_NAME": "softened_expansion_angular",
        "FORCE_SOLVER": "direct",
        "USE_EXPANSION": True,
        "EXPANSION_MODEL": "linear",
        "CONNECTION_VELOCITY_MODE": "angular",
        # If angular generated data exists, the runner will use it. Otherwise it
        # falls back to CSV_PATH and prints a warning.
        "CSV_PATH": ANGULAR_CSV_PATH,
        "EDGE_CSV_PATH": ANGULAR_EDGE_CSV_PATH,
    },
    "barnes_hut_softened": {
        "SIMULATION_MODE": "barnes_hut_softened",
        "EXPERIMENT_NAME": "barnes_hut_softened",
        "FORCE_SOLVER": "barnes_hut",
        "USE_EXPANSION": False,
        "CONNECTION_VELOCITY_MODE": "radial",
        "CSV_PATH": CSV_PATH,
        "EDGE_CSV_PATH": EDGE_CSV_PATH,
    },
    "barnes_hut_expansion": {
        "SIMULATION_MODE": "barnes_hut_expansion",
        "EXPERIMENT_NAME": "barnes_hut_expansion",
        "FORCE_SOLVER": "barnes_hut",
        "USE_EXPANSION": True,
        "EXPANSION_MODEL": "linear",
        "CONNECTION_VELOCITY_MODE": "radial",
        "CSV_PATH": CSV_PATH,
        "EDGE_CSV_PATH": EDGE_CSV_PATH,
    },
    "barnes_hut_expansion_angular": {
        "SIMULATION_MODE": "barnes_hut_expansion_angular",
        "EXPERIMENT_NAME": "barnes_hut_expansion_angular",
        "FORCE_SOLVER": "barnes_hut",
        "USE_EXPANSION": True,
        "EXPANSION_MODEL": "linear",
        "CONNECTION_VELOCITY_MODE": "angular",
        "CSV_PATH": ANGULAR_CSV_PATH,
        "EDGE_CSV_PATH": ANGULAR_EDGE_CSV_PATH,
    },
}


def as_dict() -> dict:
    """Return the runtime-relevant config values as a mutable dictionary."""
    keys = [
        "PROJECT_ROOT", "DATA_DIR", "GENERATED_DATA_DIR", "OUTPUT_ROOT", "RUNS_DIR",
        "USE_TIMESTAMPED_RUN_DIRS", "RUN_ID", "RUN_OUTPUT_ROOT", "RUN_STARTED_AT", "RUN_NOTES",
        "RAW_CSV_PATH", "CSV_PATH", "EDGE_CSV_PATH", "MAX_PARTICLES", "RANDOM_SEED", "G", "DT",
        "STEPS", "SAVE_EVERY", "SOFTENING", "SIMULATION_MODE", "EXPERIMENT_NAME",
        "FORCE_SOLVER", "USE_EXPANSION", "EXPANSION_MODEL", "H0",
        "CONNECTION_VELOCITY_MODE", "CONNECTION_VELOCITY_SCALE",
        "USE_CONNECTION_MASS_STRENGTH", "BARNES_HUT_THETA",
        "BARNES_HUT_MAX_PARTICLES_PER_LEAF", "BARNES_HUT_MAX_DEPTH",
        "BARNES_HUT_IMPLEMENTATION", "BARNES_HUT_USE_NUMBA",
        "BARNES_HUT_DIRECT_FALLBACK_N",
        "SAVE_GIF", "SAVE_VVVV_CSV", "SAVE_EDGE_CSV", "SAVE_METRICS",
        "SAVE_SUMMARY", "SAVE_INTERACTIVE_HTML", "INTERACTIVE_HTML_MAX_FRAMES",
        "INTERACTIVE_HTML_MAX_PARTICLES", "INTERACTIVE_HTML_INCLUDE_PLOTLYJS",
        "GIF_FPS", "COLLAPSE_RADIUS_FRACTION",
        "METRICS_EXACT_POTENTIAL_MAX_N", "METRICS_NEAREST_NEIGHBOR_MAX_N",
        "METRICS_POTENTIAL_SCALING", "ANIMATION_DPI",
        "ANIMATION_POINT_SIZE_MIN", "ANIMATION_POINT_SIZE_MAX",
    ]
    return {key: globals()[key] for key in keys}


def experiment_overrides(name: str) -> dict:
    """Return overrides for a named experiment."""
    if name not in EXPERIMENTS:
        valid = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment '{name}'. Valid experiments: {valid}")
    return dict(EXPERIMENTS[name])


def output_dir_for(experiment_name: str, run_id: str | None = None, run_output_root: Path | None = None) -> Path:
    """Return the experiment output folder.

    With a run_id, experiments are placed under:
        output/experiments/runs/<run_id>/<experiment_name>/

    This prevents new runs from overwriting old outputs. Without a run_id,
    the legacy behavior is preserved for compatibility.
    """
    if run_output_root is not None:
        return Path(run_output_root) / experiment_name
    if run_id:
        return RUNS_DIR / run_id / experiment_name
    return OUTPUT_ROOT / experiment_name
