# Virialized Velocity + Adaptive Softening Patch

This patch is built for the uploaded `src_last.zip` / `feature/comoving-barnes-hut-experiments` style codebase.

## Added features

### 1. Virialized initial velocity scaling
Scales the initial velocity field toward a target virial ratio:

```text
Q = 2K / |U|
```

Default target:

```text
TARGET_VIRIAL_RATIO = 1.0
```

It preserves existing velocity directions, including connection-based angular velocity, but rescales magnitude.

### 2. Adaptive gravitational softening
Adds optional per-particle softening values. Default mode is `density_boost`, which increases softening in locally dense regions based on nearest-neighbor distance.

## Changed files

- `config.py`
- `main.py`
- `src/physics.py`
- `src/simulation.py`
- `src/barnes_hut_fast.py`
- `src/export.py`
- `run_anti_collapse_experiments.py`
- `run_anti_collapse_experiments_WITH_RENDER.py`

## New file

- `run_virial_adaptive_experiments.py`

## Quick usage

Run one normal experiment with both extensions enabled:

```powershell
python main.py --experiment barnes_hut_expansion_angular --max-particles 3426 --steps 1000 --save-every 2 --no-gif --no-html --virial --adaptive-softening
```

Run the focused follow-up benchmark:

```powershell
python run_virial_adaptive_experiments.py --run-id virial_adaptive_no_visual_01 --max-particles 3426 --steps 1000 --save-every 2
```

Render one selected follow-up candidate:

```powershell
python run_virial_adaptive_experiments.py --only osc_va_virial_adaptive_k1 --run-id virial_adaptive_k1_html_01 --max-particles 3426 --steps 3000 --save-every 5 --with-html
```

List available follow-up experiments:

```powershell
python run_virial_adaptive_experiments.py --list
```
