# Forbes N-body Simulation Upgrade

This project models Forbes billionaire data as a 3D N-body simulation. Each billionaire is represented as a particle:

- **Position**: `x`, `y`, `z` columns from the input CSV
- **Mass**: derived from `NetWorth`
- **Connections**: parsed from the `connections` column
- **Initial velocity**: generated from connection data, either radial or angular
- **Evolution**: computed with direct softened gravity or Barnes-Hut accelerated gravity

The current upgrade implements Sebastian's suggested roadmap: softened gravity, comoving expansion, angular connection-based initial velocities, metrics, visual outputs, and Barnes-Hut support.

---

## 1. Input CSV

The package is configured for the uploaded CSV:

```text
data/forbes_billionaires_WorthPosConnectionsColor_20260610.csv
```

Expected columns:

```text
ID
Name
NetWorth
x
y
z
industries
connections
hue color value
```

Notes:

- `connections` may contain IDs separated by `/`, for example `4/395/87`.
- IDs such as `4.0` are normalized to `4`, so they match connection entries correctly.
- `NetWorth` is converted automatically into simulation mass. If values are large raw dollar values, they are converted to billions.

---

## 2. Main Features

Implemented features:

- Current direct-solver baseline
- Explicit softened gravity
- Comoving-coordinate expansion
- Expansion damping term `-2H(t)v`
- Radial, angular, or no connection-based initial velocity
- Kinetic energy, potential energy, virial ratio, radius, clustering, and collapse metrics
- Full pipeline runner with timestamped output folders
- Experiment runner for the four requested Sebastian comparisons
- Parameter sweep for `epsilon`, `H0`, and expansion model
- 3D Barnes-Hut octree solver for higher particle counts
- Barnes-Hut theta sweep
- CSV, JSON, GIF, optional comparison GIF outputs, and timestamped run folders

---

## 3. Project Structure

```text
config.py
main.py
run_full_pipeline.py
run_experiments.py
requirements.txt

scripts/
  generate_connection_velocity.py

src/
  animation.py
  barnes_hut.py
  cosmology.py
  data_loader.py
  export.py
  metrics.py
  physics.py
  simulation.py

docs/
  IMPLEMENTATION_NOTES.md

data/
  forbes_billionaires_WorthPosConnectionsColor_20260610.csv
  generated/
    radial/
    angular/

output/
  experiments/
    runs/
      <timestamp_or_custom_run_id>/
        RUN_README.md
        run_manifest.json
        run_index.json
        run_index.csv
        <experiment_name>/
          positions.csv
          edges.csv
          metrics.csv
          summary.json
          experiment_manifest.json
          animation.gif
```

---

## 4. Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
```

Required packages:

```text
numpy
pandas
matplotlib
pillow
```

---

## 5. Generate Simulation Input CSVs

The raw uploaded CSV does not contain `vx`, `vy`, and `vz`. These are generated from the connection data.

### Radial connection velocity

Radial mode points particles toward the mass-weighted barycenter of their connected particles.

```bash
python scripts/generate_connection_velocity.py --mode radial
```

Outputs:

```text
data/generated/radial/minimal.csv
data/generated/radial/rich.csv
data/generated/radial/edges.csv
```

### Angular connection velocity

Angular mode gives particles tangential velocity around the mass-weighted barycenter of their connected particles.

```bash
python scripts/generate_connection_velocity.py --mode angular
```

Outputs:

```text
data/generated/angular/minimal.csv
data/generated/angular/rich.csv
data/generated/angular/edges.csv
```

### No connection velocity

```bash
python scripts/generate_connection_velocity.py --mode none
```

---

## 6. Simulation Modes

The main simulation modes are configured through `config.py` and `run_experiments.py`.

| Mode | Solver | Gravity | Expansion | Initial velocity |
|---|---|---|---|---|
| `current_direct` | direct | current/direct softened | no | radial/current |
| `softened_no_expansion` | direct | softened | no | radial |
| `softened_expansion` | direct | softened | yes | radial |
| `softened_expansion_angular` | direct | softened | yes | angular |
| `barnes_hut_softened` | Barnes-Hut | softened | no | radial |
| `barnes_hut_expansion` | Barnes-Hut | softened | yes | radial |
| `barnes_hut_expansion_angular` | Barnes-Hut | softened | yes | angular |

---

## 7. Physics Model

### Softened gravity

The direct softened acceleration is:

```text
acc_i = sum_j G * m_j * (x_j - x_i) / (|x_j - x_i|^2 + epsilon^2)^(3/2)
```

The softening length `epsilon` prevents singular close-range accelerations.

### Comoving expansion

The expansion model follows Sebastian's suggested equation:

```text
x_i'' + 2H(t)x_i' = 1/a(t)^3 * sum_j Gm_j * (x_j - x_i) / (|x_j - x_i|^2 + epsilon^2)^(3/2)
```

Implementation form:

```text
acceleration = softened_gravity / a(t)^3 - 2 * H(t) * velocity
```

Default linear expansion:

```text
a(t) = 1 + H0 * t
H(t) = H0 / a(t)
```

Optional exponential expansion:

```text
a(t) = exp(H0 * t)
H(t) = H0
```

---

## 8. Running Experiments

The recommended entry point is now:

```bash
python run_full_pipeline.py --preset quick
```

This runs the full project pipeline in one command:

```text
1. Check the raw Forbes CSV.
2. Generate radial velocity simulation input.
3. Generate angular velocity simulation input.
4. Run the Sebastian comparison experiments.
5. Write all outputs into a timestamped run folder.
6. Write run-level metadata files so previous results are not overwritten.
```

### Quick smoke test

Use this first to confirm that the full pipeline works:

```bash
python run_full_pipeline.py --preset quick --no-gif
```

Quick mode uses fewer particles and fewer steps. It is intended for testing, debugging, and checking that all files are generated correctly.

To test quick mode with GIF generation:

```bash
python run_full_pipeline.py --preset quick --with-gif
```

### Full run

Use this for real experiment outputs:

```bash
python run_full_pipeline.py --preset full
```

For a faster full run without GIF rendering:

```bash
python run_full_pipeline.py --preset full --no-gif
```

### Full run with sweeps

To include parameter sweeps and Barnes-Hut theta sweeps:

```bash
python run_full_pipeline.py --preset full --include-sweep --include-theta --no-gif
```

### Custom run ID and notes

Each run is automatically written to a timestamped folder. You can also provide your own run ID:

```bash
python run_full_pipeline.py --preset full --run-id seminar_test_01
```

You can add notes that will be saved into the run metadata:

```bash
python run_full_pipeline.py --preset full --run-id seminar_test_01 --notes "first full comparison for seminar"
```

### Direct experiment runner

You can still run experiment sets directly:

```bash
python run_experiments.py --set sebastian
```

Sebastian's four requested comparisons are:

```text
current_direct
softened_no_expansion
softened_expansion
softened_expansion_angular
```

Other direct experiment commands:

```bash
python run_experiments.py --set sebastian --max-particles 50 --steps 200 --save-every 5 --no-gif
python run_experiments.py --set barnes-hut --max-particles 500 --no-gif
python run_experiments.py --set sweep --no-gif
python run_experiments.py --set theta --no-gif
```

---

## 9. Timestamped Outputs and Run Tracking

Outputs are now written into a separate run folder each time the pipeline is executed. This prevents new runs from overwriting previous results.

Default structure:

```text
output/experiments/runs/<timestamp>_<preset>/
```

Example:

```text
output/experiments/runs/20260614_153012_quick/
```

If you provide a custom run ID:

```bash
python run_full_pipeline.py --preset full --run-id seminar_test_01
```

the output is written to:

```text
output/experiments/runs/seminar_test_01/
```

### Run-level files

Each run folder contains:

```text
RUN_README.md
run_manifest.json
run_index.json
run_index.csv
```

#### `RUN_README.md`

A human-readable summary of the run. This is the first file to open after a run finishes.

It includes:

```text
run ID
start time
end time
preset
notes
raw CSV path
generated input files
experiments executed
particle count
step count
save interval
GIF setting
output paths
```

#### `run_manifest.json`

A machine-readable record of the whole run.

It stores:

```text
run metadata
command-line settings
preset
notes
start and end timestamps
input data paths
list of experiment outputs
```

#### `run_index.json`

A JSON index of all experiments executed during that run.

#### `run_index.csv`

A spreadsheet-friendly table summarizing all experiments in the run. This is useful for quickly checking what was run and where the outputs are.

### Experiment-level files

Each experiment is written into its own folder inside the run folder:

```text
output/experiments/runs/<run_id>/<experiment_name>/
```

Example:

```text
output/experiments/runs/20260614_153012_quick/softened_expansion_angular/
```

Each experiment can produce:

```text
positions.csv
edges.csv
metrics.csv
summary.json
experiment_manifest.json
animation.gif
```

If `--no-gif` is used, `animation.gif` is not generated.

### `positions.csv`

Contains particle positions, velocities, speed, mass, and color over time.

Typical columns:

```text
frame
time
id
name
x
y
z
vx
vy
vz
mass
speed
color
```

### `metrics.csv`

Contains quantitative analysis values:

```text
frame
time
kinetic_energy
potential_energy_raw
potential_energy_effective
virial_ratio
mean_radius
median_radius
max_radius
nearest_neighbor_mean
scale_factor
expansion_rate
collapsed
```

### `summary.json`

Stores final experiment parameters and summary metrics.

It is useful for checking:

```text
solver type
simulation mode
expansion settings
softening
H0
Barnes-Hut theta
particle count
runtime
collapse time
final virial ratio
final radius values
```

### `experiment_manifest.json`

Stores metadata for one specific experiment, including when it was run and which configuration was used.

Open this file when you want to know exactly what happened in a specific experiment folder.

---

## 10. Metrics

Implemented metrics:

- Kinetic energy
- Softened potential energy
- Virial ratio `2K / |U|`
- Mean radius from barycenter
- Median radius from barycenter
- Maximum radius from barycenter
- Mean nearest-neighbor distance
- Collapse time estimate
- Scale factor `a(t)`
- Expansion rate `H(t)`

Collapse time is defined as the first time when:

```text
mean_radius < COLLAPSE_RADIUS_FRACTION * initial_mean_radius
```

The default collapse threshold is:

```python
COLLAPSE_RADIUS_FRACTION = 0.3
```

---

## 11. Barnes-Hut Solver

Barnes-Hut is added as a performance-oriented force solver. It does not replace the physical model; it approximates the force computation so larger particle counts can be tested.

The direct solver scales as:

```text
O(N^2)
```

Barnes-Hut usually scales closer to:

```text
O(N log N)
```

Because the simulation is 3D, the implementation uses an octree.

Important config values:

```python
FORCE_SOLVER = "barnes_hut"
BARNES_HUT_THETA = 0.5
BARNES_HUT_MAX_PARTICLES_PER_LEAF = 1
```

Theta controls the speed/accuracy tradeoff:

```text
smaller theta -> more accurate, slower
larger theta  -> faster, less accurate
```

Recommended test values:

```text
theta = 0.3, 0.5, 0.8, 1.0
```

---

## 12. Sebastian Request Coverage

| Sebastian's request | Implementation |
|---|---|
| Do not use Sundman transformations | Sundman is not included |
| Current direct Newtonian baseline | `current_direct` |
| Softened gravity without expansion | `softened_no_expansion` |
| Softened gravity with expansion | `softened_expansion` |
| Expansion + angular connection velocities | `softened_expansion_angular` |
| Expansion/damping term | `src/physics.py`, `src/cosmology.py` |
| Connections only as initial velocity | `scripts/generate_connection_velocity.py` |
| Angular/tangential connection velocity | `--mode angular` |
| Collapse/clustering behavior | `src/metrics.py` |
| Kinetic vs. potential energy | `src/metrics.py` |
| Virial ratio | `src/metrics.py` |
| Effect of epsilon, H(t), a(t) | `run_experiments.py --set sweep` |
| Visual animation comparison | GIF outputs and comparison GIF support |
| Barnes-Hut | `src/barnes_hut.py` |
| Higher particle count | Barnes-Hut experiments |

---

## 13. Recommended Workflow

### Step 1: Run a quick smoke test

```bash
python run_full_pipeline.py --preset quick --no-gif
```

After it finishes, open:

```text
output/experiments/runs/<latest_run>/RUN_README.md
```

Check that the expected experiments completed and that each experiment folder contains `summary.json` and `metrics.csv`.

### Step 2: Test quick GIF rendering

```bash
python run_full_pipeline.py --preset quick --with-gif
```

This verifies that the animation pipeline works and that colors from `hue color value` are visible in the GIFs.

### Step 3: Run the full Sebastian comparison

```bash
python run_full_pipeline.py --preset full --no-gif
```

Use `--no-gif` first because GIF rendering is the slowest part.

### Step 4: Run the full comparison with GIFs

```bash
python run_full_pipeline.py --preset full
```

### Step 5: Run sweeps if needed

```bash
python run_full_pipeline.py --preset full --include-sweep --include-theta --no-gif
```

### Step 6: Use custom run IDs for important runs

For important runs, use a clear run ID and notes:

```bash
python run_full_pipeline.py --preset full --run-id seminar_final_01 --notes "final full run for Sebastian comparison"
```

This creates:

```text
output/experiments/runs/seminar_final_01/
```

and prevents confusion between multiple runs.

---

## 14. Strategic Summary

The upgrade strategy is:

1. Preserve the original direct-solver behavior as a baseline.
2. Make softened gravity explicit and measurable.
3. Add comoving expansion and damping.
4. Use connections only for initial velocity generation.
5. Add angular connection-based velocity to reduce direct inward collapse.
6. Add quantitative metrics for comparison.
7. Add Barnes-Hut as a separate performance solver for higher particle counts.

This keeps the main project structure intact while directly addressing Sebastian's feedback.


## Visualization colors

The uploaded CSV column `hue color value` is preserved in the generated radial/angular simulation inputs. These values are used only for visualization in `animation.gif` and optional `comparison.gif`. They are not passed to the physics solver and do not affect gravity, expansion, damping, Barnes-Hut, or metrics.

If you already generated `data/generated/radial/minimal.csv` and `data/generated/angular/minimal.csv` before this change, regenerate them:

```bash
python scripts/generate_connection_velocity.py --mode radial
python scripts/generate_connection_velocity.py --mode angular
```
