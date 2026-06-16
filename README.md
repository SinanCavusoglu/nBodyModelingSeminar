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
- Experiment runner for the four requested Sebastian comparisons
- Parameter sweep for `epsilon`, `H0`, and expansion model
- 3D Barnes-Hut octree solver for higher particle counts
- Barnes-Hut theta sweep
- CSV, JSON, GIF, and optional comparison GIF outputs

---

## 3. Project Structure

```text
config.py
main.py
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

### Sebastian's four requested comparisons

```bash
python run_experiments.py --set sebastian
```

This runs:

```text
current_direct
softened_no_expansion
softened_expansion
softened_expansion_angular
```

### Faster smoke test

```bash
python run_experiments.py --set sebastian --max-particles 50 --steps 200 --save-every 5 --no-gif
```

### Barnes-Hut experiments

```bash
python run_experiments.py --set barnes-hut --max-particles 500 --no-gif
```

### Parameter sweep

```bash
python run_experiments.py --set sweep --no-gif
```

### Barnes-Hut theta sweep

```bash
python run_experiments.py --set theta --no-gif
```

---

## 9. Outputs

Experiment outputs are written to:

```text
output/experiments/<experiment_name>/
```

Each experiment can produce:

```text
positions.csv
edges.csv
metrics.csv
animation.gif
summary.json
```

`positions.csv` contains particle positions and speeds over time.

`metrics.csv` contains quantitative analysis values:

```text
frame
time
kinetic_energy
potential_energy
virial_ratio
mean_radius
median_radius
max_radius
nearest_neighbor_mean
scale_factor
expansion_rate
```

`summary.json` stores final experiment parameters and summary metrics.

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

For the first full test:

```bash
python scripts/generate_connection_velocity.py --mode radial
python scripts/generate_connection_velocity.py --mode angular
python run_experiments.py --set sebastian --max-particles 50 --steps 200 --save-every 5 --no-gif
```

If this works, run the full comparison:

```bash
python run_experiments.py --set sebastian
```

Then test Barnes-Hut:

```bash
python run_experiments.py --set barnes-hut --max-particles 500 --no-gif
```

---


---

## Interactive HTML Visualization

In addition to `animation.gif`, each experiment now also exports an interactive Plotly file:

```text
interactive_3d.html
```

It is written next to the GIF inside each experiment folder:

```text
output/experiments/runs/<run_id>/<experiment_name>/
  animation.gif
  interactive_3d.html
  positions.csv
  metrics.csv
  summary.json
```

Open `interactive_3d.html` in a browser to rotate, zoom, pan, hover over particles, and use the frame slider. Particle colors come from the original `hue color value` column and remain visualization-only. They do not affect physics, gravity, expansion, Barnes-Hut, or metrics.

The HTML export is enabled by default. To disable it:

```bash
python run_full_pipeline.py --preset quick --no-html
```

To limit HTML size for large runs:

```bash
python run_full_pipeline.py --preset full --html-max-frames 100 --html-max-particles 250
```

You can also generate an interactive HTML file later from an existing experiment folder:

```bash
python make_interactive_html.py output/experiments/runs/<run_id>/<experiment_name>
```

The default HTML is self-contained and works offline, but the file can be large. For a smaller file that loads Plotly from the internet, use:

```bash
python make_interactive_html.py output/experiments/runs/<run_id>/<experiment_name> --cdn
```

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

## Timestamped run folders and metadata

Every execution now creates a new timestamped run folder instead of overwriting the previous experiment outputs.

Example:

```bash
python run_full_pipeline.py --preset quick --no-gif
```

Output structure:

```text
output/experiments/runs/20260614_153012_quick/
    RUN_README.md
    run_manifest.json
    run_index.json
    run_index.csv
    current_direct/
        summary.json
        experiment_manifest.json
        metrics.csv
        positions.csv
        edges.csv
    softened_no_expansion/
        ...
    softened_expansion/
        ...
    softened_expansion_angular/
        ...
```

Useful files:

- `RUN_README.md`: human-readable overview of what was run and when.
- `run_manifest.json`: metadata for the whole run, including preset, command, raw CSV, git info, and common overrides.
- `run_index.csv`: spreadsheet-friendly list of all experiments in the run.
- `run_index.json`: machine-readable list of all experiments in the run.
- `experiment_manifest.json`: per-experiment metadata inside each experiment folder.
- `summary.json`: per-experiment simulation summary and final metrics.

You can also choose a custom run id:

```bash
python run_full_pipeline.py --preset full --run-id seminar_test_01 --notes "first full run for class comparison"
```

This will create:

```text
output/experiments/runs/seminar_test_01/
```

---

## Oscillation Reduction Experiments

Sebastian suggested that if the simulation still shows a strong initial oscillation, we should test very small initial velocities or no initial velocity at all. The code now includes a dedicated oscillation experiment set and automatic data-analysis outputs.

### Run the oscillation experiment set

Fast analysis run without GIF rendering:

```bash
python run_full_pipeline.py --preset oscillation --no-gif --html-max-particles 200 --html-max-frames 80
```

With GIF rendering:

```bash
python run_full_pipeline.py --preset oscillation --with-gif --html-max-particles 300 --html-max-frames 120
```

You can also run it through `run_experiments.py`:

```bash
python run_experiments.py --set oscillation --max-particles 100 --steps 1000 --save-every 5 --no-gif
```

### What this experiment set tests

The oscillation set compares:

| Group | Purpose |
|---|---|
| Velocity source test | radial vs. angular vs. no initial velocity |
| Velocity scale test | normal scale `0.05` vs. small scale `0.01` |
| Softening test | `epsilon = 2, 5, 8, 10` |
| Expansion test | `H0 = 0, 0.01, 0.027, 0.05` |
| Economic growth test | `H0 = 0.027`, based on Sebastian's 2.7% global growth suggestion |
| Timestep test | `DT = 0.02, 0.01, 0.005` |
| Barnes-Hut candidate | best candidate with the Barnes-Hut solver |

### Generated velocity input folders

The oscillation run automatically creates additional input CSVs:

```text
data/generated/radial_scale_0p01/minimal.csv
data/generated/angular_scale_0p01/minimal.csv
data/generated/none/minimal.csv
```

The normal-scale radial/angular files are still kept at:

```text
data/generated/radial/minimal.csv
data/generated/angular/minimal.csv
```

You can manually generate scaled inputs as well:

```bash
python scripts/generate_connection_velocity.py --mode angular --velocity-scale 0.01
python scripts/generate_connection_velocity.py --mode radial --velocity-scale 0.01
python scripts/generate_connection_velocity.py --mode none
```

### Oscillation analysis outputs

Every oscillation run creates these run-level files:

```text
output/experiments/runs/<run_id>/oscillation_summary.csv
output/experiments/runs/<run_id>/oscillation_summary.json
output/experiments/runs/<run_id>/OSCILLATION_ANALYSIS.md
output/experiments/runs/<run_id>/plots/oscillation_mean_radius.png
output/experiments/runs/<run_id>/plots/oscillation_kinetic_energy.png
output/experiments/runs/<run_id>/plots/oscillation_virial_ratio.png
output/experiments/runs/<run_id>/plots/oscillation_nearest_neighbor.png
```

### How to read `oscillation_summary.csv`

The most important columns are:

| Column | Meaning |
|---|---|
| `rank` | automatic ranking; lower rank is better |
| `stability_score` | combined score; lower means smoother/more stable behavior |
| `radius_oscillation_score` | how strongly `mean_radius` zigzags over time |
| `kinetic_spike_score` | how strong kinetic-energy spikes are |
| `virial_spike_score` | how strongly the virial ratio spikes |
| `collapse_time` | first collapse time; empty/null means no collapse detected |
| `collapse_penalty` | penalty for early collapse |
| `connection_velocity_mode` | radial, angular, or none |
| `connection_velocity_scale` | initial velocity multiplier |
| `softening` | gravitational softening length |
| `H0` | expansion/economic-growth parameter |
| `dt` | timestep |

Lower `stability_score` is better. The score is not meant to be a perfect physical law; it is a practical ranking tool for detecting which setup reduces the strong oscillation most clearly.

### Recommended first interpretation

Start by comparing:

```text
osc_radial_0p05
osc_angular_0p05
osc_radial_0p01
osc_angular_0p01
osc_none
```

If `osc_none` or `osc_angular_0p01` is much smoother, then the strong oscillation is strongly influenced by the initial velocity field. If softening or `H0 = 0.027` improves the score further, then close-range gravitational collapse and expansion damping are also important.
