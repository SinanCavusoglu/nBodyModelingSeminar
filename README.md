# 3D Billionaires N-body Simulation

This project converts a Forbes billionaire dataset into a 3D particle simulation.
Each row is one person. The person starts at normalized `x`, `y`, `z` coordinates,
uses `NetWorth` as mass, and uses network connections to calculate the initial velocity.

## Data model

The joined source dataset should be here:

```text
data/forbes_billionaires_JOINED_connections_industries_v2.csv
```

Required input columns:

```text
ID, Name, NetWorth, x, y, z, industries, connections
```

`connections` must contain base-CSV IDs separated with `/`, for example:

```text
4/395/87/1574
```

Only IDs that exist in the dataset are used. Self-connections are removed.

## Initial velocity logic

For each person `i`, the preprocessing script reads the connected people `j`.

For every connection:

```text
direction_ij = normalize(position_j - position_i)
mass_j = NetWorth_j in billions
```

Then the initial velocity is calculated as:

```text
weighted_average_vector_i = sum(direction_ij * mass_j) / sum(mass_j)
mass_strength_i = log1p(sum(mass_j)) / log1p(max_total_connected_mass)
velocity_i = weighted_average_vector_i * CONNECTION_VELOCITY_SCALE * mass_strength_i
```

This means:

- connected people define the movement direction
- richer connected people pull the direction more strongly
- people connected to more / richer people get stronger initial speed
- opposite connection directions can cancel each other
- rows with no valid connections default to zero initial velocity

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Generate simulation CSVs

Run this first whenever the joined dataset changes:

```bash
python scripts/generate_connection_velocity.py
```

It creates:

```text
data/forbes_billionaires_simulation_minimal_v4.csv
data/forbes_billionaires_simulation_rich_v4.csv
data/forbes_billionaires_edges_v4.csv
```

The minimal file is the best input for the simulation.
The rich file keeps extra metadata for tooltips, color mapping, filtering, and later visual layers.
The edge file is for drawing network connections in vvvv or another 3D tool.

## Run the simulation

```bash
python main.py
```

Outputs are saved into:

```text
output/
```

Main outputs:

```text
output/billionaires_nbody_for_vvvv.csv
output/billionaires_edges_for_vvvv.csv
output/billionaires_nbody_3d.gif
```

## Important settings

Edit `config.py`.

For quick GIF tests:

```python
MAX_PARTICLES = 250
SAVE_GIF = True
SAVE_VVVV_CSV = True
SAVE_EDGE_CSV = True
```

For full-dataset vvvv export:

```python
MAX_PARTICLES = None
SAVE_GIF = False
SAVE_VVVV_CSV = True
SAVE_EDGE_CSV = True
STEPS = 1000
SAVE_EVERY = 10
```

The simulation is still all-to-all gravity, so it is `O(n^2)`. The physics step is
chunked and vectorized, but a full 3400-person simulation with many frames can still
be heavy. Barnes-Hut is the next major performance upgrade.

## vvvv usage

Particle CSV:

```text
frame, id, name, x, y, z, vx, vy, vz, mass, mass_normalized, speed
```

Edge CSV:

```text
source_id, target_id, source_name, target_name,
source_x, source_y, source_z, target_x, target_y, target_z,
dx, dy, dz, distance, source_mass_billions, target_mass_billions
```

A simple vvvv setup can:

1. read the particle CSV
2. select a frame
3. use `x`, `y`, `z` for point positions
4. use `mass_normalized` for sphere size
5. use `speed`, `industries`, or metadata from the rich CSV for color
6. read the edge CSV and draw lines from `source_id` to `target_id`

## Files to use

Use this as the simulation input:

```text
forbes_billionaires_simulation_minimal_v4.csv
```

Keep this as the master metadata file:

```text
forbes_billionaires_simulation_rich_v4.csv
```
