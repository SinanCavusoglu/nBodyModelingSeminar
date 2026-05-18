# 3D Billionaires N-body Simulation

This project turns a CSV file of billionaire data into a simple 3D particle simulation.
Each row in the CSV becomes one moving sphere.

The main idea is:

- `X`, `Y`, `Z` define the starting position.
- `NetWorth_Billions` defines the mass and visual size of the sphere.
- `ForceBasedOnAge` defines the initial speed.
- The initial movement direction is partly random and partly pulled toward the center.

The project saves two useful outputs:

- a GIF animation
- a CSV file that can be imported into vvvv "we can change later for vvvv!!!"


## Setup in VS Code

Open the project folder in VS Code.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required modules:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## CSV format

The CSV file should be placed here:

```text
data/BillionairePositionsMapped.csv
```

It must include these columns:

```text
Name, NetWorth_Billions, X, Y, Z, ForceBasedOnAge
```

## Main settings

Edit `config.py` to change the simulation.

Useful settings:

```python
MAX_PARTICLES = 1000
STEPS = 5000
SAVE_EVERY = 3
VELOCITY_SCALE = 0.05
G = 0.005
DT = 0.02
SOFTENING = 2.0
```

If the spheres move too fast or fly away, try:

```python
VELOCITY_SCALE = 0.01
G = 0.001
DT = 0.01
SOFTENING = 3.0
```

If the GIF is too long or too large, reduce `STEPS` or increase `SAVE_EVERY`.


A simple vvvv setup can do this:

1. Read the CSV.
2. Select one frame number.
3. Filter rows where `frame` equals the selected frame.
4. Use `x`, `y`, `z` as sphere positions.
5. Use `mass` as sphere size.
6. Use `speed` for color, brightness, or trails.

## To-do

- Add Barnes-Hut algorithm for larger particle counts.
- Add benchmark mode to compare performance.

in VVVV
- Add optional trails for each particle.
- Add color mapping based on industry, source, or speed.
