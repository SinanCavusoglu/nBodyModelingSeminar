# Final Cross-Check Runner

This patch adds `run_final_cross_check_experiments.py`.

It is designed for a strong PC. GIF and HTML rendering are disabled, but this version saves `positions.csv` for each experiment by default. It also keeps metrics, summaries, manifests, ranked CSV files, and plots.

## What it tests

The runner performs a coarse cross-check over the main collapse-control parameters. Expansion is not varied anymore; it is kept ON with `H0=0.027` because earlier benchmarks showed it was useful:

- position scale
- mass model
- gravitational softening
- angular velocity scale
- fixed expansion rate `H0=0.027`
- virialized initial velocity scaling
- adaptive gravitational softening

## Recommended final run

```powershell
python run_final_cross_check_experiments.py --profile final --run-id final_cross_check_01 --max-particles 3426 --steps 3000 --save-every 5
```

The `final` profile runs 288 experiments:

```text
2 position scales × 4 mass models × 3 softening values × 3 angular velocity scales × 1 fixed H0 value × 4 stabilization states = 288
```

This is large. It avoids GIF/HTML rendering, but it writes full `positions.csv` histories by default.

## Smaller test run

```powershell
python run_final_cross_check_experiments.py --profile smoke --max-particles 100 --steps 20
```

## Focused run

```powershell
python run_final_cross_check_experiments.py --profile focused --run-id focused_cross_check_01 --max-particles 3426 --steps 1500 --save-every 5
```

The `focused` profile runs 48 experiments.

## Very large run

```powershell
python run_final_cross_check_experiments.py --profile full --run-id full_cross_check_01 --max-particles 3426 --steps 3000 --save-every 5
```

The `full` profile runs 432 experiments, including adaptive `k=2` states. Expansion is still fixed at `H0=0.027`.

## Useful options

List experiments without running:

```powershell
python run_final_cross_check_experiments.py --profile final --list
```

Dry run:

```powershell
python run_final_cross_check_experiments.py --profile final --dry-run
```

Run only first N experiments:

```powershell
python run_final_cross_check_experiments.py --profile final --max-experiments 20
```

Resume an interrupted run:

```powershell
python run_final_cross_check_experiments.py --profile final --run-id final_cross_check_01 --resume
```

Disable positions if you only want a lighter analysis-only run:

```powershell
python run_final_cross_check_experiments.py --profile final --no-save-positions
```

## Output files

The run folder is created under:

```text
output/experiments/runs/<run_id>/
```

Important files:

- `final_cross_check_summary.csv`: best ranked table with parameters and stability score
- `oscillation_summary.csv`: raw automatic stability ranking
- `run_index.csv`: all experiments and output paths
- `*/positions.csv`: full saved particle positions for each experiment folder
- `FINAL_CROSS_CHECK_ANALYSIS.md`: top-20 readable summary
- `plots/`: metric comparison plots

