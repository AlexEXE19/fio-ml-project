# FIO Machine Learning Toolkit

CLI-driven utilities for exploring fio-based storage experiments, plotting time series, and training simple regression and XGBoost models on dispatch queue data.

## What this project does

- Loads fio experiments from `datasets/1.openloop` (see [datasets/README.md](datasets/README.md) for details)
- Fits 2D or 3D polynomial regressions to study relationships between `wiops`, `dispatch`, and `latency`
- Trains XGBoost regressors for latency prediction with alternative feature subsets
- Plots per-column time series for single runs and combined datasets
- Saves plots and trained models in timestamped folders under `outputs/`

## Repository layout

- `index.py` entrypoint wiring the CLI
- `src/utils/cli_utils.py` interactive menu and data selection helpers
- `src/utils/pred_plot_utils.py` data loading, plotting, and model helpers
- `datasets/` raw experiment data (open loop and closed loop)
- `outputs/` generated plots and models (created at runtime)
- `tests/` pytest coverage for core helpers

## Getting started

Requirements: Python 3.10+ and `pip`.

1. Create and activate a virtual environment (example using `venv`):
   - Windows: `python -m venv venv && venv\Scripts\activate`
   - Linux/macOS: `python3 -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the CLI from the repo root:
   - Windows: `python -m index`
   - Linux/macOS: `python3 -m index`

## Using the CLI

On launch you will see:

```
1. Predict and plot a regression model
2. Plot timeseries for each column
3. Predict and plot an XGB model
q. Quit
```

- Option 1 (regression): choose training files by index or range, select 2D or 3D regression, then pick columns and optionally force polynomial degrees. Models and plots land in `outputs/plots/regressions/...`.
- Option 2 (timeseries): generates per-column time series for every dataset plus a combined view, saved under `outputs/plots/timeseries/...`.
- Option 3 (XGB): trains latency predictors with all features, `time+wiops`, and `time+dispatch`, saving plots in `outputs/plots/xgb/...`.

All runs create timestamped folders so results from different sessions stay separate.

## Data expectations

- Open loop CSVs expose columns: `time`, `wiops`, `latency`, `dispatch`.
- Closed loop files are documented but not used by the current CLI flow.
- Config files describe controller parameters (see dataset README for an example schema).

## Outputs

- Plots: `outputs/plots/` organized by task (`regressions/2d`, `regressions/3d`, `timeseries/`, `xgb/`), each run in a timestamped subfolder.
- Models: `outputs/models/xgb/<timestamp>/` for trained XGBoost artifacts.
- Metrics: regression utilities store the best degrees and scores in JSON helpers within `outputs/` when applicable.

## Testing

Run the test suite with pytest from the repo root:

```
python -m pytest
```

## Troubleshooting

- If you add or remove datasets, restart the CLI so file discovery re-runs.
- Ensure the `outputs/` directory is writable; the CLI will create nested folders automatically.
