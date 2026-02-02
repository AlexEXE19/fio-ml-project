# Dataset guide

Experiments were run on a single local system using fio. Two scenarios are captured: open loop runs (fixed commands) and closed loop runs (PI-controlled). The CLI in `index.py` currently loads only the open loop folder.

## Folder structure

```
datasets/
├── 1.openloop/
│   ├── measures0.csv
│   ├── ...
│   └── measures9.csv
└── 2.closed-loop-10runs/
    ├── run-0/
    │   ├── config.json
    │   ├── control.csv
    │   └── sensor-dispatch.csv
    ├── ...
    └── run-9/
```

## Open loop (1.openloop)

- 10 CSVs (`measures0.csv` ... `measures9.csv`), each a separate run.
- Columns: `time`, `wiops`, `latency`, `dispatch`.
- Example rows (truncated):
  - `1730818413390259424,50,1.6831,0`
  - `1730818413490259592,50,1.7270,49.36`

## Closed loop (2.closed-loop-10runs)

Each `run-*` directory contains (not used by the CLI, available for manual analysis):

- `config.json`: PI controller parameters (fields include `kp`, `ki`, `ks`, `dt`, `output_range`, `sensor`, `actuator`, `target`, `device`, `a`, `b`, `mp`).
- `control.csv`: controller outputs with columns `time`, `target`, `error`, `control_output`, `wiops_output`.
- `sensor-dispatch.csv`: measured queue size with columns `time`, `dispatch`.

## Notes and usage

- Timestamps are stored as large integers; retain original units when plotting or training.
- The CLI defaults to loading all open loop files. You can select subsets or ranges when prompted.
- Closed loop files are available for offline analysis or custom pipelines; they are not auto-loaded by the current CLI menu.
- Outputs produced by the CLI are written to `outputs/` and do not modify these source files.
