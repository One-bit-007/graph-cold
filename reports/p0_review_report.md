# P0 Submission-Blocker Review

## Scope

This branch fixes only the P0 blockers identified by the simulated Computers & Security review:

- FATAL-1: non-real experiment artifacts were removed from the submission path.
- FATAL-2: D5 no longer writes generated or placeholder experiment numbers when raw data are missing.
- FATAL-3: ERR now measures binary evidence retention on clean informative samples.

## Before / After

| Blocker | Before | After |
|---|---:|---:|
| Non-real D5 artifacts | `results/` and D6 tables contained non-real result rows and submission PDFs referenced them | Invalid `results/`, `tables/`, `figures/`, and submission PDFs were removed; D5 requires real files |
| Implausible result profile | Graph-CoLD 97.2% Macro-F1, OpTC 100.0% Macro-F1, p=9.04e-93 in removed artifacts | No result table is produced without real CICIDS-2017, MALTLS-22, and OpTC inputs |
| ERR direction | Removed Table 2 showed full ERR 69.4% lower than `ablation_hard` 100.0% | Unit fixture now gives soft ERR=1.000 and hard-delete ERR=0.048 on the same clean informative set |

## Code Changes

- `src/data/loaders.py`: missing CICIDS-2017/MALTLS-22 files raise `FileNotFoundError` with placement instructions; metadata records `data_source` and `data_version`.
- `src/experiments/d5.py`: removed generated-data branch; datasets are `cicids2017`, `maltls22`, and `optc`; rows include source/version columns; methods train on real feature matrices when data exist.
- `src/enterprise/optc_case.py`: real `data/optc/events.csv` is required; no generated enterprise case is created.
- `src/metrics.py`: ERR uses `retained(v)=1[w(v) >= tau_ret]` over `clean AND informative`; `clean_mask` must match `~flip_mask`.
- `src/models/cold_baseline.py`: CoLD representation remains tied to the D2 multi-view encoder, not PCA.

## Reproduction Commands

Missing-data gate:

```bash
python -m pytest -q tests/test_d5_experiments.py::test_d5_p0_fails_loud_when_real_data_is_missing
```

ERR gate:

```bash
python -m pytest -q tests/test_err.py
```

Full local validation:

```bash
python -m pytest -q
```

Real-data D5 matrix after placing datasets:

```bash
python -c "from src.experiments.d5 import run_d5_experiments; run_d5_experiments(out_dir='results', configs_dir='configs')"
```

Required local inputs:

- `data/cicids2017/` with CICIDS-2017 tables and `Label`.
- `data/maltls22/` with MALTLS-22 tables and `label`.
- `data/optc/events.csv` with `host_id`, `process_id`, `parent_process_id`, `src_ip`, `dst_ip`, `timestamp`, `event_type`, `alert_type`, `label`, and `risk_score`.

## Verification

- `pytest -q`: 36 passed, 2 skipped.
- Skips are expected locally because real D5 result tables are absent after removing non-real artifacts.
- Current P0 state is intentionally not submission-ready until the real-data D5 matrix has been rerun and D6/D7 tables/manuscripts are regenerated from those real outputs.
