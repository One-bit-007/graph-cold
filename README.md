# Graph-CoLD

**Evidence-Preserving Collaborative Graph Denoising and Prioritization for SOC Alert Analysis**

Target venue: *Computers & Security* (Elsevier).

Graph-CoLD extends the CoLD framework (NDSS'26, sample-level label-noise
purification for IDS) to the **SOC alert level**. Instead of *deleting* noisy
samples, Graph-CoLD builds a **multi-view heterogeneous temporal graph**, scores
inconsistency with a **Graph Causal Divergence Metric (Graph-CDM)**, applies an
**evidence-preserving soft reweighting** (keeps low-frequency / early-APT /
boundary samples), trains a robust classifier, and produces a **Top-K alert
prioritization**.

## Why not just reuse CoLD?
| CoLD | Graph-CoLD |
|---|---|
| Feature-subset views | Five-view heterogeneous temporal graph |
| GMM-CDM | Graph-CDM (prediction + neighborhood + view consistency) |
| **Hard deletion** of noisy samples | **Evidence-preserving soft reweighting** |
| No ranking | SOC priority Top-K + alert compression |
| — | Evidence Retention Rate (ERR) metric |

## Status
Bootstrap scaffold. No official CoLD code exists; the CoLD baseline is
re-implemented here from scratch (`src/models/cold_baseline.py`) and serves both
as a comparison baseline and as the object of extension.

## Repo map
```
configs/     yaml configs (datasets / noise / model / experiment)
src/data/    dataset loading + noise injection (sym / asym / graph-consistency)
src/graph/   five-view heterogeneous + temporal graph construction
src/models/  CoLD baseline, heterogeneous encoders, Graph-CDM + soft reweighting
src/ranking/ SOC alert priority Top-K
src/         train.py / eval.py / metrics.py
scripts/     one-command reproduction scripts
docs/        design specs (graph-consistency noise, method implementation)
experiments/ produced result tables & figures
paper/       manuscript sources (Computers & Security)
```

## Quickstart (to be filled by Codex)
```bash
python -m venv .venv && . .venv/Scripts/activate
pip install -r requirements.txt
bash scripts/run_all.sh   # or scripts/run_all.ps1 on Windows
```

## Datasets
- **CICIDS-2017** (refined, 9 classes) — main benchmark
- **MALTLS-22** (encrypted, 23 classes) — main benchmark
- **OpTC** — enterprise case study only (Flash/Argus + Graph-CoLD plug-in)

## Division of labor
- **Codex**: all `src/`, `configs/`, `scripts/`, experiment runs.
- **GPT-pro**: `docs/` math finalization, `paper/`, figures, rebuttal.
