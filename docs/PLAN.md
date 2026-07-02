# Graph-CoLD — 7-Day Execution Plan & Division of Labor

Target: *Computers & Security*. Baseline: CoLD (NDSS'26, no official code → self
re-implemented). Compute: GPT-pro + Codex, GPU ready. Sync: github.com/One-bit-007.

## Scope (locked)
- Main experiments: CICIDS-2017 + MALTLS-22 with constructed multi-view graphs.
- OpTC: single enterprise case study (Flash/Argus + Graph-CoLD plug-in).

## Roles
- **Codex** — all code under `src/`, `configs/`, `scripts/`; runs experiments.
- **GPT-pro** — `docs/` math finalization, `paper/`, Figure 1, rebuttal.
- **Copilot (coordinator)** — turns specs into Codex-ready contracts, reviews
  code correctness, checks experiment consistency.

## Daily plan
| Day | Codex | GPT-pro |
|-----|-------|---------|
| D1 | repo, data pipeline (CICIDS/MALTLS), sym/asym noise, CoLD baseline skeleton | finalize Graph-CDM / soft-weight math, graph-noise definition (docs/) |
| D2 | five-view graph build + multi-view representation (stage 1); CoLD first F1 | draft Method §4.1–4.2 |
| D3 | Graph-CDM + evidence soft weights + weighted classifier (end-to-end) | draft Method §4.3 + ERR definition |
| D4 | OpTC ingest (Flash/Argus feats, XGBoost plug-in); graph-consistency noise | draft §6 enterprise + §3 motivation |
| D5 | full matrix: 3 datasets × noise × 8 baselines + ablations → csv | — |
| D6 | ERR / compression / Top-K / overhead figures; paired t-tests | write §5 results |
| D7 | README, reproduction scripts, release tag | Intro/Related/Discussion/Conclusion + Fig.1 + rebuttal; submission package |

## Baselines to reproduce/compare
CoLD (self), MCRe, MORSE, FINE, Co-Teaching(+/+), Decoupling, Flash, Argus.
Utilities: cleanlab (confident learning), noisy-label toolkits.

## Metrics
Macro-F1, FPR, FNR, alert compression ratio, Evidence Retention Rate (ERR),
noise-detection P/R/F1, time/storage overhead.

## Git workflow
- `main` stable, `dev` daily. Tag `d1`…`d7`. Push per completed module.

## Pending math corrections (must fold into D1 finalization before D3/D4)
Coordinator flagged in GPT-pro's revised D1; do NOT bake into code until fixed:
1. **D_pred target (pre-D3, CK-4)**: must compare per-view predicted label to the
   *observed* label `y_v`, i.e. `D_pred(v)=(1/M)Σ_k 1[ỹ_v^(k) ≠ y_v]` — NOT to the
   fused vote `ŷ_v` (that overlaps D_view and drops CoLD-CDM noise-transfer
   semantics). Authoritative: `docs/spec_method_impl.md §3`.
2. **Graph-noise β=0 property (pre-D4, CK-2)**: transition must keep the target
   ratio `r`; total flips `= r·N`, split into `β·rN` consistency-driven and
   `(1-β)·rN` random, so that `β=0 ⇒ symmetric(r)`. GPT-pro's
   `(1-β)δ_ij + β·π_ij` gives zero noise at β=0 and drops `r` — reject.
   Authoritative: `docs/spec_graph_noise.md §4`.
