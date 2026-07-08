# Codex Prompt — P2d DECISIVE (clean de-oracled full re-run + core-contribution re-verification)

> P2c proved the frozen results were oracle/leakage-contaminated (clean labels used
> in evidence + graph; 24% duplicate CICIDS rows). The runner is fixed but the full
> canonical matrix has NOT been regenerated. Until it is, NO table is submission-valid
> and NO manuscript writing may start. This round regenerates everything cleanly and,
> critically, re-checks whether the CORE contribution (evidence preservation / ERR /
> Graph-CDM) still holds without the oracle. Branch `revise/p2d-clean-rerun`, push,
> tag `rev-p2d`.
> INTEGRITY: real data only; NEVER fabricate. If the evidence-preservation gain
> largely vanishes after de-oracling, REPORT it honestly — do not hide; we would then
> re-scope the contribution. Numbers may drop; that is acceptable and expected.

## Files to read
src/experiments/* (the fixed D5 runner), src/models/{graph_cdm,evidence,loss}.py,
src/graph/build.py, src/data/{loaders,cicids_policy}.py, src/analysis/protocol.py,
reports/p2c_leakage_audit.md, reports/p2c_leakage_and_perdataset.md,
tables/table_p2c_*.csv, all currently-frozen canonical tables (now stale).

============================================================
# PART A — P2c GATE (verify the fixes are actually in the pipeline)
============================================================
Confirm and report:
- The de-oracled runner is the ONE used by the canonical matrix (no legacy runner path).
- Graph-CDM terms (D_pred/D_neigh/D_view) use ONLY the observed noisy training label
  `ỹ_v` — never `flip_mask` or clean `y_v`.
- Evidence score and graph construction use NO clean-label / flip_mask signal.
- Split-crossing edges = 0; CICIDS exact-dedup applied.
If any check fails, fix before PART B.

============================================================
# PART B — P2d GOALS
============================================================

## P2d-G1 — Full clean canonical re-run (mandatory; invalidates all stale tables)
- Re-run the ENTIRE matrix with the fixed runner: datasets {CICIDS(deduped), CESNET,
  UNSW} × noise {clean, symmetric, asymmetric, graph-consistency} × rates
  {10,20,40,60(,80)}% × all baselines {CoLD, ablation_hard, MCRe, MORSE, full
  Co-Teaching, FINE, Decoupling, Confident-Learning, Noisy-Supervised} × seeds {0,1,2}.
- Regenerate EVERY paper-facing table/figure from the canonical protocol and update
  frozen hashes. Delete or clearly supersede stale oracle-era tables so none can be
  cited by accident.
- Re-run number-consistency test on the fresh outputs.
**Acceptance**: all canonical tables regenerated from the de-oracled runner; stale
tables removed/superseded; consistency test green; hashes updated.

## P2d-G2 — CORE-CONTRIBUTION re-verification (the decisive question)
Does evidence preservation still help once the oracle is gone?
- Recompute, on the clean runner, full Graph-CoLD vs `ablation_hard` (ρ=0 + hard
  threshold, same graph/representation/CDM):
  Macro-F1, ERR_final, Tail-ERR, tail-class recall, high-noise FNR — per dataset and
  pooled, with paired significance.
- Explicitly test that ERR is NOT trivially 1.0 due to any residual clean-label path;
  if ERR is still exactly 1.0 for Graph-CoLD, justify it purely from soft-weight
  thresholding (no oracle), or report the corrected ERR.
- Regenerate the contribution-decomposition (graph+representation vs evidence
  preservation) on clean numbers.
**Acceptance**: a clean table showing whether evidence preservation still yields a
positive, significant ERR/tail-recall/FNR benefit. STATE the verdict plainly:
(i) benefit survives → core claim intact; (ii) benefit shrinks but positive →
re-scope to the surviving magnitude; (iii) benefit vanishes → flag for contribution
redefinition. Do not overstate.

## P2d-G3 — Honest per-dataset headline on clean numbers
- Regenerate `table_p2d_per_dataset_vs_cold.csv`: Graph-CoLD vs CoLD (and vs
  ablation_hard) per dataset (CICIDS-deduped, CESNET, UNSW) + pooled, with per-dataset
  paired tests + Holm correction + bootstrap CI.
- Reconfirm the corrected pattern (expected ≈ CICIDS +~6.7pp, CESNET ceiling ~0,
  UNSW ~−1.3pp) or report whatever the clean run actually shows.
**Acceptance**: per-dataset table with per-dataset significance; pooled never stands
alone; numbers match the fresh canonical run.

## P2d-G4 — Regenerate downstream artifacts on clean numbers
- Refresh: graph-informativeness diagnostic + informativeness↔margin correlation
  (UNSW boundary result), prioritization/compression tables, runtime/memory, all
  figures. Everything must trace to the fresh canonical CSVs.
**Acceptance**: all figures/tables regenerated and consistent with G1 outputs.

## P2d-G5 — Anti-regression guard test
- Add `tests/test_no_oracle_leakage.py` asserting that evidence score, graph
  construction, and Graph-CDM receive NO clean-label / flip_mask input (e.g., via
  interface checks or a canary run where corrupting clean labels does not change
  those components' outputs). This prevents silent reintroduction of the oracle.
**Acceptance**: guard test committed and green.

## P2d-G6 — Final clean claims-input for the manuscript
- Produce the definitive numbers GPT-pro will write from: clean per-dataset deltas,
  clean evidence-preservation benefit (ERR/tail-recall/FNR), clean pooled stat, and
  2–3 evidence-backed framing sentences (gains where graph is informative; neutral at
  ceiling; slight loss when views are weak; evidence preservation = retention value).
**Acceptance**: a compact "final claims-input" block with only clean numbers.

============================================================
# REPORTING BACK — reports/p2d_clean_rerun.md (commit)
============================================================
1. P2c gate note (all fixes confirmed in the live pipeline).
2. G1: fresh canonical table paths + hashes; list of stale tables removed/superseded.
3. G2: the CORE verdict (survives / shrinks / vanishes) with clean numbers + significance.
4. G3: clean per-dataset headline table + per-dataset p-values/CIs.
5. G4: refreshed downstream artifacts.
6. G5: guard test result.
7. G6: final clean claims-input + framing sentences.
8. Honest post-P2d reject-risk re-estimate + residual weaknesses.
9. Reproduction commands.

## Git
- branch: `revise/p2d-clean-rerun`; scoped commits per goal.
- merge main, tag `rev-p2d`, push origin main/dev/tags.

## Definition of Done
- De-oracled runner confirmed live; guard test prevents regression.
- Entire canonical matrix regenerated clean; stale oracle tables removed; consistency green.
- Core evidence-preservation benefit re-verified on clean numbers with an explicit verdict.
- Per-dataset honest headline + downstream artifacts all trace to the fresh run.
- reports/p2d_clean_rerun.md delivered with final clean claims-input and honest risk estimate.
