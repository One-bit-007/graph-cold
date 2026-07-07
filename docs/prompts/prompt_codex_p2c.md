# Codex Prompt — P2c CRITICAL (CICIDS leakage audit + per-dataset honesty + UNSW attribution)

> This is the last empirical gate before manuscript rewrite. The pooled "+9.23 pp
> Graph-CoLD vs CoLD" headline is carried almost entirely by CICIDS (~+28 pp, and
> ~0.99 FLAT across 60% noise), is a near-tie on CESNET (ceiling), and is a ~2 pp
> LOSS on UNSW. A CICIDS result that barely degrades under 60% noise is a red flag
> for structural label leakage via the graph. We must audit this honestly BEFORE
> writing. Branch `revise/p2c-leakage-perdataset`, push, tag `rev-p2c`.
> INTEGRITY: real data only; NEVER fabricate; if leakage is found, REPORT and CORRECT
> the numbers — do NOT hide it; Graph-CoLD's margin may shrink and that is acceptable.

## Files to read
src/graph/build.py, src/data/{loaders,contracts,cicids_policy,unsw_policy,paths}.py,
src/models/graph_cdm.py, src/analysis/protocol.py, the canonical matrix runners,
tables/table_p2_canonical_headline.csv, tables/table_p2b_baseline_noise_robustness.csv,
reports/p2b_baseline_fidelity.md.

============================================================
# PART A — P2b GATE (verify → report)
============================================================
- Confirm `tables/table_p2b_baseline_noise_robustness.csv` regenerates and the
  MCRe/MORSE caveat (outcome B) is recorded with `result_numbers_changed=false`.
- Confirm number-consistency test still green and frozen hashes intact.
Short pass/fail note at top of the P2c report.

============================================================
# PART B — P2c GOALS
============================================================

## P2c-G1 — CICIDS structural-leakage audit (THE decisive check)
Determine whether Graph-CoLD's ~0.99-flat CICIDS robustness is genuine or an
artifact of the graph connecting label-revealing records.

Tasks:
1. **Split-boundary check**: prove NO edge in any view connects a training node to a
   test node, and that Graph-CDM/denoising never sees test labels. Report edge counts
   crossing the split (must be 0).
2. **Near-duplicate / same-flow audit**: quantify, in CICIDS-postfilter11, the rate of
   (a) exact-duplicate feature rows, (b) near-duplicate rows (e.g., cosine/Euclidean
   under a small threshold), and (c) rows that are likely the same flow/session split
   across records. Report how many graph edges connect such pairs.
3. **Leakage ablation**: rebuild the CICIDS graph with near-duplicate / same-session
   edges REMOVED (and, separately, with a deduplicated training set), then re-run
   Graph-CoLD and CoLD across the noise rates. Report whether the ~0.99-flat curve
   survives.
   - If it collapses toward CoLD → the original gain was leakage; UPDATE the canonical
     CICIDS numbers to the de-leaked version and re-report honestly.
   - If it largely survives → provide the evidence that neighborhoods carry legitimate
     structural signal (e.g., homophily among genuinely distinct flows), and keep the
     numbers with an explicit justification.
4. **Neighborhood-denoising sanity**: measure how much of D_neigh's benefit comes from
   neighbors that share the SAME underlying (pre-noise) record vs genuinely different
   records.

Acceptance: `reports/p2c_leakage_audit.md` + a de-leaked CICIDS per-rate table; a
clear verdict (leakage / legitimate) backed by numbers; canonical tables/hashes
updated if de-leaking changed results.

## P2c-G2 — Per-dataset honest reporting (replace pooled-only headline)
The pooled average masks per-dataset inconsistency. Make per-dataset primary.
Tasks:
- Emit `tables/table_p2c_per_dataset_vs_cold.csv`: Graph-CoLD vs CoLD (and vs
  ablation_hard) Macro-F1 / ERR / FNR, BROKEN DOWN by dataset (CICIDS, CESNET, UNSW)
  and noise, with paired tests PER DATASET.
- Keep the pooled number but always present it beside the per-dataset breakdown; never
  report the pooled lift alone.
Acceptance: per-dataset table with per-dataset significance; the true pattern
(CICIDS large / CESNET ceiling / UNSW slight loss, and whatever G1 changes CICIDS to)
is explicit.

## P2c-G3 — UNSW attribution → turn the loss into a boundary result
Explain why Graph-CoLD ≤ CoLD on UNSW.
Tasks:
- Compute a **graph-informativeness diagnostic** per dataset: e.g., neighborhood
  label homophily / purity of each active view, and view coverage (UNSW = temporal +
  process only, no host/IP). Correlate this diagnostic with Graph-CoLD's per-dataset
  margin over CoLD.
- Show that low graph informativeness (UNSW) predicts no/negative margin, high
  informativeness (CICIDS, post-audit) predicts positive margin, ceiling (CESNET)
  predicts neutral. This reframes the method honestly: "Graph-CoLD helps when views
  carry structural signal; it is neutral-to-slightly-negative when they do not."
Acceptance: `tables/table_p2c_graph_informativeness.csv` + a short correlation/plot
linking informativeness to margin; UNSW's result is explained, not hidden.

## P2c-G4 — Honest headline restatement inputs
Provide the numbers GPT-pro will use to rewrite claims:
- The corrected (post-audit) per-dataset Graph-CoLD vs CoLD deltas.
- One or two sentences of evidence-backed framing: gains concentrate where graph
  structure is informative; neutral under ceiling; no gain when views are weak.
Acceptance: a compact "claims-input" block in the report with final numbers.

============================================================
# REPORTING BACK — reports/p2c_leakage_and_perdataset.md (commit)
============================================================
1. P2b gate note.
2. G1 leakage verdict with all counts (split-crossing edges=0, duplicate rates,
   de-leaked CICIDS curve) and whether canonical numbers changed.
3. G2 per-dataset table paths + key deltas + per-dataset p-values.
4. G3 informativeness diagnostic + correlation with margin; UNSW explanation.
5. G4 corrected claims-input numbers + framing sentences.
6. Honest post-P2c reject-risk re-estimate + any residual weakness.
7. Reproduction commands.

## Git
- branch: `revise/p2c-leakage-perdataset`; scoped commits per goal.
- merge main, tag `rev-p2c`, push origin main/dev/tags.

## Definition of Done
- Split-crossing edges proven 0; duplicate/same-session leakage quantified; CICIDS
  de-leak ablation run; verdict stated; canonical numbers corrected if needed.
- Per-dataset Graph-CoLD-vs-CoLD table with per-dataset significance delivered;
  pooled number never stands alone.
- Graph-informativeness diagnostic explains the UNSW non-gain as a boundary result.
- reports/p2c_leakage_and_perdataset.md committed with corrected claims-input and
  honest residual-risk estimate.
