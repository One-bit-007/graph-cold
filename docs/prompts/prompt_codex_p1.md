# Codex Prompt — P1 LONG-RANGE (with P0 acceptance gate) → push C&S acceptance to ~60–70%

> You are currently executing REVISION P0 (docs/prompts/prompt_codex_revise_p0.md).
> This prompt is your NEXT assignment. It has two parts:
>   PART A — verify your own P0 deliverables against a hard gate and remediate/report.
>   PART B — execute P1 (four goals) that lift the paper from "Major Revision, ~40%
>            reject" toward "acceptable, ~60–70%".
> Do PART A first. Do NOT start PART B until every PART A gate passes (or is
> explicitly reported as blocked with reasons). Branch `revise/p1-acceptance`
> (branch off after P0 merged to main). Push, tag `rev-p1`. No scope creep beyond
> the goals below.

## Files to read first
docs/prompts/prompt_codex_revise_p0.md (your P0 spec), docs/spec_method_impl.md,
docs/spec_graph_noise.md, docs/PLAN.md, src/data/{loaders,contracts,unsw_policy,
cicids_policy,cesnet_policy,noise,paths}.py, src/ranking/{ranking,prioritize}.py,
src/analysis/{stat_tests,result_sanity}.py, src/graph/build.py,
src/baselines/{registry,base}.py, src/experiments/*.py, src/paper/*.py, configs/*.yaml.

============================================================
# PART A — P0 ACCEPTANCE GATE (verify → remediate → report)
============================================================
For each P0 goal, run the check, and if it fails, fix it before PART B. Produce a
short "P0 Gate Report" table (pass/fail + evidence) at the end.

A1. **Evidence-isolating ablation (P0-G1)**
- Check: in the result matrix, `ablation_hard` row is NUMERICALLY DISTINCT from the
  standalone `CoLD` row on Macro-F1 and ERR; and `ablation_hard` shares the same
  graph views + encoder object as full Graph-CoLD (only ρ=0 + hard threshold differ).
- Fail → fix so hard deletion is applied on top of Graph-CoLD's graph+representation,
  not by reverting to the CoLD baseline. Re-run affected cells.
- Report: before/after `ablation_hard` vs `CoLD` numbers; assert Full ERR >
  ablation_hard ERR > CoLD ERR is coherent.

A2. **Strong baselines present (P0-G2)**
- Check: MCRe, MORSE, a FAITHFUL Co-Teaching (not "lite"), and FINE all appear in
  the main matrix on both datasets, all seeds/noise, with mean/std + paired tests.
- Fail → implement/repair missing ones per base.py contract; register in registry.py.
- Report: which baselines run, their headline numbers, and any that beat Graph-CoLD
  on any metric (report honestly).

A3. **ERR de-tautologized (P0-G3)**
- Check: a "retention → downstream benefit" table/figure exists showing tail-class
  recall↑ and/or high-noise FNR↓ for Graph-CoLD vs hard deletion, with paired
  significance; plus the counterfactual "retained-and-correctly-classified" check.
- Fail → add the analysis; if benefit is small, still report honestly.
- Report: the effect sizes + p-values; one sentence on whether ERR now has a
  demonstrated downstream consequence.

A4. **Jargon purge (P0-G4)**
- Check: `grep -rIn -E "D[0-9]+(\.[0-9]+)?|reinforced|smoke|repository candidate|before (journal )?upload|risk-clarification"` over paper-facing generated
  artifacts and src/paper/*.py returns NOTHING paper-facing.
- Fail → purge remaining strings (labels only, never numbers). Regenerate artifacts.
- Report: the grep result (should be empty) + list of files cleaned.

**GATE**: proceed to PART B only if A1–A4 pass. If any is blocked, stop and report
the blocker precisely (file, reason, what input/data is missing).

============================================================
# PART B — P1 GOALS (four)
============================================================

## P1-G1 — Third real dataset (UNSW-NB15) to remove single-dataset dependence
**Reviewer objection (M2)**: accuracy gains effectively rest on CICIDS-2017 alone
(CESNET has a ceiling effect). One effective dataset is thin for C&S.
**Target**:
- Finish `src/data/unsw_policy.py` + loader wiring so UNSW-NB15 loads under the same
  Dataset contract (postfilter policy, min-class filter, 8:2 split, active views =
  whatever fields verifiably exist; disable unsupported views honestly).
- Add UNSW-NB15 to the FULL matrix: all methods (incl. MCRe/MORSE/full Co-Teaching/
  FINE), all noise settings (symmetric/asymmetric/graph-consistency), seeds {0,1,2}.
- If UNSW cannot be sourced/licensed cleanly, DO NOT fabricate; instead report the
  blocker and propose an alternative already-available third dataset.
**Acceptance**: UNSW rows in main table + high-noise table + ablation, with mean/std
and paired tests; a per-dataset robustness statement is now backed by ≥2 datasets
that show a real (non-ceiling) accuracy margin.

## P1-G2 — Real prioritization evaluation (make the title honest)
**Reviewer objection (M8)**: the paper is titled "…Alert Prioritization" but only
evaluates ranking indirectly via compression/ERR. No ranking-quality metric.
**Target** (extend src/ranking/ranking.py + metrics + a results table):
- Using the priority score P(v), compute ranking-quality metrics on the test/alert
  set: **Top-K precision**, **Top-K recall (coverage of true malicious)**,
  **precision@review-budget**, and **alert-compression-at-fixed-recall** (e.g.
  fraction of queue needed to recover 90%/95% of true attacks).
- Compare Graph-CoLD ranking vs CoLD/ablation_hard/at least one baseline.
- Add a figure: recall vs review-budget (queue-load) curve.
**Acceptance**: a Prioritization table + queue-load curve with paired significance;
the priority claim is now directly measured, not proxied.

## P1-G3 — Statistical rigor (independence + multiple comparisons)
**Reviewer objection (M6)**: n=102 paired cells are correlated (shared datasets/
seeds/scenarios); no multiple-comparison control; effective sample size overstated.
**Target** (extend src/analysis/stat_tests.py):
- Replace/augment the naive paired test with a **cluster-robust or mixed-effects**
  paired comparison (random effects for dataset and seed), OR report a
  scenario-level paired test where each (dataset,noise,rate,beta) contributes ONE
  aggregated observation across seeds (avoids seed-level pseudo-replication).
- Apply **Holm–Bonferroni or Benjamini–Hochberg** correction across the family of
  method comparisons; report corrected p-values.
- Report effective sample size and confidence intervals (bootstrap) alongside dz.
**Acceptance**: the stats table now shows corrected p-values, CIs, and an
independence-aware test; the headline 1.83pp lift survives (or is honestly
re-stated) under the stricter test.

## P1-G4 — Graph reproducibility + graph-consistency noise validation
**Reviewer objection (M9 + wasted contribution)**: graph construction is
under-specified (kNN? threshold? k?), and the novel graph-consistency noise lacks a
β=0⇒symmetric verification.
**Target**:
- Expose ALL graph-construction hyperparameters in configs (per view: similarity
  metric, k / threshold, edge budget) and document them in src/graph/build.py
  docstrings; ensure the run is fully reconstructable from config + seed.
- Add a **verification test + figure** for graph-consistency noise: empirically show
  that at β=0 the induced transition matrix matches symmetric noise (within
  tolerance), and that increasing β concentrates flips on locally-consistent
  cross-class edges. Emit the transition-matrix heatmaps.
**Acceptance**: config-driven, seed-reproducible graphs; a β-sweep figure + a
passing unit test `tests/test_graph_noise.py::test_beta0_matches_symmetric`.

============================================================
# REPORTING BACK (required, structured)
============================================================
Produce `reports/p1_status.md` (this is an internal report file, fine to commit)
containing:
1. **P0 Gate Report**: A1–A4 pass/fail table with evidence (numbers + grep output).
   For any remediation you performed, list before→after.
2. **P1 Results Delta**: for G1–G4, the new tables/figures produced (paths), the key
   numbers, and paired/corrected significance.
3. **Acceptance-risk self-assessment**: re-estimate reject risk after P1 and name any
   remaining reviewer-facing weakness you could NOT close (be honest — do not
   overclaim). If a goal is blocked (e.g., UNSW licensing), state it explicitly with
   the exact blocker.
4. **Reproduction commands** for every new/changed table and figure.

## Constraints
- Real data only; if a dataset/baseline cannot be obtained, fail loud + report — NEVER
  fabricate or fall back to synthetic.
- Change labels/strings freely; NEVER alter recorded numeric results to look better.
- Deterministic over seeds {0,1,2}; report mean/std; keep frozen result hashes updated.

## Git
- branch: `revise/p1-acceptance` (off main after P0 merged)
- commits: scoped per goal, e.g. "revise(p1-g1): add UNSW-NB15 to full matrix"
- merge to main, tag `rev-p1`, push origin main/dev/tags.

## Definition of Done
- PART A gate fully passed and reported.
- G1: ≥2 datasets with a real (non-ceiling) accuracy margin, or UNSW blocker reported
  with alternative.
- G2: direct prioritization metrics + queue-load curve.
- G3: independence-aware, multiple-comparison-corrected stats with CIs.
- G4: config-reproducible graphs + β=0⇒symmetric verification test/figure.
- `reports/p1_status.md` delivered with honest residual-risk assessment.
