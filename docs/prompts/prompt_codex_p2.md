# Codex Prompt — P2 (with P1 acceptance gate) → close NEW-1/NEW-2/M2/M8, target ~60–70% acceptance

> You just finished P1. This is P2. Two parts:
>   PART A — verify P1 deliverables + re-confirm P0 gates, remediate/report.
>   PART B — five goals that close the reviewer risks found after P1.
> Do PART A first; do not start PART B goals whose gate is failing until fixed
> (except independent goals may proceed in parallel). Branch `revise/p2-acceptance`
> (off main). Push, tag `rev-p2`. Real data only; NEVER fabricate; only labels/strings
> may change, NEVER alter recorded numeric results to look better.

## Files to read first
docs/prompts/prompt_codex_p1.md, reports/p1_status.md,
reports/revision_p0_acceptance_report.md, docs/spec_method_impl.md,
src/data/{loaders,contracts,unsw_policy,cicids_policy,cesnet_policy,paths}.py,
src/baselines/{registry,base,mcre,morse,coteaching,fine_style,decoupling}.py,
src/experiments/{formal_matrix,baseline_matrix}.py or the current matrix runners,
src/analysis/{stat_tests,prioritization,evidence_downstream}.py,
src/ranking/{ranking,prioritize}.py, src/paper/*.py, configs/*.yaml.

============================================================
# PART A — P1 ACCEPTANCE GATE (verify → remediate → report)
============================================================
A1 (P0 still holds): re-run and confirm ERR ordering Graph-CoLD > ablation_hard >
   CoLD, and `ablation_hard` still shares Graph-CoLD graph+encoder (not reverted).
A2 (G2 prioritization): confirm table_prioritization.csv + queue-load curve exist
   and are regenerable.
A3 (G3 stats): confirm scenario-level paired test + CI + Holm are wired and used in
   the paper-facing stats table (not just naive n=102).
A4 (G4 graph noise): confirm β=0⇒symmetric test passes and β-sweep figure regenerates.
Produce a short "P1 Gate Report" table (pass/fail + evidence). Fix any fail first.

============================================================
# PART B — P2 GOALS
============================================================

## P2-G1 — NUMBER CONSISTENCY (NEW-2): one protocol, one number per method
**Reviewer risk**: CoLD Macro-F1 appears as 0.9719 in the P0 report but 0.8423 in
p1_status A1. Inconsistent baseline numbers across tables destroy credibility.
**Target**:
- Define ONE canonical evaluation protocol (aggregation over dataset×noise×rate×
  seed, and the exact averaging used for each table). Document it in one place
  (e.g., src/analysis/protocol.py + a paragraph the paper can cite).
- Regenerate EVERY paper-facing table from that single protocol so each method has a
  consistent headline number everywhere. Eliminate the CoLD 0.97/0.84 discrepancy.
- Add `tests/test_number_consistency.py`: asserts each method's headline metric is
  identical across all generated tables (within float tolerance) and matches the
  frozen result hash.
**Acceptance**: no method's headline metric differs across tables; frozen hashes
updated; consistency test green.

## P2-G2 — BASELINE FAITHFULNESS (NEW-1): calibrate strong baselines
**Reviewer risk**: Graph-CoLD beats MCRe/MORSE/FINE by ~27–37 pp. In their own
papers these methods reach ~75–99%. A 30 pp gap reads as weakened, unfair baselines.
**Target**:
- Add a **clean/low-noise sanity gate**: on 0% and 10% noise, MCRe, MORSE, FINE, and
  full Co-Teaching must reach performance comparable to their published ranges on
  CICIDS (e.g., Macro-F1 broadly in the high-0.8 to 0.9+ region on clean labels). If
  a baseline is far below, FIX the adapter (hyperparameters, training length,
  representation) until it is competitive on clean data.
- If a genuine protocol reason keeps a baseline lower (e.g., tabular vs image origin),
  DOCUMENT it explicitly with evidence; do not silently ship a crippled baseline.
- Re-run the full matrix; Graph-CoLD's lead over now-competitive baselines will be
  smaller and MORE credible. Report honestly even if the margin shrinks.
**Acceptance**: `tests/test_baseline_sanity.py` asserts each strong baseline clears a
clean-label floor; matrix re-run; a before/after margin table shows the corrected,
credible gaps.

## P2-G3 — THIRD DATASET AUTO-INGEST (M2): UNSW-NB15, trigger on presence
**Reviewer risk**: accuracy margin effectively rests on CICIDS alone (CESNET ceiling).
**Target**:
- Harden `src/data/unsw_policy.py` + loader so UNSW-NB15 loads under the Dataset
  contract. Support BOTH layouts placed at `E:\graphcold-data\unsw_nb15\`:
  (a) partition CSVs `UNSW_NB15_training-set.csv` + `UNSW_NB15_testing-set.csv`
      (~257k rows, 49 features, `attack_cat`+`label`; activate temporal +
      feature-block views; disable host/IP views honestly if no IP columns);
  (b) full CSVs `UNSW-NB15_1..4.csv` (+ `NUSW-NB15_features.csv`) with
      `srcip/dstip/Stime/Ltime` (activate IP/host/temporal; use a deterministic
      audit-window subsample to bound memory, mirroring CESNET).
- Auto-detect: if either layout is present and passes the contract, UNSW is
  AUTOMATICALLY added to the full matrix (all methods, all noise, seeds {0,1,2}); if
  absent, skip gracefully with a logged message — NO fabrication.
- Emit `reports/unsw_ingest.md` describing exactly which files/columns were detected,
  which views activated, and row counts.
**Acceptance**: with data present, UNSW rows appear in main/high-noise/ablation
tables + paired tests; with data absent, the pipeline runs clean and logs the skip.
Provide the exact command the user runs after dropping the CSVs in.

## P2-G4 — PRIORITIZATION REFRAME (M8): make the ranking claim honest and useful
**Reviewer risk**: Top-K precision is ~0.999 for Graph-CoLD, CoLD, and ablation_hard
alike — the audit-window ranking is too separable, so "prioritization" shows no
advantage despite being in the title.
**Target** (pick the honest path, implement, and report):
- Harder/realistic ranking evaluation: evaluate ranking on the FULL noisy test/queue
  (not only the separable audit window), and/or under high-noise where structure
  should help; report **alert-compression-at-fixed-recall (90%/95%)** and
  **precision@review-budget** where methods actually differ.
- If Graph-CoLD still ties on Top-K precision, REPORT it honestly and shift the
  ranking claim to evidence-aware compression (fewer alerts to review at equal
  recall) rather than raw precision; flag to GPT-pro to soften the title/claims
  accordingly.
**Acceptance**: a prioritization result where either Graph-CoLD shows a measured
advantage on some operational metric, OR the paper's ranking claim is explicitly
rescoped to what the data supports (with the supporting table).

## P2-G5 — CONTRIBUTION DECOMPOSITION (REFRAME): back the honest narrative
**Reviewer-facing reframe**: the ablation implies most Macro-F1 gain comes from the
graph/representation (ablation_hard ≈ 0.97 vs CoLD ≈ 0.84), while evidence
preservation adds ~2 pp F1 but drives the ERR/tail-recall/FNR benefit.
**Target**: emit a single decomposition table/figure attributing gains to
(i) graph+representation vs (ii) evidence preservation, across Macro-F1 AND
ERR/tail-recall/FNR. This lets the paper claim, truthfully, that evidence
preservation's value is retention (rare-attack evidence), not headline F1.
**Acceptance**: a decomposition table + figure Codex-generated; numbers consistent
with P2-G1 protocol.

============================================================
# REPORTING BACK — reports/p2_status.md (commit it)
============================================================
1. P1 Gate Report (A1–A4 pass/fail + evidence).
2. P2 Results Delta for G1–G5: paths of new/updated tables & figures, key numbers,
   corrected significance; for G2 the before→after baseline margins; for G3 the UNSW
   ingest result or the exact blocker; for G4 whether an advantage was found or the
   claim was rescoped.
3. Honest post-P2 reject-risk re-estimate + any residual weakness you could not close.
4. Reproduction commands for every changed table/figure.

## Git
- branch: `revise/p2-acceptance`; scoped commits per goal.
- merge to main, tag `rev-p2`, push origin main/dev/tags.

## Definition of Done
- PART A gate passed & reported.
- G1: every method has ONE consistent headline number across all tables (test green).
- G2: strong baselines clear a clean-label sanity floor; margins re-reported honestly.
- G3: UNSW auto-ingested when present (or precise blocker); graceful skip when absent.
- G4: prioritization advantage measured OR claim honestly rescoped, with evidence.
- G5: contribution-decomposition table/figure delivered.
- reports/p2_status.md committed with honest residual-risk assessment.
