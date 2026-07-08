# Codex Prompt — P2f TIGHTEN (kill the tautological recovery metric + power the tail tests + honest re-verdict)

> P2e reported "salvaged", but coordinator review of the full tables shows: (a) the
> tail-Macro-F1 gain of soft vs hard is significant ONLY on CESNET (p=6e-4); on CICIDS
> (p=0.074) and UNSW (p=0.061) it is NOT significant; (b) `rare_recovery_rate` = 1.0
> (soft) vs 0.0 (hard) is TAUTOLOGICAL — soft "recovers" by merely retaining, hard
> "loses" by definition, identical to the ERR collapse we already fixed. This round
> removes the tautology, properly powers the CICIDS/UNSW tail tests, and issues an
> honest re-verdict. Branch `revise/p2f-tighten`, push, tag `rev-p2f`.
> INTEGRITY: real data only; NEVER fabricate; pre-register the criterion BEFORE reading
> final numbers; if the tightened result is weaker, REPORT it — a narrow honest result
> is the goal, not a headline.

## Files to read
src/metrics.py, src/paper/p2e_salvage.py, src/models/{evidence,loss,graph_cdm}.py,
results/p2e_tail_salvage.csv, tables/table_p2e_success_tests.csv,
reports/p2e_salvage.md.

============================================================
# PART A — P2e GATE (verify → report)
============================================================
- Confirm the evidence-rescue soft weight and `test_soft_not_hard.py` are intact
  (soft ≠ hard on high-evidence/high-CDM samples).
- Confirm the tension gate result (max tension 0.375) still regenerates.
Short pass/fail note.

============================================================
# PART B — P2f GOALS
============================================================

## P2f-G1 — Redefine rare-evidence recovery to be NON-tautological (mandatory)
Current `rare_recovery_rate` counts a sample as "recovered" if it is merely retained
→ soft=1.0, hard=0.0 by construction. Replace with a definition that requires DOWNSTREAM
CORRECTNESS so hard and soft can genuinely differ on merit:

  recovered(v) := (v is a CLEAN sample of a rare/tail class)
                  AND (v is flagged suspicious, GraphCDM(v) > θ)
                  AND (the FINAL trained classifier predicts v's TRUE class correctly)

  rare_recovery_rate := |recovered| / |clean-rare-suspicious|

- Hard deletion: deleted samples are absent from training but STILL evaluated here
  (they exist in the data); recovery is whether the resulting model classifies them
  correctly — hard deletion is NOT automatically 0, and soft is NOT automatically 1.
- Compute identically for CoLD / hard / soft / semisup so the metric is method-agnostic.
**Acceptance**: `tests/test_rare_recovery_nontautological.py` asserts hard deletion can
score > 0 and soft can score < 1 on real data; the metric is no longer a constant.

## P2f-G2 — Properly power the tail-F1 comparison on ALL datasets
The P2e tail tests used test≈1000, n=9 (3 rates × 3 seeds), tail counts ~12–35 → high
variance, likely underpowered on CICIDS/UNSW.
- Increase statistical power: more seeds (e.g., {0..9}), larger/representative test
  split, and/or more asymmetric-noise rates ≥40%, keeping paced runtime.
- Recompute soft-vs-hard and semisup-vs-hard on tail-Macro-F1 AND the corrected
  rare-recovery, per dataset, with paired tests + Holm + bootstrap CI + effect size.
**Acceptance**: per-dataset tests with clearly reported n and power; result states, per
dataset, whether the tail benefit is significant after proper powering.

## P2f-G3 — Pre-registered honest re-verdict
Write `reports/p2f_preregistration.md` BEFORE reading final numbers:
- Criterion: soft (or semisup) beats hard on tail-Macro-F1 OR the CORRECTED
  rare-recovery, paired p<0.05 (Holm-corrected) AND dz≥0.3, under asymmetric noise
  ≥40%, with no significant aggregate-Macro-F1 regression.
- Record, per dataset, PASS/FAIL, and assign a global verdict:
  `robust` (significant on ≥2 datasets) / `narrow` (significant on 1 dataset only) /
  `null` (none). Do NOT relabel a narrow result as "salvaged".

## P2f-G4 — Honest claims-input for the manuscript
Produce the exact bounded statement GPT-pro will write, e.g. one of:
- robust → "evidence-aware soft retention significantly improves rare-class tail-F1
  over hard deletion under high asymmetric noise on N datasets";
- narrow → "…on CESNET; on CICIDS/UNSW the trend is consistent but not significant";
- null → recommend fallback A (audit/boundary paper), report the null honestly.
Include the corrected rare-recovery numbers so the tautology is gone from the paper.

============================================================
# REPORTING BACK — reports/p2f_tighten.md (commit)
============================================================
1. P2e gate note.
2. G1: corrected rare-recovery definition + proof it is non-constant (hard>0 / soft<1
   examples) + before/after numbers.
3. G2: powered per-dataset tail results (n, p_holm, dz, CI) for soft & semisup.
4. G3: pre-registration text + per-dataset PASS/FAIL + global verdict
   (robust/narrow/null).
5. G4: final bounded claims-input.
6. Honest reject-risk re-estimate + residual weaknesses.
7. Reproduction commands.

## Git
- branch: `revise/p2f-tighten`; scoped commits per goal; merge main, tag `rev-p2f`,
  push origin main/dev/tags.

## Definition of Done
- rare-recovery redefined to require correct classification; non-tautology test green;
  hard is no longer automatically 0 nor soft automatically 1.
- Tail comparison re-run with proper power; per-dataset significance reported with n.
- Pre-registered verdict recorded; result labeled robust / narrow / null honestly
  (no "salvaged" overclaim).
- reports/p2f_tighten.md delivered with bounded claims-input and honest risk estimate.
