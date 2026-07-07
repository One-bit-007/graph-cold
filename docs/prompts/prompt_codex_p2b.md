# Codex Prompt — P2b LIGHTWEIGHT (MCRe/MORSE noise-robustness fidelity)

> Focused verification, not a new feature round. Branch `revise/p2b-baseline-noise`,
> push, tag `rev-p2b`. Real data only; NEVER fabricate; only labels/strings may
> change, NEVER alter recorded numeric results.

## Why (reviewer risk)
In the canonical matrix, MCRe (0.620) and MORSE (0.620) trail CoLD (0.762) and even
Co-Teaching (0.705) on noise-averaged Macro-F1. But MCRe/MORSE are dataset-
purification methods known for NOISE ROBUSTNESS; in the CoLD paper they stay high
(MCRe ≈0.84–0.88, MORSE ≈0.70–0.77) even at 60% noise. Their clean-label sanity
already passed (≥0.85), so the question is robustness UNDER noise, not clean
performance. A reviewer familiar with these methods will call the current ranking
(MCRe below Co-Teaching) implausible and suspect a weakened adapter.

## Files to read
src/baselines/{mcre,morse,base,registry}.py, the matrix runners
(src/experiments/*), src/analysis/protocol.py (canonical protocol),
tables/table_p2_canonical_headline.csv, reports/p2_status.md.

## Task (do exactly this, minimal scope)
1. **Per-noise-rate robustness breakdown**: emit a small table of Macro-F1 vs
   noise rate {0,10,20,40,60(,80)}% for MCRe, MORSE, CoLD, Co-Teaching, Graph-CoLD
   on each real dataset (CICIDS at minimum). This exposes WHERE MCRe/MORSE degrade.
2. **Diagnose**: determine whether the low averages come from a genuine adapter
   weakness (e.g., purification threshold, split ratio not set to the actual noise
   rate for MORSE, representation under-training, distance metric) or from the
   protocol. Check specifically:
   - MORSE split ratio should track the ACTUAL injected noise rate (as MORSE does);
   - MCRe purification should not collapse recall on tail classes under noise.
3. **Then pick ONE honest outcome**:
   - (A) If a real adapter bug is found: fix it so MCRe/MORSE noise-robustness is
     plausible (broadly ≥ CoLD under moderate noise, consistent with their design).
     Re-run affected cells; update canonical tables + frozen hashes.
   - (B) If they are genuinely lower in this tabular/label-noise protocol for a
     defensible reason: produce the EVIDENCE (per-rate curve + short diagnosis) and
     write ONE caveat sentence the manuscript can cite, e.g. explaining the
     tabular-IDS / protocol difference from their original image/flow settings.
   Do NOT silently keep an implausible number with no explanation.

## Acceptance
- `tables/table_p2b_baseline_noise_robustness.csv` (per-rate breakdown) exists and
  regenerates.
- Either MCRe/MORSE numbers are corrected to plausible robustness (A), OR
  `reports/p2b_baseline_fidelity.md` documents the diagnosis + one-sentence caveat (B).
- If (A) changes any headline number, canonical tables and frozen hashes are updated
  and the number-consistency test still passes.
- Graph-CoLD's lead is re-reported honestly (it may shrink; that is fine and expected).

## Report back — reports/p2b_baseline_fidelity.md
1. Per-rate table (paths + key numbers).
2. Diagnosis: bug-fixed (A) or protocol-explained (B), with evidence.
3. The exact one-sentence manuscript caveat (if B) or before→after numbers (if A).
4. Updated Graph-CoLD-vs-MCRe/MORSE margins under the canonical protocol.
5. Reproduction commands.

## Git
- branch: `revise/p2b-baseline-noise`; commit "revise(p2b): MCRe/MORSE noise-robustness fidelity";
  merge main, tag `rev-p2b`, push origin main/dev/tags.
