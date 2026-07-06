# Codex Prompt — REVISION P0 (raise C&S acceptance from ~35% toward ~60%)

> A senior C&S reviewer simulation rated the current draft "honest but incremental",
> most-likely Major Revision with ~40% reject risk. This prompt fixes the four
> code-side blockers behind that risk. GPT-pro owns prose cleanup separately.
> Work on branch `revise/p0-acceptance`, push, tag `rev-p0`. Do ONLY these four
> goals; no scope creep.

## 读取（先看，勿改契约语义）
docs/spec_method_impl.md §3–§6, docs/PLAN.md, src/models/graph_cdm.py,
src/models/evidence.py, src/models/loss.py, src/metrics.py,
src/experiments/smoke_ablation.py, src/experiments/d5*.py,
src/experiments/d9_5_baseline_*.py, src/baselines/* (base.py, registry.py,
coteaching.py), src/paper/*.py (caption/CSV generation).

---

## GOAL 1 — Fix `ablation_hard` so it truly isolates evidence preservation
**Reviewer objection (M4)**: In Table 2/Table 5, `ablation_hard` is numerically
identical to CoLD, i.e. it reverts to the CoLD baseline instead of being
"Graph-CoLD minus evidence preservation". The central ablation therefore proves
nothing beyond "Graph-CoLD vs CoLD".

**Target**: `ablation_hard` must share the SAME graph views + representation +
Graph-CDM as full Graph-CoLD, and differ ONLY by disabling evidence preservation:
set `rho = 0` AND replace the soft weight with a hard retention threshold on
GraphCDM (binary keep/drop), matching a CoLD-style purifier applied on top of the
graph diagnostic. It must NOT call the standalone CoLD baseline.

**Acceptance**:
- `ablation_hard` Macro-F1/ERR are DISTINCT from the standalone CoLD row (no more
  identical rows).
- Full Graph-CoLD ERR > `ablation_hard` ERR > CoLD ERR is coherent and reported.
- Unit test asserts `ablation_hard` uses the same graph+representation objects as
  full Graph-CoLD (shared encoder), only the weighting differs.

## GOAL 2 — Add strong dataset-purification baselines (MCRe, MORSE) + full Co-Teaching
**Reviewer objection (M3)**: The strongest same-family baselines from the CoLD
paper (MCRe, MORSE, FINE) are absent; Co-Teaching is only a "lite" approximation.
A 1.83pp lead is not competitive without them.

**Target**:
- Implement `src/baselines/mcre.py` (self-supervised representation + distance-based
  purification) and `src/baselines/morse.py` (semi-supervised, treats suspected
  noisy as unlabeled; use the actual noise ratio as split ratio, as MORSE does).
  Follow the `BaselineModel` contract in `src/baselines/base.py`; register both in
  `src/baselines/registry.py`.
- Upgrade `src/baselines/coteaching.py` to a faithful Co-Teaching (dual networks,
  small-loss co-selection, rate schedule) and rename away from "lite"; keep
  Co-Teaching+ if feasible.
- FINE: keep it in the matrix; if it still fails the real-data stability check,
  report its number WITH a stability caveat instead of excluding it (excluding a
  standard baseline reads as cherry-picking to reviewers).

**Acceptance**:
- MCRe, MORSE, full Co-Teaching, FINE all appear in the main result matrix on both
  datasets, all seeds, all noise settings; numbers have mean/std.
- Graph-CoLD vs each is reported with paired tests; if any beats Graph-CoLD on a
  metric, report it honestly (do not hide).

## GOAL 3 — De-tautologize ERR: link retention to a downstream, measurable gain
**Reviewer objection (M1)**: ERR≈1.0 is near-guaranteed by construction (soft
weights ≥ retention threshold), so the headline "win" is self-fulfilling. Must
show retained evidence yields a real downstream benefit.

**Target**: add analysis proving evidence preservation → measurable gains:
- Compute, per dataset/noise/seed: **tail-class recall** (recall on low-frequency
  malicious classes) and **high-noise FNR** for full Graph-CoLD vs `ablation_hard`
  (hard deletion). Hypothesis: preserving clean-info tail samples improves tail
  recall and lowers FNR under high noise.
- Add a results table + figure (`src/analysis/…` + `src/paper/…`): "Evidence
  preservation → downstream benefit", with paired significance.
- Add a **counterfactual check**: for samples retained by soft-weighting but
  deleted by hard-deletion, measure how many are (a) clean and (b) correctly
  classified downstream — this shows retention is *useful*, not just non-zero.

**Acceptance**:
- New table/figure shows tail-recall↑ and/or high-noise FNR↓ for Graph-CoLD vs
  hard deletion with paired p-values and effect sizes.
- Narrative-ready CSV emitted; if the benefit is small, report it honestly (the
  goal is to move ERR from "by definition" to "with downstream consequence").

## GOAL 4 — Purge development-process jargon from all generated artifacts
**Reviewer objection (M5)**: Manuscript leaks internal pipeline jargon ("D9.5
reinforced matrix", "D11 risk-clarification patch", "pre-registered smoke gate",
"repository candidate package", "D5/D5.5"). This signals an uncleaned auto-generated
paper and lowers professionalism.

**Target**: in every code path that emits captions, table titles, CSV headers,
figure titles, or metadata consumed by the manuscript (`src/paper/*.py`, result
writers), remove day/stage identifiers and pipeline jargon. Replace with neutral
scientific phrasing, e.g.:
- "the verified D9.5 reinforced result matrix" → "the evaluation matrix"
- "D6 Table 1" → "Table 1"
- "smoke gate / smoke-passed" → "verified implementation" (or drop)
- "repository candidate package / before journal upload" → remove
Do NOT change numeric results — only strings/labels.

**Acceptance**:
- `grep -rIn -E "D[0-9]+(\.[0-9]+)?|reinforced|smoke|repository candidate|before (journal )?upload|risk-clarification"` over generated artifacts and paper-facing
  code returns nothing paper-facing.
- Regenerated tables/figures/captions contain no dev-process identifiers.

---

## Git
- branch: `revise/p0-acceptance`
- commit(s): "revise(p0): true evidence-ablation, MCRe/MORSE/full-CoTeaching, ERR downstream benefit, jargon purge"
- merge to main, tag `rev-p0`, push origin main/dev/tags.

## Definition of Done (all four must hold)
1. `ablation_hard` ≠ CoLD numerically and shares Graph-CoLD's graph+encoder; ablation isolates evidence preservation.
2. MCRe, MORSE, full Co-Teaching, FINE all in the main matrix with mean/std and paired tests.
3. New "retention → downstream benefit" table+figure with paired significance; counterfactual retained-and-useful check reported.
4. Zero dev-process jargon in any manuscript-facing generated artifact.
5. Output a one-page revision report: before/after numbers for GOAL 1–3 and the grep proof for GOAL 4, plus reproduction commands.
