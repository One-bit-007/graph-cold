# P5 Close-Out Report — Final Pre-Submission (Plan A)

Date: 2026-08-10. Branch: `fix/p5-final-presubmission`.
Scope: execute `docs/prompts/prompt_kim_p5.md` after the coordinator adopted the
pre-review recommendation to narrow B2 to two external anchors (cleanlab
Confident Learning + full Co-Teaching) and, per the user's directive, continue
with Plan A (honest repositioning) after CL was found to beat Graph-CoLD.

## Headline experimental finding (drives repositioning)

- 180/180 scenarios ran, 0 failures; dataset hashes match the frozen P2f
  matrices; smoke check bit-reproduced the frozen value
  (CoLD/CICIDS/seed0/40% macro_f1 = 0.5854746529538196).
- **Confident Learning significantly beats Graph-CoLD-soft on all three
  datasets on all three metrics** (Holm-corrected). ΔTail-F1: CICIDS +0.171,
  CESNET +0.234, UNSW +0.081; CL prunes only 7–24% of rows.
- **Full Co-Teaching is significantly worse everywhere**; tail-F1 ≈ 0 at
  60–80% noise on all three datasets (forgetting mechanism sacrifices rare
  classes — empirically supports the SOC motivation).
- Repositioned narrative: diagnostic precision decides everything (precise
  pruning > soft weighting > coarse deletion). Contribution downgraded to:
  leakage audit + mechanism-level evidence-preservation finding +
  graph-informativeness boundary.

## Item-by-item status

### PART A (experiments & integration) — DONE

- A1: `results/p5_external_baselines.csv` (180 scenarios, CL + full CT).
- A2: `tables/table_p5_external_tests.csv` (Holm-corrected tests).
- A3: `tables/table_p5_tail_by_rate.csv`.
- A4: Manuscript honest repositioning — abstract (CL/CT ordering sentence +
  closing downgrade), contribution 2 → mechanism-level with anchor, §5.2
  baseline paragraph, new `tab:external` table + §6.3 "External anchors",
  discussion "fair comparison" rewrite + new diagnostic-precision paragraph,
  limitations (CL-superiority statement), conclusion rewrite, Fig. 3 caption.
- A5: `reports/p5_external_baselines.{json,md}`.

### PART B (manuscript fixes) — DONE

- B1 dz 0.92→0.91 (3 places); B2 duplicate `\subsection{Discussion}` removed;
- B3 §6.4 reconciliation sentence; B4 process→behavioral view (§4.1 honest
  definition, Table 1, §6.5, abstract, contribution line, Fig. 1 regenerated
  and visually verified); B5 three downgrades + abstract magnitude cap
  (≤ +0.01); B6 Table 2 six-method mean±std; B7 reference style unified;
- B8 Fig. 1 refreshed.

### PART C (submission package) — DONE

- C1–C3, C5: `paper/submission/` — highlights, cover letter, declaration of
  competing interest, funding statement, CRediT, AI-use declaration.
- C4: `paper/submission/data_availability.md` + in-manuscript
  `\section*{Data availability}` before the bibliography
  (`[REPO/DOI PLACEHOLDER]` retained intentionally).
- C6: `GRAPHCOLD_DATA_ROOT` parameterization (`src/data/paths.py`,
  `src/experiments/d5.py::_resolve_recorded_data_path`); verified that setting
  the variable resolves correctly and that unset behavior is unchanged.
  `paper/submission/reproducibility_export_plan.md` (include/exclude lists;
  the untracked internal company document directory explicitly excluded),
  `CITATION.cff` (author metadata
  placeholders), `paper/submission/zenodo_archive_readme.md`.
- C7: `reproducibility/README.md` rewritten to match reality (stale
  `graph_cold_cas_submission.tex`, missing CSV/figure names, MALTLS-22/OpTC
  claims removed); `README_realdata.md` fixed (dangling
  `run_d5_baseline_expansion.ps1` and `paper/elsevier/build_elsevier.ps1`
  references removed, `GRAPHCOLD_DATA_ROOT` documented, final tex name added).

### PART D (verification) — DONE

- D1: full pytest archived to `reports/p5_pytest_full.log`:
  **205 passed, 3 skipped, 20 failed**. All 20 failures are pre-existing
  legacy-pipeline tests (D6/D7/D8/D11/smoke-gate) asserting on paths that no
  longer exist (`paper/elsevier/...`, `paper/sections/...`,
  `graph_cold_cas_realdata_d11.tex`) or on `d7_assemble.py`'s outdated
  two-dataset `FORMAL_DATASETS` check. None reference
  `paper/graph_cold_cas_final.tex`; none were introduced by P5 changes.
  (pypdf was added to the project venv to unblock test collection.)
- D2: `pdflatex` × 2 (MiKTeX 25.12) — exit 0 both passes, **0 undefined
  references/citations, 0 errors, 14 pages** (261,074 bytes).
- D3: consistency greps — no stray dz 0.92 (only a 0.92\linewidth minipage);
  zero "process view"; "state-of-the-art" only in negated/bounded contexts;
  MALTLS-22/OpTC mentioned only to state they are *not* evaluated; one
  intentional `[REPO/DOI PLACEHOLDER]`.
- D4: all 25 frozen files in `reports/p5_pre_run_hashes.json` — **unchanged
  (25/25, 0 changed, 0 missing)**.

## Honest assessment of the CL result

The Confident Learning result materially weakens any accuracy-superiority
claim; the manuscript now says so explicitly (abstract, §6.3, limitations,
conclusion). What remains is: (1) the leakage audit and corrected protocol;
(2) the mechanism-level finding that evidence preservation protects rare
classes where hard deletion and full co-teaching fail; (3) the
graph-informativeness boundary explaining when graph denoising helps at all;
(4) the audited external-baseline ordering itself. These are bounded claims,
and the text now treats them as such.

## Remaining human to-dos (block `submission_ready=true`)

1. ~~Fill in author metadata~~ — DONE 2026-08-10 (five authors, front matter,
   `CITATION.cff`, `paper/submission/*.md`; funding confirmed as none;
   CRediT mapping and competing-interest disclosure marked for author sign-off).
2. Execute `paper/submission/reproducibility_export_plan.md`: create the public
   repo (exclude the untracked internal company document directory), tag,
   archive on Zenodo, obtain DOI. A sanitized export tree has been prepared at
   `workspace/graph-cold-public-export/` (0 internal-directory references).
3. Replace `[REPO/DOI PLACEHOLDER]` in the tex and
   `paper/submission/data_availability.md`; recompile (2× pdflatex).
4. Only then set `submission_ready=true`.
5. Optional cleanup (not blocking): retire or update the 20 legacy-pipeline
   tests so the suite is green for future reviewers.

`submission_ready` is **false** at the time of this report, by design.

## Addendum — 2026-08-10 (post-publication)

All blocking human to-dos are now resolved:

- Author metadata filled from the author information sheet (five authors,
  corresponding author Tianqi Guo); funding confirmed as none; CRediT and
  competing-interest drafts marked for author sign-off.
- Public repository live: https://github.com/One-bit-007/graph-cold
  (`main` at `00bb451`, tag `rev-p5`).
- Zenodo archive minted: version DOI `10.5281/zenodo.21873516` (rev-p5),
  concept DOI `10.5281/zenodo.21873515`. Both DOIs are now cited in the
  manuscript Data availability section and in
  `paper/submission/data_availability.md`; the PDF was recompiled
  (14 pages, 0 undefined references).
- `CITATION.cff` carries the real repository URL and the version DOI; only a
  license choice (MIT or CC-BY-4.0 suggested) remains open and non-blocking.

Per the P5 prompt, only the human author may authorize flipping
`submission_ready` to true. With the above items resolved, that authorization
is the single remaining action before upload to the C&S editorial system.

**`submission_ready = true`** — authorized by the human author (Tianqi Guo,
corresponding author) on 2026-08-10, 20:53 local time, after confirmation that
author metadata, funding (none), the public repository, and the Zenodo DOIs
are all in place. Remaining non-blocking items: license choice for
`CITATION.cff`/GitHub, and co-author sign-off on the CRediT and
competing-interest drafts.

**Re-authorized for the P5b+ version** — on 2026-08-12, 19:39 local time, the
corresponding author re-confirmed `submission_ready = true` for commit
`847587b` (evidence-curation reframing, risk-aware evidence interpretation,
reference style unification, TikZ Fig. 1), after an independent audit of the
diff verified: no results/figures data changed (25/25 frozen hashes intact),
the "never zero" weight claim is consistent with Eq. (4), the risk-aware
$R_{\text{delete}}$ interpretation mirrors the implemented evidence score, all
cross-references resolve, and the build is clean (14 pages, 0 undefined).
License (MIT) added 2026-08-10; CRediT/COI statements finalized 2026-08-10.
