# KIM Prompt — P5 FINAL PRE-SUBMISSION ROUND (close B1/B2/B3 + text fixes)

> Your P5 pre-submission review was verified by the coordinator (both spot-checks
> confirmed: CSV dz=0.914879→paper must say 0.91; duplicate Discussion heading at
> tex L661/L666). All findings are ADOPTED with one calibration: B2 is scoped to
> exactly TWO external baselines (cleanlab Confident Learning + full Co-teaching),
> NOT the full MCRe/MORSE/FINE family — P2b already showed those adapters have
> contested fidelity in this protocol, and reopening that front adds risk, not value.
> Branch `fix/p5-final-presubmission`, push, tag `rev-p5`.
> INTEGRITY: real data only; never fabricate; never alter existing recorded results;
> submission_ready stays false until the human author confirms metadata.

## Files to read first
paper/graph_cold_cas_final.tex, tables/table_p2f_powered_tests.csv,
tables/table_p2f_summary.csv, reports/p2f_tighten.md (+ .json),
reports/p2c_leakage_audit.md, src/experiments/* (P2f runner),
src/baselines/{confident_learning,coteaching,base,registry}.py,
paper/archive_do_not_submit/ (old D11 support materials, for reference only),
reproducibility/README_realdata.md, configs/*.yaml,
your own review: graph_cold_pre_submission_review.md.

============================================================
# PART A — P5-A EXPERIMENT SUPPLEMENT (start first; the only re-run)
============================================================
Goal: close B2 with two external anchors under the EXACT P2f clean protocol
(3 datasets, dedup, de-oracled runner, train 10k/test 5k audit windows,
tail-asymmetric {40,60,80}%, seeds 0–9, scenario-paired stats).

A1. Run **Confident Learning (cleanlab)** under the P2f runner. cleanlab is already
    in requirements.txt. Use the same Dataset contract, same splits/seeds/noise.
A2. Run **full Co-teaching** (dual networks, small-loss co-selection, rate schedule
    tracking the true injected noise rate) under the P2f runner. Upgrade from the
    lite version; name it "Co-teaching" only if faithful, otherwise keep "-lite"
    and say so.
A3. Emit results into a NEW file `results/p5_external_baselines.csv` (do NOT modify
    any existing results file). Compute the same metrics (aggregate Macro-F1,
    tail-F1, tail recall) and scenario-paired tests vs Graph-CoLD-soft with Holm
    correction + bootstrap CI.
A4. Manuscript integration: add the two methods to Table 2 (per-dataset means) and
    a new row-pair in the powered-comparison table OR an appendix table
    `tab:external` — whichever keeps Tables 2/4 readable. Report honestly even if
    an external baseline beats Graph-CoLD on any metric.
A5. Also produce the per-noise-rate stratified tail-F1 view you recommended
    (data already exists in results/p2f_tail_powered.csv — presentation only,
    no re-run): `tables/table_p5_tail_by_rate.csv` + a short paragraph in §6.3.

Acceptance: two external baselines present under identical protocol with paired
stats; existing result files untouched (hash-check them before/after).

============================================================
# PART B — P5-B MANUSCRIPT CORRECTIONS (parallel with A)
============================================================
All edits in paper/graph_cold_cas_final.tex unless stated.

B1. **dz 0.92 → 0.91** in exactly three places: abstract (L~50, interval becomes
    $d_z\in[0.91,1.29]$), §6.3 running text (L~583), Table 4 row (L~601).
    Source of truth: cohens_dz=0.914879 in table_p2f_powered_tests.csv.
B2. **Delete the duplicate heading**: remove `\subsection{Discussion}` at L~666,
    keep `\section{Discussion}`; promote any sub-structure to plain paragraphs
    or meaningful subsection titles (e.g., "Operational scope", "Fair comparison",
    "Lessons from the leakage audit").
B3. **§6.4 reconciliation sentence**: after the +6.7/+0.3/−1.3 audit deltas, add
    one sentence stating these come from the P2c audit protocol and differ in
    aggregation from the final P2f matrix (Tables 2–4); cite both reports.
B4. **UNSW view rename**: "process view" → "behavioral view" for UNSW everywhere
    (Table 1, §4.1, §6.5, Fig.1 caption). In §4.1 define it honestly: derived from
    service/protocol/state behavioral feature blocks, NOT process lineage. Keep
    the code contract untouched; this is prose-level renaming. If the paper
    claims must match `expected_view_support`, state that the behavioral view is
    a feature-block view, not host-process telemetry.
B5. **Claim downgrades**:
    - Fig.3 / contribution 3: "predicts" → "is consistent with" (n=3 datasets, no
      inferential value); adjust §6.5 wording accordingly.
    - RQ5 memory comparison vs "strong purification baselines": annotate that the
      memory numbers come from the earlier protocol, or re-measure under P2f if
      trivially available from the A-runs.
    - Abstract RQ1: qualify "statistically significant aggregate Macro-F1
      improvement on all three datasets" with magnitude bounds (CESNET/UNSW gains
      ≤ ~1pp).
B6. **Table 2 uncertainty**: add std (or bootstrap 95% CI) columns from
    table_p2f_summary source data; if width becomes an issue, use ± compact form.
B7. **Reference style unification**: make all 39 bibitems consistent (quotes,
    journal abbreviations, page formats); no content changes, style only.
B8. **Figure 1**: remove/replace the embedded draft caption text ("Fig1. ...
    submission-scope workflow" and the two-dataset label block) — if the PDF
    cannot be regenerated, crop/re-export or, at minimum, ensure the LaTeX caption
    supersedes and the in-figure stale text is noted for the camera-ready fix.

============================================================
# PART C — P5-C SUBMISSION PACKAGE REBUILD
============================================================
Create fresh, three-dataset-scope support materials under `paper/submission/`
(do NOT resurrect archive_do_not_submit content; write anew, consistent with the
final manuscript):
C1. `highlights.md` — 3–5 bullets, 85 chars max each, three-dataset scope,
    bounded claims (tail-F1 gains, leakage audit, informativeness boundary).
C2. `cover_letter.md` — addressed to Computers & Security EiC; states scope,
    contributions, the leakage-audit honesty angle, and confirms no dual
    submission. Leave author names as `[AUTHOR NAMES]` placeholders.
C3. `declaration_of_competing_interest.md`, `funding_statement.md`,
    `credit_author_statement.md` — templates with `[TO BE CONFIRMED BY AUTHORS]`
    placeholders; CRediT roles pre-mapped to [Author 1..n] slots.
C4. `data_availability.md` + an in-manuscript Data Availability section (before
    References): "Raw datasets are public (CICIDS-2017, CESNET-TLS-Year22 Zenodo
    10608607, UNSW-NB15); code and frozen result hashes at [REPO/DOI PLACEHOLDER]."
C5. `ai_use_declaration.md` — updated three-dataset version: generative AI
    assisted code/text drafting; authors reviewed and take responsibility.
C6. **Public reproducibility entry**: prepare (but do not publish without human
    approval) an anonymized export plan: which folders go public, a `CITATION.cff`,
    and a Zenodo-ready README. Replace hardcoded `E:\graphcold-data` with a
    config/env variable (`GRAPHCOLD_DATA_ROOT`) in loaders/paths — code change
    limited strictly to path resolution, covered by existing data-contract tests.
C7. Fix `reproducibility/README_realdata.md` stale items: correct tex filename
    references, remove MALTLS-22/OpTC as required inputs, document
    GRAPHCOLD_DATA_ROOT.

============================================================
# PART D — P5-D VERIFICATION & CLOSE-OUT
============================================================
D1. Full pytest in a torch-enabled environment; archive the log to
    `reports/p5_pytest_full.log` (all tests including the 2 torch-dependent ones).
D2. pdflatex twice; zero undefined citations/references; page count reported.
D3. Re-run the three P3 consistency greps + one NEW grep: no occurrence of
    "process view" tied to UNSW remains.
D4. Hash-check: all pre-existing results/*.csv/json unchanged
    (compare against reports/d9 + p2f frozen hashes).
D5. Report `reports/p5_close_out.md`: per-item status for A1–A5, B1–B8, C1–C7,
    D1–D4; external-baseline numbers with honest commentary; remaining
    human-only items (author metadata, repo/DOI publication, submission_ready).

## Git
- branch: fix/p5-final-presubmission; scoped commits per part (A/B/C/D).
- merge main, tag rev-p5, push origin main/dev/tags.

## Hard constraints
- Never modify existing results/*.csv/json (new results go to results/p5_*).
- Never fabricate numbers, references, DOIs, or baseline results.
- If cleanlab or full Co-teaching fails on real data, report the failure honestly
  with logs — do not silently drop or substitute synthetic numbers.
- submission_ready remains false; only the human author may authorize flipping it
  after B1-metadata items are filled.
