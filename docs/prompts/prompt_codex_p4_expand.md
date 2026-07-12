# Codex Prompt — P4 EXPAND (two phases: pre-artifacts, then merge). Target: C&S Full Article length.

> Goal: raise the manuscript from ~9 pages / 25 refs (Short-Communication size) to
> C&S Full-Article size (~16–20 pages, ~40–45 refs) WITHOUT new experiments and
> WITHOUT changing any results. Run PHASE 1 now; PHASE 2 runs AFTER GPT-pro returns
> the expanded prose. Branch `expand/p4-fullarticle`, push, tag `rev-p4`.
> INTEGRITY: no model/CDM/encoder/noise/metric code changes; no results/*.csv/json
> changes; no fabricated numbers; do NOT set submission_ready=true.

## Files to read
paper/graph_cold_cas_final.tex, tables/table_p2d_per_dataset_vs_cold.csv,
tables/table_p2f_summary.csv, reports/p2f_tighten.json,
docs/prompts/prompt_gptpro_p4_expand.md (the paired GPT-pro prompt, for key alignment).

============================================================
# PHASE 1 — Produce artifacts GPT-pro will cite (run NOW)
============================================================

## P4-1 New results table: noise-type breakdown (NO re-experiment)
Currently the paper only shows the tail-noise analysis. Build ONE additional table
from EXISTING data in `tables/table_p2d_per_dataset_vs_cold.csv`, which already
contains rows for noise_type ∈ {clean, symmetric, asymmetric, graph_consistency}
per dataset with Graph-CoLD vs CoLD/hard deltas.
- Emit `tables/table_p4_noise_type_breakdown.csv` by aggregating the existing rows
  (mean over rates/beta per dataset × noise_type) — pure re-grouping of existing
  numbers, no new runs.
- Add a LaTeX table `\label{tab:noisetype}` into graph_cold_cas_final.tex in §6
  (RQ1 area) showing Graph-CoLD vs CoLD Macro-F1 per dataset × noise-type, with the
  source line `Source: tables/table_p4_noise_type_breakdown.csv`.
- Verify every printed number equals the source CSV (add note in report).

## P4-2 Add verified references as \bibitem (agree keys with GPT-pro prompt)
Append these REAL, verifiable entries to the `thebibliography` block. Use EXACTLY
these cite keys so GPT-pro can reference them:
- guerra2022datasets  — J. L. Guerra, C. Catania, E. Veas, "Datasets are not enough:
  Challenges in labeling network traffic," Computers & Security 120 (2022) 102810.
- qing2023lowquality  — Y. Qing et al., "Low-quality training data only? A robust
  framework for detecting encrypted malicious network traffic," arXiv:2309.04798, 2023.
- zhao2021enhancing   — Z. Zhao et al., "Enhancing robustness of on-line learning
  models on highly noisy data," IEEE TDSC 18(5) (2021) 2177–2192.
- wang2022feco        — N. Wang et al., "FeCo: Boosting intrusion detection capability
  in IoT networks via contrastive learning," in IEEE INFOCOM, 2022, pp. 1409–1418.
- yue2022cleid        — Y. Yue et al., "Contrastive learning enhanced intrusion
  detection," IEEE TNSM 19(4) (2022) 4232–4247.
- liu2025deepdrac     — Y. Liu et al., "DeepDRAC: Disposition recommendation for alert
  clusters based on security event patterns," IEEE TIFS 20 (2025) 6443–6458.
- shon2023semisup     — H. G. Shon, Y. Lee, M. Yoon, "Semi-supervised alert filtering
  for network security," Electronics 12(23) (2023) 4755.
- wang2024alertagg    — W. Wang et al., "Transformer-based framework for alert
  aggregation and attack prediction in a multi-stage attack," Computers & Security
  136 (2024) 103533.
- northcutt2021cleanlab — (if not already present separately) same as confident
  learning; DO NOT duplicate if [9] already covers it — instead skip.
- xia2021sample       — X. Xia et al., "Sample selection with uncertainty of losses
  for learning with noisy labels," arXiv:2106.00445, 2021.
- liu2021metric       — C. Liu et al., "Noise-resistant deep metric learning with
  ranking-based instance selection," in CVPR, 2021, pp. 6811–6820.
- yang2024recda       — S. Yang et al., "RecDA: Concept drift adaptation with
  representation enhancement for network intrusion detection," in ACM SIGKDD, 2024,
  pp. 3818–3828.
- andresini2021insomnia — G. Andresini et al., "Insomnia: Towards concept-drift
  robustness in network intrusion detection," in ACM AISec, 2021, pp. 111–122.
- hynek2024zenodo     — CESNET-TLS-Year22 Zenodo record 10608607, 2024 (dataset).
- moustafa2015unswb   — N. Moustafa, J. Slay, "The evaluation of network anomaly
  detection systems: UNSW-NB15," Information Security Journal 25 (2016) 18–31.
Only include entries you can format correctly; do NOT invent DOIs/pages. If any is
uncertain, mark it plainly and leave it out rather than fabricate.

## P4-3 Insert anchor comments for GPT-pro prose
In graph_cold_cas_final.tex, insert clearly-marked LaTeX comment anchors where the
expanded prose will go, so PHASE 2 is a clean paste:
- `% <<P4-RELATED-WORK>>` at the end of §2 (before §3).
- `% <<P4-METHOD-ALGORITHM>>` at the end of §4 (before §5) — for a pseudocode block.
- `% <<P4-RESULTS-NOISETYPE>>` in §6 right after the new table (RQ1 area).
- `% <<P4-DISCUSSION>>` at the start of §7 body.

## PHASE-1 output — paper/p4_phase1_manifest.txt (commit)
List: new table label + printed cells vs CSV check; the FINAL list of cite keys
actually added (some may be skipped); the anchor line numbers. This manifest is the
handoff to GPT-pro.

Commit PHASE 1: "expand(p4-1): noise-type table + refs + prose anchors".

============================================================
# PHASE 2 — Merge GPT-pro prose, compile, verify (run AFTER GPT-pro returns)
============================================================
Input: GPT-pro's per-section LaTeX blocks (Related Work, Method-algorithm,
Results-noise narrative, Discussion), which reference the P4-2 cite keys and the
`tab:noisetype` label.

Steps:
1. Paste each block at its matching `% <<P4-...>>` anchor.
2. `pdflatex -interaction=nonstopmode graph_cold_cas_final.tex` twice.
3. Verify: "Output written", ZERO undefined citation/reference; every P4-2 key is
   actually cited at least once (no orphan refs); page count now ~16–20.
4. Re-run the P3 consistency greps and confirm they still pass:
   - MCRe/MORSE/FINE/Decoupling not claimed as Tables-2/3 baselines;
   - Co-Teaching only as "Co-Teaching-lite" in experimental context;
   - no "two-dataset" result-scope phrasing.
5. Also update Figure 1 workflow caption/labels to list all THREE datasets
   (CICIDS / CESNET / UNSW-NB15) if the figure text still says only two — text/label
   change only, do not regenerate the figure if not feasible; otherwise note it.

## PHASE-2 output — paper/p4_phase2_report.txt (commit)
Compile result (pages, zero undefined), orphan-citation check, grep re-verification,
and any GPT-pro block that could not be placed cleanly.

Commit PHASE 2: "expand(p4-2): merge expanded prose, compile, verify".
Merge main, tag rev-p4, push origin main/dev/tags.

## Constraints (both phases)
- No results/*.csv/json edits; new table is a re-grouping of existing rows only.
- No fabricated references, DOIs, or numbers.
- submission_ready stays false.
