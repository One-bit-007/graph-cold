# Codex Prompt — P4 PHASE 1 (artifacts for GPT-pro)

> STOP after this phase. Do NOT touch prose or merge anything yet.
> GPT-pro runs next; you run again in PHASE 2.
> Branch `expand/p4-phase1-artifacts`. Commit, push, tag `rev-p4-phase1`.
> INTEGRITY: no model/CDM/encoder/noise/metric code changes; no results/*.csv/json
> changes; no fabricated numbers; do NOT set submission_ready=true.

## Why Phase 1 exists (read this first)
GPT-pro must know the exact cite-keys and table label BEFORE it writes prose,
otherwise it will generate undefined citations. You produce the "raw materials"
(refs + table + anchors), hand them over via `paper/p4_phase1_manifest.txt`, and
then GPT-pro expands the prose using those known keys. You merge in Phase 2.

## Files to read
paper/graph_cold_cas_final.tex,
tables/table_p2d_per_dataset_vs_cold.csv,
tables/table_p2f_summary.csv,
reports/p2f_tighten.json.

============================================================
# TASK 1 — Build noise-type breakdown table (NO new experiments)
============================================================
`tables/table_p2d_per_dataset_vs_cold.csv` already contains rows for
noise_type ∈ {clean, symmetric, asymmetric, graph_consistency} per dataset.
Pure re-grouping — no new runs.

Steps:
1. Aggregate: for each (dataset, noise_type) pair, compute mean of
   `graphcold_macro_f1`, `cold_macro_f1`, `delta_macro_f1_vs_cold` over all rows
   that share that (dataset, noise_type) (collapsing rate, beta, seed).
2. Write the result to `tables/table_p4_noise_type_breakdown.csv`.
3. Verify: every printed value in the CSV equals the arithmetic mean of the source
   rows (add verification note in the Phase-1 manifest).
4. Insert a LaTeX table into `paper/graph_cold_cas_final.tex` immediately AFTER
   Table 2 (still in §6.1 / RQ1). Use `\label{tab:noisetype}` and caption:
   "Graph-CoLD vs. CoLD Macro-F1 by noise type (mean over noise rates, seeds 0–9).
   Source: \texttt{tables/table\_p4\_noise\_type\_breakdown.csv}."
   — Columns: Dataset | Noise type | CoLD | Graph-CoLD-soft | $\Delta$
   — Rows ordered: dataset, then within each dataset: clean / symmetric /
     asymmetric / graph\_consistency.
   — Do NOT invent numbers; use only the computed means.

============================================================
# TASK 2 — Add verified references as \bibitem
============================================================
Append to the `\begin{thebibliography}` block in `graph_cold_cas_final.tex`.
Use EXACTLY these cite keys (GPT-pro will reference them by name):

  guerra2022datasets   — J. L. Guerra, C. Catania, E. Veas, "Datasets are not
    enough: Challenges in labeling network traffic," Computers & Security 120
    (2022) 102810.
  qing2023lowquality   — Y. Qing et al., "Low-quality training data only? A robust
    framework for detecting encrypted malicious network traffic," arXiv preprint
    arXiv:2309.04798, 2023.
  zhao2021enhancing    — Z. Zhao et al., "Enhancing robustness of on-line learning
    models on highly noisy data," IEEE TDSC 18(5) (2021) 2177–2192.
  wang2022feco         — N. Wang et al., "FeCo: Boosting intrusion detection
    capability in IoT networks via contrastive learning," in IEEE INFOCOM, 2022,
    pp. 1409–1418.
  yue2022cleid         — Y. Yue et al., "Contrastive learning enhanced intrusion
    detection," IEEE TNSM 19(4) (2022) 4232–4247.
  liu2025deepdrac      — Y. Liu et al., "DeepDRAC: Disposition recommendation for
    alert clusters based on security event patterns," IEEE TIFS 20 (2025) 6443–6458.
  shon2023semisup      — H. G. Shon, Y. Lee, M. Yoon, "Semi-supervised alert
    filtering for network security," Electronics 12(23) (2023) 4755.
  wang2024alertagg     — W. Wang et al., "Transformer-based framework for alert
    aggregation and attack prediction in a multi-stage attack," Computers & Security
    136 (2024) 103533.
  xia2021sample        — X. Xia et al., "Sample selection with uncertainty of losses
    for learning with noisy labels," arXiv preprint arXiv:2106.00445, 2021.
  liu2021metric        — C. Liu et al., "Noise-resistant deep metric learning with
    ranking-based instance selection," in CVPR, 2021, pp. 6811–6820.
  yang2024recda        — S. Yang et al., "RecDA: Concept drift adaptation with
    representation enhancement for network intrusion detection," in ACM SIGKDD,
    2024, pp. 3818–3828.
  andresini2021insomnia — G. Andresini et al., "Insomnia: Towards concept-drift
    robustness in network intrusion detection," in ACM AISec, 2021, pp. 111–122.
  hynek2024zenodo      — K. Hynek et al., CESNET-TLS-Year22 dataset, Zenodo record
    10608607, 2024.
  moustafa2016unsw     — N. Moustafa, J. Slay, "The evaluation of network anomaly
    detection systems: UNSW-NB15," Information Security Journal 25(1–3) (2016)
    18–31.

Rules:
- Do NOT duplicate a ref already in the file (e.g., if [9] already is northcutt2021).
- If you cannot format a bib entry correctly, omit it and note it in the manifest.
- Do NOT invent DOIs, page numbers, or author lists.

============================================================
# TASK 3 — Insert four prose anchors in the tex file
============================================================
Insert these exact comment lines (they guide Phase 2 paste locations):
- `% <<P4-RELATED-WORK>>`   → at the end of §2 Related Work, BEFORE \section{Problem}
- `% <<P4-METHOD-ALGORITHM>>` → at the end of §4 Method, BEFORE \section{Experimental}
- `% <<P4-RESULTS-NOISETYPE>>` → immediately AFTER the new \label{tab:noisetype} table
- `% <<P4-DISCUSSION>>`     → at the very start of §7 Discussion body (after heading)

============================================================
# OUTPUT — paper/p4_phase1_manifest.txt (commit this file)
============================================================
Plain-text report containing:
1. Table: noise-type table printed cells vs CSV means (numeric verification).
2. Final confirmed cite-key list (keys actually added; keys skipped with reason).
3. Anchor line numbers after insertion (one line per anchor).
4. Any warnings or skipped items.

This manifest is the HANDOFF to GPT-pro. GPT-pro reads it before writing prose.

## Git
- branch: expand/p4-phase1-artifacts
- commit: "expand(p4-phase1): noise-type table, refs, prose anchors"
- push origin expand/p4-phase1-artifacts; tag rev-p4-phase1
- DO NOT merge to main yet (that happens at end of Phase 2).

## Constraints
- No model, CDM, encoder, noise model, metrics code touched.
- No results/*.csv/json changed.
- Noise-type table values are arithmetic means of existing rows only.
- Do NOT set submission_ready=true.
