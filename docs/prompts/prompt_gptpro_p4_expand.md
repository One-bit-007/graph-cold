# GPT-pro Prompt — P4 EXPAND (prose) → C&S Full-Article depth

> Run AFTER Codex PHASE 1 (docs/prompts/prompt_codex_p4_expand.md). Codex has
> already: (a) added a noise-type breakdown table with label `tab:noisetype`;
> (b) added verified references with the cite keys listed below; (c) inserted
> anchor comments in paper/graph_cold_cas_final.tex. Read
> `paper/p4_phase1_manifest.txt` for the FINAL confirmed cite-key list and anchor
> locations before writing — only cite keys that Codex actually added.
> You OUTPUT drop-in LaTeX blocks per section; Codex will paste them (PHASE 2).
> Do NOT invent numbers, citations, or results; every quantitative statement must
> already exist in the manuscript tables or the P2c/P2d/P2f reports.

## Goal
Expand the manuscript from ~5,000 to ~9,000–11,000 words (C&S Full Article), keeping
the bounded, honest narrative intact. Preserve all existing claims and their
tables/figures; you are adding depth, not new results.

## Deliver four LaTeX blocks, each for its anchor

### BLOCK A — for `% <<P4-RELATED-WORK>>` (target ~1,500–2,000 words, 3 subsections)
Rewrite/expand §2 Related Work into three labelled subsections:
1. \subsubsection*{Noisy-label learning for intrusion detection} — deepen the
   robust-training vs purification split; cite guerra2022datasets, qing2023lowquality,
   zhao2021enhancing, xia2021sample, liu2021metric alongside existing CoLD/FINE/MORSE/MCRe.
2. \subsubsection*{Graph learning and contrastive representation for security} —
   position label-space diagnostics vs embedding-distance; cite wang2022feco,
   yue2022cleid, plus existing GNN/SimCLR refs.
3. \subsubsection*{SOC alert triage, prioritization, and concept drift} — connect the
   operational framing; cite liu2025deepdrac, shon2023semisup, wang2024alertagg,
   yang2024recda, andresini2021insomnia, plus existing Vaarandi/Chavali/CADE.
End with one paragraph that sharply states the delta of this paper vs all of the above
(label-space + evidence preservation + leakage audit).

### BLOCK B — for `% <<P4-METHOD-ALGORITHM>>` (target ~500–800 words)
Add:
1. A boxed algorithm (\begin{algorithm}...\end{algorithm}, use the algorithm2e or
   algorithmic package — instruct Codex which to \usepackage if needed) summarizing
   the two-stage Graph-CoLD pipeline: inputs (views, noisy labels), stage-1
   representation, Graph-CDM scoring, evidence-rescue weight, weighted classifier.
2. A short "Design rationale" paragraph justifying the three key choices with reasons
   already in the paper: (i) why label space not embedding distance; (ii) why the
   additive-rescue weight form rather than a floor; (iii) why hard deletion is the
   central comparator. No new claims.

### BLOCK C — for `% <<P4-RESULTS-NOISETYPE>>` (target ~250–400 words)
Narrate the new `tab:noisetype` table: describe Graph-CoLD vs CoLD Macro-F1 across
clean / symmetric / asymmetric / graph-consistency noise per dataset, using ONLY the
numbers Codex put in that table. Emphasize that gains are largest on CICIDS and under
asymmetric noise, consistent with the informativeness story; keep it bounded.

### BLOCK D — for `% <<P4-DISCUSSION>>` (target ~600–1,000 words, 3–4 paragraphs)
Expand §7 Discussion:
1. Operational meaning for SOC practitioners: when to deploy Graph-CoLD (informative
   views, rare-attack retention priority) and when not (weak views, ceiling regimes).
2. Proactive defense of the bounded scope (address reviewer concern W1): explain why
   the clean-protocol comparison is CoLD + ablation_hard, why mixing earlier-protocol
   SOTA numbers would be unsound, and what a fair future SOTA comparison requires.
3. The leakage audit as a transferable methodological lesson for graph-based security
   ML (duplicates + clean-label leakage inflation).
4. Threats/future work bridge (streaming, full-archive, faithful SOTA re-run).

## Constraints
- Cite ONLY keys confirmed in p4_phase1_manifest.txt; every key you cite must exist.
- Reference the new table as \ref{tab:noisetype}.
- No new numbers beyond existing tables/reports; no overclaiming; keep "Co-Teaching-lite"
  and the three-dataset P2f scope consistent with the current manuscript.
- Output must be clean LaTeX blocks, each headed by a comment naming its target anchor
  (e.g., "% BLOCK A -> <<P4-RELATED-WORK>>"), so Codex can paste mechanically.

## After you return
Codex PHASE 2 pastes your blocks at the anchors, compiles, checks zero undefined
citations/refs, verifies no orphan references, re-runs the consistency greps, and
reports page count (~16–20).
