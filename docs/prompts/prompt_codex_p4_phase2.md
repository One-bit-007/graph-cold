# Codex Prompt — P4 PHASE 2 (merge GPT-pro prose, compile, verify)

> Run ONLY AFTER GPT-pro has returned all four LaTeX blocks AND Phase 1 is
> committed (tag rev-p4-phase1 exists on the remote).
> Branch `expand/p4-phase2-merge` (off the Phase 1 branch or main after merge).
> Commit, push, tag `rev-p4`.
> INTEGRITY: no model/CDM/encoder/noise/metric code changes; no results/*.csv/json
> changes; do NOT set submission_ready=true.

## What GPT-pro produced (your input)
GPT-pro has returned four LaTeX blocks, each headed by a comment naming its anchor:
  - % BLOCK A -> <<P4-RELATED-WORK>>
  - % BLOCK B -> <<P4-METHOD-ALGORITHM>>
  - % BLOCK C -> <<P4-RESULTS-NOISETYPE>>
  - % BLOCK D -> <<P4-DISCUSSION>>

These blocks use only cite keys confirmed in `paper/p4_phase1_manifest.txt` and
reference the table label `tab:noisetype` inserted in Phase 1.

## Files to read
paper/graph_cold_cas_final.tex,
paper/p4_phase1_manifest.txt,
GPT-pro's returned LaTeX blocks (provided alongside this prompt).

============================================================
# TASK 1 — Paste each block at its anchor
============================================================
For each of the four anchors in `paper/graph_cold_cas_final.tex`:
1. Locate the comment line (e.g., `% <<P4-RELATED-WORK>>`).
2. INSERT the corresponding GPT-pro block IMMEDIATELY AFTER the anchor comment,
   preserving the comment line itself so the file remains auditable.
3. Do NOT delete any existing prose from Phase 1; only ADD the new block after
   its anchor. The anchor comment stays.
4. If a block contains `\usepackage{...}` lines for algorithm typesetting, move
   those to the preamble (before `\begin{document}`), not inline in the body.

============================================================
# TASK 2 — Compile and verify
============================================================
Run `pdflatex -interaction=nonstopmode paper/graph_cold_cas_final.tex` TWICE.

Check all of the following and report each:
A. "Output written on graph_cold_cas_final.pdf" with page count ~16–20. If still 9,
   the blocks were not pasted — abort and report.
B. ZERO "undefined citation" lines in the log.
C. ZERO "undefined reference" lines in the log.
D. Orphan-citation check: every cite key listed in p4_phase1_manifest.txt appears
   at least once in a \cite{} call (no key added but never cited).
E. Orphan-reference check: every \ref{} in the GPT-pro blocks resolves (no
   undefined cross-references).

If B or C fails, identify which key/ref is undefined and fix:
- Undefined citation → check if the key is in the thebibliography; if missing, add
  the correct \bibitem from the manifest or GPT-pro block; if the key is wrong,
  correct it in the prose.
- Undefined reference → check if the label exists; correct the \label or \ref.

============================================================
# TASK 3 — Re-run P3 consistency greps
============================================================
These must still pass after adding the new prose. Run each grep and report PASS/FAIL:

1. `grep -n "MCRe\|MORSE\|FINE\|Decoupling" paper/graph_cold_cas_final.tex`
   PASS = zero occurrences inside §5.2 or any results-claim sentence (Related Work
   and Bibliography occurrences are expected and fine).

2. `grep -in "co-teaching[^+\-]" paper/graph_cold_cas_final.tex`
   PASS = in §5+, every hit is followed by "-lite" or "lightweight approximation";
   §2 Related Work occurrences with "Han et al." are fine.

3. `grep -in "two.dataset\|two dataset" paper/graph_cold_cas_final.tex`
   PASS = the only hit is the explicit §8 Threats phrase "two-dataset lock recorded
   in the D9 submission audit was an intermediate checkpoint".

If any grep FAILS, fix the offending sentence in the GPT-pro block (edit only the
newly added text, not existing prose).

============================================================
# TASK 4 — Update Figure 1 dataset labels (text only)
============================================================
Figure 1 was generated externally and its PDF cannot be regenerated here.
However, its CAPTION in graph_cold_cas_final.tex may still list only two datasets.
- Find the Figure 1 caption and update it so the word "CICIDS postfilter11 /
  CESNET postfilter25" becomes "CICIDS-2017, CESNET-TLS-Year22, and UNSW-NB15
  (active views vary by dataset; see Table 1)."
- Do NOT modify the PDF figure itself.

============================================================
# OUTPUT — paper/p4_phase2_report.txt (commit)
============================================================
Plain-text report:
1. Which anchor each block was pasted at (line numbers before and after).
2. Compile result: page count, zero-undefined confirmation.
3. Orphan citation/reference check results.
4. P3 grep results (PASS/FAIL for each of the three checks).
5. Figure 1 caption update confirmation.
6. Any block that could not be placed cleanly and why.

## Git
- branch: expand/p4-phase2-merge
- commits:
  "expand(p4-phase2): paste GPT-pro blocks at anchors"
  "expand(p4-phase2): compile, verify, consistency greps"
- merge to main; tag rev-p4; push origin main/dev/tags.

## Constraints
- Only paper/graph_cold_cas_final.tex and paper/p4_phase2_report.txt are modified.
- No model, CDM, encoder, noise model, metrics code touched.
- No results/*.csv/json changed.
- Do NOT set submission_ready=true.
