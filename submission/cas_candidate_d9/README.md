# Graph-CoLD C&S Candidate Package D9

This is a submission-ready candidate package, not an automatic journal upload.
`submission_ready` remains false until named authors confirm metadata, funding,
competing-interest declarations, and final editorial approval.

## Contents

- `author/`: author manuscript source, PDF, figures, tables, and BibTeX.
- `review/`: anonymous review manuscript source, PDF, figures, tables, and BibTeX.
- `submission_materials/`: cover letter candidate, highlights, CRediT, funding,
  competing-interest, and data-availability statements.
- `source_trace/`: D8 audit/risk notes and hashes for frozen source artifacts.

## Formal Scope

- Datasets: CICIDS-2017 postfilter11; CESNET-TLS-Year22 postfilter25.
- Methods: Graph-CoLD, CoLD, ablation_hard, Noisy-Supervised,
  Confident-Learning, and Co-Teaching-lite.
- Excluded: MALTLS-22, OpTC, UNSW-NB15, USTC-TFC2016, FINE, MCRe, MORSE,
  Flash, Argus, Decoupling, and full Co-Teaching.

## Headline Numbers

- Graph-CoLD vs CoLD Macro-F1 lift: 1.83 percentage points.
- Paired grouped p-value: 5.60e-06; Cohen dz=0.458; n=102.
- ERR_final: Graph-CoLD 1.0000; ablation_hard 0.8953.

## Lock Rule

Do not modify `results/*.csv` or `results/*.json` inside this package flow. Any
new experiment requires returning to the D5/D5.5 gate and regenerating all
downstream D6-D9 artifacts.
