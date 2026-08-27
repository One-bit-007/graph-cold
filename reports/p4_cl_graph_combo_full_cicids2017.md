# P0-4 CL + Graph-CDM Full Report

- Rows: 150 / expected 150
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; rates: [0.4, 0.6, 0.8]
- Primary G2 method: `CL+Graph-soft` vs `Confident-Learning`.
- Frozen family B size: 18 rate-specific tests; unrun cells conservatively enter Holm as p=1.
- Per-rate G2 pass requires positive Tail-F1, Holm p<0.05, dz>=0.3, CI lower bound >0, Macro-F1 non-inferiority at delta=0.01, and no class mean F1 loss beyond 0.05.

## Mean metrics

| dataset | protocol | method | macro F1 | tail F1 | tail recall | retained |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| cicids2017 | C1-tier1 + S1 temporal-by-source-file | CL+Graph-soft | 0.1337 | 0.3416 | 0.2912 | 0.960 |
| cicids2017 | C1-tier1 + S1 temporal-by-source-file | CL-issue-intersection | 0.1194 | 0.2636 | 0.2038 | 0.998 |
| cicids2017 | C1-tier1 + S1 temporal-by-source-file | CL-issue-union | 0.1011 | 0.2309 | 0.1788 | 0.958 |
| cicids2017 | C1-tier1 + S1 temporal-by-source-file | Confident-Learning | 0.1370 | 0.3554 | 0.3107 | 0.960 |
| cicids2017 | C1-tier1 + S1 temporal-by-source-file | Graph-CoLD-soft | 0.1364 | 0.3624 | 0.2731 | 1.000 |

## G2 status

- **Global verdict: null**
- **cicids2017**: dataset pass=False.
  - rate=0.4: tail delta +0.0040, Holm p=1, dz=0.341, CI=[+0.0000, +0.0116]; macro delta +0.0115, NI p=0.0782; worst class delta +0.0000; pass=False.
  - rate=0.6: tail delta -0.0422, Holm p=1, dz=-0.316, CI=[-0.1265, +0.0000]; macro delta -0.0230, NI p=0.732; worst class delta -0.0574; pass=False.
  - rate=0.8: tail delta -0.0032, Holm p=1, dz=-0.316, CI=[-0.0095, +0.0000]; macro delta +0.0016, NI p=0.0136; worst class delta -0.0032; pass=False.
  - Mixed-effect robustness: average tail delta -0.0138, one-sided p=0.835, CI=[-0.0415, +0.0139], converged=True; warnings=1.

## Protocol limits

- **cicids2017** — C1-tier1 + S1 temporal-by-source-file; limitation: S2 unavailable because the local copy lacks entity columns.
