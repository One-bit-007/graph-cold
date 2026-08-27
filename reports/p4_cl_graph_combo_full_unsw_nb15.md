# P0-4 CL + Graph-CDM Full Report

- Rows: 150 / expected 150
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; rates: [0.4, 0.6, 0.8]
- Primary G2 method: `CL+Graph-soft` vs `Confident-Learning`.
- Frozen family B size: 18 rate-specific tests; unrun cells conservatively enter Holm as p=1.
- Per-rate G2 pass requires positive Tail-F1, Holm p<0.05, dz>=0.3, CI lower bound >0, Macro-F1 non-inferiority at delta=0.01, and no class mean F1 loss beyond 0.05.

## Mean metrics

| dataset | protocol | method | macro F1 | tail F1 | tail recall | retained |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| unsw_nb15 | C1b-tier2 + official fixed partition | CL+Graph-soft | 0.4830 | 0.2565 | 0.2829 | 0.738 |
| unsw_nb15 | C1b-tier2 + official fixed partition | CL-issue-intersection | 0.4815 | 0.2238 | 0.2428 | 0.943 |
| unsw_nb15 | C1b-tier2 + official fixed partition | CL-issue-union | 0.4626 | 0.2086 | 0.2414 | 0.723 |
| unsw_nb15 | C1b-tier2 + official fixed partition | Confident-Learning | 0.4850 | 0.2632 | 0.2912 | 0.738 |
| unsw_nb15 | C1b-tier2 + official fixed partition | Graph-CoLD-soft | 0.4840 | 0.2308 | 0.2389 | 1.000 |

## G2 status

- **Global verdict: null**
- **unsw_nb15**: dataset pass=False.
  - rate=0.4: tail delta -0.0102, Holm p=1, dz=-0.590, CI=[-0.0208, -0.0001]; macro delta -0.0035, NI p=0.00732; worst class delta -0.0227; pass=False.
  - rate=0.6: tail delta -0.0109, Holm p=1, dz=-0.568, CI=[-0.0222, -0.0003]; macro delta -0.0031, NI p=0.00736; worst class delta -0.0252; pass=False.
  - rate=0.8: tail delta +0.0010, Holm p=1, dz=0.070, CI=[-0.0062, +0.0095]; macro delta +0.0006, NI p=0.000114; worst class delta -0.0320; pass=False.
  - Mixed-effect robustness: average tail delta -0.0067, one-sided p=0.978, CI=[-0.0132, -0.0002], converged=True; warnings=1.

## Protocol limits

- **unsw_nb15** — C1b-tier2 + official fixed partition; limitation: S1/S2 unavailable because the fixed files lack timestamp/entity columns.
