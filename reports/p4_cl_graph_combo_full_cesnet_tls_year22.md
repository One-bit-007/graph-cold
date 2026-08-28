# P0-4 CL + Graph-CDM Full Report

- Rows: 150 / expected 150
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; rates: [0.4, 0.6, 0.8]
- Primary G2 method: `CL+Graph-soft` vs `Confident-Learning`.
- Frozen family B size: 18 rate-specific tests; unrun cells conservatively enter Holm as p=1.
- Per-rate G2 pass requires positive Tail-F1, Holm p<0.05, dz>=0.3, CI lower bound >0, Macro-F1 non-inferiority at delta=0.01, and no class mean F1 loss beyond 0.05.

## Mean metrics

| dataset | protocol | method | macro F1 | tail F1 | tail recall | retained |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| cesnet_tls_year22 | C1b-tier2 + S1∩S2 temporal/entity-disjoint | CL+Graph-soft | 0.3956 | 0.2995 | 0.1953 | 0.515 |
| cesnet_tls_year22 | C1b-tier2 + S1∩S2 temporal/entity-disjoint | CL-issue-intersection | 0.4354 | 0.2423 | 0.1365 | 0.925 |
| cesnet_tls_year22 | C1b-tier2 + S1∩S2 temporal/entity-disjoint | CL-issue-union | 0.3804 | 0.2707 | 0.1760 | 0.482 |
| cesnet_tls_year22 | C1b-tier2 + S1∩S2 temporal/entity-disjoint | Confident-Learning | 0.3934 | 0.2951 | 0.1942 | 0.515 |
| cesnet_tls_year22 | C1b-tier2 + S1∩S2 temporal/entity-disjoint | Graph-CoLD-soft | 0.4508 | 0.2628 | 0.1497 | 1.000 |

## G2 status

- **Global verdict: null**
- **cesnet_tls_year22**: dataset pass=False.
  - rate=0.4: tail delta +0.0038, Holm p=1, dz=0.146, CI=[-0.0102, +0.0200]; macro delta +0.0023, NI p=0.00393; worst class delta -0.0320; pass=False.
  - rate=0.6: tail delta +0.0118, Holm p=1, dz=0.454, CI=[-0.0034, +0.0256]; macro delta +0.0072, NI p=0.000391; worst class delta -0.0192; pass=False.
  - rate=0.8: tail delta -0.0024, Holm p=1, dz=-0.112, CI=[-0.0149, +0.0101]; macro delta -0.0027, NI p=0.0486; worst class delta -0.0354; pass=False.
  - Mixed-effect robustness: average tail delta +0.0044, one-sided p=0.192, CI=[-0.0055, +0.0144], converged=True; warnings=1.

## Protocol limits

- **cesnet_tls_year22** — C1b-tier2 + S1∩S2 temporal/entity-disjoint; limitation: audit frame spans about seven hours.
