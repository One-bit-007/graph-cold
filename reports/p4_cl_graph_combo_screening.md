# P0-4 CL + Graph-CDM Screening Report

- Rows: 45 / expected 45
- Seeds: [0, 1, 2]; rates: [0.6]
- Primary G2 method: `CL+Graph-soft` vs `Confident-Learning`.
- G2 pass requires Holm-significant positive Tail-F1 with CI lower bound > 0 and Macro-F1 non-inferiority at delta=0.01.

## Mean metrics

| dataset | protocol | method | macro F1 | tail F1 | tail recall | retained |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| cesnet_tls_year22 | C1b-tier2 + S1∩S2 temporal/entity-disjoint | CL+Graph-soft | 0.4054 | 0.3272 | 0.2152 | 0.464 |
| cesnet_tls_year22 | C1b-tier2 + S1∩S2 temporal/entity-disjoint | CL-issue-intersection | 0.4421 | 0.2563 | 0.1317 | 0.917 |
| cesnet_tls_year22 | C1b-tier2 + S1∩S2 temporal/entity-disjoint | CL-issue-union | 0.3893 | 0.3008 | 0.1904 | 0.439 |
| cesnet_tls_year22 | C1b-tier2 + S1∩S2 temporal/entity-disjoint | Confident-Learning | 0.3994 | 0.3236 | 0.2093 | 0.464 |
| cesnet_tls_year22 | C1b-tier2 + S1∩S2 temporal/entity-disjoint | Graph-CoLD-soft | 0.4495 | 0.2609 | 0.1420 | 1.000 |
| cicids2017 | C1-tier1 + S1 temporal-by-source-file | CL+Graph-soft | 0.0766 | 0.1923 | 0.1328 | 0.951 |
| cicids2017 | C1-tier1 + S1 temporal-by-source-file | CL-issue-intersection | 0.0959 | 0.1920 | 0.1325 | 0.998 |
| cicids2017 | C1-tier1 + S1 temporal-by-source-file | CL-issue-union | 0.0758 | 0.1896 | 0.1314 | 0.949 |
| cicids2017 | C1-tier1 + S1 temporal-by-source-file | Confident-Learning | 0.0782 | 0.1923 | 0.1328 | 0.951 |
| cicids2017 | C1-tier1 + S1 temporal-by-source-file | Graph-CoLD-soft | 0.0812 | 0.1921 | 0.1326 | 1.000 |
| unsw_nb15 | C1b-tier2 + official fixed partition | CL+Graph-soft | 0.4772 | 0.2424 | 0.2671 | 0.736 |
| unsw_nb15 | C1b-tier2 + official fixed partition | CL-issue-intersection | 0.4779 | 0.2159 | 0.2282 | 0.943 |
| unsw_nb15 | C1b-tier2 + official fixed partition | CL-issue-union | 0.4638 | 0.2056 | 0.2417 | 0.719 |
| unsw_nb15 | C1b-tier2 + official fixed partition | Confident-Learning | 0.4787 | 0.2468 | 0.2715 | 0.736 |
| unsw_nb15 | C1b-tier2 + official fixed partition | Graph-CoLD-soft | 0.4808 | 0.2229 | 0.2387 | 1.000 |

## G2 status

- **cesnet_tls_year22**: tail delta +0.0035, Holm p=1; macro delta +0.0060, NI p=0.0868; G2 pass=False.
- **cicids2017**: tail delta +0.0000, Holm p=1; macro delta -0.0016, NI p=0.0176; G2 pass=False.
- **unsw_nb15**: tail delta -0.0044, Holm p=1; macro delta -0.0015, NI p=0.0485; G2 pass=False.

## Protocol limits

- **cicids2017** — C1-tier1 + S1 temporal-by-source-file; limitation: S2 unavailable because the local copy lacks entity columns.
- **cesnet_tls_year22** — C1b-tier2 + S1∩S2 temporal/entity-disjoint; limitation: audit frame spans about seven hours.
- **unsw_nb15** — C1b-tier2 + official fixed partition; limitation: S1/S2 unavailable because the fixed files lack timestamp/entity columns.
