# D9 Reviewer Risk Pack v2

This risk pack consolidates the D8 reviewer risk register, the D9 submission
lock audit, and the D9.5 baseline reinforcement update. It is an internal
pre-submission reviewer-risk artifact, not an experiment result file.

## Scope Lock

- Formal datasets: CICIDS-2017 postfilter11 and CESNET-TLS-Year22 postfilter25.
- Formal methods after D9.5: Graph-CoLD, CoLD, ablation_hard,
  Noisy-Supervised, Confident-Learning, Co-Teaching-lite, and Decoupling.
- Excluded datasets remain out of scope: MALTLS-22, OpTC, UNSW-NB15, and
  USTC-TFC2016.
- Excluded methods remain out of formal comparison unless independently
  implemented and real-data smoke-passed: MCRe, MORSE, Flash, Argus, full FINE,
  full Co-Teaching, and FINE-style.

## Reviewer-Risk Clusters

1. Baseline breadth: D9.5 adds Decoupling, but the comparison remains bounded
   to implemented and smoke-passed baselines. The manuscript should avoid
   broad benchmark claims.
2. FINE-style exclusion: FINE-style was implemented but excluded because it
   failed the D9.5 real-data smoke gate on CICIDS-2017 symmetric noise. This
   should be explained directly if the manuscript discusses excluded methods.
3. CESNET scope: CESNET-TLS-Year22 is a deterministic postfilter25 audit-window
   subset, not a full archive evaluation. Claims should frame CESNET primarily
   as a cross-domain stability and evidence-retention check.
4. SOC operational claim: compression ratio and ERR are operational proxies,
   not an analyst-in-the-loop validation study.
5. ERR interpretation: ERR_final can reach 1.0000 because it measures retention
   over clean informative samples under a retained-mask threshold; it is not a
   detection-accuracy score.
6. Co-Teaching-lite naming: the implementation must remain labeled
   Co-Teaching-lite and not be described as full Co-Teaching.
7. Submission declarations: author list, affiliations, funding, competing
   interest, AI declaration review, final PDF review, and journal upload remain
   human-confirmation tasks.

## D9/D9.5 Evidence Pointers

- D9 submission lock audit: `reports/d9/d9_submission_lock_audit.json`
- D8 risk register: `reports/d8/reviewer_risk_register_v1.md`
- D9.5 baseline update: `reports/d9_5/reviewer_risk_update_baselines.md`
- D9.5 final audit: `reports/d9_5/d9_5_final_audit.json`

## Submission Stance

The D9/D9.5 package is suitable as a candidate package for author review and
rebuttal preparation. It is not submission-ready until the remaining human
metadata and declaration tasks are completed. `submission_ready` must remain
false in derived D11 artifacts.
