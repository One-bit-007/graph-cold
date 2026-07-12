# Cover Letter Draft v1.0

Dear Editors,

We submit the v1.0 draft of Graph-CoLD for consideration by Computers & Security.
The manuscript studies noisy-label intrusion detection and SOC alert
prioritization using verified CICIDS-2017, CESNET-TLS-Year22, and UNSW-NB15
real-data settings. The core contribution is a graph label-denoising method that
uses label-space consistency over active graph views and audits evidence
retention against hard deletion.

The evaluation is deliberately bounded to verified real-data baselines.
Graph-CoLD improves Macro-F1 over the aligned CoLD baseline by
2.17 percentage points in a paired scenario-level test
(p=1.27e-04). Mean ERR_final changes by
0.00 percentage points relative to hard deletion, so the
clean rerun does not claim a positive ERR-retention lift.

The manuscript explicitly states that CESNET-TLS-Year22 is a deterministic
audit-window subset, UNSW-NB15 is a partition-layout robustness check, and
MALTLS-22 and OpTC are not reported.

Sincerely,

The Graph-CoLD authors
