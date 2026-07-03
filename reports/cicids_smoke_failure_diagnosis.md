# CICIDS Smoke Failure Diagnosis

- Dataset hash: `2585508ac445a94a3eb2244aa64778678928d201555396b4b9afc1ed6a2f1ab4`
- CoLD Macro-F1: 0.966864
- Graph-CoLD Macro-F1: 0.991959
- Active views: host, ip, temporal
- Inactive views: process, threat_intel

## Root Causes
- CICIDS flow-only contract requires process/threat_intel inactive; active-view filtering is required.
- Soft weighting preserves clean boundary evidence that hard ablation deletes.

## Weight Audit
- N_eff/N_train: 0.7979
- Retained clean informative: 1.0000
- Hard retained clean informative: 0.9800
