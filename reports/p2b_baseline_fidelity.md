# P2b Baseline Fidelity Report

## Per-rate Table
- Table: `tables/table_p2b_baseline_noise_robustness.csv`
- Source: `results/table_main_expanded.csv`
- Source SHA-256: `b9a7f26563e27bced0c2e77b8864bcfe19521bbe1cda7424afad261e63c113a9`

## Key Numbers

### CICIDS-2017 symmetric

| noise_rate | MCRe | MORSE | CoLD | Co-Teaching | Graph-CoLD |
| --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.6191 | 0.5861 | 0.6789 | 0.6620 | 0.7705 |
| 0.2000 | 0.4815 | 0.4886 | 0.6325 | 0.5730 | 0.6805 |
| 0.4000 | 0.3436 | 0.3410 | 0.5420 | 0.4260 | 0.5497 |
| 0.6000 | 0.2409 | 0.2361 | 0.3624 | 0.2988 | 0.3690 |

### CESNET-TLS-Year22 symmetric

| noise_rate | MCRe | MORSE | CoLD | Co-Teaching | Graph-CoLD |
| --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.9433 | 0.9402 | 0.9561 | 0.9440 | 0.9676 |
| 0.2000 | 0.9053 | 0.9024 | 0.9544 | 0.8965 | 0.9527 |
| 0.4000 | 0.8603 | 0.8199 | 0.8990 | 0.8050 | 0.8966 |
| 0.6000 | 0.7983 | 0.7464 | 0.7857 | 0.6164 | 0.7935 |

### UNSW-NB15 symmetric

| noise_rate | MCRe | MORSE | CoLD | Co-Teaching | Graph-CoLD |
| --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.4799 | 0.4842 | 0.4846 | 0.4854 | 0.4841 |
| 0.2000 | 0.4674 | 0.4471 | 0.4797 | 0.4722 | 0.4872 |
| 0.4000 | 0.3792 | 0.3761 | 0.4431 | 0.4370 | 0.4430 |
| 0.6000 | 0.3105 | 0.2979 | 0.3738 | 0.3214 | 0.3695 |

## Diagnosis
- Outcome: `B_protocol_explained`
- Adapter bug found: False
- MORSE split ratio tracks actual injected noise rate: True
- MCRe retain fraction tracks actual injected noise rate: True
- MCRe clean-informative/tail collapse observed: True
- MORSE retains evidence but classifier degrades: True

MCRe/MORSE are not globally broken: clean-label sanity passes and CESNET symmetric/graph-noise curves stay high. The low canonical means come from CICIDS/UNSW high-noise tabular settings, where MCRe's class-wise centroid filter removes many clean informative samples and MORSE retains evidence but propagates weak pseudo-label decision boundaries.

## Manuscript Caveat
MCRe and MORSE are reported as faithful tabular adapters in this real-data label-noise protocol, but their noise robustness is not claimed to reproduce the original papers because centroid-based purification degrades on CICIDS/UNSW high-noise tabular class geometry while remaining strong on CESNET symmetric and graph-noise settings.

## Updated Canonical Margins
- Graph-CoLD minus MCRe Macro-F1: 0.0263
- Graph-CoLD minus MORSE Macro-F1: 0.0362

## Reproduction Commands
- `python -m src.paper.p2b_baseline_fidelity`
- `python -m pytest tests/test_p2b_baseline_fidelity.py tests/test_number_consistency.py -q`
