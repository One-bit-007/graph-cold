# P2b Baseline Fidelity Report

## Per-rate Table
- Table: `tables/table_p2b_baseline_noise_robustness.csv`
- Source: `results/table_main_expanded.csv`
- Source SHA-256: `125ce9e63442e82deced11caf3391dd9b4ac103a6f1f3c1a7776ca7499d493cf`

## Key Numbers

### CICIDS-2017 symmetric

| noise_rate | MCRe | MORSE | CoLD | Co-Teaching | Graph-CoLD |
| --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.6290 | 0.6379 | 0.7092 | 0.7644 | 0.9903 |
| 0.2000 | 0.5110 | 0.5132 | 0.7071 | 0.6998 | 0.9895 |
| 0.4000 | 0.3562 | 0.3617 | 0.7082 | 0.5829 | 0.9886 |
| 0.6000 | 0.2515 | 0.2393 | 0.7098 | 0.4699 | 0.9858 |

### CESNET-TLS-Year22 symmetric

| noise_rate | MCRe | MORSE | CoLD | Co-Teaching | Graph-CoLD |
| --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.9740 | 0.9713 | 0.9921 | 0.9898 | 0.9956 |
| 0.2000 | 0.9383 | 0.9394 | 0.9926 | 0.9685 | 0.9958 |
| 0.4000 | 0.8687 | 0.8675 | 0.9916 | 0.9052 | 0.9950 |
| 0.6000 | 0.8219 | 0.8170 | 0.9916 | 0.7682 | 0.9937 |

### UNSW-NB15 symmetric

| noise_rate | MCRe | MORSE | CoLD | Co-Teaching | Graph-CoLD |
| --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.4822 | 0.5153 | 0.6029 | 0.5638 | 0.5844 |
| 0.2000 | 0.4606 | 0.4691 | 0.6007 | 0.5448 | 0.5843 |
| 0.4000 | 0.3983 | 0.4192 | 0.5939 | 0.4896 | 0.5785 |
| 0.6000 | 0.3635 | 0.3537 | 0.5873 | 0.4238 | 0.5708 |

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
- Graph-CoLD minus MCRe Macro-F1: 0.2344
- Graph-CoLD minus MORSE Macro-F1: 0.2350

## Reproduction Commands
- `python -m src.paper.p2b_baseline_fidelity`
- `python -m pytest tests/test_p2b_baseline_fidelity.py tests/test_number_consistency.py -q`
