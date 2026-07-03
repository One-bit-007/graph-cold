# CICIDS Final Protocol Freeze

- Dataset: `cicids2017`
- Reported as: CICIDS-2017
- Dataset hash: `2585508ac445a94a3eb2244aa64778678928d201555396b4b9afc1ed6a2f1ab4`
- Selected policy: `postfilter11`
- Retained classes: 11
- No refined9 claim: true

## Removed Classes

- Heartbleed: 11
- Infiltration: 36
- Web Attack replacement-character Sql Injection: 21
- Web Attack replacement-character XSS: 652

## Views

- Active: host | ip | temporal
- Inactive: process | threat_intel

## Noise And Splits

- Symmetric rates: 0.1, 0.2, 0.4, 0.6
- Asymmetric rates: 0.1, 0.2, 0.4, 0.6
- Graph beta settings: 0.0, 0.3, 0.6, 1.0
- Seeds: 0, 1, 2
- Split policy: 8:2 stratified split when class counts permit
- Noise is injected only into training labels.

## Fairness Protocol

CoLD, ablation_hard, and Graph-CoLD must use the same real CICIDS split and
noise protocol. `ablation_hard` is the rho=0 hard-retention comparator.
Graph-CoLD adds Graph-CDM and evidence weighting on top of the same data
protocol.

## ERR Definition

ERR is measured on clean informative samples using
`retained(v)=1[w(v)>=tau_ret]` and Tail-ERR; final ERR is the average of ERR and
Tail-ERR.

## D5 Status

Formal D5 is not allowed yet because no verified second dataset is ready.
