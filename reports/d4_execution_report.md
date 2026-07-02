# D4 Execution Report

## 1. Implementation summary

D4 adds graph-consistency label noise, D3 regression audit coverage, and an OpTC-style enterprise mini-case. The implementation preserves D2 Stage-1 encoder code and keeps Graph-CDM in label / prediction space.

## 2. D3 regression audit result

The D3 audit passed. D_pred still compares view labels to observed labels, D_neigh remains label-space KL with normalization and empty-neighbor fallback, and D_view remains mode disagreement. ERR was updated to the D4 formula with Tail-ERR.

## 3. Graph-consistency noise design and beta=0 proof

Total flips are `floor(rN)`. The implementation splits flips into `floor(beta * floor(rN))` graph-biased flips and the remaining symmetric random flips. When `beta=0`, graph-biased flips are zero and the function delegates directly to `inject_symmetric(y, r, C, rng)`, so the total flip count and transition mechanism exactly match symmetric noise for the same seed.

## 4. OpTC enterprise case description

The OpTC case uses a deterministic synthetic SOC/provenance event table when real OpTC files are absent. Events include host, process, parent process, source/destination IP, timestamp, event type, alert type, label, and risk score. The five views map to host co-occurrence, IP communication, process lineage, temporal windows, and threat-intel/risk similarity. D_chain is enabled with `lambda4=0.1`.

## 5. Tests executed

`python -m pytest -q` passed with 31 tests and 0 failures.

New D4 tests:

- `tests/test_d4_d3_regression_audit.py`
- `tests/test_graph_consistency_noise.py`
- `tests/test_optc_case.py`

## 6. Known issues / TODO before D5

Real OpTC files are not present locally, so the enterprise case runs in synthetic mode. Flash/Argus adapters expose the comparison interface but do not reimplement those papers. D5 should run the full matrix on real CICIDS-2017 and MALTLS-22 data once available.
