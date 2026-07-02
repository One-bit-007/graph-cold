# Graph-CoLD D7 Reproducibility Package

This package rebuilds the submission artifacts from the tracked D5/D6 outputs.
It does not retrain models, rerun the experimental matrix, or modify model code.

## Inputs

- `results/table_main.csv`
- `results/table_main_raw.csv`
- `results/table_ablation.csv`
- `results/table_optc.csv`
- `results/stat_tests.json`
- `tables/table_1_main_results.csv`
- `tables/table_2_ablation.csv`
- `tables/table_3_optc.csv`
- `figures/fig2_macro_f1_vs_noise_rate.png`
- `figures/fig3_err_vs_compression_ratio.png`
- `figures/fig4_ablation_drop_bar.png`
- `figures/fig5_optc_soc_ranking.png`

## Rebuild command

From the repository root:

```bash
bash reproducibility/run_all.sh
```

If the system `python` command is a Windows Store stub, provide a real runtime:

```bash
PYTHON_BIN=/path/to/python bash reproducibility/run_all.sh
```

The script checks required D5 result files, regenerates D6 paper-ready tables and
figures using `src.paper.d6_prep`, validates the test suite, and compiles
`paper/graph_cold_cas_submission.tex` when `latexmk` or `pdflatex` is available.

## Outputs

- `paper/graph_cold_cas_submission.tex`
- `paper/graph_cold_cas_submission.pdf`
- `tables/table_1_main_results.csv`
- `tables/table_2_ablation.csv`
- `tables/table_3_optc.csv`
- `reports/d6_statistical_narrative.md`
- `reports/d7_final_checklist.json`

## Traceability note

All numeric claims in the manuscript are taken from `results/`, `tables/`, and
`reports/d6_statistical_narrative.md`. In P0 mode, D5 requires real CICIDS-2017,
MALTLS-22, and OpTC files and stops before writing results if any required input
is missing.
