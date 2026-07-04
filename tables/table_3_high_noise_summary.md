# Table 3. High-noise robustness summary

Source: `results/table_main_expanded.csv`.

| Dataset           | Method             | Macro-F1 mean | FPR mean | FNR mean | ERR mean | Compression ratio mean | Scenario count |
| ----------------- | ------------------ | ------------- | -------- | -------- | -------- | ---------------------- | -------------- |
| CICIDS-2017       | Graph-CoLD         | 0.9852        | 0.0021   | 0.0016   | 1.0000   | 0.9970                 | 24             |
| CICIDS-2017       | CoLD               | 0.9333        | 0.0020   | 0.0059   | 0.8228   | 0.9989                 | 24             |
| CICIDS-2017       | ablation_hard      | 0.9333        | 0.0020   | 0.0059   | 0.8228   | 0.9989                 | 24             |
| CICIDS-2017       | Noisy-Supervised   | 0.4082        | 0.2585   | 0.2183   | 1.0000   | 1.0000                 | 24             |
| CICIDS-2017       | Confident-Learning | 0.5576        | 0.2002   | 0.1021   | 0.6819   | 0.9995                 | 24             |
| CICIDS-2017       | Co-Teaching-lite   | 0.5623        | 0.2774   | 0.0813   | 0.6217   | 0.9913                 | 24             |
| CESNET-TLS-Year22 | Graph-CoLD         | 0.9945        | 0.0130   | 0.0001   | 1.0000   | 0.9821                 | 24             |
| CESNET-TLS-Year22 | CoLD               | 0.9944        | 0.0136   | 0.0001   | 0.9459   | 0.9824                 | 24             |
| CESNET-TLS-Year22 | ablation_hard      | 0.9944        | 0.0136   | 0.0001   | 0.9459   | 0.9824                 | 24             |
| CESNET-TLS-Year22 | Noisy-Supervised   | 0.5891        | 0.2120   | 0.1619   | 1.0000   | 0.9999                 | 24             |
| CESNET-TLS-Year22 | Confident-Learning | 0.8434        | 0.0598   | 0.0374   | 0.8444   | 0.9952                 | 24             |
| CESNET-TLS-Year22 | Co-Teaching-lite   | 0.7992        | 0.1804   | 0.0076   | 0.8237   | 0.9933                 | 24             |
