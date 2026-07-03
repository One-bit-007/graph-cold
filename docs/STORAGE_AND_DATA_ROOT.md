# Storage and External Data Root

Graph-CoLD supports keeping large real datasets outside the repository.

Configured external root:

```text
E:/graphcold-data
```

Expected CESNET-TLS-Year22 layout:

```text
E:/graphcold-data/
  _downloads/
    cesnet_tls_year22/
      CESNET-TLS-Year22.zip
  tls_alternative/
    cesnet_tls_year22/
      <real extracted CSV/Parquet/DataZoo export files>
```

Resolution priority:

1. CLI `--data-root`
2. Environment variable `GRAPH_COLD_DATA_ROOT`
3. `configs/paths.yaml`
4. Repository-local `data/`

Before large downloads, run:

```bash
python scripts/check_storage.py --data-root E:/graphcold-data --required-gb 80
```

CESNET download and gate commands:

```bash
python scripts/download_tls_alternative.py --candidate cesnet_tls_year22 --mode auto --data-root E:/graphcold-data --download-cache E:/graphcold-data/_downloads --out E:/graphcold-data/tls_alternative/cesnet_tls_year22 --min-free-gb 80 --confirm-large-download
python -m src.data.audit --dataset cesnet_tls_year22 --data-root E:/graphcold-data
python -m src.experiments.smoke_realdata --dataset cesnet_tls_year22 --configs configs --out reports --data-root E:/graphcold-data
python -m src.experiments.cesnet_mini_matrix --dataset cesnet_tls_year22 --configs configs --out results --reports reports --data-root E:/graphcold-data
```

Raw external data and downloaded archives must not be committed to Git.
