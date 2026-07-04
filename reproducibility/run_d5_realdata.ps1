$ErrorActionPreference = "Stop"
python -m src.data.audit
python scripts/check_data_ready.py
python -m src.experiments.d5 --out results --configs configs
python -m src.experiments.d5_baseline_expansion --out results --configs configs --reports reports
