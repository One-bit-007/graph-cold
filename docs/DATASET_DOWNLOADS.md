# Dataset Download and Preparation Commands

This project permits only real audited datasets to enter smoke tests and D5.
These commands prepare files and reports; none of them runs the D5 full matrix.

## CICIDS-2017

Official sources:

- UNB CIC dataset page: https://www.unb.ca/cic/datasets/ids-2017.html
- CIC download form linked from UNB: https://cicresearch.ca/CICDataset/CIC-IDS-2017/

The UNB page states that `GeneratedLabelledFlows.zip` and
`MachineLearningCSV.zip` are publicly available for researchers. In practice the
download link may require the CIC form or a redirect, so auto mode is best-effort
and intentionally conservative.

Print instructions only:

```bash
python scripts/download_cicids2017.py --out data/cicids2017 --mode instructions
```

Try official-page auto discovery:

```bash
python scripts/download_cicids2017.py --out data/cicids2017 --mode auto
```

Prepare a manually downloaded zip:

```bash
python scripts/download_cicids2017.py --out data/cicids2017 --zip path/to/MachineLearningCSV.zip --mode local-zip
```

The local-zip mode extracts and renames only the eight expected CSV files:

```text
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## MALTLS-22

MALTLS-22 remains source-unverified in this repository. Do not download another
TLS dataset into `data/maltls22/`, and do not report MALTLS-22 results until a
verifiable source/access route is documented.

## TLS Alternative Candidates

Print CESNET-TLS-Year22 instructions:

```bash
python scripts/download_tls_alternative.py --candidate cesnet_tls_year22 --mode instructions
```

Large-download guard:

```bash
python scripts/download_tls_alternative.py --candidate cesnet_tls_year22 --mode auto --out data/tls_alternative/cesnet_tls_year22
```

The command above will not download large CESNET files unless
`--confirm-large-download` is supplied. Even after download, the dataset must be
reported as `CESNET-TLS-Year22`, not MALTLS-22, and needs its own audited
contract before use in experiments.

DataZoo guarded mode:

```bash
python scripts/download_tls_alternative.py --candidate cesnet_tls_year22 --mode datazoo --out data/tls_alternative/cesnet_tls_year22 --confirm-large-download
```

Prepare a manually downloaded Zenodo/DataZoo archive or exported table:

```bash
python scripts/download_tls_alternative.py --candidate cesnet_tls_year22 --mode local-archive --archive path/to/cesnet_tls_year22_archive_or_table --out data/tls_alternative/cesnet_tls_year22
```

Official sources:

- Zenodo record: https://zenodo.org/records/10608607
- CESNET DataZoo docs: https://cesnet.github.io/cesnet-datazoo/
- CESNET DataZoo GitHub: https://github.com/CESNET/cesnet-datazoo

After local files are present, inspect schema and set `configs/datasets.yaml`
`cesnet_tls_year22.label_col` if the label column is not named `label`, then run:

```bash
python -m src.data.audit --dataset cesnet_tls_year22
python -m src.experiments.smoke_realdata --dataset cesnet_tls_year22 --configs configs --out reports
python -m src.experiments.cesnet_mini_matrix --dataset cesnet_tls_year22 --configs configs --out results --reports reports
```

## UNSW-NB15 Backup Candidate

Official source:

- UNSW Canberra Cyber dataset page: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Recommended local path:

```text
E:\graphcold-data\unsw_nb15\
```

Download the official CSV files and the feature-list file manually from the
UNSW page, then place the CSV files under the local path above. Do not commit
raw CSVs or archives.

Print instructions:

```bash
python scripts/download_unsw_nb15.py --mode instructions --out E:/graphcold-data/unsw_nb15
```

Write a local manifest:

```bash
python scripts/download_unsw_nb15.py --mode manifest --out E:/graphcold-data/unsw_nb15
```

Refresh audit and policy reports after files are present:

```bash
python -m src.data.audit --dataset unsw_nb15 --data-root E:/graphcold-data
python -m src.data.unsw_policy --data-root E:/graphcold-data --out reports
python -m src.experiments.second_dataset_selection --reports reports
```

Default loader policy:

- multiclass `attack_cat`
- drop empty `attack_cat`
- `postfilter` with `min_class_count = 1000`
- active views: `ip | temporal`

If UNSW-NB15 is selected as the second dataset, D5 and the manuscript must
report the dataset as `UNSW-NB15`.

## USTC-TFC2016 Candidate

USTC-TFC2016 is currently candidate-only. The repository records an audit stub
and candidate report, but it must not enter D5 until the user confirms a
download route, license, file format, label mapping, and audit pass.

## OpTC

Print instructions:

```bash
python scripts/download_optc.py --mode instructions
```

Write a manifest only:

```bash
python scripts/download_optc.py --mode manifest
```

Prepare a converted events table:

```bash
python scripts/download_optc.py --mode local-events --events path/to/events.csv --out data/optc
```

Required `events.csv` columns:

```text
host_id
process_id
parent_process_id
src_ip
dst_ip
timestamp
event_type
alert_type
label
risk_score
```

OpTC source material is large and may require manual conversion from eCAR, Bro,
or JSON provenance records. This repository does not start a full raw OpTC
download by default.

## Unified Entry Points

```bash
python scripts/prepare_datasets.py --dataset cicids2017 --mode instructions
python scripts/prepare_datasets.py --dataset cicids2017 --mode local-zip --zip path/to/MachineLearningCSV.zip
python scripts/prepare_datasets.py --dataset tls_alternative --candidate cesnet_tls_year22 --mode instructions
python scripts/prepare_datasets.py --dataset unsw_nb15 --mode instructions --out E:/graphcold-data/unsw_nb15
python scripts/prepare_datasets.py --dataset optc --mode instructions
python scripts/check_data_ready.py
```

Smoke may run only after audit passes:

```bash
python -m src.data.audit
python scripts/check_data_ready.py
python -m src.experiments.smoke_realdata --dataset cicids2017 --configs configs --out reports
```

Forbidden at this stage:

```bash
python -c "from src.experiments.d5 import run_d5_experiments; run_d5_experiments(...)"
```
