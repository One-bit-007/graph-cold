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

