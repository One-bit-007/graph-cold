# Graph-CoLD Dataset Preparation

Graph-CoLD is now gated on real local datasets. Raw data must be downloaded by
the user and placed under `data/`, which is ignored by Git. The audit commands
below verify files, schema, hashes, label distributions, and view support before
any D5 experiment is allowed.

Download/preparation commands are maintained in `docs/DATASET_DOWNLOADS.md`.

```bash
python -m src.data.audit
python -m src.experiments.smoke_realdata --dataset cicids2017 --configs configs --out reports
```

Do not commit raw datasets, packet captures, compressed CSVs, Parquet files, or
derived formal experiment tables until the audit and smoke gate pass on real
inputs.

## CICIDS-2017

Official dataset name: CICIDS-2017 / Intrusion Detection Evaluation Dataset
(CICIDS2017), Canadian Institute for Cybersecurity.

Download page:

- https://www.unb.ca/cic/datasets/ids-2017.html

Recommended input: CSV flow files. Prefer the generated flow CSVs for this
project; PCAP files are large and are not the default ingestion format.

Place the eight expected CSV files exactly as:

```text
data/
  cicids2017/
    Monday-WorkingHours.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    Friday-WorkingHours-Morning.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Required label column: `Label`.

Required field families:

- Flow numeric features.
- Source/destination IP.
- Source/destination port.
- Protocol.
- Timestamp or flow timestamp, if present.

Column names are stripped before audit because CICIDS-2017 CSVs are known to
contain leading or trailing spaces in some releases. Timestamp absence does not
hard-fail the audit, but the temporal view is marked `derived_limited`. If all
IP/port/protocol identifiers are absent, the audit blocks the dataset.

Graph view support:

- `host`: expected from host/IP identifiers.
- `ip`: expected from IP/port/protocol fields.
- `temporal`: expected when timestamp is present; otherwise limited.
- `process`: not expected for CICIDS-2017.
- `threat_intel`: not expected for CICIDS-2017.

## MALTLS-22

MALTLS-22 is referenced by CoLD as a 23-class encrypted malicious traffic
benchmark containing benign traffic plus 22 malicious TLS traffic categories.
The current repository does not contain a verified official download URL,
license, or access procedure. Until this is resolved:

- `MALTLS-22 source not verified`
- `MALTLS-22 must not be reported as evaluated`
- `Do not fabricate MALTLS-22 results`

Expected local path if the source is later verified:

```text
data/
  maltls22/
    <verified original MALTLS-22 tables>
```

Expected schema:

- Label column: `label`.
- TLS/flow/packet feature columns.
- Timestamp/time column.
- At least 10,000 rows and at least two classes.

The MALTLS-22 contract is conditional and has `source_verified=False` by
default, so D5 remains blocked even if a local folder appears, until the source
investigation is updated with a verified acquisition path.

## TLS Alternative Candidates

If MALTLS-22 cannot be obtained, a replacement can be proposed, but it must be
reported under its own name and audited under a separate contract. It must never
be renamed as MALTLS-22.

Candidate TLS/encrypted-traffic datasets:

- CESNET-TLS-Year22.
- USTC-TFC2016.
- Malicious_TLS.
- Other public TLS or encrypted traffic datasets with clear labels, license, and
  reproducible download instructions.

Replacement rules:

- The replacement dataset is not MALTLS-22.
- Paper dataset names and claims must be rewritten if a replacement is used.
- A replacement needs its own contract and source investigation.
- It can enter experiments only after real files pass audit.

### CESNET-TLS-Year22

Official dataset name to report: `CESNET-TLS-Year22`.

Local key used by this repository:

```text
cesnet_tls_year22
```

Local path:

```text
data/
  tls_alternative/
    cesnet_tls_year22/
      <real CESNET-TLS-Year22 CSV/Parquet/DataZoo export files>
```

Official sources:

- Zenodo record: https://zenodo.org/records/10608607
- CESNET DataZoo: https://cesnet.github.io/cesnet-datazoo/
- DataZoo GitHub: https://github.com/CESNET/cesnet-datazoo

The dataset must be reported as `CESNET-TLS-Year22`, never as MALTLS-22. It is
a replacement candidate for the second main dataset slot only after real local
files pass audit and smoke/mini-matrix gates.

Expected schema is configurable because DataZoo exports and Zenodo files may
differ. The contract accepts label-like columns (`label`, `class`, `service`,
`app`, `target`, `category`), TLS/flow feature columns (`tls`, `sni`, `ja3`,
`flow`, `packet`, `bytes`, `duration`), and timestamp-like columns.

Graph view policy:

- `ip`: expected from TLS/flow features.
- `temporal`: expected from timestamp/date fields.
- `host`: optional only if endpoint-like fields exist and are explicitly enabled.
- `process`: unsupported.
- `threat_intel`: unsupported.

## OpTC Enterprise Case

The OpTC case study requires real provenance events at:

```text
data/
  optc/
    events.csv
```

Required columns:

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

If `data/optc/events.csv` is missing, the enterprise case is marked
`not_available`, and no OpTC result may be reported. Flash/Argus adapters can be
used only after this real provenance table passes audit.

## Git Hygiene

The repository `.gitignore` excludes `/data/`, `/datasets/`, `*.pcap`,
`*.csv.gz`, and `*.parquet`. Keep raw data outside version control. Audit
reports may include file hashes and schema summaries, but not raw rows.
