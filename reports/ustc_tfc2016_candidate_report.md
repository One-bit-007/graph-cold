# USTC-TFC2016 Candidate Report

- Dataset: `ustc_tfc2016`
- Reported as: USTC-TFC2016
- Current status: `candidate_only`
- May enter D5: false

## Source Candidate

USTC-TFC2016 is tracked only as a possible public traffic-classification
candidate. The current repository has not completed source, license, file
layout, label, or schema verification.

## Expected Format

- Packet/flow tables or converted PCAP-derived features.
- Explicit class/family labels.
- Reproducible split seed or documented train/test split.

## Likely Preprocessing

- Convert raw traffic artifacts into tabular flow features.
- Normalize the label taxonomy.
- Verify benign/malicious class mapping.
- Audit class counts and feature columns.

## Recommendation

Do not select USTC-TFC2016 for D5 unless the user confirms the download route,
license, schema, and a passing audit.
