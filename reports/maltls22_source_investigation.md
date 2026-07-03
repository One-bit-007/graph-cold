# MALTLS-22 Source Investigation

## Summary

Status: `MALTLS-22 source not verified`.

The current repository does not contain a verified MALTLS-22 dataset package,
download URL, license, or access procedure. MALTLS-22 must remain disabled for
D5 reporting until a verifiable source is documented and local files pass
`src.data.audit`.

## Local Repository Findings

1. CoLD paper PDF, BibTeX, or reference files in the repository:
   - No local CoLD PDF is tracked in `paper/`.
   - Previously generated paper artifacts were removed during P0.
   - Local docs and prompts mention MALTLS-22 as planned scope, but do not
     provide an authoritative download URL or license.
2. Local MALTLS-22 source:
   - No source package is present under `data/maltls22/`.
   - No local manifest or checksum file was found.
3. Local licensing/access requirements:
   - Not found.

## Web Investigation

The CoLD NDSS 2026 paper states that MALTLS-22 contains 22 realistic encrypted
malicious traffic types plus benign traffic, captured from 2018 to 2021 and
labeled with threat intelligence. Its reference [13] points to MCRe:

- Q. Yuan, G. Gou, Y. Zhu, Y. Zhu, G. Xiong, and Y. Wang, "MCRe: A unified
  framework for handling malicious traffic with noise labels based on
  multidimensional constraint representation," IEEE Transactions on Information
  Forensics and Security, 2023.

The public search performed for this gate found CoLD/slide references to
MALTLS-22 but did not confirm a direct official dataset download or licensing
page. Therefore, the source remains unverified.

Relevant public pages inspected:

- https://www.ndss-symposium.org/wp-content/uploads/2026-s1950-paper.pdf
- https://www.ndss-symposium.org/wp-content/uploads/s1950-yang-slides.pdf

## Gate Decision

- `MALTLS-22 source not verified`
- `MALTLS-22 must not be reported as evaluated`
- `Do not fabricate MALTLS-22 results`

The `MALTLS22_CONTRACT` remains conditional with `source_verified=False`.
Consequently, readiness reports must set `maltls22.ready_for_d5=false` until a
verified source and local files are available.

## Candidate Alternatives

If MALTLS-22 cannot be obtained, use a replacement only under its own dataset
name and after a separate contract/audit:

- CESNET-TLS-Year22.
- USTC-TFC2016.
- Malicious_TLS.
- Another public TLS/encrypted-traffic dataset with clear labels, license, and
  reproducible acquisition instructions.

These alternatives are not MALTLS-22. If one is selected, the manuscript dataset
name, experimental design, and claims must be rewritten accordingly.

