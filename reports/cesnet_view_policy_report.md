# CESNET View Policy Report

- Active views: ip | temporal
- Unsupported views: process | threat_intel
- Host optional columns present: True

## View Columns
- ip: TLS_SNI, TLS_JA3, FLOW_ENDREASON_IDLE, FLOW_ENDREASON_ACTIVE, FLOW_ENDREASON_END, FLOW_ENDREASON_OTHER, PACKETS, PACKETS_REV, BYTES, BYTES_REV, DURATION, PPI_DURATION
- temporal: TIME_FIRST, TIME_LAST
- host: SRC_IP, PHIST_SRC_IPT, DST_IP, PHIST_DST_IPT
- process: none
- threat_intel: none
