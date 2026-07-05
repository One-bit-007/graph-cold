$ErrorActionPreference = "Stop"
python -m src.paper.d8_harden
paper\elsevier\build_elsevier.ps1
python -m src.paper.d8_harden --audit-only
