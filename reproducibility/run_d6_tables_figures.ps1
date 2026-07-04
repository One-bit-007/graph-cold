$ErrorActionPreference = "Stop"
python -m src.paper.d6_prep
python -m src.paper.d7_assemble
paper\elsevier\build_elsevier.ps1
python -m src.paper.d7_assemble --audit-only
