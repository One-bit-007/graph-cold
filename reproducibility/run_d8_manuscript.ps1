$ErrorActionPreference = "Stop"
if (-not $env:GRAPH_COLD_PYTHON) { $env:GRAPH_COLD_PYTHON = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" }
& $env:GRAPH_COLD_PYTHON -m src.paper.d8_harden
powershell -ExecutionPolicy Bypass -File paper\elsevier\build_elsevier.ps1
& $env:GRAPH_COLD_PYTHON -m src.paper.d8_harden --audit-only
