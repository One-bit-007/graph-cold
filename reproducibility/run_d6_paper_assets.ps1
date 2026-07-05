$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Test-PythonCandidate {
    param([string]$Path)
    if (-not $Path) { return $false }
    try { & $Path --version *> $null; return $LASTEXITCODE -eq 0 } catch { return $false }
}
$PythonCandidates = @()
if ($env:GRAPH_COLD_PYTHON) { $PythonCandidates += $env:GRAPH_COLD_PYTHON }
$PythonCandidates += (Join-Path $RepoRoot ".venv\Scripts\python.exe")
$PythonCandidates += (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")
$PathPython = Get-Command python -ErrorAction SilentlyContinue
if ($PathPython) { $PythonCandidates += $PathPython.Source }
$Python = $null
foreach ($Candidate in $PythonCandidates) {
    if (Test-PythonCandidate $Candidate) { $Python = $Candidate; break }
}
if (-not $Python) { throw "No usable Python interpreter found. Set GRAPH_COLD_PYTHON to Python 3.10+ and rerun." }

# Equivalent manual command: python -m src.paper.d6_prep
& $Python -m src.paper.d6_prep
