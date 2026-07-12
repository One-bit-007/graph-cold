$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
pdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
bibtex graph_cold_cas_realdata
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
pdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
pdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
