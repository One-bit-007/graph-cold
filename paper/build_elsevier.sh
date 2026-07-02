#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/elsevier"

pdflatex graph_cold_cas_final.tex
bibtex graph_cold_cas_final
pdflatex graph_cold_cas_final.tex
pdflatex graph_cold_cas_final.tex
