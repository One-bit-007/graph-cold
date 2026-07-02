#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/ieee"

pdflatex graph_cold_ieee_twocolumn.tex
bibtex graph_cold_ieee_twocolumn
pdflatex graph_cold_ieee_twocolumn.tex
pdflatex graph_cold_ieee_twocolumn.tex
