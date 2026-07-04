#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex
bibtex graph_cold_cas_realdata
pdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex
pdflatex --disable-installer -halt-on-error -interaction=nonstopmode graph_cold_cas_realdata.tex
