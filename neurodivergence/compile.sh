#!/bin/bash
# Compile the LaTeX report. Run from neurodivergence/
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex   # second pass for refs
echo "Done: report.pdf"
