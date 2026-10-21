#!/usr/bin/env bash

export PATH=/cluster/apps/texlive/2026/bin/x86_64-linux:$PATH

pdflatex slides && biber slides && pdflatex slides && pdflatex slides
