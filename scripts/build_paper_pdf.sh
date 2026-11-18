#!/usr/bin/env bash
# Flatten paper/MAST.md -> paper/arxiv.md -> paper/arxiv.tex -> paper/MAST.pdf (pdfLaTeX + BibTeX).
# Adapted from catskillsresearch/lyasoff2016 scripts/build_arxiv_pdf.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${TMPDIR:-/tmp}"

if [[ "${1:-}" != "--no-flatten" ]]; then
  echo "==> Flattening paper/MAST.md note tree"
  python3 scripts/compile_md_tree.py --md-only
  echo "==> Building paper/arxiv.tex"
  python3 scripts/build_paper_tex.py
fi

cd paper
TEX="arxiv.tex"
command -v bibtex >/dev/null || { echo "BibTeX is required" >&2; exit 1; }
test -s references.bib || { echo "Missing paper/references.bib" >&2; exit 1; }
latexmk -C "$TEX" >/dev/null 2>&1 || true
rm -f arxiv.aux arxiv.out arxiv.toc arxiv.bbl arxiv.blg
echo "==> Compiling PDF (pdfLaTeX + BibTeX via latexmk)"
latexmk -pdf -bibtex \
  -pdflatex="pdflatex -interaction=nonstopmode -halt-on-error" \
  -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null || {
  echo "latexmk reported errors; tail of log:" >&2
  tail -n 60 arxiv.log >&2 || true
  exit 1
}
test -s arxiv.bbl || { echo "BibTeX produced no bibliography" >&2; exit 1; }
if grep -Eq 'Citation .* undefined|There were undefined citations|There were undefined references|Reference .* undefined' arxiv.log; then
  echo "Unresolved references/citations:" >&2
  grep -E 'undefined' arxiv.log >&2 || true
  exit 1
fi
if grep -Eq 'Missing character' arxiv.log; then
  echo "Missing glyphs (undeclared Unicode):" >&2
  grep -E 'Missing character' arxiv.log | sort | uniq -c >&2
  exit 1
fi
cp -f arxiv.pdf MAST.pdf
pages=$(grep -aoE '/Type */Pages.*?/Count [0-9]+' MAST.pdf | grep -oE '[0-9]+$' | sort -n | tail -1 || true)
echo "wrote paper/MAST.pdf ($(du -h MAST.pdf | cut -f1)${pages:+, $pages pages})"
