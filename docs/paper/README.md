# Paper source

Self-contained NeurIPS-style preprint layout (no venue `.sty` needed). Source of record is
`../WRITEUP-LPFT.md`. Edit the Markdown first, then
mirror the change here.

Build (tectonic, installed via Homebrew):

    tectonic -X compile main.tex

Overleaf: upload `main.tex`, `refs.bib`, `figures/`; compiles with pdfLaTeX or XeLaTeX.

Submitting to a venue: replace the preamble with the venue's official style file and author
block; the body needs no changes beyond the venue's section limits.
