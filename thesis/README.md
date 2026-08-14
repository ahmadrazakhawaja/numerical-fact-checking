# Master's Thesis — Language-Invariant Reason Ranking in Numerical Fact-Checking

LaTeX source for the thesis *"Last in Translation? Evaluating Language-Invariant Reason Ranking in
Numerical Fact-Checking"* (University of Göttingen).

## Layout

```
thesis/
├── 00_main.tex        # main document: preamble, metadata, chapter includes
├── mainbib.bib        # single bibliography for the whole thesis
├── acmart.cls         # document class (local copy — not installed system-wide)
├── ACM-Reference-Format.bst
├── Makefile
├── chapters/
│   ├── titlepage.tex
│   ├── introduction.tex     (currently commented out in 00_main.tex)
│   ├── related_work.tex
│   ├── background.tex
│   ├── methodology.tex
│   ├── results.tex
│   └── appendix.tex         # full per-model result tables
├── tables/
│   ├── table-results-summary.tex       # condensed table, used in the body
│   ├── table-results-main.tex          # full per-model accuracy (appendix)
│   └── table-results-consistency.tex   # full per-model consistency (appendix)
└── figures/
    ├── *.svg                # all diagrams and plots
    └── UniGoettingen_Logo.pdf
```

`00_main.tex` sets `\graphicspath{{figures/}}` and `\svgpath{{figures/}}`, so chapter files
reference figures by bare name — e.g. `\includesvg[...]{results-tradeoff}`, not
`figures/results-tradeoff`.

## Building

```bash
make
```

That runs the full `pdflatex → bibtex → pdflatex → pdflatex` cycle and produces `00_main.pdf`.

**All intermediate files go to `build/`**, so the source folder is never littered with `.aux`,
`.log`, `.toc`, `.bbl` or the Inkscape SVG cache. The finished PDF is copied back to
`./00_main.pdf`. `build/` is git-ignored in full.

Manually, the equivalent is:

```bash
mkdir -p build && pdflatex -shell-escape -output-directory=build 00_main && (cd build && BIBINPUTS=..: bibtex 00_main) && pdflatex -shell-escape -output-directory=build 00_main && pdflatex -shell-escape -output-directory=build 00_main && cp build/00_main.pdf .
```

### Requirements

- A TeX distribution (TeX Live / MacTeX) with `pdflatex` and `bibtex`.
- **Inkscape**, required by the `svg` package to convert the `.svg` figures. Without it the build
  fails on the first `\includesvg`.
- **`-shell-escape` is mandatory**, because the `svg` package shells out to Inkscape.

### Other targets

```bash
make quick      # single pass — fast for prose edits (refs may be stale)
make clean      # clear intermediates, keep the Inkscape cache so rebuilds stay fast
make distclean  # delete build/ and 00_main.pdf entirely
make watch      # rebuild automatically on save (requires latexmk)
```

`make clean` deliberately keeps `build/svg-inkscape/`: re-converting all 16 SVGs through Inkscape is
the slowest part of the build. Use `make distclean` for a true from-scratch rebuild.

## Notes

- Everything the build generates (`.aux`, `.bbl`, `.log`, `svg-inkscape/`, …) is covered by
  `.gitignore`. Only source files are tracked.
- `00_main.pdf` **is** committed so the current draft is readable directly from the repository. To
  stop tracking it, uncomment the `00_main.pdf` line in `.gitignore`.
- The figures in `figures/results-*.svg` are generated from the experiment results; regenerating
  them is a separate step outside this folder.
