# Thesis build.
# Requires pdflatex, bibtex and Inkscape (for the svg package).
#
# All intermediate files (.aux, .log, .toc, .bbl, the Inkscape cache, ...)
# are written to build/ so the source folder stays clean. The finished
# PDF is copied back to ./$(MAIN).pdf.

MAIN  := 00_main
BUILD := build
# NOTE: no -halt-on-error. The svg package generates figure PDFs *during* the
# first pass, and cross-references resolve only on later passes, so transient
# first-pass errors are normal. nonstopmode lets the multi-pass cycle complete;
# real errors are reported by the `errors` target below.
LATEX := pdflatex -shell-escape -interaction=nonstopmode -output-directory=$(BUILD)

SOURCES := $(MAIN).tex $(wildcard chapters/*.tex) $(wildcard tables/*.tex) \
           $(wildcard figures/*.svg) mainbib.bib

.PHONY: all quick clean distclean watch errors

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	@mkdir -p $(BUILD)
	$(LATEX) $(MAIN)
	cd $(BUILD) && BIBINPUTS=..: bibtex $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	@cp $(BUILD)/$(MAIN).pdf $(MAIN).pdf
	@echo "==> $(MAIN).pdf  ($$(pdfinfo $(MAIN).pdf 2>/dev/null | awk '/Pages/{print $$2}') pages)"
	@grep -cE '^! ' $(BUILD)/$(MAIN).log | awk '{if ($$1>0) print "    "$$1" LaTeX error(s) - run: make errors"}' || true
	@grep -ci 'undefined' $(BUILD)/$(MAIN).log | awk '{if ($$1>0) print "    "$$1" undefined reference/citation warning(s)"}' || true

# Single pass: fast for prose edits. Cross-references may be stale.
quick:
	@mkdir -p $(BUILD)
	$(LATEX) $(MAIN)
	@cp $(BUILD)/$(MAIN).pdf $(MAIN).pdf

# Remove intermediates but keep the Inkscape cache so rebuilds stay fast.
clean:
	rm -f $(BUILD)/$(MAIN).aux $(BUILD)/$(MAIN).bbl $(BUILD)/$(MAIN).blg \
	      $(BUILD)/$(MAIN).log $(BUILD)/$(MAIN).out $(BUILD)/$(MAIN).toc \
	      $(BUILD)/$(MAIN).lof $(BUILD)/$(MAIN).lot $(BUILD)/$(MAIN).fls \
	      $(BUILD)/$(MAIN).fdb_latexmk $(BUILD)/$(MAIN).synctex.gz

# Remove the entire build directory and the output PDF.
distclean:
	rm -rf $(BUILD) $(MAIN).pdf

watch:
	latexmk -pdf -pvc -shell-escape -output-directory=$(BUILD) $(MAIN).tex

# Show the real errors and undefined references from the last build.
errors:
	@echo "--- LaTeX errors ---";        grep -E '^! ' -A3 $(BUILD)/$(MAIN).log || echo "  none"
	@echo "--- undefined refs/cites ---"; grep -i 'undefined' $(BUILD)/$(MAIN).log | sort -u || echo "  none"
