# Thesis build.
# Requires pdflatex, bibtex and Inkscape (for the svg package).
#
# All intermediate files (.aux, .log, .toc, .bbl, the Inkscape cache, ...)
# are written to build/ so the source folder stays clean. The finished
# PDF is copied back to ./$(MAIN).pdf.

MAIN  := 00_main
BUILD := build
LATEX := pdflatex -shell-escape -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD)

SOURCES := $(MAIN).tex $(wildcard chapters/*.tex) $(wildcard tables/*.tex) \
           $(wildcard figures/*.svg) mainbib.bib

.PHONY: all quick clean distclean watch

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	@mkdir -p $(BUILD)
	$(LATEX) $(MAIN)
	cd $(BUILD) && BIBINPUTS=..: bibtex $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	@cp $(BUILD)/$(MAIN).pdf $(MAIN).pdf
	@echo "==> $(MAIN).pdf  ($$(pdfinfo $(MAIN).pdf 2>/dev/null | awk '/Pages/{print $$2}') pages)"

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
