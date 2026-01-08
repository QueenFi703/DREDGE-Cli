# Makefile for µH-iOS Whitepaper
# Compiles LaTeX to PDF for academic submission

PAPER = uhios-whitepaper
TEXFILE = $(PAPER).tex
PDFFILE = $(PAPER).pdf

.PHONY: all clean view

all: $(PDFFILE)

$(PDFFILE): $(TEXFILE)
	@echo "Compiling LaTeX whitepaper..."
	pdflatex -interaction=nonstopmode $(TEXFILE)
	pdflatex -interaction=nonstopmode $(TEXFILE)  # Second pass for references
	@echo "PDF generated: $(PDFFILE)"

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz
	rm -f $(PDFFILE)

view: $(PDFFILE)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDFFILE); \
	elif command -v open > /dev/null; then \
		open $(PDFFILE); \
	else \
		echo "PDF viewer not found. Please open $(PDFFILE) manually."; \
	fi

help:
	@echo "µH-iOS Whitepaper Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Compile whitepaper to PDF"
	@echo "  make clean    - Remove generated files"
	@echo "  make view     - Open PDF in default viewer"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex (TeX Live or MacTeX)"
	@echo "  - Standard LaTeX packages (amsmath, tikz, listings, etc.)"
