PDF = main.pdf

all: $(PDF)

$(PDF): main.tex refs.bib metrics.tex tables.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp main.pdf engagegraph-report.pdf

metrics.tex tables.tex: ../../reports/results.json ../../artifacts/runs/benchmark.json ../../scripts/make_paper_tables.py
	cd ../.. && .venv/bin/python scripts/make_paper_tables.py

clean:
	latexmk -C
	rm -f metrics.tex tables.tex

.PHONY: all clean
