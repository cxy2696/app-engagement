PDF = main.pdf

all: $(PDF)

$(PDF): main.tex refs.bib results.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp main.pdf engagegraph-report.pdf

results.tex: ../reports/results.json ../artifacts/runs/benchmark.json ../scripts/make_paper_tables.py
	cd .. && .venv/bin/python scripts/make_paper_tables.py

clean:
	latexmk -C
	rm -f results.tex

.PHONY: all clean
