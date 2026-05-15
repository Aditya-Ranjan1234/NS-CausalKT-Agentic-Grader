# IEEE Paper Draft

This folder contains an IEEE-style LaTeX draft for the NS-CausalKT project. The draft has been expanded to target approximately 8 IEEE conference pages when compiled with the included figures and tables.

## Files

- `main.tex` - main IEEE conference paper source
- `references.bib` - BibTeX references
- `figures/` - copied project figures from `visualizations/` and `report/`

## Build

From this folder, run:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

`pdflatex` was not available in the current local shell, so the PDF was not generated here.
