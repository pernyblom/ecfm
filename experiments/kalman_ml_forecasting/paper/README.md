# Kalman ML Forecasting Paper Draft

This folder contains a workshop-paper draft for the Kalman ML forecasting experiments.

## Files

- `main.tex`: paper entry point.
- `sections/01_introduction.tex`: motivation and contributions.
- `sections/02_related_work.tex`: placeholder related-work structure.
- `sections/03_problem_setup.tex`: dataset, notation, forecast protocol, and metrics.
- `sections/04_method.tex`: Kalman baseline, residual network, input branches, cutouts, and decorrelation.
- `sections/05_experiments.tex`: planned experiment sequence and result-table templates.
- `sections/06_discussion.tex`: interpretation structure for correlation and decorrelation results.
- `sections/07_conclusion.tex`: short conclusion placeholder.
- `references.bib`: starter bibliography placeholders.

## Compile

From this folder:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft intentionally uses TODO markers for numerical results, figure paths, workshop name, and final claims.
