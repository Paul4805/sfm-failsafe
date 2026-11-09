# outputs/

Generated experiment artifacts are written here (ignored by git by default).

- `runs/<run>/` — baseline COLMAP run, analysis, features, logs.
- `experiments/<run>/adaptive/` — baseline vs adaptive comparison.
- `failure_model/{pre,post}/` — real failure models and metrics.
- `failure_model_synthetic/{pre,post}/` — synthetic validation models.
- `experiments_synthetic/` — synthetic adaptive comparison.
- `REPORT.md` — detailed report generated from saved artifacts.
- `REPORT.pdf` — PDF report with embedded visualizations.
- `REPORT_SYNTHETIC.md` / `.pdf` — separate synthetic validation reports.

Large binaries (`database.db`, `sparse/`, refined models) stay here and are not
committed. Reports and small metric tables can be committed explicitly.
