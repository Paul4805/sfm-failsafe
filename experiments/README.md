# experiments/

This directory is reserved for experiment notes and launch recipes. Generated
runs live under `outputs/` and are ignored by git.

Recommended real experiment sequence:

```bash
python scripts/run_baseline.py --config configs/courtyard.yaml \
  --run-name clean --condition clean --overwrite

python scripts/make_challenging_subsets.py --config configs/courtyard.yaml \
  --conditions blur,limited_views,degraded_matches \
  --input-database outputs/runs/clean/database.db

python scripts/run_baseline.py --images data/courtyard/subsets/blur/images \
  --run-name blur --condition blur
python scripts/run_baseline.py --images data/courtyard/subsets/limited_views/images \
  --run-name limited_views --condition limited_views
python scripts/run_baseline.py --images data/courtyard/images \
  --run-name degraded_matches --condition degraded_matches \
  --input-database data/courtyard/subsets/degraded_matches/database.db \
  --reuse-database

python scripts/train_failure_model.py --runs-root outputs/runs --mode pre \
  --out outputs/failure_model/pre
python scripts/run_adaptive.py --run-dir outputs/runs/clean \
  --out outputs/experiments/clean/adaptive
python scripts/make_report.py --config configs/courtyard.yaml
```

Synthetic validation is separate and must never be reported as real data:

```bash
python scripts/run_all_experiments.py --synthetic
```
