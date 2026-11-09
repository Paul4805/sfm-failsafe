# SfM-FailSafe

[![tests](https://github.com/Paul4805/sfm-failsafe/actions/workflows/tests.yml/badge.svg)](https://github.com/Paul4805/sfm-failsafe/actions/workflows/tests.yml)

**Failure prediction and adaptive track weighting for sparse 3D reconstruction.**

This repository studies when COLMAP Structure-from-Motion (SfM) fails or becomes
unreliable, and whether reliability-weighted feature tracks improve an existing
sparse reconstruction. The target dataset is the **ETH3D High-Resolution
Multi-View** benchmark, initially the **Courtyard** scene (38 DSLR images).

COLMAP is the reconstruction backend. The repository does **not** replace
COLMAP with a toy SfM implementation. It adds:

1. A reproducible COLMAP baseline pipeline and run summary.
2. Track/image/pair analysis and leakage-free pre-reconstruction candidate
   tracks from the COLMAP database.
3. A failure-prediction model trained over reconstruction runs.
4. Reliability scoring and weighted post-mapper refinement of the same model.
5. Baseline-vs-adaptive comparison, plots, tests, and a generated report.

> **Scope warning.** Adaptive weighting is implemented as a **post-mapper
> refinement**. It is not integrated into COLMAP's incremental mapper. The
> scipy fallback fixes intrinsics, uses the first camera as gauge, and only
> optimizes tracks selected by track length. This limitation is intentional and
> documented, not hidden.

---

## 1. What is implemented

| Component | Status |
|---|---|
| COLMAP CLI feature extraction / matching / mapper | Implemented, used for real ETH3D runs |
| Largest sub-model selection + TXT model I/O | Implemented (pycolmap optional, not required) |
| Baseline summary, labels, command/timing logs | Implemented |
| Track/image/pair analysis | Implemented |
| Pre-reconstruction candidate tracks from DB | Implemented (match graph connected components) |
| Failure prediction (pre + post modes) | Implemented; pre mode is leakage-free |
| Reliability weights + scipy weighted BA | Implemented with analytic sparse Jacobian |
| Unselected-track re-triangulation after pose refinement | Implemented and reported |
| pyceres / Ceres backend | Not available here; scipy fallback used and reported |
| COLMAP-filter + `bundle_adjuster` approximation | Implemented (`--engine colmap_filtered`) |
| ETH3D ground-truth scan evaluation | **Not implemented**; internal SfM metrics only |
| Full mapper integration | **Not implemented**; post-mapper only |

## 1b. Results in this checkout (2026-11-09)

These are the actual artifacts generated in this checkout. They are **one scene**
(ETH3D Courtyard) and must not be generalized. The authoritative numbers and
artifact links are in [`outputs/REPORT.md`](outputs/REPORT.md), the PDF version
is [`outputs/REPORT.pdf`](outputs/REPORT.pdf), and synthetic validation is
separate in [`outputs/REPORT_SYNTHETIC.md`](outputs/REPORT_SYNTHETIC.md)
([PDF](outputs/REPORT_SYNTHETIC.pdf)).

Hardware/software: Windows 11, Python 3.12.10, RTX 3060 Laptop 6 GB,
COLMAP 3.9.1 CUDA, `SiftExtraction.max_image_size=2000`.

| Run | Input | Registered | Points | Mean reproj (px) | Label |
|---|---:|---:|---:|---:|---|
| clean | 38 | 38 | 20,445 | 0.7544 | success |
| blur (σ=3) | 12 | 9 | 2,060 | 0.6269 | success |
| blur_mild (σ=1.5) | 12 | 9 | 2,031 | 0.6370 | success |
| blur_severe (σ=6) | 12 | 9 | 3,284 | 0.5470 | success |
| limited_views (8) | 8 | 6 | 2,270 | 0.3820 | success |
| limited_views_severe (4) | 4 | 0 | 0 | n/a | failure (mapper) |
| degraded_matches (60% dropped) | 38 | 20 | 2,836 | 0.7130 | failure (ratio 0.53) |
| degraded_matches_severe (90%) | 38 | 2 | 128 | 0.3858 | failure (ratio/points) |

Adaptive comparison on the clean run (full 20,445 tracks, same initial model):

| Variant | Engine | Points | Mean reproj (px) | Median (px) | Runtime |
|---|---|---:|---:|---:|---:|
| baseline COLMAP | COLMAP mapper | 20,445 | 0.7544 | 0.6431 | 93.0 s |
| uniform BA (control) | scipy | 20,445 | 0.7454 | 0.6366 | 38.0 s |
| length-only weights | scipy | 20,445 | 0.7465 | 0.6375 | 29.5 s |
| full reliability weights | scipy | 20,445 | 0.7465 | 0.6371 | 39.7 s |

Interpretation: post-mapper refinement improves the full-model mean reprojection
error, but the **uniform control is marginally better than the weighted variants**
on this scene. This is reported as-is; no claim is made that reliability
weighting outperforms uniform BA. The per-refinement diagnostics use a robust
weighted objective and can show raw selected-observation RMSE not improving even
when the full-model mean improves.

Failure prediction (9 real runs, 5 success / 4 failure, **preliminary**):

| Mode | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Note |
|---|---|---:|---:|---:|---:|---:|---|
| pre (leakage-free) | logistic regression | 0.667 | 0.500 | 1.000 | 0.667 | 1.000 | tiny test set |
| post (diagnostic) | gradient boosting | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | uses mapper output; not leakage-free |

`python -m pytest -q` passes **50 tests** without ETH3D or COLMAP.

## 2. Repository layout

```
sfm-failsafe/
  README.md
  requirements.txt
  environment.yml
  pyproject.toml
  configs/
    default.yaml
    courtyard.yaml
  scripts/
    download_eth3d.sh
    prepare_courtyard.py
    make_challenging_subsets.py
    run_baseline.py
    analyze_tracks.py
    train_failure_model.py
    run_adaptive.py
    run_all_experiments.py
    make_report.py                 # detailed artifact-indexed Markdown report
    make_pdf_report.py             # PDF report with embedded plots
    generate_synthetic_dataset.py  # tests/demo only
    make_demo_runs.py              # clearly-labelled synthetic validation
  src/sfm_failsafe/
    config.py  colmap_runner.py  dataset.py  model_io.py
    database.py  track_analysis.py  features.py  reliability.py
    weighted_ba.py  adaptive_weighting.py  failure_model.py
    evaluation.py  visualization.py  report.py  synthetic.py  utils.py
  tests/
  experiments/      # experiment notes / optional launch scripts
  outputs/          # generated runs, models, metrics, plots, reports
```

## 3. Environment

Python 3.10+ (tested with 3.12). Core Python packages:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Conda:

```bash
conda env create -f environment.yml
conda activate sfm-failsafe
```

Optional:

- `pycolmap` — preferred binary model I/O if your Python environment allows it.
  The code falls back to COLMAP's CLI `model_converter` + TXT parser.
- `pyceres` — a native weighted-BA backend if a working build is available.
  This checkout uses the required scipy fallback and reports `engine_used`.

### COLMAP

Install COLMAP and make `colmap` available on `PATH`, or set:

```yaml
colmap:
  executable: /absolute/path/to/colmap
```

or the `COLMAP_EXECUTABLE` environment variable. Verify:

```bash
colmap -h
```

Portable Windows archives work: point `colmap.executable` at
`.../bin/colmap.exe`; the runner automatically adds the sibling `lib/` folder
to `PATH` so DLLs load without the batch wrapper.

> **Windows Smart App Control note.** On a machine with Smart App Control /
> enforced code integrity, unsigned new COLMAP releases and pycolmap DLLs may
> be blocked (`0xC0E90002`). This checkout used the older, widely distributed
> **COLMAP 3.9.1 Windows CUDA** build, which ran successfully. If COLMAP fails
> to start, check Windows Security → App & browser control → Smart App Control.
> Do not disable security features unless you understand and accept the risk.

## 4. Dataset

ETH3D Courtyard high-resolution training archive (~0.5 GB compressed):

```bash
bash scripts/download_eth3d.sh data/raw
python scripts/prepare_courtyard.py --extract
```

The download script uses:

```
https://www.eth3d.net/data/courtyard_dslr_undistorted.7z
```

`prepare_courtyard.py` supports common layouts through `dataset.image_glob`,
detects `dslr_calibration_undistorted/` if present, hardlinks images into
`data/courtyard/images`, and writes `manifest.json` + `manifest.csv`.
Use `--dry-run`, `--max-images`, and `--mode copy` as needed.

The ETH3D data is governed by the ETH3D terms of use: <https://www.eth3d.net/>.

## 5. Configs

`configs/default.yaml` holds portable defaults; `configs/courtyard.yaml` holds
the real ETH3D values (38 images, PINHOLE cameras, COLMAP path, all-point BA
cap). Relative paths are resolved against the repository root.

Important keys:

- `thresholds.*` — success/failure labels (registration ratio, point count,
  mean/median reprojection error, track length).
- `colmap.*` — executable, GPU, camera model, image size, mapper options.
- `features.*` — candidate-track construction.
- `failure_model.*` — split, model list, threshold.
- `weighting.*` — `w_min`, sigmoid/quantile mapping, feature weights and refs.
- `weighted_ba.*` — max points, iterations, robust loss, re-triangulation.
- `analysis.*` — blur/image sampling and track-angle sampling.

## 6. Exact commands

All scripts support `--dry-run` and print actionable errors if COLMAP/data are
missing.

### Baseline

```bash
python scripts/run_baseline.py \
  --config configs/courtyard.yaml \
  --run-name clean --condition clean --overwrite
```

Fast smoke test (subset + downscale):

```bash
python scripts/run_baseline.py --run-name smoke --max-images 6 --downscale 0.5 --overwrite
```

Outputs under `outputs/runs/<run-name>/`:

- `database.db`, `sparse/`, selected `model/`
- `reconstruction_summary.json`, `label.json`, `run_manifest.json`
- `tracks.csv`, `images.csv`, `pairs.csv`
- `candidate_tracks.csv`, `features_pre.json`
- `logs/commands.log`, `logs/timing.json`

### Analysis of an existing model

```bash
python scripts/analyze_tracks.py \
  --config configs/courtyard.yaml \
  --run-dir outputs/runs/clean
# or: --model path/to/sparse/0 --db database.db --images image_dir
```

### Challenging subsets

```bash
python scripts/make_challenging_subsets.py \
  --config configs/courtyard.yaml \
  --conditions clean,blur,limited_views,degraded_matches \
  --blur-sigma 3 --limit-count 8 --drop-fraction 0.6 \
  --input-database outputs/runs/clean/database.db
```

`degraded_matches` copies the database before dropping matches; the source is
never modified. Run it with:

```bash
python scripts/run_baseline.py --images data/courtyard/images \
  --run-name degraded_matches --condition degraded_matches \
  --input-database data/courtyard/subsets/degraded_matches/database.db \
  --reuse-database
```

### Failure prediction

```bash
python scripts/train_failure_model.py \
  --config configs/courtyard.yaml \
  --runs-root outputs/runs --mode pre \
  --out outputs/failure_model/pre
```

`--mode pre` is leakage-free. `--mode post` is diagnostic and explicitly marked
as not leakage-free. Outputs: dataset CSV, metrics JSON, `failure_model.joblib`,
split JSON, feature importances, ROC/PR/confusion/importance plots.

### Adaptive weighting

```bash
python scripts/run_adaptive.py \
  --config configs/courtyard.yaml \
  --run-dir outputs/runs/clean \
  --out outputs/experiments/clean/adaptive \
  --modes uniform,length,full
```

Outputs: `track_weights.csv`, `track_weights_length.csv`,
`track_weights_uniform.csv`, refined model directories, per-mode tracks and
summaries, `comparison.csv`, `comparison.md`, `adaptive_summary.json`, plots.

Approximate COLMAP-only path (clearly labelled, uses COLMAP BA after filtering):

```bash
python scripts/run_adaptive.py ... --engine colmap_filtered
```

### Full experiment + report

```bash
python scripts/run_all_experiments.py --config configs/courtyard.yaml
python scripts/make_report.py --config configs/courtyard.yaml
python scripts/make_pdf_report.py --config configs/courtyard.yaml
```

The PDF report embeds available run plots, failure-model diagnostics, and
adaptive comparison plots. It is generated directly from saved artifacts.

Synthetic validation (not real results):

```bash
python scripts/make_demo_runs.py --out outputs/demo_runs --seeds 0,1
python scripts/train_failure_model.py --runs-root outputs/demo_runs \
  --mode pre --out outputs/failure_model_synthetic/pre
python scripts/run_all_experiments.py --synthetic
```

The synthetic generator uses a sphere scene and a **simulated mapper**. Every
artifact is marked `data_kind: synthetic` / `engine: synthetic_simulated_mapper`.
Do not report these numbers as COLMAP or ETH3D results.

## 7. Method details

### Baseline labels

A run is a **failure** if any configured condition holds: mapper failure,
registration ratio below threshold, too few 3D points, mean/median reprojection
error above threshold, or median track length below threshold. See
`thresholds.*` and `evaluation.label_summary`.

### Leakage-free pre-reconstruction features

`features.candidate_tracks_from_database` builds a keypoint match graph from
verified two-view geometries (falling back to putative matches), unions
keypoints into connected components, and computes candidate track length,
image diversity, match support, inlier consistency, and keypoint/image spread.
It does **not** use 3D points or camera poses. Post-reconstruction features are
available for diagnostics but are not leakage-free.

### Reliability and weights

For each track:

- longer tracks, lower reprojection error, larger triangulation angle, wider
  image spread, higher verified-inlier consistency, and sharper source images
  increase reliability;
- component scores are normalized by configurable reference values;
- available components are renormalized per track;
- a sigmoid/quantile/linear map produces reliability in [0, 1], then
  `weight = w_min + (1 - w_min) * reliability`.

`length_only` is an ablation. Weights are saved per track.

### Weighted BA

The scipy backend minimizes `sum_i w_i * rho(||proj(X, R, t) - uv|| / sigma)`
with a robust loss, analytic sparse Jacobian, angle-axis/SE(3)-style local
updates, fixed intrinsics, and the first camera fixed as gauge. Distorted
observations are undistorted to pinhole pixels before optimization.

Because only selected tracks are jointly optimized, moving cameras can worsen
unselected tracks. The implementation therefore re-triangulates unselected
tracks with cameras fixed after the weighted stage, and reports
`retriangulation.num_updated`, `rmse_before`, and `rmse_after`. With
`courtyard.yaml`, `max_points: 25000` covers all ~20k Courtyard tracks; the
default `1500` is a faster, documented approximation.

**This is a post-mapper refinement, not a weighted incremental mapper.**

## 8. Tests

```bash
python -m pytest -q
```

Tests do not need ETH3D or COLMAP. They use a tiny generated sphere scene and a
COLMAP-compatible SQLite database written by the fixture generator. Coverage
includes config merging, TXT model round-trip, camera projection/undistortion,
track statistics, candidate tracks, reliability monotonicity, labels, metrics,
weighted BA + finite-difference Jacobian check, database corruption safety,
failure-model training on cached runs, PDF report generation, and CLI dry-runs.

## 9. Outputs and report

- `outputs/runs/<run>/` — one directory per reconstruction run.
- `outputs/experiments/<run>/adaptive/` — baseline vs adaptive artifacts.
- `outputs/failure_model/{pre,post}/` — real failure models.
- `outputs/failure_model_synthetic/{pre,post}/` — synthetic validation models.
- `outputs/experiments_synthetic/` — synthetic adaptive comparison.
- `outputs/REPORT.md` — detailed real-experiment report generated from saved
  artifacts.
- `outputs/REPORT.pdf` — PDF version with embedded tables and plots.
- `outputs/REPORT_SYNTHETIC.md` — separate synthetic validation report.
- `outputs/REPORT_SYNTHETIC.pdf` — PDF synthetic validation report.

A report is built **only from saved JSON/CSV files**; it does not recompute or
invent numbers. If a heavy experiment was not run, the report says so.

## 10. Known limitations

1. Adaptive weighting is post-mapper; COLMAP's incremental mapper does not
   consume the weights.
2. scipy weighted BA fixes intrinsics and gauge, optimizes a configurable subset
   of tracks, and does not add tracks.
3. Unselected points are re-triangulated after pose refinement; this changes the
   model but is reported explicitly.
4. pyceres/Ceres is not available in this checkout; no Ceres result is claimed.
5. pycolmap may be unavailable/blocked; binary models are converted with the
   COLMAP CLI and parsed as TXT.
6. Failure labels are threshold-based and depend on the number/diversity of
   runs. With few runs, prediction metrics are marked preliminary.
7. Only the Courtyard scene is used; cross-scene generalization is not tested.
8. ETH3D ground-truth accuracy/completeness is not evaluated; only internal SfM
   metrics and labels are reported.

## 11. Hardware / reproducibility notes

- A CUDA GPU is strongly recommended for SIFT on 38 high-resolution images.
  This checkout used an NVIDIA RTX 3060 6 GB with COLMAP 3.9.1 CUDA and
  `SiftExtraction.max_image_size=2000`; CPU-only COLMAP works but is slower.
- Seeds are fixed in configs and synthetic generation. COLMAP itself is
  deterministic given the same version/options/database, but mapper results can
  vary slightly across versions.
- Always keep the original ETH3D images and databases read-only. The runner
  copies/stages inputs; it never edits the source database in place.
- `--max-images` selects images evenly across the sorted list; use it for smoke
  tests, not for final numbers.

## 12. License

No open-source license has been chosen for this coursework project. The code and
results are published for review and reproducibility; contact the repository
owner before reuse or redistribution. ETH3D data remains subject to the ETH3D
terms of use.
