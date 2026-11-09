# SfM-FailSafe experiment report

PDF version (with embedded plots): [REPORT.pdf](REPORT.pdf)

- Generated: 2026-11-10 00:20:59
- Config: `configs/courtyard.yaml`
- Runs root: `runs`
- Experiments root: `experiments`
- Python: 3.12.10 / Windows-11-10.0.26300-SP0
- Packages: numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1, opencv 5.0.0

## Executive summary

- Real COLMAP runs recorded: **9**
- Synthetic validation runs recorded: **16**
- Clean baseline: 38/38 cameras, 20445 points, mean reprojection error 0.7544 px, label `success`.
- Real labels: 5 success / 4 failure.
- Adaptive best variant by mean reprojection: `uniform` (0.7454 px).
- Baseline mean reprojection: 0.7544 px.
- Real pre-reconstruction failure model: `logistic_regression`, F1=0.667, ROC-AUC=1.000, preliminary=True.

## Data and model provenance

- Dataset manifest: `data/courtyard/manifest.json`
- Scene: courtyard
- Source images: 38
- Camera model: PINHOLE
- Calibration reference: `data/courtyard/calibration`
- COLMAP version: COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo
- Real run image staging: `data/courtyard/subsets/blur/images`

## Real experiment matrix

| run | condition | status | images | registered | ratio | points | mean_err_px | median_err_px | track_median | label | runtime_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur | blur | success | 12 | 9 | 0.7500 | 2060 | 0.6269 | 0.5391 | 3.0000 | success | 17.8339 |
| blur_mild | blur_mild | success | 12 | 9 | 0.7500 | 2031 | 0.6370 | 0.5466 | 3.0000 | success | 12.9992 |
| blur_severe | blur_severe | success | 12 | 9 | 0.7500 | 3284 | 0.5470 | 0.4237 | 3.0000 | success | 26.6491 |
| clean | clean | success | 38 | 38 | 1.0000 | 20445 | 0.7544 | 0.6431 | 4.0000 | success | 92.9708 |
| degraded_matches | degraded_matches | success | 38 | 20 | 0.5263 | 2836 | 0.7130 | 0.6198 | 3.0000 | failure | 152.0113 |
| degraded_matches_severe | degraded_matches_severe | success | 38 | 2 | 0.0526 | 128 | 0.3858 | 0.2319 | 2.0000 | failure | 7.9154 |
| limited_views | limited_views | success | 8 | 6 | 0.7500 | 2270 | 0.3820 | 0.2726 | 2.0000 | success | 13.0456 |
| limited_views_severe | limited_views_severe | failed | 4 | 0 | 0.0000 | 0 | n/a | n/a | n/a | failure | 1.9998 |
| smoke | smoke | success | 6 | 3 | 0.5000 | 544 | 0.1277 | 0.0735 | 2.0000 | failure | 4.3078 |


## Per-run details

### `blur`

- Condition: `blur`
- Data kind: `real`
- Engine: `colmap_cli`
- Status: `success`
- Registered: 9/12 (0.750)
- 3D points: 2060
- Reprojection error mean/median: 0.6269 / 0.5391 px
- Track length mean/median/p90: 3.28 / 3.00 / 4.00
- Success label: **success**
- COLMAP: COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo
- Input images: 12
- Timings: feature_extraction=1.96s, matching=1.05s, mapping=11.97s
- Total runtime: 17.8 s

| artifact | path | size |
| --- | --- | --- |
| database | runs/blur/database.db | 16.4 MB |
| model cameras | runs/blur/model/cameras.txt | 648.0 B |
| model images | runs/blur/model/images.txt | 2.4 MB |
| model points | runs/blur/model/points3D.txt | 197.3 KB |
| tracks.csv | runs/blur/tracks.csv | 707.7 KB |
| images.csv | runs/blur/images.csv | 1.8 KB |
| pairs.csv | runs/blur/pairs.csv | 4.4 KB |
| candidate_tracks.csv | runs/blur/candidate_tracks.csv | 670.9 KB |
| features_pre.json | runs/blur/features_pre.json | 1.9 KB |
| summary | runs/blur/reconstruction_summary.json | 1.9 KB |
| command log | runs/blur/logs/commands.log | 141.1 KB |


Plots: [reproj_error.png](runs/blur/plots/reproj_error.png), [sparse_pointcloud.png](runs/blur/plots/sparse_pointcloud.png), [track_length.png](runs/blur/plots/track_length.png)

### `blur_mild`

- Condition: `blur_mild`
- Data kind: `real`
- Engine: `colmap_cli`
- Status: `success`
- Registered: 9/12 (0.750)
- 3D points: 2031
- Reprojection error mean/median: 0.6370 / 0.5466 px
- Track length mean/median/p90: 3.32 / 3.00 / 4.00
- Success label: **success**
- COLMAP: COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo
- Input images: 12
- Timings: feature_extraction=2.20s, matching=1.18s, mapping=6.36s
- Total runtime: 13.0 s

| artifact | path | size |
| --- | --- | --- |
| database | runs/blur_mild/database.db | 17.3 MB |
| model cameras | runs/blur_mild/model/cameras.txt | 645.0 B |
| model images | runs/blur_mild/model/images.txt | 2.6 MB |
| model points | runs/blur_mild/model/points3D.txt | 195.1 KB |
| tracks.csv | runs/blur_mild/tracks.csv | 697.4 KB |
| images.csv | runs/blur_mild/images.csv | 1.8 KB |
| pairs.csv | runs/blur_mild/pairs.csv | 4.5 KB |
| candidate_tracks.csv | runs/blur_mild/candidate_tracks.csv | 668.9 KB |
| features_pre.json | runs/blur_mild/features_pre.json | 1.9 KB |
| summary | runs/blur_mild/reconstruction_summary.json | 1.9 KB |
| command log | runs/blur_mild/logs/commands.log | 107.4 KB |


Plots: [reproj_error.png](runs/blur_mild/plots/reproj_error.png), [sparse_pointcloud.png](runs/blur_mild/plots/sparse_pointcloud.png), [track_length.png](runs/blur_mild/plots/track_length.png)

### `blur_severe`

- Condition: `blur_severe`
- Data kind: `real`
- Engine: `colmap_cli`
- Status: `success`
- Registered: 9/12 (0.750)
- 3D points: 3284
- Reprojection error mean/median: 0.5470 / 0.4237 px
- Track length mean/median/p90: 3.12 / 3.00 / 4.00
- Success label: **success**
- COLMAP: COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo
- Input images: 12
- Timings: feature_extraction=1.96s, matching=1.09s, mapping=20.66s
- Total runtime: 26.6 s

| artifact | path | size |
| --- | --- | --- |
| database | runs/blur_severe/database.db | 14.3 MB |
| model cameras | runs/blur_severe/model/cameras.txt | 649.0 B |
| model images | runs/blur_severe/model/images.txt | 2.3 MB |
| model points | runs/blur_severe/model/points3D.txt | 309.1 KB |
| tracks.csv | runs/blur_severe/tracks.csv | 1.1 MB |
| images.csv | runs/blur_severe/images.csv | 1.8 KB |
| pairs.csv | runs/blur_severe/pairs.csv | 4.5 KB |
| candidate_tracks.csv | runs/blur_severe/candidate_tracks.csv | 744.8 KB |
| features_pre.json | runs/blur_severe/features_pre.json | 1.9 KB |
| summary | runs/blur_severe/reconstruction_summary.json | 1.9 KB |
| command log | runs/blur_severe/logs/commands.log | 236.2 KB |


Plots: [reproj_error.png](runs/blur_severe/plots/reproj_error.png), [sparse_pointcloud.png](runs/blur_severe/plots/sparse_pointcloud.png), [track_length.png](runs/blur_severe/plots/track_length.png)

### `clean`

- Condition: `clean`
- Data kind: `real`
- Engine: `colmap_cli`
- Status: `success`
- Registered: 38/38 (1.000)
- 3D points: 20445
- Reprojection error mean/median: 0.7544 / 0.6431 px
- Track length mean/median/p90: 4.48 / 4.00 / 8.00
- Success label: **success**
- COLMAP: COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo
- Input images: 38
- Timings: feature_extraction=9.79s, matching=9.03s, mapping=52.43s
- Total runtime: 93.0 s

| artifact | path | size |
| --- | --- | --- |
| database | runs/clean/database.db | 61.5 MB |
| model cameras | runs/clean/model/cameras.txt | 2.4 KB |
| model images | runs/clean/model/images.txt | 11.8 MB |
| model points | runs/clean/model/points3D.txt | 2.1 MB |
| tracks.csv | runs/clean/tracks.csv | 7.0 MB |
| images.csv | runs/clean/images.csv | 6.5 KB |
| pairs.csv | runs/clean/pairs.csv | 55.1 KB |
| candidate_tracks.csv | runs/clean/candidate_tracks.csv | 4.1 MB |
| features_pre.json | runs/clean/features_pre.json | 1.9 KB |
| summary | runs/clean/reconstruction_summary.json | 1.9 KB |
| command log | runs/clean/logs/commands.log | 205.8 KB |


Plots: [reproj_error.png](runs/clean/plots/reproj_error.png), [sparse_pointcloud.png](runs/clean/plots/sparse_pointcloud.png), [track_length.png](runs/clean/plots/track_length.png)

### `degraded_matches`

- Condition: `degraded_matches`
- Data kind: `real`
- Engine: `colmap_cli`
- Status: `success`
- Registered: 20/38 (0.526)
- 3D points: 2836
- Reprojection error mean/median: 0.7130 / 0.6198 px
- Track length mean/median/p90: 4.25 / 3.00 / 8.00
- Success label: **failure**
- Failure reasons: registration_ratio 0.526 < 0.600
- COLMAP: COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo
- Input images: 38
- Timings: feature_extraction=0.00s, matching=0.00s, mapping=141.69s
- Total runtime: 152.0 s

| artifact | path | size |
| --- | --- | --- |
| database | runs/degraded_matches/database.db | 61.5 MB |
| model cameras | runs/degraded_matches/model/cameras.txt | 1.3 KB |
| model images | runs/degraded_matches/model/images.txt | 6.2 MB |
| model points | runs/degraded_matches/model/points3D.txt | 294.0 KB |
| tracks.csv | runs/degraded_matches/tracks.csv | 975.0 KB |
| images.csv | runs/degraded_matches/images.csv | 4.5 KB |
| pairs.csv | runs/degraded_matches/pairs.csv | 24.3 KB |
| candidate_tracks.csv | runs/degraded_matches/candidate_tracks.csv | 1.6 MB |
| features_pre.json | runs/degraded_matches/features_pre.json | 1.9 KB |
| summary | runs/degraded_matches/reconstruction_summary.json | 2.0 KB |
| command log | runs/degraded_matches/logs/commands.log | 1.1 MB |


Plots: [reproj_error.png](runs/degraded_matches/plots/reproj_error.png), [sparse_pointcloud.png](runs/degraded_matches/plots/sparse_pointcloud.png), [track_length.png](runs/degraded_matches/plots/track_length.png)

### `degraded_matches_severe`

- Condition: `degraded_matches_severe`
- Data kind: `real`
- Engine: `colmap_cli`
- Status: `success`
- Registered: 2/38 (0.053)
- 3D points: 128
- Reprojection error mean/median: 0.3858 / 0.2319 px
- Track length mean/median/p90: 2.00 / 2.00 / 2.00
- Success label: **failure**
- Failure reasons: registration_ratio 0.053 < 0.600; num_points3D 128 < 1000
- COLMAP: COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo
- Input images: 38
- Timings: feature_extraction=0.00s, matching=0.00s, mapping=0.33s
- Total runtime: 7.9 s

| artifact | path | size |
| --- | --- | --- |
| database | runs/degraded_matches_severe/database.db | 61.5 MB |
| model cameras | runs/degraded_matches_severe/model/cameras.txt | 220.0 B |
| model images | runs/degraded_matches_severe/model/images.txt | 687.1 KB |
| model points | runs/degraded_matches_severe/model/points3D.txt | 11.3 KB |
| tracks.csv | runs/degraded_matches_severe/tracks.csv | 44.9 KB |
| images.csv | runs/degraded_matches_severe/images.csv | 2.6 KB |
| pairs.csv | runs/degraded_matches_severe/pairs.csv | 1.9 KB |
| candidate_tracks.csv | runs/degraded_matches_severe/candidate_tracks.csv | 113.5 KB |
| features_pre.json | runs/degraded_matches_severe/features_pre.json | 1.9 KB |
| summary | runs/degraded_matches_severe/reconstruction_summary.json | 2.0 KB |
| command log | runs/degraded_matches_severe/logs/commands.log | 5.4 KB |


Plots: [reproj_error.png](runs/degraded_matches_severe/plots/reproj_error.png), [sparse_pointcloud.png](runs/degraded_matches_severe/plots/sparse_pointcloud.png), [track_length.png](runs/degraded_matches_severe/plots/track_length.png)

### `limited_views`

- Condition: `limited_views`
- Data kind: `real`
- Engine: `colmap_cli`
- Status: `success`
- Registered: 6/8 (0.750)
- 3D points: 2270
- Reprojection error mean/median: 0.3820 / 0.2726 px
- Track length mean/median/p90: 2.50 / 2.00 / 4.00
- Success label: **success**
- COLMAP: COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo
- Input images: 8
- Timings: feature_extraction=2.11s, matching=0.70s, mapping=6.70s
- Total runtime: 13.0 s

| artifact | path | size |
| --- | --- | --- |
| database | runs/limited_views/database.db | 12.3 MB |
| model cameras | runs/limited_views/model/cameras.txt | 458.0 B |
| model images | runs/limited_views/model/images.txt | 1.9 MB |
| model points | runs/limited_views/model/points3D.txt | 204.7 KB |
| tracks.csv | runs/limited_views/tracks.csv | 783.8 KB |
| images.csv | runs/limited_views/images.csv | 1.3 KB |
| pairs.csv | runs/limited_views/pairs.csv | 2.0 KB |
| candidate_tracks.csv | runs/limited_views/candidate_tracks.csv | 363.5 KB |
| features_pre.json | runs/limited_views/features_pre.json | 1.9 KB |
| summary | runs/limited_views/reconstruction_summary.json | 1.9 KB |
| command log | runs/limited_views/logs/commands.log | 96.9 KB |


Plots: [reproj_error.png](runs/limited_views/plots/reproj_error.png), [sparse_pointcloud.png](runs/limited_views/plots/sparse_pointcloud.png), [track_length.png](runs/limited_views/plots/track_length.png)

### `limited_views_severe`

- Condition: `limited_views_severe`
- Data kind: `real`
- Engine: `colmap_cli`
- Status: `failed`
- Registered: 0/4 (0.000)
- 3D points: 0
- Reprojection error mean/median: n/a / n/a px
- Track length mean/median/p90: n/a / n/a / n/a
- Success label: **failure**
- Failure reasons: mapper_failed; registration_ratio 0.000 < 0.600; num_points3D 0 < 1000
- COLMAP: COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo
- Input images: 4
- Timings: feature_extraction=1.12s, matching=0.45s
- Total runtime: 2.0 s
- Error: `COLMAP mapper failed with exit code 1.
Command: "tools\colmap391cuda\COLMAP-3.9.1-windows-cuda\bin\colmap.exe" mapper --database_path "outputs\runs\limited_views_severe\database.db" --image_path "data\courtyard\subsets\limited_views_severe\images" --output_path "outputs\runs\limited_views_severe\sparse" --Mapper.min_num_matches 15 --Mapper.ba_global_images_ratio 1.1 --Mapper.ba_global_points_ratio 1.1 --Mapper.ba_local_max_refinements 3 --Mapper.max_num_models 10
Log: outputs\runs\limited_views_severe\logs\commands.log
stdout:

stderr:
I20261109 23:47:50.559350 10320 misc.cc:198] 
==============================================================================
Loading database
==============================================================================
I20261109 23:47:50.561344 10320 database_cache.cc:54] Loading cameras...
I20261109 23:47:50.561344 10320 database_cache.cc:64]  4 in 0.000s
I20261109 23:47:50.561344 10320 database_cache.cc:72] Loading matches...
I20261109 23:47:50.562345 10320 database_cache.cc:78]  6 in 0.000s
I20261109 23:47:50.562345 10320 database_cache.cc:94] Loading images...
I20261109 23:47:50.564345 10320 database_cache.cc:143]  4 in 0.002s (connected 4)
I20261109 23:47:50.564345 10320 database_cache.cc:154] Building correspondence graph...
I20261109 23:47:50.565356 10320 database_cache.cc:190]  in 0.001s (ignored 0)
I20261109 23:47:50.565356 10320 timer.cc:91] Elapsed time: 0.000 [minutes]
I20261109 23:47:50.566351 10320 misc.cc:198] 
==============================================================================
Finding good initial image pair
==============================================================================
I20261109 23:47:50.603359 10320 incremental_mapper.cc:404] => No good initial image pair found.
I20261109 23:47:50.603359 10320 incremental_mapper.cc:310] => Relaxing the initialization constraints.
I20261109 23:47:50.604387 10320 misc.cc:198] 
==============================================================================
Finding good initial image pair
==============================================================================
I20261109 23:47:50.636423 10320 incremental_mapper.cc:404] => No good initial image pair found.
I20261109 23:47:50.638010 10320 incremental_mapper.cc:318] => Relaxing the initialization constraints.
I20261109 23:47:50.638010 10320 misc.cc:198] 
==============================================================================
Finding good initial image pair
==============================================================================
I20261109 23:47:50.673544 10320 incremental_mapper.cc:404] => No good initial image pair found.
I20261109 23:47:50.674531 10320 incremental_mapper.cc:310] => Relaxing the initialization constraints.
I20261109 23:47:50.674531 10320 misc.cc:198] 
==============================================================================
Finding good initial image pair
==============================================================================
I20261109 23:47:50.779067 10320 incremental_mapper.cc:404] => No good initial image pair found.
I20261109 23:47:50.779067 10320 incremental_mapper.cc:318] => Relaxing the initialization constraints.
I20261109 23:47:50.780063 10320 misc.cc:198] 
==============================================================================
Finding good initial image pair
==============================================================================
I20261109 23:47:50.884704 10320 incremental_mapper.cc:404] => No good initial image pair found.
I20261109 23:47:50.885702 10320 timer.cc:91] Elapsed time: 0.005 [minutes]
E20261109 23:47:50.885702 24980 sfm.cc:266] failed to create sparse model
`

| artifact | path | size |
| --- | --- | --- |
| database | runs/limited_views_severe/database.db | 6.4 MB |
| model cameras | - | - |
| model images | - | - |
| model points | - | - |
| tracks.csv | - | - |
| images.csv | - | - |
| pairs.csv | - | - |
| candidate_tracks.csv | runs/limited_views_severe/candidate_tracks.csv | 238.7 KB |
| features_pre.json | runs/limited_views_severe/features_pre.json | 1.9 KB |
| summary | runs/limited_views_severe/reconstruction_summary.json | 4.9 KB |
| command log | runs/limited_views_severe/logs/commands.log | 7.8 KB |


### `smoke`

- Condition: `smoke`
- Data kind: `real`
- Engine: `colmap_cli`
- Status: `success`
- Registered: 3/6 (0.500)
- 3D points: 544
- Reprojection error mean/median: 0.1277 / 0.0735 px
- Track length mean/median/p90: 2.11 / 2.00 / 3.00
- Success label: **failure**
- Failure reasons: registration_ratio 0.500 < 0.600; num_points3D 544 < 1000
- COLMAP: COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo
- Input images: 6
- Timings: feature_extraction=0.84s, matching=0.76s, mapping=0.60s
- Total runtime: 4.3 s

| artifact | path | size |
| --- | --- | --- |
| database | runs/smoke/database.db | 8.2 MB |
| model cameras | runs/smoke/model/cameras.txt | 274.0 B |
| model images | runs/smoke/model/images.txt | 772.1 KB |
| model points | runs/smoke/model/points3D.txt | 48.4 KB |
| tracks.csv | runs/smoke/tracks.csv | 189.4 KB |
| images.csv | runs/smoke/images.csv | 857.0 B |
| pairs.csv | runs/smoke/pairs.csv | 844.0 B |
| candidate_tracks.csv | runs/smoke/candidate_tracks.csv | 85.0 KB |
| features_pre.json | runs/smoke/features_pre.json | 1.9 KB |
| summary | runs/smoke/reconstruction_summary.json | 1.9 KB |
| command log | runs/smoke/logs/commands.log | 34.2 KB |


Plots: [reproj_error.png](runs/smoke/plots/reproj_error.png), [sparse_pointcloud.png](runs/smoke/plots/sparse_pointcloud.png), [track_length.png](runs/smoke/plots/track_length.png)

## Failure prediction

### Pre-reconstruction model (primary, leakage-free)

- Mode: `pre`
- Leakage-free: `True`
- Runs: 9 (classes {'0': 5, '1': 4})
- Features: 38
- Split: `stratified_holdout` (train 6, test 3)
- Best model: `logistic_regression`
- Threshold (tuned on train): 0.5000
- Preliminary: `True`
- Preliminary reason: Small number of runs / conditions; treat metrics as pipeline validation, not a scientific result.

| model | accuracy | precision | recall | f1 | roc_auc | avg_precision | threshold | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.6667 | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 0.5000 | 1 | 1 | 1 | 0 |
| random_forest | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0 | 1 | 1 | 1 |
| gradient_boosting | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 0.5000 | 0 | 0 | 2 | 1 |


Leave-one-condition-out (random forest):

| n | threshold | accuracy | precision | recall | f1 | specificity | roc_auc | average_precision | tp | tn | fp | fn | predicted_positive | actual_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9.0000 | 0.5000 | 0.6667 | 0.6667 | 0.5000 | 0.5714 | 0.8000 | 0.7500 | 0.7470 | 2.0000 | 4.0000 | 1.0000 | 2.0000 | 3.0000 | 4.0000 |


Top features:

| feature | importance |
| --- | --- |
| candidate_mean_image_diversity | 0.0612 |
| pair_inlier_mean | 0.0503 |
| blur_mean | 0.0498 |
| candidate_mean_inlier_consistency | 0.0493 |
| blur_median | 0.0483 |
| keypoints_per_image | 0.0473 |
| blur_p10 | 0.0473 |
| candidate_edges_per_node | 0.0452 |
| candidate_mean_kp_spread_norm | 0.0441 |
| num_keypoints | 0.0403 |
| num_cameras | 0.0380 |
| num_input_images | 0.0380 |
| candidate_frac_len_ge3 | 0.0344 |
| pair_inlier_median | 0.0342 |
| candidate_track_p90_len | 0.0323 |


Plots: [class_balance.png](failure_model/pre/plots/class_balance.png), [confusion_matrix.png](failure_model/pre/plots/confusion_matrix.png), [feature_importance.png](failure_model/pre/plots/feature_importance.png), [roc_pr.png](failure_model/pre/plots/roc_pr.png), [roc_pr_leave_one_condition_out.png](failure_model/pre/plots/roc_pr_leave_one_condition_out.png)

- Dataset: [runs_dataset_pre.csv](failure_model/pre/runs_dataset_pre.csv)
- Model: [failure_model.joblib](failure_model/pre/failure_model.joblib)

### Post-reconstruction diagnostic model (NOT leakage-free)

- Mode: `post`
- Leakage-free: `False`
- Runs: 9 (classes {'0': 5, '1': 4})
- Features: 23
- Split: `stratified_holdout` (train 6, test 3)
- Best model: `gradient_boosting`
- Threshold (tuned on train): 0.5000
- Preliminary: `True`
- Preliminary reason: Small number of runs / conditions; treat metrics as pipeline validation, not a scientific result.

| model | accuracy | precision | recall | f1 | roc_auc | avg_precision | threshold | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 0.5000 | 0 | 2 | 0 | 1 |
| random_forest | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0 | 1 | 1 | 1 |
| gradient_boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1 | 2 | 0 | 0 |


Leave-one-condition-out (random forest):

| n | threshold | accuracy | precision | recall | f1 | specificity | roc_auc | average_precision | tp | tn | fp | fn | predicted_positive | actual_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9.0000 | 0.5000 | 0.5556 | 0.5000 | 0.2500 | 0.3333 | 0.8000 | 0.6500 | 0.7153 | 1.0000 | 4.0000 | 1.0000 | 3.0000 | 2.0000 | 4.0000 |


Top features:

| feature | importance |
| --- | --- |
| registration_ratio | 1.0000 |
| num_input_images | 0.0000 |
| num_registered_images | 0.0000 |
| num_points3D | 0.0000 |
| num_observations | 0.0000 |
| mean_reproj_error | 0.0000 |
| median_reproj_error | 0.0000 |
| p95_reproj_error | 0.0000 |
| track_length_mean | 0.0000 |
| track_length_median | 0.0000 |
| track_length_p90 | 0.0000 |
| num_tracks_ge3 | 0.0000 |
| frac_tracks_ge3 | 0.0000 |
| mean_observations_per_registered_image | 0.0000 |
| num_covis_pairs | 0.0000 |


Plots: [class_balance.png](failure_model/post/plots/class_balance.png), [confusion_matrix.png](failure_model/post/plots/confusion_matrix.png), [feature_importance.png](failure_model/post/plots/feature_importance.png), [roc_pr.png](failure_model/post/plots/roc_pr.png), [roc_pr_leave_one_condition_out.png](failure_model/post/plots/roc_pr_leave_one_condition_out.png)

- Dataset: [runs_dataset_post.csv](failure_model/post/runs_dataset_post.csv)
- Model: [failure_model.joblib](failure_model/post/failure_model.joblib)

Failure labels are threshold-based reconstruction outcomes. With few runs, metrics validate the pipeline but are not a scientific result.

Threshold-dependent metrics (accuracy/precision/recall/F1) can be unstable when the held-out set is tiny or contains a single class; ROC-AUC and average precision are threshold-free and should be preferred in that case.

## Baseline vs adaptive track weighting

- Baseline model: `runs/clean/model`
- Engine requested: `auto`
- Modes: uniform, length, full
- Total runtime: 132.9 s

| variant | engine | num_registered_images | num_points3D | mean_reproj_error | median_reproj_error | track_length_median | weight_mean | success_label | runtime_seconds | delta_mean_reproj_error | delta_num_points3D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | colmap_baseline | 38 | 20445 | 0.7544 | 0.6431 | 4.0000 | 0.8544 | 0 | 92.9708 | 0.0000 | 0.0000 |
| uniform | scipy_least_squares | 38 | 20445 | 0.7454 | 0.6366 | 4.0000 | 1.0000 | 0 | 39.0343 | -0.0090 | 0.0000 |
| length | scipy_least_squares | 38 | 20445 | 0.7465 | 0.6375 | 4.0000 | 0.8431 | 0 | 31.0078 | -0.0079 | 0.0000 |
| full | scipy_least_squares | 38 | 20445 | 0.7465 | 0.6371 | 4.0000 | 0.8544 | 0 | 38.2791 | -0.0079 | 0.0000 |


Weight distributions:

| mode | count | weight_mean | weight_std | weight_min | weight_p10 | weight_median | weight_max | reliability_mean | fraction_below_0_1 | fraction_above_0_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 20445 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| length | 20445 | 0.8431 | 0.0982 | 0.6159 | 0.7924 | 0.8830 | 0.9549 | 0.8348 | 0.0000 | 0.3286 |
| full | 20445 | 0.8544 | 0.0834 | 0.2587 | 0.7572 | 0.8819 | 0.9487 | 0.8467 | 0.0000 | 0.3189 |


Refinement diagnostics:

| mode | status | engine | selected_points | observations | rmse_before | rmse_after | improved | retri_points | runtime_s | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | ok | scipy_least_squares | 20445 | 91619 | 1.0041 | 1.0071 | False | None | 39.0343 |  |
| length | ok | scipy_least_squares | 20445 | 91619 | 1.0041 | 1.0066 | False | None | 31.0078 |  |
| full | ok | scipy_least_squares | 20445 | 91619 | 1.0041 | 1.0062 | False | None | 38.2791 |  |


Plots: [comparison_bars.png](experiments/clean/adaptive/plots/comparison_bars.png), [reliability_scatter.png](experiments/clean/adaptive/plots/reliability_scatter.png), [reproj_error_baseline.png](experiments/clean/adaptive/plots/reproj_error_baseline.png), [track_length_baseline.png](experiments/clean/adaptive/plots/track_length_baseline.png), [weight_distribution.png](experiments/clean/adaptive/plots/weight_distribution.png)

- Comparison table: [comparison.csv](experiments/clean/adaptive/comparison.csv)
- Track weights: [track_weights.csv](experiments/clean/adaptive/track_weights.csv)
- Adaptive summary: [adaptive_summary.json](experiments/clean/adaptive/adaptive_summary.json)
- Refined models: uniform: `experiments/clean/adaptive/models/uniform`, length: `experiments/clean/adaptive/models/length`, full: `experiments/clean/adaptive/models/full`

- Baseline is the unmodified COLMAP sparse model.
- Adaptive variants are post-mapper refinements from the same initial model; they are not integrated into COLMAP's incremental mapper.
- Uniform refinement is a control: same BA, all weights = 1.
- Length-only is an ablation of the full reliability score.
- Selected tracks receive the weighted robust BA; unselected tracks are re-triangulated against the refined cameras when the point cap excludes any.
- errors_before/after in refinement diagnostics cover selected observations with the robust objective; the comparison table uses full-model track mean/median reprojection error.
- data_kind=real

## Synthetic pipeline validation

> These runs use a generated sphere scene and a **simulated mapper**. They validate code paths only and are not COLMAP/ETH3D results.

| run_name | run_dir | condition | status | label | label_name | reasons | num_input_images | num_registered_images | num_points3D | mean_reproj_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_s0 | demo_runs/clean_s0 | clean | success | 0 | success | [] | 12 | 12 | 2672 | 0.6598 |
| clean_s1 | demo_runs/clean_s1 | clean | success | 0 | success | [] | 12 | 12 | 2660 | 0.6470 |
| blur_mild_s0 | demo_runs/blur_mild_s0 | blur_mild | success | 0 | success | [] | 12 | 12 | 2672 | 1.1547 |
| blur_mild_s1 | demo_runs/blur_mild_s1 | blur_mild | success | 0 | success | [] | 12 | 12 | 2660 | 1.1323 |
| blur_severe_s0 | demo_runs/blur_severe_s0 | blur_severe | success | 1 | failure | ['mean_reproj_error 2.309 > 2.000', 'median_reproj_error 1.790 > 1.500'] | 12 | 12 | 2672 | 2.3095 |
| blur_severe_s1 | demo_runs/blur_severe_s1 | blur_severe | success | 1 | failure | ['mean_reproj_error 2.265 > 2.000', 'median_reproj_error 1.792 > 1.500'] | 12 | 12 | 2660 | 2.2646 |
| limited_views_s0 | demo_runs/limited_views_s0 | limited_views | success | 0 | success | [] | 6 | 6 | 2439 | 0.9902 |
| limited_views_s1 | demo_runs/limited_views_s1 | limited_views | success | 0 | success | [] | 6 | 6 | 2452 | 0.9639 |
| limited_views_severe_s0 | demo_runs/limited_views_severe_s0 | limited_views_severe | success | 1 | failure | ['registration_ratio 0.500 < 0.600', 'num_points3D 302 < 1000'] | 4 | 2 | 302 | 1.6820 |
| limited_views_severe_s1 | demo_runs/limited_views_severe_s1 | limited_views_severe | success | 0 | success | [] | 4 | 4 | 1200 | 1.6507 |
| degraded_matches_mild_s0 | demo_runs/degraded_matches_mild_s0 | degraded_matches_mild | success | 0 | success | [] | 12 | 12 | 2672 | 1.1547 |
| degraded_matches_mild_s1 | demo_runs/degraded_matches_mild_s1 | degraded_matches_mild | success | 0 | success | [] | 12 | 12 | 2660 | 1.1323 |
| degraded_matches_severe_s0 | demo_runs/degraded_matches_severe_s0 | degraded_matches_severe | success | 1 | failure | ['registration_ratio 0.167 < 0.600', 'num_points3D 737 < 1000', 'mean_reproj_error 2.726 > 2.000', 'median_reproj_error 1.885 > 1.500'] | 12 | 2 | 737 | 2.7263 |
| degraded_matches_severe_s1 | demo_runs/degraded_matches_severe_s1 | degraded_matches_severe | success | 1 | failure | ['registration_ratio 0.083 < 0.600', 'num_points3D 0 < 1000'] | 12 | 1 | 0 | n/a |
| combined_severe_s0 | demo_runs/combined_severe_s0 | combined_severe | success | 1 | failure | ['registration_ratio 0.333 < 0.600', 'num_points3D 512 < 1000', 'median_reproj_error 1.533 > 1.500'] | 6 | 2 | 512 | 1.8956 |
| combined_severe_s1 | demo_runs/combined_severe_s1 | combined_severe | success | 1 | failure | ['registration_ratio 0.333 < 0.600', 'num_points3D 518 < 1000'] | 6 | 2 | 518 | 1.9477 |


### Synthetic pre-reconstruction model

- Mode: `pre`
- Leakage-free: `True`
- Runs: 16 (classes {'0': 9, '1': 7})
- Features: 38
- Split: `group_held_out` (train 10, test 6)
- Best model: `gradient_boosting`
- Threshold (tuned on train): 0.5000
- Preliminary: `False`

| model | accuracy | precision | recall | f1 | roc_auc | avg_precision | threshold | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.6250 | 0.5000 | 0.5000 | 0 | 4 | 0 | 2 |
| random_forest | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4167 | 0.5000 | 0 | 4 | 0 | 2 |
| gradient_boosting | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.7500 | 0.6667 | 0.5000 | 0 | 4 | 0 | 2 |


Leave-one-condition-out (random forest):

| n | threshold | accuracy | precision | recall | f1 | specificity | roc_auc | average_precision | tp | tn | fp | fn | predicted_positive | actual_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16.0000 | 0.5000 | 0.7500 | 0.7143 | 0.7143 | 0.7143 | 0.7778 | 0.7460 | 0.7993 | 5.0000 | 7.0000 | 2.0000 | 2.0000 | 7.0000 | 7.0000 |


Top features:

| feature | importance |
| --- | --- |
| candidate_nodes_per_image | 0.2982 |
| candidate_mean_image_span_norm | 0.2839 |
| candidate_largest_component_frac | 0.2314 |
| pair_inlier_p10 | 0.1866 |
| num_input_images | 0.0000 |
| num_cameras | 0.0000 |
| candidate_pair_count_used | 0.0000 |
| candidate_total_edges | 0.0000 |
| candidate_total_components | 0.0000 |
| candidate_total_nodes | 0.0000 |
| candidate_track_p90_len | 0.0000 |
| candidate_track_count | 0.0000 |
| candidate_track_mean_len | 0.0000 |
| candidate_pairs_truncated | 0.0000 |
| candidate_frac_len_ge5 | 0.0000 |


Plots: [class_balance.png](failure_model_synthetic/pre/plots/class_balance.png), [confusion_matrix.png](failure_model_synthetic/pre/plots/confusion_matrix.png), [feature_importance.png](failure_model_synthetic/pre/plots/feature_importance.png), [roc_pr.png](failure_model_synthetic/pre/plots/roc_pr.png), [roc_pr_leave_one_condition_out.png](failure_model_synthetic/pre/plots/roc_pr_leave_one_condition_out.png)

- Dataset: [runs_dataset_pre.csv](failure_model_synthetic/pre/runs_dataset_pre.csv)
- Model: [failure_model.joblib](failure_model_synthetic/pre/failure_model.joblib)

### Synthetic post-reconstruction diagnostic model

- Mode: `post`
- Leakage-free: `False`
- Runs: 16 (classes {'0': 9, '1': 7})
- Features: 23
- Split: `group_held_out` (train 10, test 6)
- Best model: `logistic_regression`
- Threshold (tuned on train): 0.5000
- Preliminary: `False`

| model | accuracy | precision | recall | f1 | roc_auc | avg_precision | threshold | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | 0 | 4 | 0 | 2 |
| random_forest | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | 0 | 4 | 0 | 2 |
| gradient_boosting | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3333 | 0.5000 | 0 | 4 | 0 | 2 |


Leave-one-condition-out (random forest):

| n | threshold | accuracy | precision | recall | f1 | specificity | roc_auc | average_precision | tp | tn | fp | fn | predicted_positive | actual_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16.0000 | 0.5000 | 0.7500 | 0.8000 | 0.5714 | 0.6667 | 0.8889 | 0.8095 | 0.7488 | 4.0000 | 8.0000 | 1.0000 | 3.0000 | 5.0000 | 7.0000 |


Top features:

| feature | importance |
| --- | --- |
| registration_ratio | 0.1807 |
| num_points3D | 0.0947 |
| mean_observations_per_registered_image | 0.0844 |
| num_registered_images | 0.0624 |
| median_reproj_error | 0.0573 |
| num_observations | 0.0516 |
| mean_reproj_error | 0.0499 |
| track_error_mean | 0.0499 |
| track_weight_median | 0.0381 |
| weight_mean | 0.0375 |
| reliability_mean | 0.0375 |
| track_weight_p10 | 0.0352 |
| num_input_images | 0.0351 |
| num_covis_pairs | 0.0250 |
| track_length_max | 0.0200 |


Plots: [class_balance.png](failure_model_synthetic/post/plots/class_balance.png), [confusion_matrix.png](failure_model_synthetic/post/plots/confusion_matrix.png), [feature_importance.png](failure_model_synthetic/post/plots/feature_importance.png), [roc_pr.png](failure_model_synthetic/post/plots/roc_pr.png), [roc_pr_leave_one_condition_out.png](failure_model_synthetic/post/plots/roc_pr_leave_one_condition_out.png)

- Dataset: [runs_dataset_post.csv](failure_model_synthetic/post/runs_dataset_post.csv)
- Model: [failure_model.joblib](failure_model_synthetic/post/failure_model.joblib)

### Baseline vs adaptive track weighting

> Data kind: **synthetic**.

- Baseline model: `demo_runs/clean_s0/model`
- Engine requested: `auto`
- Modes: uniform, length, full
- Total runtime: 31.4 s

| variant | engine | num_registered_images | num_points3D | mean_reproj_error | median_reproj_error | track_length_median | weight_mean | success_label | runtime_seconds | delta_mean_reproj_error | delta_num_points3D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | colmap_baseline | 12 | 2672 | 0.6598 | 0.5113 | 4.0000 | 0.8936 | 0 | 2.1466 | 0.0000 | 0.0000 |
| uniform | scipy_least_squares | 12 | 2672 | 0.5630 | 0.4111 | 4.0000 | 1.0000 | 0 | 9.8999 | -0.0968 | 0.0000 |
| length | scipy_least_squares | 12 | 2672 | 0.5641 | 0.4112 | 4.0000 | 0.8857 | 0 | 8.5332 | -0.0957 | 0.0000 |
| full | scipy_least_squares | 12 | 2672 | 0.5653 | 0.4108 | 4.0000 | 0.8936 | 0 | 8.5959 | -0.0946 | 0.0000 |


Weight distributions:

| mode | count | weight_mean | weight_std | weight_min | weight_p10 | weight_median | weight_max | reliability_mean | fraction_below_0_1 | fraction_above_0_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 2672 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| length | 2672 | 0.8857 | 0.0684 | 0.6159 | 0.7924 | 0.8830 | 0.9296 | 0.8796 | 0.0000 | 0.4764 |
| full | 2672 | 0.8936 | 0.0372 | 0.6014 | 0.8500 | 0.9044 | 0.9339 | 0.8880 | 0.0000 | 0.5954 |


Refinement diagnostics:

| mode | status | engine | selected_points | observations | rmse_before | rmse_after | improved | retri_points | runtime_s | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | ok | scipy_least_squares | 2672 | 11487 | 1.3937 | 1.1959 | True | None | 9.8999 |  |
| length | ok | scipy_least_squares | 2672 | 11487 | 1.3937 | 1.1924 | True | None | 8.5332 |  |
| full | ok | scipy_least_squares | 2672 | 11487 | 1.3937 | 1.1872 | True | None | 8.5959 |  |


Plots: [comparison_bars.png](experiments_synthetic/clean_s0/adaptive/plots/comparison_bars.png), [reliability_scatter.png](experiments_synthetic/clean_s0/adaptive/plots/reliability_scatter.png), [reproj_error_baseline.png](experiments_synthetic/clean_s0/adaptive/plots/reproj_error_baseline.png), [track_length_baseline.png](experiments_synthetic/clean_s0/adaptive/plots/track_length_baseline.png), [weight_distribution.png](experiments_synthetic/clean_s0/adaptive/plots/weight_distribution.png)

- Comparison table: [comparison.csv](experiments_synthetic/clean_s0/adaptive/comparison.csv)
- Track weights: [track_weights.csv](experiments_synthetic/clean_s0/adaptive/track_weights.csv)
- Adaptive summary: [adaptive_summary.json](experiments_synthetic/clean_s0/adaptive/adaptive_summary.json)
- Refined models: uniform: `experiments_synthetic/clean_s0/adaptive/models/uniform`, length: `experiments_synthetic/clean_s0/adaptive/models/length`, full: `experiments_synthetic/clean_s0/adaptive/models/full`

- Baseline is the unmodified COLMAP sparse model.
- Adaptive variants are post-mapper refinements from the same initial model; they are not integrated into COLMAP's incremental mapper.
- Uniform refinement is a control: same BA, all weights = 1.
- Length-only is an ablation of the full reliability score.
- COLMAP baseline points are not re-triangulated by the scipy refinement.
- data_kind=synthetic

## Commands

```bash
bash scripts/download_eth3d.sh data/raw
python scripts/prepare_courtyard.py --extract
python scripts/run_baseline.py --config configs/courtyard.yaml --run-name clean --condition clean
python scripts/analyze_tracks.py --config configs/courtyard.yaml --run-dir outputs/runs/clean
python scripts/make_challenging_subsets.py --config configs/courtyard.yaml
python scripts/train_failure_model.py --config configs/courtyard.yaml --runs-root outputs/runs --mode pre
python scripts/run_adaptive.py --config configs/courtyard.yaml --run-dir outputs/runs/clean
python scripts/run_all_experiments.py --config configs/courtyard.yaml
python -m pytest -q
```

## Outputs inventory

| path | size |
| --- | --- |
| .gitkeep | 0.0 B |
| demo_runs/_demo_summary.json | 7.0 KB |
| demo_runs/blur_mild_s0/candidate_tracks.csv | 238.1 KB |
| demo_runs/blur_mild_s0/database.db | 512.0 KB |
| demo_runs/blur_mild_s0/database_pairs.json | 8.8 KB |
| demo_runs/blur_mild_s0/features_post.json | 1.1 KB |
| demo_runs/blur_mild_s0/features_pre.json | 1.9 KB |
| demo_runs/blur_mild_s0/images/image_0001.png | 335.6 KB |
| demo_runs/blur_mild_s0/images/image_0002.png | 344.0 KB |
| demo_runs/blur_mild_s0/images/image_0003.png | 343.5 KB |
| demo_runs/blur_mild_s0/images/image_0004.png | 337.6 KB |
| demo_runs/blur_mild_s0/images/image_0005.png | 328.0 KB |
| demo_runs/blur_mild_s0/images/image_0006.png | 317.2 KB |
| demo_runs/blur_mild_s0/images/image_0007.png | 301.9 KB |
| demo_runs/blur_mild_s0/images/image_0008.png | 294.1 KB |
| demo_runs/blur_mild_s0/images/image_0009.png | 291.9 KB |
| demo_runs/blur_mild_s0/images/image_0010.png | 298.6 KB |
| demo_runs/blur_mild_s0/images/image_0011.png | 312.5 KB |
| demo_runs/blur_mild_s0/images/image_0012.png | 328.5 KB |
| demo_runs/blur_mild_s0/images.csv | 2.2 KB |
| demo_runs/blur_mild_s0/label.json | 544.0 B |
| demo_runs/blur_mild_s0/manifest.json | 892.0 B |
| demo_runs/blur_mild_s0/model/cameras.txt | 149.0 B |
| demo_runs/blur_mild_s0/model/images.txt | 365.9 KB |
| demo_runs/blur_mild_s0/model/points3D.txt | 269.1 KB |
| demo_runs/blur_mild_s0/pairs.csv | 5.2 KB |
| demo_runs/blur_mild_s0/reconstruction_summary.json | 1.9 KB |
| demo_runs/blur_mild_s0/run_manifest.json | 797.0 B |
| demo_runs/blur_mild_s0/tracks.csv | 917.6 KB |
| demo_runs/blur_mild_s1/candidate_tracks.csv | 236.6 KB |
| demo_runs/blur_mild_s1/database.db | 516.0 KB |
| demo_runs/blur_mild_s1/database_pairs.json | 8.8 KB |
| demo_runs/blur_mild_s1/features_post.json | 1.1 KB |
| demo_runs/blur_mild_s1/features_pre.json | 1.9 KB |
| demo_runs/blur_mild_s1/images/image_0001.png | 335.6 KB |
| demo_runs/blur_mild_s1/images/image_0002.png | 343.4 KB |
| demo_runs/blur_mild_s1/images/image_0003.png | 344.8 KB |
| demo_runs/blur_mild_s1/images/image_0004.png | 337.1 KB |
| demo_runs/blur_mild_s1/images/image_0005.png | 327.4 KB |
| demo_runs/blur_mild_s1/images/image_0006.png | 316.4 KB |
| demo_runs/blur_mild_s1/images/image_0007.png | 303.4 KB |
| demo_runs/blur_mild_s1/images/image_0008.png | 293.7 KB |
| demo_runs/blur_mild_s1/images/image_0009.png | 292.9 KB |
| demo_runs/blur_mild_s1/images/image_0010.png | 298.3 KB |
| demo_runs/blur_mild_s1/images/image_0011.png | 312.7 KB |
| demo_runs/blur_mild_s1/images/image_0012.png | 326.9 KB |
| demo_runs/blur_mild_s1/images.csv | 2.2 KB |
| demo_runs/blur_mild_s1/label.json | 544.0 B |
| demo_runs/blur_mild_s1/manifest.json | 892.0 B |
| demo_runs/blur_mild_s1/model/cameras.txt | 149.0 B |
| demo_runs/blur_mild_s1/model/images.txt | 365.9 KB |
| demo_runs/blur_mild_s1/model/points3D.txt | 268.1 KB |
| demo_runs/blur_mild_s1/pairs.csv | 5.2 KB |
| demo_runs/blur_mild_s1/reconstruction_summary.json | 1.9 KB |
| demo_runs/blur_mild_s1/run_manifest.json | 797.0 B |
| demo_runs/blur_mild_s1/tracks.csv | 913.7 KB |
| demo_runs/blur_severe_s0/candidate_tracks.csv | 239.9 KB |
| demo_runs/blur_severe_s0/database.db | 508.0 KB |
| demo_runs/blur_severe_s0/database_pairs.json | 8.8 KB |
| demo_runs/blur_severe_s0/features_post.json | 1.1 KB |
| demo_runs/blur_severe_s0/features_pre.json | 1.9 KB |
| demo_runs/blur_severe_s0/images/image_0001.png | 179.3 KB |
| demo_runs/blur_severe_s0/images/image_0002.png | 182.2 KB |
| demo_runs/blur_severe_s0/images/image_0003.png | 181.5 KB |
| demo_runs/blur_severe_s0/images/image_0004.png | 179.0 KB |
| demo_runs/blur_severe_s0/images/image_0005.png | 178.4 KB |
| demo_runs/blur_severe_s0/images/image_0006.png | 167.0 KB |
| demo_runs/blur_severe_s0/images/image_0007.png | 157.5 KB |
| demo_runs/blur_severe_s0/images/image_0008.png | 151.1 KB |
| demo_runs/blur_severe_s0/images/image_0009.png | 148.4 KB |
| demo_runs/blur_severe_s0/images/image_0010.png | 155.3 KB |
| demo_runs/blur_severe_s0/images/image_0011.png | 165.5 KB |
| demo_runs/blur_severe_s0/images/image_0012.png | 172.2 KB |
| demo_runs/blur_severe_s0/images.csv | 2.1 KB |
| demo_runs/blur_severe_s0/label.json | 633.0 B |
| demo_runs/blur_severe_s0/manifest.json | 894.0 B |
| demo_runs/blur_severe_s0/model/cameras.txt | 149.0 B |
| demo_runs/blur_severe_s0/model/images.txt | 365.9 KB |
| demo_runs/blur_severe_s0/model/points3D.txt | 267.7 KB |
| demo_runs/blur_severe_s0/pairs.csv | 4.8 KB |
| demo_runs/blur_severe_s0/reconstruction_summary.json | 2.0 KB |
| demo_runs/blur_severe_s0/run_manifest.json | 805.0 B |
| demo_runs/blur_severe_s0/tracks.csv | 916.8 KB |
| demo_runs/blur_severe_s1/candidate_tracks.csv | 244.6 KB |
| demo_runs/blur_severe_s1/database.db | 512.0 KB |
| demo_runs/blur_severe_s1/database_pairs.json | 8.8 KB |
| demo_runs/blur_severe_s1/features_post.json | 1.1 KB |
| demo_runs/blur_severe_s1/features_pre.json | 1.9 KB |
| demo_runs/blur_severe_s1/images/image_0001.png | 178.6 KB |
| demo_runs/blur_severe_s1/images/image_0002.png | 181.0 KB |
| demo_runs/blur_severe_s1/images/image_0003.png | 184.8 KB |
| demo_runs/blur_severe_s1/images/image_0004.png | 179.1 KB |
| demo_runs/blur_severe_s1/images/image_0005.png | 175.1 KB |
| demo_runs/blur_severe_s1/images/image_0006.png | 166.6 KB |
| demo_runs/blur_severe_s1/images/image_0007.png | 158.7 KB |
| demo_runs/blur_severe_s1/images/image_0008.png | 150.9 KB |
| demo_runs/blur_severe_s1/images/image_0009.png | 149.2 KB |
| demo_runs/blur_severe_s1/images/image_0010.png | 155.5 KB |
| demo_runs/blur_severe_s1/images/image_0011.png | 167.9 KB |
| demo_runs/blur_severe_s1/images/image_0012.png | 172.3 KB |
| demo_runs/blur_severe_s1/images.csv | 2.2 KB |
| demo_runs/blur_severe_s1/label.json | 633.0 B |
| demo_runs/blur_severe_s1/manifest.json | 894.0 B |
| demo_runs/blur_severe_s1/model/cameras.txt | 149.0 B |
| demo_runs/blur_severe_s1/model/images.txt | 365.9 KB |
| demo_runs/blur_severe_s1/model/points3D.txt | 266.8 KB |
| demo_runs/blur_severe_s1/pairs.csv | 4.7 KB |
| demo_runs/blur_severe_s1/reconstruction_summary.json | 2.0 KB |
| demo_runs/blur_severe_s1/run_manifest.json | 805.0 B |
| demo_runs/blur_severe_s1/tracks.csv | 913.1 KB |
| demo_runs/clean_s0/candidate_tracks.csv | 240.5 KB |
| demo_runs/clean_s0/database.db | 508.0 KB |
| demo_runs/clean_s0/database_pairs.json | 8.8 KB |
| demo_runs/clean_s0/features_post.json | 1.1 KB |
| demo_runs/clean_s0/features_pre.json | 1.8 KB |
| demo_runs/clean_s0/images/image_0001.png | 640.5 KB |
| demo_runs/clean_s0/images/image_0002.png | 642.3 KB |
| demo_runs/clean_s0/images/image_0003.png | 642.2 KB |
| demo_runs/clean_s0/images/image_0004.png | 640.2 KB |
| demo_runs/clean_s0/images/image_0005.png | 637.8 KB |
| demo_runs/clean_s0/images/image_0006.png | 634.4 KB |
| demo_runs/clean_s0/images/image_0007.png | 630.0 KB |
| demo_runs/clean_s0/images/image_0008.png | 627.9 KB |
| demo_runs/clean_s0/images/image_0009.png | 627.0 KB |
| demo_runs/clean_s0/images/image_0010.png | 628.8 KB |
| demo_runs/clean_s0/images/image_0011.png | 633.0 KB |
| demo_runs/clean_s0/images/image_0012.png | 637.7 KB |
| demo_runs/clean_s0/images.csv | 2.2 KB |
| demo_runs/clean_s0/label.json | 544.0 B |
| demo_runs/clean_s0/manifest.json | 888.0 B |
| demo_runs/clean_s0/model/cameras.txt | 149.0 B |
| demo_runs/clean_s0/model/images.txt | 365.9 KB |
| demo_runs/clean_s0/model/points3D.txt | 270.0 KB |
| demo_runs/clean_s0/pairs.csv | 5.1 KB |
| demo_runs/clean_s0/reconstruction_summary.json | 1.9 KB |
| demo_runs/clean_s0/run_manifest.json | 781.0 B |
| demo_runs/clean_s0/tracks.csv | 919.5 KB |
| demo_runs/clean_s1/candidate_tracks.csv | 239.7 KB |
| demo_runs/clean_s1/database.db | 512.0 KB |
| demo_runs/clean_s1/database_pairs.json | 8.8 KB |
| demo_runs/clean_s1/features_post.json | 1.1 KB |
| demo_runs/clean_s1/features_pre.json | 1.9 KB |
| demo_runs/clean_s1/images/image_0001.png | 640.2 KB |
| demo_runs/clean_s1/images/image_0002.png | 642.1 KB |
| demo_runs/clean_s1/images/image_0003.png | 641.8 KB |
| demo_runs/clean_s1/images/image_0004.png | 640.0 KB |
| demo_runs/clean_s1/images/image_0005.png | 637.7 KB |
| demo_runs/clean_s1/images/image_0006.png | 634.3 KB |
| demo_runs/clean_s1/images/image_0007.png | 630.2 KB |
| demo_runs/clean_s1/images/image_0008.png | 627.6 KB |
| demo_runs/clean_s1/images/image_0009.png | 627.2 KB |
| demo_runs/clean_s1/images/image_0010.png | 628.9 KB |
| demo_runs/clean_s1/images/image_0011.png | 632.7 KB |
| demo_runs/clean_s1/images/image_0012.png | 637.2 KB |
| demo_runs/clean_s1/images.csv | 2.2 KB |
| demo_runs/clean_s1/label.json | 544.0 B |
| demo_runs/clean_s1/manifest.json | 888.0 B |
| demo_runs/clean_s1/model/cameras.txt | 149.0 B |
| demo_runs/clean_s1/model/images.txt | 366.0 KB |
| demo_runs/clean_s1/model/points3D.txt | 269.2 KB |
| demo_runs/clean_s1/pairs.csv | 5.1 KB |
| demo_runs/clean_s1/reconstruction_summary.json | 1.9 KB |
| demo_runs/clean_s1/run_manifest.json | 781.0 B |
| demo_runs/clean_s1/tracks.csv | 915.8 KB |
| demo_runs/combined_severe_s0/candidate_tracks.csv | 27.3 KB |
| demo_runs/combined_severe_s0/database.db | 152.0 KB |
| demo_runs/combined_severe_s0/database_pairs.json | 2.2 KB |
| demo_runs/combined_severe_s0/features_post.json | 1.1 KB |
| demo_runs/combined_severe_s0/features_pre.json | 1.8 KB |
| demo_runs/combined_severe_s0/images/image_0001.png | 274.1 KB |
| demo_runs/combined_severe_s0/images/image_0002.png | 280.3 KB |
| demo_runs/combined_severe_s0/images/image_0003.png | 268.0 KB |
| demo_runs/combined_severe_s0/images/image_0004.png | 239.3 KB |
| demo_runs/combined_severe_s0/images/image_0005.png | 226.5 KB |
| demo_runs/combined_severe_s0/images/image_0006.png | 250.5 KB |
| demo_runs/combined_severe_s0/images.csv | 753.0 B |
| demo_runs/combined_severe_s0/label.json | 667.0 B |
| demo_runs/combined_severe_s0/manifest.json | 759.0 B |
| demo_runs/combined_severe_s0/model/cameras.txt | 149.0 B |
| demo_runs/combined_severe_s0/model/images.txt | 59.6 KB |
| demo_runs/combined_severe_s0/model/points3D.txt | 44.1 KB |
| demo_runs/combined_severe_s0/pairs.csv | 427.0 B |
| demo_runs/combined_severe_s0/reconstruction_summary.json | 1.9 KB |
| demo_runs/combined_severe_s0/run_manifest.json | 746.0 B |
| demo_runs/combined_severe_s0/tracks.csv | 175.8 KB |
| demo_runs/combined_severe_s1/candidate_tracks.csv | 13.6 KB |
| demo_runs/combined_severe_s1/database.db | 152.0 KB |
| demo_runs/combined_severe_s1/database_pairs.json | 2.2 KB |
| demo_runs/combined_severe_s1/features_post.json | 1.1 KB |
| demo_runs/combined_severe_s1/features_pre.json | 1.8 KB |
| demo_runs/combined_severe_s1/images/image_0001.png | 272.9 KB |
| demo_runs/combined_severe_s1/images/image_0002.png | 281.7 KB |
| demo_runs/combined_severe_s1/images/image_0003.png | 265.2 KB |
| demo_runs/combined_severe_s1/images/image_0004.png | 241.3 KB |
| demo_runs/combined_severe_s1/images/image_0005.png | 229.0 KB |
| demo_runs/combined_severe_s1/images/image_0006.png | 251.4 KB |
| demo_runs/combined_severe_s1/images.csv | 750.0 B |
| demo_runs/combined_severe_s1/label.json | 624.0 B |
| demo_runs/combined_severe_s1/manifest.json | 759.0 B |
| demo_runs/combined_severe_s1/model/cameras.txt | 149.0 B |
| demo_runs/combined_severe_s1/model/images.txt | 59.7 KB |
| demo_runs/combined_severe_s1/model/points3D.txt | 44.7 KB |
| demo_runs/combined_severe_s1/pairs.csv | 301.0 B |
| demo_runs/combined_severe_s1/reconstruction_summary.json | 1.8 KB |
| demo_runs/combined_severe_s1/run_manifest.json | 746.0 B |
| demo_runs/combined_severe_s1/tracks.csv | 177.4 KB |
| demo_runs/degraded_matches_mild_s0/candidate_tracks.csv | 245.6 KB |
| demo_runs/degraded_matches_mild_s0/database.db | 512.0 KB |
| demo_runs/degraded_matches_mild_s0/database_pairs.json | 8.8 KB |
| demo_runs/degraded_matches_mild_s0/features_post.json | 1.1 KB |
| demo_runs/degraded_matches_mild_s0/features_pre.json | 1.9 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0001.png | 640.5 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0002.png | 642.3 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0003.png | 642.2 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0004.png | 640.2 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0005.png | 637.8 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0006.png | 634.4 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0007.png | 630.0 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0008.png | 627.9 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0009.png | 627.0 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0010.png | 628.8 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0011.png | 633.0 KB |
| demo_runs/degraded_matches_mild_s0/images/image_0012.png | 637.7 KB |
| demo_runs/degraded_matches_mild_s0/images.csv | 2.2 KB |
| demo_runs/degraded_matches_mild_s0/label.json | 544.0 B |
| demo_runs/degraded_matches_mild_s0/manifest.json | 904.0 B |
| demo_runs/degraded_matches_mild_s0/model/cameras.txt | 149.0 B |
| demo_runs/degraded_matches_mild_s0/model/images.txt | 365.9 KB |
| demo_runs/degraded_matches_mild_s0/model/points3D.txt | 269.1 KB |
| demo_runs/degraded_matches_mild_s0/pairs.csv | 4.9 KB |
| demo_runs/degraded_matches_mild_s0/reconstruction_summary.json | 2.0 KB |
| demo_runs/degraded_matches_mild_s0/run_manifest.json | 846.0 B |
| demo_runs/degraded_matches_mild_s0/tracks.csv | 917.6 KB |
| demo_runs/degraded_matches_mild_s1/candidate_tracks.csv | 253.5 KB |
| demo_runs/degraded_matches_mild_s1/database.db | 512.0 KB |
| demo_runs/degraded_matches_mild_s1/database_pairs.json | 8.8 KB |
| demo_runs/degraded_matches_mild_s1/features_post.json | 1.1 KB |
| demo_runs/degraded_matches_mild_s1/features_pre.json | 1.9 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0001.png | 640.2 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0002.png | 642.1 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0003.png | 641.8 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0004.png | 640.0 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0005.png | 637.7 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0006.png | 634.3 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0007.png | 630.2 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0008.png | 627.6 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0009.png | 627.2 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0010.png | 628.9 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0011.png | 632.7 KB |
| demo_runs/degraded_matches_mild_s1/images/image_0012.png | 637.2 KB |
| demo_runs/degraded_matches_mild_s1/images.csv | 2.2 KB |
| demo_runs/degraded_matches_mild_s1/label.json | 544.0 B |
| demo_runs/degraded_matches_mild_s1/manifest.json | 904.0 B |
| demo_runs/degraded_matches_mild_s1/model/cameras.txt | 149.0 B |
| demo_runs/degraded_matches_mild_s1/model/images.txt | 365.9 KB |
| demo_runs/degraded_matches_mild_s1/model/points3D.txt | 268.1 KB |
| demo_runs/degraded_matches_mild_s1/pairs.csv | 4.8 KB |
| demo_runs/degraded_matches_mild_s1/reconstruction_summary.json | 2.0 KB |
| demo_runs/degraded_matches_mild_s1/run_manifest.json | 846.0 B |
| demo_runs/degraded_matches_mild_s1/tracks.csv | 913.7 KB |
| demo_runs/degraded_matches_severe_s0/candidate_tracks.csv | 54.6 KB |
| demo_runs/degraded_matches_severe_s0/database.db | 508.0 KB |
| demo_runs/degraded_matches_severe_s0/database_pairs.json | 8.8 KB |
| demo_runs/degraded_matches_severe_s0/features_post.json | 1.1 KB |
| demo_runs/degraded_matches_severe_s0/features_pre.json | 1.9 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0001.png | 640.5 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0002.png | 642.3 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0003.png | 642.2 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0004.png | 640.2 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0005.png | 637.8 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0006.png | 634.4 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0007.png | 630.0 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0008.png | 627.9 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0009.png | 627.0 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0010.png | 628.8 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0011.png | 633.0 KB |
| demo_runs/degraded_matches_severe_s0/images/image_0012.png | 637.7 KB |
| demo_runs/degraded_matches_severe_s0/images.csv | 1.1 KB |
| demo_runs/degraded_matches_severe_s0/label.json | 708.0 B |
| demo_runs/degraded_matches_severe_s0/manifest.json | 906.0 B |
| demo_runs/degraded_matches_severe_s0/model/cameras.txt | 149.0 B |
| demo_runs/degraded_matches_severe_s0/model/images.txt | 60.5 KB |
| demo_runs/degraded_matches_severe_s0/model/points3D.txt | 64.1 KB |
| demo_runs/degraded_matches_severe_s0/pairs.csv | 574.0 B |
| demo_runs/degraded_matches_severe_s0/reconstruction_summary.json | 1.9 KB |
| demo_runs/degraded_matches_severe_s0/run_manifest.json | 854.0 B |
| demo_runs/degraded_matches_severe_s0/tracks.csv | 254.7 KB |
| demo_runs/degraded_matches_severe_s1/candidate_tracks.csv | 57.7 KB |
| demo_runs/degraded_matches_severe_s1/database.db | 512.0 KB |
| demo_runs/degraded_matches_severe_s1/database_pairs.json | 8.8 KB |
| demo_runs/degraded_matches_severe_s1/features_post.json | 906.0 B |
| demo_runs/degraded_matches_severe_s1/features_pre.json | 1.9 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0001.png | 640.2 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0002.png | 642.1 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0003.png | 641.8 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0004.png | 640.0 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0005.png | 637.7 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0006.png | 634.3 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0007.png | 630.2 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0008.png | 627.6 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0009.png | 627.2 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0010.png | 628.9 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0011.png | 632.7 KB |
| demo_runs/degraded_matches_severe_s1/images/image_0012.png | 637.2 KB |
| demo_runs/degraded_matches_severe_s1/images.csv | 1005.0 B |
| demo_runs/degraded_matches_severe_s1/label.json | 619.0 B |
| demo_runs/degraded_matches_severe_s1/manifest.json | 906.0 B |
| demo_runs/degraded_matches_severe_s1/model/cameras.txt | 149.0 B |
| demo_runs/degraded_matches_severe_s1/model/images.txt | 29.1 KB |
| demo_runs/degraded_matches_severe_s1/model/points3D.txt | 125.0 B |
| demo_runs/degraded_matches_severe_s1/pairs.csv | 517.0 B |
| demo_runs/degraded_matches_severe_s1/reconstruction_summary.json | 1.6 KB |
| demo_runs/degraded_matches_severe_s1/run_manifest.json | 854.0 B |
| demo_runs/degraded_matches_severe_s1/tracks.csv | 273.0 B |
| demo_runs/limited_views_s0/candidate_tracks.csv | 191.0 KB |
| demo_runs/limited_views_s0/database.db | 152.0 KB |
| demo_runs/limited_views_s0/database_pairs.json | 2.2 KB |
| demo_runs/limited_views_s0/features_post.json | 1.1 KB |
| demo_runs/limited_views_s0/features_pre.json | 1.8 KB |
| demo_runs/limited_views_s0/images/image_0001.png | 640.5 KB |
| demo_runs/limited_views_s0/images/image_0002.png | 642.2 KB |
| demo_runs/limited_views_s0/images/image_0003.png | 638.0 KB |
| demo_runs/limited_views_s0/images/image_0004.png | 629.9 KB |
| demo_runs/limited_views_s0/images/image_0005.png | 627.2 KB |
| demo_runs/limited_views_s0/images/image_0006.png | 633.2 KB |
| demo_runs/limited_views_s0/images.csv | 1.2 KB |
| demo_runs/limited_views_s0/label.json | 544.0 B |
| demo_runs/limited_views_s0/manifest.json | 757.0 B |
| demo_runs/limited_views_s0/model/cameras.txt | 149.0 B |
| demo_runs/limited_views_s0/model/images.txt | 182.2 KB |
| demo_runs/limited_views_s0/model/points3D.txt | 214.8 KB |
| demo_runs/limited_views_s0/pairs.csv | 1.4 KB |
| demo_runs/limited_views_s0/reconstruction_summary.json | 1.9 KB |
| demo_runs/limited_views_s0/run_manifest.json | 738.0 B |
| demo_runs/limited_views_s0/tracks.csv | 836.5 KB |
| demo_runs/limited_views_s1/candidate_tracks.csv | 192.6 KB |
| demo_runs/limited_views_s1/database.db | 152.0 KB |
| demo_runs/limited_views_s1/database_pairs.json | 2.2 KB |
| demo_runs/limited_views_s1/features_post.json | 1.1 KB |
| demo_runs/limited_views_s1/features_pre.json | 1.9 KB |
| demo_runs/limited_views_s1/images/image_0001.png | 640.2 KB |
| demo_runs/limited_views_s1/images/image_0002.png | 642.1 KB |
| demo_runs/limited_views_s1/images/image_0003.png | 637.5 KB |
| demo_runs/limited_views_s1/images/image_0004.png | 630.2 KB |
| demo_runs/limited_views_s1/images/image_0005.png | 627.1 KB |
| demo_runs/limited_views_s1/images/image_0006.png | 633.1 KB |
| demo_runs/limited_views_s1/images.csv | 1.2 KB |
| demo_runs/limited_views_s1/label.json | 544.0 B |
| demo_runs/limited_views_s1/manifest.json | 757.0 B |
| demo_runs/limited_views_s1/model/cameras.txt | 149.0 B |
| demo_runs/limited_views_s1/model/images.txt | 182.9 KB |
| demo_runs/limited_views_s1/model/points3D.txt | 216.0 KB |
| demo_runs/limited_views_s1/pairs.csv | 1.4 KB |
| demo_runs/limited_views_s1/reconstruction_summary.json | 1.9 KB |
| demo_runs/limited_views_s1/run_manifest.json | 738.0 B |
| demo_runs/limited_views_s1/tracks.csv | 840.2 KB |
| demo_runs/limited_views_severe_s0/candidate_tracks.csv | 14.3 KB |
| demo_runs/limited_views_severe_s0/database.db | 96.0 KB |
| demo_runs/limited_views_severe_s0/database_pairs.json | 772.0 B |
| demo_runs/limited_views_severe_s0/features_post.json | 1.1 KB |
| demo_runs/limited_views_severe_s0/features_pre.json | 1.8 KB |
| demo_runs/limited_views_severe_s0/images/image_0001.png | 413.5 KB |
| demo_runs/limited_views_severe_s0/images/image_0002.png | 415.0 KB |
| demo_runs/limited_views_severe_s0/images/image_0003.png | 389.4 KB |
| demo_runs/limited_views_severe_s0/images/image_0004.png | 386.9 KB |
| demo_runs/limited_views_severe_s0/images.csv | 631.0 B |
| demo_runs/limited_views_severe_s0/label.json | 624.0 B |
| demo_runs/limited_views_severe_s0/manifest.json | 717.0 B |
| demo_runs/limited_views_severe_s0/model/cameras.txt | 149.0 B |
| demo_runs/limited_views_severe_s0/model/images.txt | 58.8 KB |
| demo_runs/limited_views_severe_s0/model/points3D.txt | 25.9 KB |
| demo_runs/limited_views_severe_s0/pairs.csv | 248.0 B |
| demo_runs/limited_views_severe_s0/reconstruction_summary.json | 1.8 KB |
| demo_runs/limited_views_severe_s0/run_manifest.json | 699.0 B |
| demo_runs/limited_views_severe_s0/tracks.csv | 103.8 KB |
| demo_runs/limited_views_severe_s1/candidate_tracks.csv | 39.9 KB |
| demo_runs/limited_views_severe_s1/database.db | 96.0 KB |
| demo_runs/limited_views_severe_s1/database_pairs.json | 772.0 B |
| demo_runs/limited_views_severe_s1/features_post.json | 1.1 KB |
| demo_runs/limited_views_severe_s1/features_pre.json | 1.8 KB |
| demo_runs/limited_views_severe_s1/images/image_0001.png | 413.8 KB |
| demo_runs/limited_views_severe_s1/images/image_0002.png | 414.0 KB |
| demo_runs/limited_views_severe_s1/images/image_0003.png | 390.4 KB |
| demo_runs/limited_views_severe_s1/images/image_0004.png | 386.6 KB |
| demo_runs/limited_views_severe_s1/images.csv | 819.0 B |
| demo_runs/limited_views_severe_s1/label.json | 544.0 B |
| demo_runs/limited_views_severe_s1/manifest.json | 717.0 B |
| demo_runs/limited_views_severe_s1/model/cameras.txt | 149.0 B |
| demo_runs/limited_views_severe_s1/model/images.txt | 118.7 KB |
| demo_runs/limited_views_severe_s1/model/points3D.txt | 102.9 KB |
| demo_runs/limited_views_severe_s1/pairs.csv | 550.0 B |
| demo_runs/limited_views_severe_s1/reconstruction_summary.json | 1.8 KB |
| demo_runs/limited_views_severe_s1/run_manifest.json | 699.0 B |
| demo_runs/limited_views_severe_s1/tracks.csv | 411.2 KB |
| environment.json | 1.0 KB |
| experiments/clean/adaptive/adaptive_summary.json | 14.0 KB |
| experiments/clean/adaptive/comparison.csv | 1.4 KB |
| experiments/clean/adaptive/comparison.md | 2.2 KB |
| experiments/clean/adaptive/models/full/cameras.txt | 2.4 KB |
| experiments/clean/adaptive/models/full/images.txt | 11.8 MB |
| experiments/clean/adaptive/models/full/points3D.txt | 2.1 MB |
| experiments/clean/adaptive/models/length/cameras.txt | 2.4 KB |
| experiments/clean/adaptive/models/length/images.txt | 11.8 MB |
| experiments/clean/adaptive/models/length/points3D.txt | 2.1 MB |
| experiments/clean/adaptive/models/uniform/cameras.txt | 2.4 KB |
| experiments/clean/adaptive/models/uniform/images.txt | 11.8 MB |
| experiments/clean/adaptive/models/uniform/points3D.txt | 2.1 MB |
| experiments/clean/adaptive/plots/comparison_bars.png | 86.1 KB |
| experiments/clean/adaptive/plots/reliability_scatter.png | 101.3 KB |
| experiments/clean/adaptive/plots/reproj_error_baseline.png | 34.6 KB |
| experiments/clean/adaptive/plots/track_length_baseline.png | 34.8 KB |
| experiments/clean/adaptive/plots/weight_distribution.png | 47.9 KB |
| experiments/clean/adaptive/reconstruction_summary_full.json | 835.0 B |
| experiments/clean/adaptive/reconstruction_summary_length.json | 835.0 B |
| experiments/clean/adaptive/reconstruction_summary_uniform.json | 801.0 B |
| experiments/clean/adaptive/track_weights.csv | 8.0 MB |
| experiments/clean/adaptive/track_weights_length.csv | 7.3 MB |
| experiments/clean/adaptive/track_weights_uniform.csv | 6.5 MB |
| experiments/clean/adaptive/tracks_full.csv | 7.0 MB |
| experiments/clean/adaptive/tracks_length.csv | 7.0 MB |
| experiments/clean/adaptive/tracks_uniform.csv | 7.0 MB |
| experiments/clean/adaptive_fulltest/adaptive_summary.json | 5.9 KB |
| experiments/clean/adaptive_fulltest/comparison.csv | 910.0 B |
| experiments/clean/adaptive_fulltest/comparison.md | 1.5 KB |
| experiments/clean/adaptive_fulltest/models/full/cameras.txt | 2.4 KB |
| experiments/clean/adaptive_fulltest/models/full/images.txt | 11.8 MB |
| experiments/clean/adaptive_fulltest/models/full/points3D.txt | 2.1 MB |
| experiments/clean/adaptive_fulltest/plots/comparison_bars.png | 68.0 KB |
| experiments/clean/adaptive_fulltest/plots/reliability_scatter.png | 102.4 KB |
| experiments/clean/adaptive_fulltest/plots/reproj_error_baseline.png | 34.3 KB |
| experiments/clean/adaptive_fulltest/plots/track_length_baseline.png | 35.3 KB |
| experiments/clean/adaptive_fulltest/plots/weight_distribution.png | 48.0 KB |
| experiments/clean/adaptive_fulltest/reconstruction_summary_full.json | 833.0 B |
| experiments/clean/adaptive_fulltest/track_weights.csv | 8.0 MB |
| experiments/clean/adaptive_fulltest/tracks_full.csv | 6.9 MB |
| experiments_synthetic/clean_s0/adaptive/adaptive_summary.json | 13.7 KB |
| experiments_synthetic/clean_s0/adaptive/comparison.csv | 1.4 KB |
| experiments_synthetic/clean_s0/adaptive/comparison.md | 2.0 KB |
| experiments_synthetic/clean_s0/adaptive/models/full/cameras.txt | 149.0 B |
| experiments_synthetic/clean_s0/adaptive/models/full/images.txt | 365.9 KB |
| experiments_synthetic/clean_s0/adaptive/models/full/points3D.txt | 270.2 KB |
| experiments_synthetic/clean_s0/adaptive/models/length/cameras.txt | 149.0 B |
| experiments_synthetic/clean_s0/adaptive/models/length/images.txt | 365.9 KB |
| experiments_synthetic/clean_s0/adaptive/models/length/points3D.txt | 270.2 KB |
| experiments_synthetic/clean_s0/adaptive/models/uniform/cameras.txt | 149.0 B |
| experiments_synthetic/clean_s0/adaptive/models/uniform/images.txt | 365.9 KB |
| experiments_synthetic/clean_s0/adaptive/models/uniform/points3D.txt | 270.1 KB |
| experiments_synthetic/clean_s0/adaptive/plots/comparison_bars.png | 86.3 KB |
| experiments_synthetic/clean_s0/adaptive/plots/reliability_scatter.png | 66.0 KB |
| experiments_synthetic/clean_s0/adaptive/plots/reproj_error_baseline.png | 35.0 KB |
| experiments_synthetic/clean_s0/adaptive/plots/track_length_baseline.png | 29.1 KB |
| experiments_synthetic/clean_s0/adaptive/plots/weight_distribution.png | 43.1 KB |
| experiments_synthetic/clean_s0/adaptive/reconstruction_summary_full.json | 820.0 B |
| experiments_synthetic/clean_s0/adaptive/reconstruction_summary_length.json | 824.0 B |
| experiments_synthetic/clean_s0/adaptive/reconstruction_summary_uniform.json | 791.0 B |
| experiments_synthetic/clean_s0/adaptive/track_weights.csv | 1.0 MB |
| experiments_synthetic/clean_s0/adaptive/track_weights_length.csv | 987.9 KB |
| experiments_synthetic/clean_s0/adaptive/track_weights_uniform.csv | 862.7 KB |
| experiments_synthetic/clean_s0/adaptive/tracks_full.csv | 921.7 KB |
| experiments_synthetic/clean_s0/adaptive/tracks_length.csv | 921.8 KB |
| experiments_synthetic/clean_s0/adaptive/tracks_uniform.csv | 921.6 KB |
| failure_model/post/failure_model.joblib | 66.0 KB |
| failure_model/post/feature_importances.csv | 538.0 B |
| failure_model/post/metrics.json | 10.0 KB |
| failure_model/post/oof_predictions.csv | 653.0 B |
| failure_model/post/plots/class_balance.png | 11.7 KB |
| failure_model/post/plots/confusion_matrix.png | 26.0 KB |
| failure_model/post/plots/feature_importance.png | 75.6 KB |
| failure_model/post/plots/roc_pr.png | 54.9 KB |
| failure_model/post/plots/roc_pr_leave_one_condition_out.png | 64.2 KB |
| failure_model/post/runs_dataset_post.csv | 3.6 KB |
| failure_model/post/split.json | 388.0 B |
| failure_model/pre/failure_model.joblib | 4.9 KB |
| failure_model/pre/feature_importances.csv | 1.5 KB |
| failure_model/pre/metrics.json | 12.9 KB |
| failure_model/pre/oof_predictions.csv | 652.0 B |
| failure_model/pre/plots/class_balance.png | 11.7 KB |
| failure_model/pre/plots/confusion_matrix.png | 22.4 KB |
| failure_model/pre/plots/feature_importance.png | 76.3 KB |
| failure_model/pre/plots/roc_pr.png | 56.4 KB |
| failure_model/pre/plots/roc_pr_leave_one_condition_out.png | 63.6 KB |
| failure_model/pre/runs_dataset_pre.csv | 5.8 KB |
| failure_model/pre/split.json | 388.0 B |
| failure_model_synthetic/post/failure_model.joblib | 4.0 KB |
| failure_model_synthetic/post/feature_importances.csv | 915.0 B |
| failure_model_synthetic/post/metrics.json | 10.5 KB |
| failure_model_synthetic/post/oof_predictions.csv | 1.4 KB |
| failure_model_synthetic/post/plots/class_balance.png | 11.8 KB |
| failure_model_synthetic/post/plots/confusion_matrix.png | 25.2 KB |
| failure_model_synthetic/post/plots/feature_importance.png | 80.3 KB |
| failure_model_synthetic/post/plots/roc_pr.png | 56.3 KB |
| failure_model_synthetic/post/plots/roc_pr_leave_one_condition_out.png | 64.5 KB |
| failure_model_synthetic/post/runs_dataset_post.csv | 6.7 KB |
| failure_model_synthetic/post/split.json | 599.0 B |
| failure_model_synthetic/pre/failure_model.joblib | 66.6 KB |
| failure_model_synthetic/pre/feature_importances.csv | 1.0 KB |
| failure_model_synthetic/pre/metrics.json | 13.2 KB |
| failure_model_synthetic/pre/oof_predictions.csv | 1.3 KB |
| failure_model_synthetic/pre/plots/class_balance.png | 11.8 KB |
| failure_model_synthetic/pre/plots/confusion_matrix.png | 25.2 KB |


_Inventory truncated to 500 of 766 files._

## Limitations and not-run items

- Adaptive weighting is a **post-mapper refinement** of the same COLMAP model; it is not integrated into COLMAP's incremental mapper.
- The scipy weighted BA keeps intrinsics fixed, fixes the first camera as gauge, and does not add/remove tracks.
- Unselected tracks are re-triangulated against the refined cameras; this is reported per refinement, not hidden.
- pyceres is not installed, so the documented scipy fallback is used. No Ceres run is claimed.
- pycolmap is unavailable in this environment, so COLMAP binary models are converted with the COLMAP CLI and parsed from TXT; no pycolmap result is claimed.
- ETH3D ground-truth laser-scan accuracy/completeness is not evaluated; only SfM internal metrics and labels are reported.
- Failure labels depend on the number and diversity of runs; with few conditions, the prediction metrics are preliminary.
- Only the ETH3D Courtyard scene is used; leave-one-scene-out generalization is not evaluated.
- The real failure model is marked preliminary because of the small run/condition count.

## Next steps

- Add more ETH3D scenes and severity levels to obtain enough runs per class.
- Add a pyceres/Ceres adapter when a signed, importable build is available.
- Integrate weights into a modified mapper / retriangulation stage.
- Evaluate against ETH3D ground-truth scans rather than only internal SfM metrics.
