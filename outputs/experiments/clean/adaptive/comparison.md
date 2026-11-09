# Baseline vs adaptive track weighting (clean)

- Baseline is the unmodified COLMAP sparse model.
- Adaptive variants are post-mapper refinements from the same initial model; they are not integrated into COLMAP's incremental mapper.
- Uniform refinement is a control: same BA, all weights = 1.
- Length-only is an ablation of the full reliability score.
- Selected tracks receive the weighted robust BA; unselected tracks are re-triangulated against the refined cameras when the point cap excludes any.
- errors_before/after in refinement diagnostics cover selected observations with the robust objective; the comparison table uses full-model track mean/median reprojection error.
- data_kind=real

| variant | status | num_input_images | num_registered_images | registration_ratio | num_points3D | num_observations | mean_reproj_error | median_reproj_error | p95_reproj_error | track_length_mean | track_length_median | track_length_p90 | weight_mean | success_label | failure_reasons | runtime_seconds | engine | weighted | approximation | delta_num_registered_images | delta_num_points3D | delta_mean_reproj_error | delta_median_reproj_error | delta_track_length_median | delta_registration_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | success | 38 | 38 | 1.0000 | 20445 | 91619 | 0.7544 | 0.6431 | 1.7687 | 4.4812 | 4.0000 | 8.0000 | 0.8544 | 0 |  | 92.9708 | colmap_baseline | False | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| uniform | success | 38 | 38 | 1.0000 | 20445 | 91619 | 0.7454 | 0.6366 | 1.7359 | 4.4812 | 4.0000 | 8.0000 | 1.0000 | 0 |  | 39.0343 | scipy_least_squares | False | False | 0.0000 | 0.0000 | -0.0090 | -0.0066 | 0.0000 | 0.0000 |
| length | success | 38 | 38 | 1.0000 | 20445 | 91619 | 0.7465 | 0.6375 | 1.7397 | 4.4812 | 4.0000 | 8.0000 | 0.8431 | 0 |  | 31.0078 | scipy_least_squares | True | False | 0.0000 | 0.0000 | -0.0079 | -0.0056 | 0.0000 | 0.0000 |
| full | success | 38 | 38 | 1.0000 | 20445 | 91619 | 0.7465 | 0.6371 | 1.7416 | 4.4812 | 4.0000 | 8.0000 | 0.8544 | 0 |  | 38.2791 | scipy_least_squares | True | False | 0.0000 | 0.0000 | -0.0079 | -0.0060 | 0.0000 | 0.0000 |

