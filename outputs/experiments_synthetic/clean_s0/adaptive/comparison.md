# Baseline vs adaptive track weighting (clean_s0)

- Baseline is the unmodified COLMAP sparse model.
- Adaptive variants are post-mapper refinements from the same initial model; they are not integrated into COLMAP's incremental mapper.
- Uniform refinement is a control: same BA, all weights = 1.
- Length-only is an ablation of the full reliability score.
- COLMAP baseline points are not re-triangulated by the scipy refinement.
- data_kind=synthetic

| variant | status | num_input_images | num_registered_images | registration_ratio | num_points3D | num_observations | mean_reproj_error | median_reproj_error | p95_reproj_error | track_length_mean | track_length_median | track_length_p90 | weight_mean | success_label | failure_reasons | runtime_seconds | engine | weighted | approximation | delta_num_registered_images | delta_num_points3D | delta_mean_reproj_error | delta_median_reproj_error | delta_track_length_median | delta_registration_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | success | 12 | 12 | 1.0000 | 2672 | 11487 | 0.6598 | 0.5113 | 1.9897 | 4.2990 | 4.0000 | 5.0000 | 0.8936 | 0 |  | 2.1466 | colmap_baseline | False | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| uniform | success | 12 | 12 | 1.0000 | 2672 | 11487 | 0.5630 | 0.4111 | 2.0322 | 4.2990 | 4.0000 | 5.0000 | 1.0000 | 0 |  | 9.8999 | scipy_least_squares | False | False | 0.0000 | 0.0000 | -0.0968 | -0.1002 | 0.0000 | 0.0000 |
| length | success | 12 | 12 | 1.0000 | 2672 | 11487 | 0.5641 | 0.4112 | 2.0398 | 4.2990 | 4.0000 | 5.0000 | 0.8857 | 0 |  | 8.5332 | scipy_least_squares | True | False | 0.0000 | 0.0000 | -0.0957 | -0.1001 | 0.0000 | 0.0000 |
| full | success | 12 | 12 | 1.0000 | 2672 | 11487 | 0.5653 | 0.4108 | 2.0543 | 4.2990 | 4.0000 | 5.0000 | 0.8936 | 0 |  | 8.5959 | scipy_least_squares | True | False | 0.0000 | 0.0000 | -0.0946 | -0.1005 | 0.0000 | 0.0000 |

