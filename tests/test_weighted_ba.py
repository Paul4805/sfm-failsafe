import numpy as np
import pytest

from sfm_failsafe import reliability, track_analysis
from sfm_failsafe.weighted_ba import (filter_tracks, refine_weighted,
                                      weights_from_dataframe)


def test_refine_reduces_reprojection_error(synthetic_rec, tiny_config):
    tracks = track_analysis.compute_track_table(synthetic_rec, tiny_config)
    weights = reliability.score_tracks(tracks, tiny_config)
    weight_map = weights_from_dataframe(weights)
    refined, report = refine_weighted(synthetic_rec, weight_map, tiny_config)
    assert report["status"] == "ok", report
    assert report["engine_used"] == "scipy_least_squares"
    assert report["num_selected_points"] > 0
    assert report["errors_after"]["rmse"] < report["errors_before"]["rmse"]
    assert report["improved"] is True
    # Gauge: fixed camera is untouched.
    fixed_id = report["fixed_camera_id"]
    assert np.allclose(refined.images[fixed_id].qvec, synthetic_rec.images[fixed_id].qvec)
    assert np.allclose(refined.images[fixed_id].tvec, synthetic_rec.images[fixed_id].tvec)
    # Points were actually updated.
    some_id = list(refined.points3D)[0]
    assert not np.allclose(refined.points3D[some_id].xyz,
                           synthetic_rec.points3D[some_id].xyz)


def test_analytic_jacobian_matches_finite_difference(synthetic_rec, tiny_config):
    from sfm_failsafe.weighted_ba import (_camera_arrays, _pose_bases,
                                          _project_residual_and_jac)
    point_ids = sorted(synthetic_rec.points3D)[:20]
    weights = {pid: 1.0 for pid in point_ids}
    data = _camera_arrays(synthetic_rec, point_ids, weights, 1.0)
    assert data is not None
    data["fixed_image"] = synthetic_rec.images[min(synthetic_rec.images)]
    rot0, t0 = _pose_bases(synthetic_rec, data["free_image_ids"])
    data["rot0"], data["t0"] = rot0, t0
    n_free = len(data["free_image_ids"])
    n_points = len(data["xyz"])
    x0 = np.zeros(6 * n_free + 3 * n_points)
    _, jacobian = _project_residual_and_jac(
        x0, data, rot0, t0, data["free_image_ids"], need_jac=True
    )
    residual0, _ = _project_residual_and_jac(
        x0, data, rot0, t0, data["free_image_ids"], need_jac=False
    )
    rng = np.random.default_rng(0)
    for column in rng.choice(jacobian.shape[1], size=6, replace=False):
        xp = x0.copy()
        xp[column] += 1e-6
        residualp, _ = _project_residual_and_jac(
            xp, data, rot0, t0, data["free_image_ids"], need_jac=False
        )
        numeric = (residualp - residual0) / 1e-6
        analytic = jacobian[:, column].toarray().ravel()
        assert np.allclose(numeric, analytic, atol=1e-4, rtol=1e-4)


def test_filter_tracks(synthetic_rec, tiny_config):
    tracks = track_analysis.compute_track_table(synthetic_rec, tiny_config)
    weight_map = {int(pid): (0.01 if i % 2 else 0.9)
                  for i, pid in enumerate(tracks["point3D_id"])}
    filtered, report = filter_tracks(synthetic_rec, weight_map, weight_threshold=0.1)
    assert report["removed_points"] > 0
    assert filtered.num_points < synthetic_rec.num_points
    for image in filtered.images.values():
        assert all(int(pid) in filtered.points3D or int(pid) == -1
                   for pid in image.point3D_ids)


def test_min_weight_skips_tracks(synthetic_rec, tiny_config):
    tracks = track_analysis.compute_track_table(synthetic_rec, tiny_config)
    weights = reliability.score_tracks(tracks, tiny_config)
    tiny_config["weighted_ba"]["min_weight"] = 0.9
    selected = int((weights["weight"] >= 0.9).sum())
    _, report = refine_weighted(
        synthetic_rec, weights_from_dataframe(weights), tiny_config
    )
    assert report["num_selected_points"] <= selected or report["status"] == "skipped"
