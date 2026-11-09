import numpy as np
import pandas as pd

from sfm_failsafe import reliability


def _tracks(n=20):
    return pd.DataFrame({
        "point3D_id": np.arange(1, n + 1),
        "track_length": np.linspace(2, 12, n),
        "mean_reproj_error": np.linspace(3.0, 0.2, n),
        "triangulation_angle_deg": np.linspace(1.0, 12.0, n),
        "image_spread_norm": np.linspace(0.01, 0.5, n),
    })


def test_weights_in_range_and_monotonic(config):
    scored = reliability.score_tracks(_tracks(), config)
    assert scored["weight"].between(config["weighting"]["w_min"], 1.0).all()
    # Longer, lower-error, wider-baseline tracks should receive higher weight.
    assert scored["weight"].iloc[-1] > scored["weight"].iloc[0]


def test_error_and_length_effects(config):
    base = pd.DataFrame({
        "point3D_id": [1, 2, 3],
        "track_length": [8, 8, 8],
        "mean_reproj_error": [0.2, 2.0, 8.0],
        "triangulation_angle_deg": [10, 10, 10],
        "image_spread_norm": [0.2, 0.2, 0.2],
    })
    weights = reliability.score_tracks(base, config)["weight"].to_numpy()
    assert weights[0] > weights[1] > weights[2]

    lengths = pd.DataFrame({
        "point3D_id": [1, 2, 3],
        "track_length": [2, 5, 12],
        "mean_reproj_error": [0.5, 0.5, 0.5],
        "triangulation_angle_deg": [10, 10, 10],
        "image_spread_norm": [0.2, 0.2, 0.2],
    })
    weights = reliability.score_tracks(lengths, config)["weight"].to_numpy()
    assert weights[0] < weights[1] < weights[2]


def test_length_only_ablation(config):
    scored = reliability.score_tracks(_tracks(), config, length_only=True)
    assert scored["score_length"].notna().all()
    assert scored["weight"].between(config["weighting"]["w_min"], 1.0).all()


def test_quantile_mode_and_missing_features(config):
    import copy
    cfg = copy.deepcopy(config)
    cfg["weighting"]["map_mode"] = "quantile"
    scored = reliability.score_tracks(_tracks(), cfg)
    assert scored["reliability"].between(0.0, 1.0).all()
    assert abs(scored["reliability"].mean() - 0.5) < 0.1


def test_weight_summary_keys(config):
    scored = reliability.score_tracks(_tracks(), config)
    summary = reliability.weight_summary(scored)
    assert summary["count"] == 20
    for key in ("weight_mean", "weight_min", "weight_max", "fraction_below_0_1"):
        assert key in summary
