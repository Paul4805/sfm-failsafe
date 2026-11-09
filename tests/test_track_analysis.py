import numpy as np
import pandas as pd

from sfm_failsafe import track_analysis


def test_track_table_statistics(synthetic_rec, tiny_config):
    tracks = track_analysis.compute_track_table(synthetic_rec, tiny_config)
    assert len(tracks) == synthetic_rec.num_points
    assert (tracks["track_length"] >= 2).all()
    assert tracks["mean_reproj_error"].notna().all()
    assert (tracks["mean_reproj_error"] >= 0).all()
    assert tracks["triangulation_angle_deg"].between(0, 180).all()
    assert tracks["weight"].between(0.05, 1.0).all()
    assert tracks["reliability"].between(0.0, 1.0).all()
    assert set(["point3D_id", "image_spread", "image_spread_norm",
                "mean_depth", "depth_ratio"]).issubset(tracks.columns)


def test_summary_and_image_pair_tables(synthetic_rec, synthetic_dataset, tiny_config):
    tracks = track_analysis.compute_track_table(synthetic_rec, tiny_config)
    summary = track_analysis.summarize_reconstruction(
        synthetic_rec, tracks, input_image_count=8
    )
    assert summary["num_input_images"] == 8
    assert summary["num_registered_images"] == synthetic_rec.num_images
    assert 0.0 <= summary["registration_ratio"] <= 1.0
    assert summary["num_points3D"] == synthetic_rec.num_points
    assert np.isfinite(summary["mean_reproj_error"])

    images = track_analysis.compute_image_table(
        synthetic_rec, tracks, tiny_config,
        image_dir=synthetic_dataset["images_dir"],
        all_image_names=synthetic_dataset["manifest"]["image_names"],
    )
    assert len(images) == 8
    assert images["registered"].all()
    assert images["blur_score"].notna().all()
    assert (images["blur_score"] > 0).all()

    pairs = track_analysis.compute_pair_table(
        synthetic_rec, tracks, db_path=synthetic_dataset["database"]
    )
    assert len(pairs) > 0
    assert "num_shared_points" in pairs.columns
    assert (pairs["num_shared_points"] > 0).any()


def test_analysis_writes_required_files(synthetic_rec, synthetic_dataset, tiny_config, tmp_path):
    analysis = track_analysis.analyze_reconstruction(
        synthetic_rec, tiny_config,
        image_dir=synthetic_dataset["images_dir"],
        db_path=synthetic_dataset["database"],
        all_image_names=synthetic_dataset["manifest"]["image_names"],
        input_image_count=8,
    )
    paths = track_analysis.write_analysis(analysis, tmp_path / "analysis")
    for key in ("tracks", "images", "pairs", "summary"):
        assert paths[key].exists()
    loaded = pd.read_csv(paths["tracks"])
    assert len(loaded) == synthetic_rec.num_points
