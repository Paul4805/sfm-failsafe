import numpy as np
import pytest

from sfm_failsafe.database import (ColmapDatabase, decode_pair_id,
                                   drop_match_fraction, encode_pair_id)
from sfm_failsafe.features import candidate_tracks_from_database


def test_database_roundtrip(synthetic_dataset):
    with ColmapDatabase(synthetic_dataset["database"]) as db:
        stats = db.stats()
        images = db.images()
        pairs = db.pair_records()
        keypoints = db.keypoints(int(images["image_id"].iloc[0]))
    assert stats["num_images"] == 8
    assert stats["num_cameras"] >= 1
    assert stats["num_pairs"] > 0
    assert keypoints.shape[0] > 0
    assert keypoints.shape[1] == 2
    assert not pairs.empty
    assert (pairs["num_inliers"] > 0).any()


def test_pair_id_roundtrip():
    pair_id = encode_pair_id(12, 5)
    assert decode_pair_id(pair_id) == (5, 12)


def test_candidate_tracks(synthetic_dataset, config):
    tracks, summary = candidate_tracks_from_database(
        synthetic_dataset["database"], min_length=2
    )
    assert len(tracks) == summary["candidate_track_count"]
    assert (tracks["track_length"] >= 2).all()
    assert (tracks["num_images"] <= tracks["track_length"]).all()
    assert tracks["inlier_consistency"].between(0, 1).all()
    assert summary["candidate_total_edges"] > 0
    assert 0.0 <= summary["candidate_mean_image_diversity"] <= 1.0


def test_drop_matches_does_not_modify_source(synthetic_dataset, tmp_path):
    source = synthetic_dataset["database"]
    with ColmapDatabase(source) as db:
        source_pairs = len(db.pair_records())
        source_matches = sum(len(db.matches(p)) for p in db.pair_ids())
    output = tmp_path / "dropped.db"
    drop_match_fraction(source, output, drop_fraction=0.7, seed=0)
    with ColmapDatabase(source) as db:
        assert len(db.pair_records()) == source_pairs
        assert sum(len(db.matches(p)) for p in db.pair_ids()) == source_matches
    with ColmapDatabase(output) as db:
        dropped_matches = sum(len(db.matches(p)) for p in db.pair_ids())
    assert dropped_matches < source_matches
