import json
from pathlib import Path

import numpy as np

from sfm_failsafe.database import ColmapDatabase
from sfm_failsafe.dataset import discover_images, prepare_eth3d_courtyard
from sfm_failsafe.model_io import read_model
from sfm_failsafe.synthetic import (generate_synthetic_dataset,
                                     simulate_registration,
                                     subset_reconstruction)


def test_synthetic_dataset_structure(tmp_path):
    out = tmp_path / "scene"
    result = generate_synthetic_dataset(
        out, num_images=6, num_points=120, width=240, height=180,
        noise_px=0.3, seed=0, write_images=True,
    )
    assert (out / "images").is_dir()
    assert (out / "model" / "cameras.txt").exists()
    assert (out / "database.db").exists()
    assert result["manifest"]["data_kind"] == "synthetic"
    rec = read_model(out / "model")
    assert rec.num_images == 6
    assert rec.num_points > 10
    assert len(list((out / "images").glob("*.png"))) == 6
    with ColmapDatabase(out / "database.db") as db:
        assert db.stats()["num_images"] == 6
        assert db.stats()["num_pairs"] > 0


def test_simulated_registration_and_subset(tmp_path):
    out = tmp_path / "scene"
    result = generate_synthetic_dataset(
        out, num_images=6, num_points=120, width=240, height=180, seed=1
    )
    rec = result["reconstruction"]
    registered = simulate_registration(rec, result["database"], min_links=1)
    assert len(registered) == 6
    subset = subset_reconstruction(rec, registered[:4])
    assert subset.num_images == 4
    assert subset.num_points <= rec.num_points
    for image in subset.images.values():
        assert len(image.xys) == len(image.point3D_ids)


def test_prepare_eth3d_layout(tmp_path):
    raw = tmp_path / "raw"
    image_dir = raw / "courtyard" / "images" / "dslr_images_undistorted"
    image_dir.mkdir(parents=True)
    import cv2
    for index in range(3):
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"DSC_{index:04d}.JPG"), image)
    images, pattern = discover_images(raw)
    assert len(images) == 3
    assert "dslr_images_undistorted" in pattern
    manifest = prepare_eth3d_courtyard(raw, tmp_path / "prepared", mode="hardlink")
    assert manifest["num_images"] == 3
    assert (tmp_path / "prepared" / "manifest.json").exists()
    assert len(list((tmp_path / "prepared" / "images").glob("*.JPG"))) == 3
