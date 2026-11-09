import numpy as np
import pytest

from sfm_failsafe.model_io import (Camera, Image, Reconstruction, read_model,
                                    rotmat_to_qvec, qvec_to_rotmat,
                                    select_largest_model, write_model)


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r = qvec_to_rotmat(q)
        q2 = rotmat_to_qvec(r)
        assert np.allclose(qvec_to_rotmat(q2), r, atol=1e-8)


def test_pinhole_projection_and_undistortion():
    camera = Camera(1, "PINHOLE", 640, 480, np.array([500.0, 500.0, 320.0, 240.0]))
    uv = camera.project(np.array([[0.0, 0.0, 2.0], [1.0, -1.0, 2.0]]))
    assert np.allclose(uv[0], [320.0, 240.0])
    assert np.allclose(camera.undistort_to_pinhole(uv), uv)


def test_opencv_undistortion_roundtrip():
    params = np.array([500.0, 500.0, 320.0, 240.0, -0.2, 0.05, 0.001, -0.001])
    camera = Camera(1, "OPENCV", 640, 480, params)
    pixels = np.array([[100.0, 120.0], [400.0, 300.0], [320.0, 240.0]])
    pinhole = camera.undistort_to_pinhole(pixels)
    # Reproject the undistorted normalized coordinate through distortion and
    # check it lands back on the original pixel.
    fx, fy = camera.focal
    cx, cy = camera.principal
    x = (pinhole[:, 0] - cx) / fx
    y = (pinhole[:, 1] - cy) / fy
    xd, yd = camera._distort_normalized(x, y)
    rebuilt = np.column_stack([fx * xd + cx, fy * yd + cy])
    assert np.allclose(rebuilt, pixels, atol=1e-6)


def test_model_roundtrip(synthetic_rec, tmp_path):
    out = write_model(synthetic_rec, tmp_path / "model")
    loaded = read_model(out)
    assert loaded.num_images == synthetic_rec.num_images
    assert loaded.num_points == synthetic_rec.num_points
    assert set(loaded.cameras) == set(synthetic_rec.cameras)
    for image_id, image in synthetic_rec.images.items():
        other = loaded.images[image_id]
        assert image.name == other.name
        assert np.allclose(image.qvec, other.qvec, atol=1e-9)
        assert np.allclose(image.tvec, other.tvec, atol=1e-9)
        assert np.allclose(image.xys, other.xys, atol=1e-9)
    for point_id, point in synthetic_rec.points3D.items():
        other = loaded.points3D[point_id]
        assert np.allclose(point.xyz, other.xyz, atol=1e-9)
        assert np.array_equal(point.image_ids, other.image_ids)


def test_select_largest_submodel(synthetic_rec, tmp_path):
    sparse = tmp_path / "sparse"
    # A small model and a larger model.
    write_model(Reconstruction(), sparse / "0")
    write_model(synthetic_rec, sparse / "1")
    selected, stats = select_largest_model(sparse)
    assert selected.name == "1"
    assert stats["num_images"] == synthetic_rec.num_images
    assert stats["num_points3D"] == synthetic_rec.num_points


def test_missing_model_actionable(tmp_path):
    with pytest.raises(Exception) as exc:
        read_model(tmp_path / "nope")
    assert "COLMAP model" in str(exc.value) or "No COLMAP model" in str(exc.value)
