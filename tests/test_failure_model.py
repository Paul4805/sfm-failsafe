import json
from pathlib import Path

import numpy as np
import pandas as pd

from sfm_failsafe import failure_model


def _write_run(run_dir: Path, index: int, label: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    failure = bool(label)
    summary = {
        "status": "success",
        "registration_ratio": 0.25 if failure else 0.95,
        "num_points3D": 150 if failure else 5000,
        "mean_reproj_error": 4.0 if failure else 0.8,
        "median_reproj_error": 3.0 if failure else 0.6,
        "track_length_median": 1.0 if failure else 4.0,
        "num_input_images": 8,
        "num_registered_images": 2 if failure else 8,
    }
    (run_dir / "reconstruction_summary.json").write_text(json.dumps(summary))
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "run_name": run_dir.name,
        "condition": f"condition_{index % 4}",
        "data_kind": "real",
        "images_dir": str(run_dir / "images"),
        "image_names": [f"image_{i}.png" for i in range(8)],
    }))
    features = {
        "num_input_images": 8,
        "num_pairs": 20 + index,
        "candidate_track_mean_len": 1.5 + 0.2 * index,
        "candidate_mean_inlier_consistency": 0.5 + 0.03 * index,
        "pair_inlier_mean": 0.4 + 0.04 * index,
        "blur_mean": 100.0 + 10 * index,
        "candidate_frac_len_ge3": 0.2 + 0.02 * index,
    }
    (run_dir / "features_pre.json").write_text(json.dumps({
        "meta": {"run_name": run_dir.name, "mode": "pre"},
        "features": features,
    }))


def test_train_failure_model_on_cached_runs(tmp_path, config):
    runs_root = tmp_path / "runs"
    for index in range(12):
        _write_run(runs_root / f"run_{index:02d}", index, label=int(index >= 6))
    out = tmp_path / "model"
    metrics = failure_model.train_failure_model(runs_root, out, config, mode="pre")
    assert metrics["status"] == "ok"
    assert metrics["n_runs"] == 12
    assert metrics["mode"] == "pre"
    assert "best_metrics" in metrics
    assert (out / "failure_model.joblib").exists()
    assert (out / "metrics.json").exists()
    assert (out / "runs_dataset_pre.csv").exists()
    assert (out / "plots" / "roc_pr.png").exists()
    bundle = __import__("joblib").load(out / "failure_model.joblib")
    rows = pd.DataFrame([metrics["feature_names"] and {
        name: 1.0 for name in metrics["feature_names"]
    }])
    probabilities = failure_model.predict_bundle(out / "failure_model.joblib", rows)
    assert probabilities.shape == (1,)
    assert 0.0 <= probabilities[0] <= 1.0


def test_split_dataset_group_aware(tmp_path, config):
    rows = []
    for group in range(5):
        for label in (0, 1):
            rows.append({
                "run_name": f"g{group}_{label}",
                "condition": f"c{group}",
                "label": label,
                "x": float(label),
            })
    df = pd.DataFrame(rows)
    split = failure_model.split_dataset(df, config)
    assert len(split["train_indices"]) > 0
    assert len(split["test_indices"]) > 0
    assert set(split["train_indices"]).isdisjoint(split["test_indices"])


def test_choose_threshold_prefers_separation():
    threshold = failure_model._choose_threshold(
        np.array([0, 0, 1, 1]), np.array([0.01, 0.2, 0.8, 0.99])
    )
    assert 0.2 <= threshold <= 0.8
