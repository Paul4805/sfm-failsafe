import numpy as np
import pandas as pd

from sfm_failsafe import evaluation


def test_perfect_classification_metrics():
    metrics = evaluation.classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_known_confusion_matrix():
    metrics = evaluation.classification_metrics(
        [0, 0, 1, 1], [0.9, 0.2, 0.4, 0.8], threshold=0.5
    )
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_single_class_auc_is_nan():
    metrics = evaluation.classification_metrics([1, 1], [0.2, 0.8])
    assert np.isnan(metrics["roc_auc"])


def test_comparison_table_deltas():
    table = evaluation.comparison_table([
        {"variant": "baseline", "num_points3D": 100, "mean_reproj_error": 1.0,
         "track_length_median": 4},
        {"variant": "full", "num_points3D": 110, "mean_reproj_error": 0.8,
         "track_length_median": 4},
    ])
    with_deltas = evaluation.add_deltas(table, "baseline")
    assert with_deltas.loc[1, "delta_num_points3D"] == 10
    assert abs(with_deltas.loc[1, "delta_mean_reproj_error"] + 0.2) < 1e-9


def test_markdown_writer(tmp_path):
    table = pd.DataFrame([{"variant": "a", "x": 1.23456}])
    path = evaluation.write_comparison_markdown(table, tmp_path / "cmp.md")
    text = path.read_text(encoding="utf-8")
    assert "variant" in text
    assert "1.2346" in text
