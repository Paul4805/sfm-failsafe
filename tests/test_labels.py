from sfm_failsafe.evaluation import label_summary


def _success_summary():
    return {
        "status": "success",
        "registration_ratio": 0.95,
        "num_points3D": 5000,
        "mean_reproj_error": 0.8,
        "median_reproj_error": 0.6,
        "track_length_median": 4.0,
        "num_registered_images": 36,
        "num_input_images": 38,
    }


def test_success_label(config):
    label = label_summary(_success_summary(), config)
    assert label["label"] == 0
    assert label["label_name"] == "success"
    assert label["reasons"] == []


def test_low_registration_ratio(config):
    summary = _success_summary()
    summary["registration_ratio"] = 0.2
    label = label_summary(summary, config)
    assert label["label"] == 1
    assert any("registration_ratio" in reason for reason in label["reasons"])


def test_too_few_points(config):
    summary = _success_summary()
    summary["num_points3D"] = 10
    label = label_summary(summary, config)
    assert label["label"] == 1
    assert any("num_points3D" in reason for reason in label["reasons"])


def test_reprojection_thresholds(config):
    summary = _success_summary()
    summary["mean_reproj_error"] = 10.0
    summary["median_reproj_error"] = 9.0
    label = label_summary(summary, config)
    assert label["label"] == 1
    assert sum("reproj_error" in reason for reason in label["reasons"]) == 2


def test_mapper_failure(config):
    label = label_summary({"status": "failed"}, config)
    assert label["label"] == 1
    assert "mapper_failed" in label["reasons"]


def test_short_tracks(config):
    summary = _success_summary()
    summary["track_length_median"] = 1.0
    label = label_summary(summary, config)
    assert label["label"] == 1
    assert any("track_length_median" in reason for reason in label["reasons"])
