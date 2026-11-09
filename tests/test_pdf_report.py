import json
from pathlib import Path

from sfm_failsafe.config import load_config, set_dotted
from sfm_failsafe.pdf_report import REPORTLAB_AVAILABLE, generate_pdf_report


def test_pdf_report_builder(tmp_path):
    assert REPORTLAB_AVAILABLE, "reportlab should be installed via requirements.txt"
    config = load_config()
    set_dotted(config, "paths.failure_model_dir", str(tmp_path / "no_model"))
    run_dir = tmp_path / "runs" / "clean"
    run_dir.mkdir(parents=True)
    (run_dir / "reconstruction_summary.json").write_text(json.dumps({
        "status": "success",
        "condition": "clean",
        "data_kind": "real",
        "num_input_images": 8,
        "num_registered_images": 8,
        "registration_ratio": 1.0,
        "num_points3D": 1000,
        "mean_reproj_error": 0.8,
        "median_reproj_error": 0.5,
        "track_length_mean": 4.0,
        "track_length_median": 4.0,
        "track_length_p90": 6.0,
        "success_label_name": "success",
        "failure_reasons": [],
        "runtime_seconds": 1.0,
    }))
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "run_name": "clean", "condition": "clean", "data_kind": "real",
    }))
    out = tmp_path / "REPORT.pdf"
    path = generate_pdf_report(config, tmp_path / "runs", tmp_path / "experiments", out)
    assert path.exists()
    assert path.stat().st_size > 4_000
    assert path.read_bytes()[:4] == b"%PDF"
