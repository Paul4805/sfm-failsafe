import json
import subprocess
import sys
from pathlib import Path

from sfm_failsafe.model_io import write_model


def _run(repo_root: Path, args: list[str], timeout: int = 120):
    return subprocess.run(
        [sys.executable, *args], cwd=repo_root, capture_output=True, text=True,
        timeout=timeout,
    )


def test_run_baseline_dry_run(repo_root, tmp_path):
    result = _run(repo_root, [
        "scripts/run_baseline.py", "--dry-run",
        "--config", "configs/default.yaml",
        "--images", str(tmp_path / "missing-images"),
        "--runs-root", str(tmp_path / "runs"),
        "--run-name", "dry",
    ])
    assert result.returncode == 0, result.stderr
    assert "DRY-RUN" in result.stdout
    assert (tmp_path / "runs" / "dry" / "dry_run_plan.json").exists()


def test_analyze_tracks_dry_run(repo_root, tmp_path):
    result = _run(repo_root, [
        "scripts/analyze_tracks.py", "--dry-run",
        "--config", "configs/default.yaml",
        "--model", str(tmp_path / "does-not-exist"),
        "--out", str(tmp_path / "out"),
    ])
    assert result.returncode == 0, result.stderr
    assert "Dry-run" in result.stdout


def test_train_failure_model_dry_run(repo_root, tmp_path):
    (tmp_path / "runs").mkdir()
    result = _run(repo_root, [
        "scripts/train_failure_model.py", "--dry-run",
        "--config", "configs/default.yaml",
        "--runs-root", str(tmp_path / "runs"),
        "--out", str(tmp_path / "model"),
    ])
    assert result.returncode == 0, result.stderr
    assert "Dry-run complete" in result.stdout


def test_run_adaptive_dry_run(repo_root, synthetic_rec, tmp_path):
    run_dir = tmp_path / "run"
    write_model(synthetic_rec, run_dir / "model")
    (run_dir / "reconstruction_summary.json").write_text(json.dumps({
        "num_input_images": 8, "num_registered_images": synthetic_rec.num_images,
        "registration_ratio": 1.0, "num_points3D": synthetic_rec.num_points,
        "mean_reproj_error": 1.0, "median_reproj_error": 0.8,
    }))
    result = _run(repo_root, [
        "scripts/run_adaptive.py", "--dry-run",
        "--config", "configs/default.yaml",
        "--run-dir", str(run_dir),
        "--out", str(tmp_path / "adaptive"),
    ])
    assert result.returncode == 0, result.stderr
    assert "Dry-run" in result.stdout


def test_run_all_synthetic_dry_run(repo_root, tmp_path):
    result = _run(repo_root, [
        "scripts/run_all_experiments.py", "--synthetic", "--dry-run",
        "--config", "configs/default.yaml",
    ])
    assert result.returncode == 0, result.stderr
    assert "Dry-run plan" in result.stdout


def test_prepare_courtyard_dry_run_actionable(repo_root, tmp_path):
    result = _run(repo_root, [
        "scripts/prepare_courtyard.py", "--dry-run",
        "--config", "configs/default.yaml",
        "--raw-root", str(tmp_path / "missing"),
    ])
    assert result.returncode == 2
    assert "No ETH3D" in result.stderr or "No ETH3D" in result.stdout
