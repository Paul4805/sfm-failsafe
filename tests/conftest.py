"""Shared pytest fixtures: tiny synthetic scene, no ETH3D/COLMAP required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sfm_failsafe.config import load_config  # noqa: E402
from sfm_failsafe.model_io import read_model  # noqa: E402
from sfm_failsafe.synthetic import generate_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("synthetic_scene")
    result = generate_synthetic_dataset(
        out, num_images=8, num_points=250, width=320, height=240,
        noise_px=0.35, outlier_fraction=0.01, pose_noise_deg=0.1, seed=3,
        write_images=True,
    )
    return result


@pytest.fixture(scope="session")
def synthetic_rec(synthetic_dataset):
    return read_model(synthetic_dataset["model_dir"])


@pytest.fixture()
def tiny_config(config) -> dict:
    import copy
    cfg = copy.deepcopy(config)
    cfg.setdefault("weighted_ba", {})["max_points"] = 200
    cfg["weighted_ba"]["max_iterations"] = 20
    cfg.setdefault("weighting", {})["w_min"] = 0.05
    return cfg
