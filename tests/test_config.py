import copy
from pathlib import Path

import pytest

from sfm_failsafe.config import (cfg_path, default_config_path, get_cfg,
                                 load_config, set_dotted)
from sfm_failsafe.utils import ActionableError


def test_default_config_loads(repo_root):
    config = load_config()
    assert config["_meta"]["repo_root"] == str(repo_root)
    assert get_cfg(config, "colmap.camera_model")
    assert get_cfg(config, "weighting.w_min") == 0.05
    assert cfg_path(config, "paths.runs_root").is_absolute()
    assert default_config_path().exists()


def test_config_deep_merge_and_override(repo_root):
    config = load_config(repo_root / "configs" / "courtyard.yaml",
                         {"thresholds.min_num_points3D": 42})
    assert config["thresholds"]["min_num_points3D"] == 42
    # Values not present in courtyard.yaml survive from defaults.
    assert config["analysis"]["max_points"] == 200_000
    assert config["dataset"]["scene"] if "scene" in config["dataset"] else True


def test_dotted_set_and_get():
    config = {"a": {"b": 1}}
    set_dotted(config, "a.c", 2)
    assert get_cfg(config, "a.c") == 2
    assert get_cfg(config, "missing.key", "fallback") == "fallback"


def test_missing_config_raises(repo_root):
    with pytest.raises(ActionableError):
        load_config(repo_root / "configs" / "does-not-exist.yaml")
