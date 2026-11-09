"""YAML configuration loading with deep-merge defaults.

All relative paths in a config are interpreted relative to the repository
root, not the current working directory.  That keeps every experiment
reproducible no matter where a script is launched from.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .utils import ActionableError, repo_root


def _deep_merge(base: dict, override: Mapping) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ActionableError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ActionableError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ActionableError(f"Config root must be a mapping: {path}")
    return data


def default_config_path() -> Path:
    return repo_root() / "configs" / "default.yaml"


def load_config(path: str | Path | None = None,
                overrides: Mapping[str, Any] | None = None) -> dict:
    """Load configs/default.yaml, deep-merge a user config, then apply overrides.

    ``overrides`` uses dotted keys, e.g. ``{"colmap.max_image_size": 1600}``.
    Values set to ``None`` are kept as ``None`` (YAML null).
    """
    root = repo_root()
    defaults = _read_yaml(default_config_path())
    config: dict = copy.deepcopy(defaults)

    if path is not None:
        config_path = Path(path)
        if not config_path.is_absolute():
            candidate = Path.cwd() / config_path
            config_path = candidate if candidate.exists() else root / config_path
        if config_path.resolve() != default_config_path().resolve():
            config = _deep_merge(config, _read_yaml(config_path))
    else:
        config_path = default_config_path()

    config["_meta"] = {
        "repo_root": str(root),
        "config_path": str(config_path),
    }

    if overrides:
        for dotted, value in overrides.items():
            set_dotted(config, dotted, value)
    return config


def set_dotted(config: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def get_cfg(config: Mapping, dotted: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def cfg_path(config: Mapping, dotted: str, default: str | None = None) -> Path:
    value = get_cfg(config, dotted, default)
    if value is None:
        raise ActionableError(f"Config key is not set: {dotted}")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(get_cfg(config, "_meta.repo_root", str(repo_root()))) / path


def resolve_path(config: Mapping, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(get_cfg(config, "_meta.repo_root", str(repo_root()))) / path


def config_to_yaml(config: Mapping, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in config.items() if k != "_meta"}
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path
