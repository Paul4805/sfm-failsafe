"""Small shared helpers used across sfm-failsafe."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


class ActionableError(RuntimeError):
    """Error whose message tells the user exactly what to fix."""


def repo_root() -> Path:
    """Return the repository root (the directory that contains configs/ and src/)."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(name: str = "sfm_failsafe", log_file: str | Path | None = None,
                 level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Avoid duplicating handlers when scripts call setup_logger more than once.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if value is pd.NA:
        return None
    return value


def dump_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    payload = json.dumps(_jsonable(obj), indent=indent, allow_nan=False)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise ActionableError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def read_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: str | Path, hint: str = "") -> Path:
    path = Path(path)
    if not path.exists():
        message = f"Required file is missing: {path}"
        if hint:
            message += f"\n{hint}"
        raise ActionableError(message)
    return path


def require_dir(path: str | Path, hint: str = "") -> Path:
    path = Path(path)
    if not path.is_dir():
        message = f"Required directory is missing: {path}"
        if hint:
            message += f"\n{hint}"
        raise ActionableError(message)
    return path


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def as_int_list(values: Iterable[Any]) -> list[int]:
    return [int(v) for v in values]


class Timer:
    """Context manager that records wall-clock seconds."""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start

    def seconds(self) -> float:
        if self.start == 0.0:
            return 0.0
        return time.perf_counter() - self.start


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def slugify(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in text.strip()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def find_images(images_dir: str | Path, pattern: str = "*") -> list[Path]:
    """Recursively find image files, sorted deterministically."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files = [p for p in images_dir.rglob(pattern) if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)
