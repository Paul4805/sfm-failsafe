"""ETH3D Courtyard ingestion and deterministic challenging-subset manifests."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .colmap_runner import IMAGE_EXTENSIONS, select_images
from .config import get_cfg
from .utils import (ActionableError, dump_json, ensure_dir, find_images,
                    load_json, slugify)

DEFAULT_IMAGE_GLOBS = [
    "**/dslr_images_undistorted/*.JPG",
    "**/dslr_images_undistorted/*.jpg",
    "**/images/dslr_images_undistorted/*.JPG",
    "**/dslr_calibration_jpg/*.JPG",
    "**/images/*.JPG",
    "**/images/*.jpg",
    "*.JPG",
    "*.jpg",
]


def discover_images(raw_root: str | Path,
                    image_glob: str | None = None) -> tuple[list[Path], str]:
    """Find ETH3D images using the configured glob and documented fallbacks."""
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        return [], ""
    patterns = ([image_glob] if image_glob else []) + DEFAULT_IMAGE_GLOBS
    best: list[Path] = []
    best_pattern = ""
    for pattern in dict.fromkeys(patterns):
        candidates = sorted(
            path for path in raw_root.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(candidates) > len(best):
            best, best_pattern = candidates, pattern
    return best, best_pattern


def find_archive(raw_root: str | Path) -> Path | None:
    raw_root = Path(raw_root)
    for pattern in ("**/*courtyard*.7z", "**/*courtyard*.zip", "**/*courtyard*undistorted*"):
        matches = sorted(raw_root.parent.glob(pattern)) + sorted(raw_root.glob(pattern))
        for match in matches:
            if match.is_file():
                return match
    return None


def extract_archive(archive: str | Path, destination: str | Path) -> Path:
    archive, destination = Path(archive), ensure_dir(Path(destination))
    if archive.suffix.lower() == ".7z":
        try:
            import py7zr
        except ImportError as exc:
            raise ActionableError(
                "py7zr is required to extract .7z archives:\n"
                "  python -m pip install py7zr\n"
                f"or extract manually: 7z x \"{archive}\" -o\"{destination}\""
            ) from exc
        with py7zr.SevenZipFile(archive, mode="r") as zf:
            zf.extractall(path=str(destination))
        return destination
    if archive.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
        return destination
    raise ActionableError(f"Unsupported archive type: {archive}")


def _materialise(source: Path, target: Path, mode: str) -> None:
    ensure_dir(target.parent)
    if target.exists():
        return
    if mode == "symlink":
        try:
            os.symlink(source, target)
            return
        except OSError:
            mode = "hardlink"
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _find_calibration(raw_root: Path) -> Path | None:
    for candidate in raw_root.rglob("cameras.txt"):
        if (candidate.parent / "images.txt").exists():
            return candidate.parent
    for candidate in raw_root.rglob("cameras.bin"):
        if (candidate.parent / "images.bin").exists():
            return candidate.parent
    return None


def prepare_eth3d_courtyard(raw_root: str | Path, out_dir: str | Path,
                            image_glob: str | None = None,
                            max_images: int | None = None,
                            mode: str = "hardlink", extract: bool = False,
                            archive: str | Path | None = None,
                            camera_model: str = "OPENCV",
                            dry_run: bool = False) -> dict[str, Any]:
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    archive_path = Path(archive) if archive else find_archive(raw_root)
    if extract and archive_path and not dry_run:
        print(f"Extracting {archive_path} -> {raw_root}")
        extract_archive(archive_path, raw_root)
    images, pattern = discover_images(raw_root, image_glob)
    if not images:
        message = (
            f"No ETH3D Courtyard images found under {raw_root}.\n"
            "Expected a layout matching one of:\n"
            "  <raw>/courtyard/dslr_images_undistorted/*.JPG\n"
            "  <raw>/courtyard/images/dslr_images_undistorted/*.JPG\n"
            f"Configured image_glob: {image_glob or '(defaults)'}\n\n"
            "Download the training archive (~0.5 GB):\n"
            "  bash scripts/download_eth3d.sh data/raw\n"
            "  python scripts/prepare_courtyard.py --extract\n"
            "Then retry."
        )
        if archive_path and not extract:
            message += f"\nFound archive {archive_path}; rerun with --extract."
        raise ActionableError(message)
    images = select_images(images, max_images)
    print(f"Found {len(images)} images using glob: {pattern}")
    if dry_run:
        for path in images[:5]:
            print(f"  {path}")
        if len(images) > 5:
            print(f"  ... ({len(images) - 5} more)")
        print("\nDry-run complete. No files were copied/linked.")
        return {"images": [str(p) for p in images], "image_glob": pattern,
                "num_images": len(images), "dry_run": True}

    out_dir = ensure_dir(out_dir)
    image_out = ensure_dir(out_dir / "images")
    names: list[str] = []
    used: set[str] = set()
    for source in images:
        name = source.name
        if name in used:
            name = f"{slugify(source.parent.name)}_{name}"
        used.add(name)
        _materialise(source, image_out / name, mode)
        names.append(name)

    calibration_src = _find_calibration(raw_root)
    calibration_dst = out_dir / "calibration"
    if calibration_src is not None:
        ensure_dir(calibration_dst)
        for pattern_file in ("cameras.txt", "images.txt", "points3D.txt",
                             "cameras.bin", "images.bin", "points3D.bin"):
            candidate = calibration_src / pattern_file
            if candidate.exists():
                shutil.copy2(candidate, calibration_dst / candidate.name)

    manifest = {
        "dataset": "eth3d_high_res_multi_view",
        "scene": "courtyard",
        "data_kind": "real",
        "condition": "clean",
        "raw_root": str(raw_root.resolve()),
        "images_dir": str(image_out.resolve()),
        "source_images_dir": str(images[0].parent.resolve()) if images else None,
        "image_glob": pattern,
        "image_names": names,
        "num_images": len(names),
        "camera_model": camera_model,
        "calibration_dir": str(calibration_dst.resolve()) if calibration_src else None,
        "archive": str(archive_path) if archive_path else None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    dump_json(manifest, out_dir / "manifest.json")
    pd.DataFrame({"image_name": names}).to_csv(out_dir / "manifest.csv", index=False)
    if calibration_src:
        print(f"Copied calibration reference -> {calibration_dst}")
    print(f"Prepared {len(names)} images -> {image_out}")
    return manifest


# ---------------------------------------------------------------------------
# Challenging subsets
# ---------------------------------------------------------------------------

def select_subset(paths: Sequence[Path], keep_every: int | None = None,
                  limit_count: int | None = None,
                  limit_mode: str = "every") -> list[Path]:
    paths = sorted(paths)
    if limit_count and limit_count > 0:
        if limit_mode == "contiguous":
            return paths[:limit_count]
        if limit_count >= len(paths):
            return list(paths)
        positions = np.linspace(0, len(paths) - 1, int(limit_count)).round().astype(int)
        return [paths[i] for i in sorted(set(positions.tolist()))]
    if keep_every and keep_every > 1:
        return paths[::keep_every]
    return list(paths)


def _blur_copy(source: Path, target: Path, sigma: float) -> None:
    import cv2

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ActionableError(f"Cannot read image: {source}")
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    ensure_dir(target.parent)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if target.suffix.lower() in (".jpg", ".jpeg") else []
    cv2.imwrite(str(target), blurred, params)


def create_subset(source_images_dir: str | Path, out_root: str | Path,
                  condition: str, max_images: int | None = None,
                  blur_sigma: float = 3.0, keep_every: int | None = None,
                  limit_count: int | None = None, limit_mode: str = "every",
                  drop_fraction: float = 0.5, seed: int = 0,
                  input_database: str | Path | None = None,
                  mode: str = "hardlink", dry_run: bool = False) -> dict[str, Any]:
    """Create one subset (clean/blur/limited_views/degraded_matches)."""
    source_images_dir = Path(source_images_dir)
    images = sorted(
        path for path in source_images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ActionableError(
            f"No images found under {source_images_dir}. Prepare the dataset first:\n"
            "  python scripts/prepare_courtyard.py --extract"
        )
    images = select_images(images, max_images)
    subset_dir = Path(out_root) / condition
    if condition == "clean":
        base_condition = "clean"
    elif condition.startswith("blur"):
        base_condition = "blur"
    elif condition.startswith("limited_views"):
        base_condition = "limited_views"
    elif condition.startswith("degraded_matches"):
        base_condition = "degraded_matches"
    else:
        raise ActionableError(
            f"Unknown condition: {condition} (use clean|blur|limited_views|degraded_matches)"
        )
    if base_condition == "clean":
        selected = images
        images_dir = source_images_dir
        image_names = [str(p.relative_to(source_images_dir)).replace("\\", "/") for p in selected]
    elif base_condition == "blur":
        selected = images
        images_dir = subset_dir / "images"
        image_names = [str(p.relative_to(source_images_dir)).replace("\\", "/") for p in selected]
        if not dry_run:
            for source, name in zip(selected, image_names):
                _blur_copy(source, Path(images_dir) / name, blur_sigma)
    elif base_condition == "limited_views":
        selected = select_subset(images, keep_every=keep_every,
                                 limit_count=limit_count, limit_mode=limit_mode)
        images_dir = subset_dir / "images"
        image_names = [str(p.relative_to(source_images_dir)).replace("\\", "/") for p in selected]
        if not dry_run:
            for source, name in zip(selected, image_names):
                _materialise(source, Path(images_dir) / name, mode)
    elif base_condition == "degraded_matches":
        selected = images
        images_dir = source_images_dir
        image_names = [str(p.relative_to(source_images_dir)).replace("\\", "/") for p in selected]
    else:  # pragma: no cover
        raise ActionableError(f"Unknown base condition: {base_condition}")

    manifest: dict[str, Any] = {
        "dataset": "eth3d_high_res_multi_view",
        "scene": "courtyard",
        "data_kind": "real",
        "condition": condition,
        "base_condition": base_condition,
        "source_images_dir": str(source_images_dir.resolve()),
        "images_dir": str(Path(images_dir).resolve()) if not dry_run else str(images_dir),
        "image_names": image_names,
        "num_images": int(len(selected)),
        "parameters": {
            "max_images": max_images,
            "blur_sigma": blur_sigma if base_condition == "blur" else None,
            "keep_every": keep_every if base_condition == "limited_views" else None,
            "limit_count": limit_count if base_condition == "limited_views" else None,
            "limit_mode": limit_mode if base_condition == "limited_views" else None,
            "drop_fraction": drop_fraction if base_condition == "degraded_matches" else None,
            "seed": seed,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if base_condition == "degraded_matches":
        from .database import drop_match_fraction

        if input_database is not None:
            source_db = Path(input_database)
            if not source_db.exists():
                raise ActionableError(f"Input database not found: {source_db}")
            if not dry_run:
                target = ensure_dir(subset_dir) / "database.db"
                drop_match_fraction(source_db, target, drop_fraction, seed=seed)
                manifest["input_database"] = str(target.resolve())
                manifest["database_corruption"] = {
                    "source": str(source_db.resolve()),
                    "drop_fraction": float(drop_fraction),
                    "seed": int(seed),
                }
        else:
            manifest["input_database"] = None
            manifest["database_instructions"] = (
                "Run baseline matching on the clean images first, then re-run this "
                "condition with --input-database <run>/database.db to create a "
                "deterministically degraded copy."
            )

    if not dry_run:
        manifests_dir = ensure_dir(Path(out_root) / "manifests")
        dump_json(manifest, manifests_dir / f"{condition}.json")
        pd.DataFrame({
            "condition": [condition] * len(image_names),
            "image_name": image_names,
        }).to_csv(manifests_dir / f"{condition}.csv", index=False)
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = load_json(path, None)
    if not isinstance(data, dict):
        raise ActionableError(f"Invalid subset manifest: {path}")
    return data


def manifest_image_dir(manifest: Mapping[str, Any]) -> Path:
    value = manifest.get("images_dir")
    if not value:
        raise ActionableError("Manifest has no images_dir")
    return Path(str(value))
