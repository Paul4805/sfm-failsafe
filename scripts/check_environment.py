#!/usr/bin/env python
"""Record environment diagnostics (Python, packages, COLMAP, GPU, dataset)."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from _bootstrap import REPO_ROOT

from sfm_failsafe.colmap_runner import ColmapRunner
from sfm_failsafe.config import cfg_path, get_cfg, load_config
from sfm_failsafe.utils import dump_json, load_json

PACKAGES = ["numpy", "pandas", "sklearn", "matplotlib", "cv2", "scipy",
            "yaml", "joblib", "pytest", "reportlab", "pycolmap", "pyceres"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Record environment diagnostics.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "courtyard.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "environment.json"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    report = {
        "python": sys.version,
        "executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {},
    }
    repo_root = Path(config.get("_meta", {}).get("repo_root", REPO_ROOT))

    def relative(value: str | Path) -> str:
        path = Path(str(value))
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except Exception:
            return path.name
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            report["packages"][name] = getattr(module, "__version__", "present")
        except Exception as exc:
            report["packages"][name] = f"unavailable ({type(exc).__name__})"
    try:
        runner = ColmapRunner(config)
        report["colmap"] = {
            "executable": relative(runner.executable),
            "version": runner.check_available(),
        }
    except Exception as exc:
        report["colmap"] = {"error": str(exc)}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        report["gpu"] = proc.stdout.strip().splitlines() if proc.returncode == 0 else []
    except Exception:
        report["gpu"] = []
    manifest_path = cfg_path(config, "dataset.manifest")
    if manifest_path.exists():
        manifest = load_json(manifest_path, {})
        report["dataset"] = {
            "manifest": relative(manifest_path),
            "scene": manifest.get("scene"),
            "num_images": manifest.get("num_images"),
            "camera_model": manifest.get("camera_model"),
            "images_dir": relative(manifest.get("images_dir", "")),
        }
    else:
        report["dataset"] = {"manifest": str(manifest_path), "exists": False}
    out = Path(args.out)
    dump_json(report, out)
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
