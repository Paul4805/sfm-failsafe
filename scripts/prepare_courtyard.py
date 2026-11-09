#!/usr/bin/env python
"""Prepare ETH3D Courtyard images and write a manifest."""

from __future__ import annotations

import argparse
import sys

from _bootstrap import REPO_ROOT

from sfm_failsafe.config import cfg_path, get_cfg, load_config
from sfm_failsafe.dataset import prepare_eth3d_courtyard
from sfm_failsafe.utils import ActionableError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest ETH3D Courtyard into data/courtyard.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "courtyard.yaml"))
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--image-glob", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--mode", default="hardlink", choices=["hardlink", "copy", "symlink"])
    parser.add_argument("--extract", action="store_true",
                        help="Extract the Courtyard .7z archive if found.")
    parser.add_argument("--archive", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    raw_root = args.raw_root or str(cfg_path(config, "dataset.raw_root"))
    out_dir = args.out_dir or str(
        cfg_path(config, "dataset.images_dir").parent
    )
    image_glob = args.image_glob or get_cfg(config, "dataset.image_glob")
    print(f"Raw root: {raw_root}")
    print(f"Out dir:  {out_dir}")
    try:
        manifest = prepare_eth3d_courtyard(
            raw_root, out_dir, image_glob=image_glob,
            max_images=args.max_images, mode=args.mode,
            extract=args.extract, archive=args.archive,
            camera_model=get_cfg(config, "dataset.camera_model", "PINHOLE"),
            dry_run=args.dry_run,
        )
    except ActionableError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    if not args.dry_run:
        print(f"Manifest:  {out_dir}/manifest.json")
        print(f"Images:    {manifest['num_images']}")
        print(f"Calibration: {manifest.get('calibration_dir') or 'not found'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
