#!/usr/bin/env python
"""Create deterministic challenging subsets: clean, blur, limited views, degraded matches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from _bootstrap import REPO_ROOT

from sfm_failsafe.config import cfg_path, get_cfg, load_config
from sfm_failsafe.dataset import create_subset
from sfm_failsafe.utils import ActionableError, dump_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Courtyard condition manifests and image copies.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "courtyard.yaml"))
    parser.add_argument("--images", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--conditions", default="clean,blur,limited_views,degraded_matches")
    parser.add_argument("--blur-sigma", type=float, default=3.0)
    parser.add_argument("--keep-every", type=int, default=3)
    parser.add_argument("--limit-count", type=int, default=None)
    parser.add_argument("--limit-mode", default="every", choices=["every", "contiguous"])
    parser.add_argument("--drop-fraction", type=float, default=0.5)
    parser.add_argument("--input-database", default=None,
                        help="Base database for degraded_matches corruption.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="hardlink", choices=["hardlink", "copy", "symlink"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    images_dir = Path(args.images) if args.images else cfg_path(config, "dataset.images_dir")
    out_root = Path(args.out) if args.out else cfg_path(config, "dataset.subsets_root")
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    valid = {"clean", "blur", "limited_views", "degraded_matches"}
    invalid = [c for c in conditions if c not in valid]
    if invalid:
        print(f"ERROR: invalid conditions {invalid}", file=sys.stderr)
        return 2
    print(f"Images:    {images_dir}")
    print(f"Output:    {out_root}")
    print(f"Conditions: {conditions}")
    if not images_dir.is_dir():
        print(f"\nERROR: image directory missing: {images_dir}\n"
              "Run scripts/prepare_courtyard.py --extract first.", file=sys.stderr)
        return 2

    manifests = []
    for condition in conditions:
        print(f"\n[{condition}]")
        try:
            manifest = create_subset(
                images_dir, out_root, condition,
                max_images=args.max_images,
                blur_sigma=args.blur_sigma,
                keep_every=args.keep_every,
                limit_count=args.limit_count,
                limit_mode=args.limit_mode,
                drop_fraction=args.drop_fraction,
                seed=args.seed,
                input_database=args.input_database,
                mode=args.mode,
                dry_run=args.dry_run,
            )
        except ActionableError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        manifests.append(manifest)
        print(f"  images: {manifest['num_images']}")
        print(f"  images_dir: {manifest['images_dir']}")
        if condition == "degraded_matches" and not manifest.get("input_database"):
            print("  NOTE: no --input-database; manifest records the corruption instructions.")
        if args.dry_run:
            print("  dry-run: no files written")

    if not args.dry_run and manifests:
        summary = pd.DataFrame([{
            "condition": m["condition"],
            "num_images": m["num_images"],
            "images_dir": m["images_dir"],
            "input_database": m.get("input_database"),
            "manifest": str(Path(out_root) / "manifests" / f"{m['condition']}.json"),
        } for m in manifests])
        out_root.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_root / "manifests.csv", index=False)
        dump_json({"manifests": manifests}, out_root / "manifests.json")
        print(f"\nSummary: {out_root / 'manifests.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
