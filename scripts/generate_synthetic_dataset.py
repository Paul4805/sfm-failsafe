#!/usr/bin/env python
"""Generate a tiny synthetic multi-view dataset for tests/demos (not ETH3D)."""

from __future__ import annotations

import argparse
import sys

from _bootstrap import REPO_ROOT

from sfm_failsafe.synthetic import generate_synthetic_dataset
from sfm_failsafe.utils import ActionableError, dump_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic sphere-scene dataset.")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "synthetic" / "scene"))
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--num-points", type=int, default=800)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--noise-px", type=float, default=0.4)
    parser.add_argument("--blur-sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = generate_synthetic_dataset(
            args.out, num_images=args.num_images, num_points=args.num_points,
            width=args.width, height=args.height, noise_px=args.noise_px,
            blur_sigma=args.blur_sigma, seed=args.seed,
            write_images=not args.no_images,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    dump_json({
        "data_kind": "synthetic",
        "note": "Generated fixture; not an ETH3D or COLMAP reconstruction result.",
        "manifest": result["manifest"],
    }, f"{args.out}/generation.json")
    print("Synthetic dataset created")
    print(f"  images: {args.out}/images")
    print(f"  model:  {args.out}/model")
    print(f"  db:     {args.out}/database.db")
    print(f"  points: {result['reconstruction'].num_points}")
    print("  data_kind: synthetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
