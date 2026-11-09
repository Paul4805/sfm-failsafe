#!/usr/bin/env python
"""Generate synthetic runs with a simulated mapper (pipeline validation only)."""

from __future__ import annotations

import argparse
import sys

from _bootstrap import REPO_ROOT

from sfm_failsafe.config import cfg_path, load_config
from sfm_failsafe.synthetic import generate_demo_runs
from sfm_failsafe.utils import dump_json, setup_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clearly-labelled synthetic runs for pipeline validation."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "demo_runs"))
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--num-points", type=int, default=3000)
    parser.add_argument("--min-links", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    print("WARNING: this generates SYNTHETIC runs with a simulated mapper.")
    print("         These are not COLMAP/ETH3D results and must not be reported as such.")
    print(f"Output: {args.out}")
    print(f"Seeds: {seeds}")
    if args.dry_run:
        print("Dry-run: would generate 8 variants x seeds; no files written.")
        return 0
    logger = setup_logger("sfm_failsafe.demo")
    created = generate_demo_runs(
        args.out, config, seeds=seeds, num_images=args.num_images,
        num_points=args.num_points, min_links=args.min_links,
        overwrite=args.overwrite, logger=logger,
    )
    dump_json({
        "data_kind": "synthetic",
        "engine": "synthetic_simulated_mapper",
        "runs": created,
    }, f"{args.out}/_demo_summary.json")
    import pandas as pd

    pd.DataFrame(created).to_csv(f"{args.out}/_demo_summary.csv", index=False)
    print(f"\nCreated {len(created)} synthetic runs")
    labels = pd.DataFrame(created)["label_name"].value_counts().to_dict()
    print(f"Labels: {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
