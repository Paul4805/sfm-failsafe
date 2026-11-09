#!/usr/bin/env python
"""Run the baseline COLMAP pipeline: extraction -> matching -> mapper -> analysis."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from _bootstrap import REPO_ROOT

from sfm_failsafe.colmap_runner import ColmapRunner
from sfm_failsafe.config import cfg_path, get_cfg, load_config, set_dotted
from sfm_failsafe.utils import ActionableError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline COLMAP and save database, sparse model, analysis, and summary.",
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "courtyard.yaml"))
    parser.add_argument("--images", default=None,
                        help="Image directory (default: dataset.images_dir from config).")
    parser.add_argument("--runs-root", default=None,
                        help="Override paths.runs_root.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--condition", default="clean",
                        help="Condition label stored in the run manifest.")
    parser.add_argument("--data-kind", default="real", choices=["real", "synthetic"])
    parser.add_argument("--max-images", type=int, default=None,
                        help="Deterministically keep at most N images (evenly spaced).")
    parser.add_argument("--downscale", type=float, default=1.0,
                        help="Resize staged images by this factor (0 < s <= 1).")
    parser.add_argument("--input-database", default=None,
                        help="Copy this database into the run (e.g. a degraded-matches DB).")
    parser.add_argument("--reuse-database", action="store_true",
                        help="Skip feature extraction/matching and map from the copied database.")
    parser.add_argument("--matcher", default=None,
                        choices=["exhaustive", "sequential", "vocab_tree"])
    parser.add_argument("--max-image-size", type=int, default=None,
                        help="COLMAP SiftExtraction.max_image_size.")
    parser.add_argument("--gpu", dest="gpu", action="store_true", default=None)
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the COLMAP commands without running or writing outputs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {}
    if args.matcher:
        overrides["colmap.matcher"] = args.matcher
    if args.max_image_size:
        overrides["colmap.max_image_size"] = int(args.max_image_size)
    if args.gpu is not None:
        overrides["colmap.use_gpu"] = bool(args.gpu)
    if args.runs_root:
        overrides["paths.runs_root"] = args.runs_root
    config = load_config(args.config, overrides)

    if not (0.0 < args.downscale <= 1.0):
        print("ERROR: --downscale must be in (0, 1].", file=sys.stderr)
        return 2
    images_dir = args.images or str(cfg_path(config, "dataset.images_dir"))
    runs_root = cfg_path(config, "paths.runs_root")
    condition = args.condition
    run_name = args.run_name or ("baseline" if condition == "clean" else f"baseline_{condition}")
    run_dir = runs_root / run_name

    print(f"Config:     {config['_meta']['config_path']}")
    print(f"Images:     {images_dir}")
    print(f"Run dir:    {run_dir}")
    print(f"Run:        {run_name} (condition={condition}, data_kind={args.data_kind})")
    if not args.dry_run and not Path(images_dir).is_dir():
        print(
            "\nERROR: image directory does not exist.\n"
            "  python scripts/prepare_courtyard.py --extract\n"
            "  bash scripts/download_eth3d.sh\n"
            "or pass --images /path/to/courtyard/images\n",
            file=sys.stderr,
        )
        return 2

    runner = ColmapRunner(config, dry_run=args.dry_run)
    try:
        summary = runner.run_baseline(
            images_dir=images_dir,
            run_dir=run_dir,
            max_images=args.max_images,
            downscale=args.downscale,
            input_database=args.input_database,
            reuse_database=args.reuse_database,
            condition=condition,
            data_kind=args.data_kind,
            run_name=run_name,
            overwrite=args.overwrite,
        )
    except ActionableError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1

    if args.dry_run:
        print("\nDry-run complete. No COLMAP process was executed.")
        return 0

    print("\nBaseline complete")
    print(f"  status:        {summary.get('status')}")
    print(f"  registered:    {summary.get('num_registered_images')}/{summary.get('num_input_images')}")
    print(f"  points3D:      {summary.get('num_points3D')}")
    print(f"  mean reproj:   {summary.get('mean_reproj_error')} px")
    print(f"  success label: {summary.get('success_label_name')} ({summary.get('success_label')})")
    if summary.get("failure_reasons"):
        print(f"  reasons:       {'; '.join(summary['failure_reasons'])}")
    print(f"  summary:       {run_dir / 'reconstruction_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
