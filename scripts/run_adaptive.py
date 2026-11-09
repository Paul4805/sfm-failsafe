#!/usr/bin/env python
"""Reliability weighting + weighted BA/refinement comparison for one baseline run."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from _bootstrap import REPO_ROOT

from sfm_failsafe.adaptive_weighting import resolve_model_dir, run_adaptive
from sfm_failsafe.config import cfg_path, get_cfg, load_config
from sfm_failsafe.utils import ActionableError, load_json, setup_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline COLMAP with reliability-weighted post-mapper refinement."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "courtyard.yaml"))
    parser.add_argument("--run-dir", required=True,
                        help="Baseline run directory containing model/, tracks.csv, summary.")
    parser.add_argument("--out", default=None,
                        help="Output dir (default: outputs/experiments/<run>/adaptive).")
    parser.add_argument("--modes", default="uniform,length,full",
                        help="Comma-separated: uniform,length,full.")
    parser.add_argument("--engine", default="auto",
                        choices=["auto", "scipy", "pyceres", "colmap_filtered"])
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.max_points:
        config.setdefault("weighted_ba", {})["max_points"] = int(args.max_points)
    if args.max_iter:
        config.setdefault("weighted_ba", {})["max_iterations"] = int(args.max_iter)
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "reconstruction_summary.json"
    summary = load_json(summary_path, {}) if summary_path.exists() else {}
    out_dir = Path(args.out) if args.out else (
        cfg_path(config, "paths.experiments_root") / run_dir.name / "adaptive"
    )
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in ("uniform", "length", "full")]
    if invalid:
        print(f"ERROR: invalid modes: {invalid}", file=sys.stderr)
        return 2

    try:
        model_dir = resolve_model_dir(run_dir, summary,
                                      get_cfg(config, "colmap.executable", "colmap"))
    except ActionableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Run:      {run_dir}")
    print(f"Model:    {model_dir}")
    print(f"Output:   {out_dir}")
    print(f"Modes:    {modes}")
    print(f"Engine:   {args.engine}")
    print(f"Weighted BA max points: {config['weighted_ba'].get('max_points')}")
    if args.dry_run:
        print("\nDry-run: would compute reliability weights, write track_weights.csv, "
              "refine the same initial model with each mode, and write comparison.csv/md. "
              "No files were written.")
        return 0
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        print(f"ERROR: output directory is not empty: {out_dir}\nUse --overwrite.", file=sys.stderr)
        return 2

    logger = setup_logger("sfm_failsafe.adaptive")
    try:
        payload = run_adaptive(
            run_dir, out_dir, config, modes=modes, engine=args.engine, logger=logger,
        )
    except ActionableError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1

    print("\nAdaptive comparison complete")
    for row in payload["variants"]:
        print(f"  {str(row['variant']):10s} points={row['num_points3D']} "
              f"mean_reproj={row['mean_reproj_error']} "
              f"track_median={row['track_length_median']} "
              f"label={row['success_label']} engine={row.get('engine')}")
    print(f"  comparison: {out_dir / 'comparison.csv'}")
    print(f"  weights:    {out_dir / 'track_weights.csv'}")
    print(f"  summary:    {out_dir / 'adaptive_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
