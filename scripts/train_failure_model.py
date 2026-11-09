#!/usr/bin/env python
"""Train and evaluate a failure-prediction model over reconstruction runs."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from _bootstrap import REPO_ROOT

from sfm_failsafe import features, failure_model
from sfm_failsafe.config import cfg_path, load_config
from sfm_failsafe.utils import ActionableError, load_json, setup_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pre- or post-reconstruction failure model.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "courtyard.yaml"))
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--out", default=None,
                        help="Output directory (default: paths.failure_model_dir).")
    parser.add_argument("--mode", default="pre", choices=["pre", "post"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    runs_root = Path(args.runs_root) if args.runs_root else cfg_path(config, "paths.runs_root")
    out_dir = Path(args.out) if args.out else cfg_path(config, "paths.failure_model_dir")
    logger = setup_logger("sfm_failsafe.failure")

    run_dirs = features.iter_run_dirs(runs_root)
    print(f"Runs root: {runs_root}")
    print(f"Output:    {out_dir}")
    print(f"Mode:      {args.mode} "
          f"({'leakage-free database features' if args.mode == 'pre' else 'NOT leakage-free'})")
    print(f"Runs found: {len(run_dirs)}")
    for run_dir in run_dirs:
        feature_file = run_dir / ("features_pre.json" if args.mode == "pre" else "features_post.json")
        db_file = run_dir / "database.db"
        print(f"  - {run_dir.name}: "
              f"{'features cached' if feature_file.exists() else 'features will be extracted'}; "
              f"db={'yes' if db_file.exists() else 'NO'}")
    if args.dry_run:
        print("\nDry-run complete. No model was trained and no files were written.")
        return 0
    if not run_dirs:
        print("\nERROR: no runs found. Run scripts/run_baseline.py first.", file=sys.stderr)
        return 2

    try:
        metrics = failure_model.train_failure_model(
            runs_root, out_dir, config, mode=args.mode, logger=logger,
        )
    except ActionableError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1

    best = metrics.get("best_metrics", {})
    print("\nTraining complete")
    print(f"  best model:  {metrics.get('best_model')}")
    print(f"  evaluation:  {metrics.get('split', {}).get('evaluation_mode')}")
    print(f"  n runs:      {metrics.get('n_runs')}")
    print(f"  accuracy:    {best.get('accuracy')}")
    print(f"  precision:   {best.get('precision')}")
    print(f"  recall:      {best.get('recall')}")
    print(f"  f1:          {best.get('f1')}")
    print(f"  roc_auc:     {best.get('roc_auc')}")
    print(f"  preliminary: {metrics.get('preliminary')}")
    print(f"  metrics:     {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
