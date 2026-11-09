#!/usr/bin/env python
"""Build outputs/REPORT.md from all saved experiment artifacts."""

from __future__ import annotations

import argparse
import sys

from _bootstrap import REPO_ROOT

from sfm_failsafe.config import cfg_path, load_config
from sfm_failsafe.report import generate_report
from sfm_failsafe.utils import ActionableError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a detailed report from saved outputs.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "courtyard.yaml"))
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="Report on the synthetic demo runs instead of the real runs.")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Do not include the synthetic validation section.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.synthetic:
        runs_root = REPO_ROOT / "outputs" / "demo_runs"
        experiments_root = REPO_ROOT / "outputs" / "experiments_synthetic"
        out = REPO_ROOT / "outputs" / "REPORT_SYNTHETIC.md"
    else:
        runs_root = cfg_path(config, "paths.runs_root")
        experiments_root = cfg_path(config, "paths.experiments_root")
        out = REPO_ROOT / "outputs" / "REPORT.md"
    if args.runs_root:
        runs_root = args.runs_root
    if args.experiments_root:
        experiments_root = args.experiments_root
    if args.out:
        out = args.out
    try:
        path = generate_report(
            config, runs_root, experiments_root, out,
            synthetic_runs_root=(None if args.no_synthetic else
                                 (runs_root if args.synthetic
                                  else REPO_ROOT / "outputs" / "demo_runs")),
            synthetic_models_root=REPO_ROOT / "outputs" / "failure_model_synthetic",
            synthetic_experiments_root=REPO_ROOT / "outputs" / "experiments_synthetic",
            title="SfM-FailSafe synthetic pipeline validation"
            if args.synthetic else "SfM-FailSafe experiment report",
            synthetic_only=bool(args.synthetic),
        )
    except ActionableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Report written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
