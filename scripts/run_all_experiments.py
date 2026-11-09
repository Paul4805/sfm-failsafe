#!/usr/bin/env python
"""Run the full experiment matrix and write outputs/REPORT.md.

Real mode uses COLMAP + ETH3D.  ``--synthetic`` runs the clearly-labelled
synthetic validation pipeline (simulated mapper) when real COLMAP/dataset runs
are unavailable.  The report always records which mode produced each number.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from _bootstrap import REPO_ROOT

from sfm_failsafe import evaluation, failure_model
from sfm_failsafe.adaptive_weighting import run_adaptive
from sfm_failsafe.colmap_runner import ColmapRunner
from sfm_failsafe.config import cfg_path, get_cfg, load_config
from sfm_failsafe.dataset import create_subset
from sfm_failsafe.synthetic import generate_demo_runs
from sfm_failsafe.utils import (ActionableError, dump_json, ensure_dir, format_seconds,
                                load_json, setup_logger)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline, training, adaptive comparison, and report.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "courtyard.yaml"))
    parser.add_argument("--synthetic", action="store_true",
                        help="Run the labelled synthetic validation pipeline instead of COLMAP/ETH3D.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit images for the clean baseline (smoke test).")
    parser.add_argument("--subset-max-images", type=int, default=12,
                        help="Image limit for blur/limited subsets (real mode).")
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument("--conditions", default="clean,blur,limited_views,degraded_matches")
    parser.add_argument("--seeds", default="0,1", help="Synthetic demo seeds.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-adaptive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }
    try:
        env["packages"] = subprocess.run(
            [sys.executable, "-c",
             "import numpy,pandas,sklearn,matplotlib,cv2,scipy;"
             "print('numpy',numpy.__version__);print('pandas',pandas.__version__);"
             "print('scikit-learn',sklearn.__version__);print('matplotlib',matplotlib.__version__);"
             "print('opencv',cv2.__version__);print('scipy',scipy.__version__)"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip().splitlines()
    except Exception:
        env["packages"] = []
    return env


def _summary_rows(run_dirs: list[Path], config) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        path = run_dir / "reconstruction_summary.json"
        if not path.exists():
            continue
        summary = load_json(path, {})
        rows.append(evaluation.comparison_row(
            run_dir.name, summary, config,
            runtime_seconds=summary.get("runtime_seconds"),
            extra={
                "condition": summary.get("condition", "unknown"),
                "data_kind": summary.get("data_kind", "real"),
                "status": summary.get("status"),
            },
        ))
    return pd.DataFrame(rows)


def _write_report(path: Path, config, mode: str, runs: pd.DataFrame,
                  training: dict, adaptive: dict, blockers: list[str],
                  commands: list[str], environment: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# SfM-FailSafe experiment report")
    lines.append("")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Mode: **{mode}**")
    if mode == "synthetic":
        lines.append("")
        lines.append("> **WARNING: synthetic pipeline validation.** These runs use a simulated "
                     "mapper on a generated sphere scene. They are not COLMAP or ETH3D results "
                     "and must not be reported as scientific reconstruction performance.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Python: {environment.get('python')}")
    lines.append(f"- Platform: {environment.get('platform')}")
    for package in environment.get("packages", []):
        lines.append(f"- {package}")
    if blockers:
        lines.append("")
        lines.append("## Blockers / not run")
        lines.append("")
        for blocker in blockers:
            lines.append(f"- {blocker}")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    lines.append("```")
    for command in commands:
        lines.append(command)
    lines.append("```")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    if runs.empty:
        lines.append("_No runs completed._")
    else:
        display = runs.copy()
        columns = [c for c in [
            "variant", "condition", "status", "num_registered_images",
            "num_input_images", "num_points3D", "mean_reproj_error",
            "track_length_median", "success_label", "runtime_seconds",
        ] if c in display.columns]
        lines.append(evaluation.table_to_markdown(display[columns]))
    lines.append("")
    lines.append("## Failure prediction")
    lines.append("")
    if not training:
        lines.append("_Not run._")
    for model_mode, payload in training.items():
        lines.append(f"### {model_mode} mode")
        lines.append("")
        if payload.get("status") != "ok":
            lines.append(f"Not trained: {payload.get('error') or payload.get('message')}")
        else:
            best = payload.get("best_metrics", {})
            lines.append(f"- Best model: `{payload.get('best_model')}`")
            lines.append(f"- Runs: {payload.get('n_runs')} (classes: {payload.get('class_counts')})")
            lines.append(f"- Evaluation: `{payload.get('split', {}).get('evaluation_mode')}`")
            lines.append(f"- Accuracy: {best.get('accuracy')}")
            lines.append(f"- Precision: {best.get('precision')}")
            lines.append(f"- Recall: {best.get('recall')}")
            lines.append(f"- F1: {best.get('f1')}")
            lines.append(f"- ROC-AUC: {best.get('roc_auc')}")
            lines.append(f"- Preliminary: {payload.get('preliminary')}")
            if payload.get("preliminary"):
                lines.append("- Interpretation: pipeline validation only; too few runs/conditions.")
            lines.append(f"- Metrics: `{payload.get('model_path')}`")
        lines.append("")
    lines.append("## Baseline vs adaptive refinement")
    lines.append("")
    if not adaptive:
        lines.append("_Not run._")
    else:
        comparison_path = Path(adaptive.get("outputs", {}).get("comparison_csv", ""))
        if comparison_path.exists():
            try:
                table = pd.read_csv(comparison_path)
                lines.append(evaluation.table_to_markdown(table))
            except Exception:
                lines.append(f"Comparison written to `{comparison_path}`")
        else:
            lines.append(f"Comparison written to `{comparison_path}`")
        lines.append("")
        lines.append("Limitations:")
        for note in adaptive.get("limitations", []):
            lines.append(f"- {note}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Adaptive weighting is a **post-mapper refinement** of the same COLMAP model, "
                 "not integrated into COLMAP's incremental mapper.")
    lines.append("- Intrinsics are fixed in the scipy weighted BA; the gauge is fixed by holding "
                 "the first camera constant.")
    lines.append("- `scipy` refinement only optimizes selected tracks (default max 1500 points) "
                 "and does not re-triangulate or add tracks.")
    lines.append("- Failure labels are threshold-based reconstruction outcomes; they are only as "
                 "meaningful as the number and diversity of runs.")
    lines.append("- The pre-reconstruction model uses database/matches/candidate tracks; the post "
                 "model uses mapper outputs and is not leakage-free.")
    lines.append("")
    lines.append("## Next steps")
    lines.append("")
    lines.append("- Add more ETH3D scenes/conditions to increase runs per class.")
    lines.append("- Use pyceres (if a signed/available build exists) for full weighted BA.")
    lines.append("- Integrate weights into a modified mapper / retriangulation stage.")
    lines.append("- Evaluate against ETH3D ground-truth laser scans (not only SfM internal metrics).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_synthetic(config, args, runs_root: Path, experiments_root: Path,
                   commands: list[str], logger) -> tuple[pd.DataFrame, dict, dict, list[str]]:
    blockers: list[str] = []
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    commands.append(f"python scripts/make_demo_runs.py --out {runs_root} --seeds {args.seeds}")
    demo = generate_demo_runs(
        runs_root, config, seeds=seeds, overwrite=args.overwrite, logger=logger,
    )
    dump_json({"data_kind": "synthetic", "runs": demo}, runs_root / "_demo_summary.json")
    run_dirs = [Path(row["run_dir"]) for row in demo]
    runs = _summary_rows(run_dirs, config)
    training: dict = {}
    if not args.skip_training:
        for mode in ("pre", "post"):
            model_out = REPO_ROOT / "outputs" / "failure_model_synthetic" / mode
            commands.append(
                f"python scripts/train_failure_model.py --runs-root {runs_root} "
                f"--mode {mode} --out {model_out}"
            )
            try:
                training[mode] = failure_model.train_failure_model(
                    runs_root, model_out, config, mode=mode, logger=logger,
                )
            except ActionableError as exc:
                training[mode] = {"status": "error", "error": str(exc)}
            except Exception as exc:
                training[mode] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    adaptive_payload: dict = {}
    if not args.skip_adaptive:
        clean_runs = [Path(row["run_dir"]) for row in demo
                      if row["condition"] == "clean" and row["status"] == "success"]
        if clean_runs:
            run_dir = sorted(clean_runs)[0]
            out = experiments_root / run_dir.name / "adaptive"
            commands.append(f"python scripts/run_adaptive.py --run-dir {run_dir} --out {out}")
            try:
                adaptive_payload = run_adaptive(run_dir, out, config, logger=logger)
            except Exception as exc:
                blockers.append(f"Adaptive comparison failed: {type(exc).__name__}: {exc}")
                logger.error("Adaptive comparison failed: %s", exc)
    return runs, training, adaptive_payload, blockers


def _run_real(config, args, runs_root: Path, experiments_root: Path,
              commands: list[str], logger) -> tuple[pd.DataFrame, dict, dict, list[str]]:
    blockers: list[str] = []
    images_dir = cfg_path(config, "dataset.images_dir")
    if not images_dir.is_dir():
        raise ActionableError(
            f"Dataset images not found: {images_dir}\n"
            "Run: python scripts/prepare_courtyard.py --extract"
        )
    runner = ColmapRunner(config, logger=logger)
    version = runner.check_available()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    clean_run = runs_root / "clean"
    run_dirs: list[Path] = []
    clean_summary = None
    if clean_run.exists() and (clean_run / "reconstruction_summary.json").exists() and not args.overwrite:
        logger.info("Reusing existing clean run: %s", clean_run)
        clean_summary = load_json(clean_run / "reconstruction_summary.json", {})
    else:
        commands.append(
            f"python scripts/run_baseline.py --config {args.config} --run-name clean "
            f"--condition clean --overwrite"
        )
        clean_summary = runner.run_baseline(
            images_dir=images_dir, run_dir=clean_run, max_images=args.max_images,
            downscale=args.downscale, condition="clean", data_kind="real",
            run_name="clean", overwrite=True,
        )
    run_dirs.append(clean_run)
    clean_db = clean_run / "database.db"

    subset_specs = {
        "blur": {"blur_sigma": 3.0},
        "blur_mild": {"blur_sigma": 1.5},
        "blur_severe": {"blur_sigma": 6.0},
        "limited_views": {"limit_count": 8},
        "limited_views_severe": {"limit_count": 4},
        "degraded_matches": {"drop_fraction": 0.6},
        "degraded_matches_severe": {"drop_fraction": 0.9},
    }
    for condition in conditions:
        if condition == "clean":
            continue
        if condition not in subset_specs:
            blockers.append(f"Unknown condition skipped: {condition}")
            continue
        run_dir = runs_root / condition
        if run_dir.exists() and (run_dir / "reconstruction_summary.json").exists() and not args.overwrite:
            logger.info("Reusing existing run: %s", run_dir)
            run_dirs.append(run_dir)
            continue
        spec = subset_specs[condition]
        input_db = clean_db if condition.startswith("degraded_matches") else None
        # Degraded-matching runs reuse the full clean database, so their image
        # set must match that database; blur/limited subsets are small.
        current_max = None if condition.startswith("degraded_matches") else args.subset_max_images
        manifest = create_subset(
            images_dir, cfg_path(config, "dataset.subsets_root"), condition,
            max_images=current_max, blur_sigma=float(spec.get("blur_sigma", 3.0)),
            limit_count=spec.get("limit_count"),
            drop_fraction=float(spec.get("drop_fraction", 0.5)),
            input_database=input_db, seed=0,
        )
        commands.append(
            f"python scripts/run_baseline.py --config {args.config} "
            f"--images \"{manifest['images_dir']}\" --run-name {condition} "
            f"--condition {condition}" +
            (f" --input-database \"{manifest['input_database']}\" --reuse-database"
             if condition.startswith("degraded_matches") and manifest.get("input_database")
             else "")
        )
        try:
            runner.run_baseline(
                manifest["images_dir"], run_dir,
                max_images=None,  # manifest already defines the subset
                downscale=args.downscale,
                input_database=manifest.get("input_database"),
                reuse_database=bool(condition.startswith("degraded_matches") and manifest.get("input_database")),
                condition=condition, data_kind="real", run_name=condition,
                overwrite=True,
            )
        except ActionableError as exc:
            blockers.append(f"{condition}: {exc}")
            logger.error("%s failed: %s", condition, exc)
            continue
        run_dirs.append(run_dir)

    runs = _summary_rows(run_dirs, config)
    training: dict = {}
    if not args.skip_training:
        for mode in ("pre", "post"):
            commands.append(
                f"python scripts/train_failure_model.py --runs-root {runs_root} --mode {mode} "
                f"--out {cfg_path(config, 'paths.failure_model_dir') / mode}"
            )
            try:
                training[mode] = failure_model.train_failure_model(
                    runs_root, cfg_path(config, "paths.failure_model_dir") / mode,
                    config, mode=mode, logger=logger,
                )
            except ActionableError as exc:
                training[mode] = {"status": "error", "error": str(exc)}
            except Exception as exc:
                training[mode] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    adaptive_payload: dict = {}
    if not args.skip_adaptive:
        out = experiments_root / clean_run.name / "adaptive"
        commands.append(f"python scripts/run_adaptive.py --run-dir {clean_run} --out {out}")
        try:
            adaptive_payload = run_adaptive(clean_run, out, config, logger=logger)
        except Exception as exc:
            blockers.append(f"Adaptive comparison failed: {type(exc).__name__}: {exc}")
            logger.error("Adaptive comparison failed: %s", exc)
    if clean_summary and clean_summary.get("status") != "success":
        blockers.append(f"Clean COLMAP baseline failed: {clean_summary.get('error', 'unknown')}")
    return runs, training, adaptive_payload, blockers


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {}
    if args.downscale != 1.0:
        overrides["dataset.downscale"] = args.downscale
    config = load_config(args.config, overrides)
    experiments_root = cfg_path(config, "paths.experiments_root")
    runs_root = cfg_path(config, "paths.runs_root")
    if args.synthetic:
        runs_root = REPO_ROOT / "outputs" / "demo_runs"
        experiments_root = REPO_ROOT / "outputs" / "experiments_synthetic"
    report_path = cfg_path(config, "paths.report")
    logger = setup_logger("sfm_failsafe.experiments")
    commands = [f"python scripts/run_all_experiments.py {' '.join(sys.argv[1:])}"]
    mode = "synthetic" if args.synthetic else "real"
    if args.synthetic:
        report_path = REPO_ROOT / "outputs" / "REPORT_SYNTHETIC.md"

    print(f"Mode: {mode}")
    print(f"Runs root: {runs_root}")
    print(f"Experiments root: {experiments_root}")
    print(f"Report: {report_path}")
    if args.dry_run:
        print("\nDry-run plan:")
        if args.synthetic:
            print("  1. generate synthetic demo runs (simulated mapper)")
            print("  2. train pre/post failure models")
            print("  3. adaptive comparison on a clean synthetic run")
            print("  4. write report")
        else:
            print("  1. prepare Courtyard (must already be done)")
            print("  2. COLMAP baseline for clean + selected conditions")
            print("  3. train pre/post failure models")
            print("  4. adaptive comparison on clean run")
            print("  5. write report")
        print("\nNo files were written.")
        return 0

    environment = _environment()
    blockers: list[str] = []
    training: dict = {}
    adaptive: dict = {}
    runs = pd.DataFrame()
    exit_code = 0
    try:
        if args.synthetic:
            runs, training, adaptive, blockers = _run_synthetic(
                config, args, runs_root, experiments_root, commands, logger)
            blockers.insert(0, "Synthetic mode: no COLMAP/ETH3D run was performed.")
        else:
            runs, training, adaptive, blockers = _run_real(
                config, args, runs_root, experiments_root, commands, logger)
    except ActionableError as exc:
        blockers.append(str(exc))
        logger.error("%s", exc)
        exit_code = 2
    except Exception as exc:
        blockers.append(f"Unhandled error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        exit_code = 1

    _write_report(report_path, config, mode, runs, training, adaptive, blockers,
                  commands, environment)
    # Also generate the detailed artifact-indexed report (overwrites/extends the
    # basic one with per-run details, weight diagnostics, and an inventory).
    try:
        from sfm_failsafe.report import generate_report
        detailed_out = (REPO_ROOT / "outputs" / ("REPORT_SYNTHETIC.md" if args.synthetic else "REPORT.md"))
        generate_report(
            config,
            runs_root,
            experiments_root,
            detailed_out,
            synthetic_runs_root=(runs_root if args.synthetic
                                 else REPO_ROOT / "outputs" / "demo_runs"),
            synthetic_models_root=REPO_ROOT / "outputs" / "failure_model_synthetic",
            synthetic_experiments_root=REPO_ROOT / "outputs" / "experiments_synthetic",
            title="SfM-FailSafe synthetic pipeline validation"
            if args.synthetic else "SfM-FailSafe experiment report",
            synthetic_only=bool(args.synthetic),
        )
        print(f"Detailed report: {detailed_out}")
        try:
            from sfm_failsafe.pdf_report import generate_pdf_report
            pdf_out = detailed_out.with_suffix(".pdf")
            generate_pdf_report(
                config, runs_root, experiments_root, pdf_out,
                synthetic_runs_root=(runs_root if args.synthetic
                                     else REPO_ROOT / "outputs" / "demo_runs"),
                synthetic_models_root=REPO_ROOT / "outputs" / "failure_model_synthetic",
                synthetic_experiments_root=REPO_ROOT / "outputs" / "experiments_synthetic",
                synthetic_only=bool(args.synthetic),
            )
            print(f"PDF report: {pdf_out}")
        except Exception as exc:
            logger.warning("PDF report generation failed: %s", exc)
    except Exception as exc:
        logger.warning("Detailed report generation failed: %s", exc)
    print(f"\nReport: {report_path}")
    if not runs.empty:
        print(f"Runs:   {len(runs)}")
        print(runs[[c for c in ["variant", "status", "num_points3D", "mean_reproj_error",
                                "success_label"] if c in runs.columns]].to_string(index=False))
    for model_mode, payload in training.items():
        if payload.get("status") == "ok":
            print(f"Model {model_mode}: {payload.get('best_model')} "
                  f"F1={payload.get('best_metrics', {}).get('f1')} "
                  f"preliminary={payload.get('preliminary')}")
        else:
            print(f"Model {model_mode}: not trained ({payload.get('error') or payload.get('message')})")
    if adaptive:
        print(f"Adaptive: {adaptive.get('outputs', {}).get('comparison_csv')}")
    for blocker in blockers:
        print(f"BLOCKER: {blocker}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
