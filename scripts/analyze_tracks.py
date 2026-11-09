#!/usr/bin/env python
"""Analyze a COLMAP sparse model: tracks.csv, images.csv, pairs.csv, summary."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from _bootstrap import REPO_ROOT

from sfm_failsafe import evaluation, features, model_io, track_analysis, visualization
from sfm_failsafe.adaptive_weighting import resolve_model_dir
from sfm_failsafe.config import cfg_path, get_cfg, load_config
from sfm_failsafe.utils import ActionableError, dump_json, load_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze COLMAP tracks and reconstruction statistics.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "courtyard.yaml"))
    parser.add_argument("--run-dir", default=None,
                        help="Run directory produced by run_baseline.py.")
    parser.add_argument("--model", default=None, help="COLMAP model directory (TXT or binary).")
    parser.add_argument("--db", default=None, help="COLMAP database (for pair match statistics).")
    parser.add_argument("--images", default=None, help="Image directory (for blur scores).")
    parser.add_argument("--out", default=None, help="Output directory (default: run dir).")
    parser.add_argument("--input-image-count", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--no-pre", action="store_true",
                        help="Do not extract pre-reconstruction candidate tracks.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.max_points:
        config.setdefault("analysis", {})["max_points"] = int(args.max_points)
    run_dir = Path(args.run_dir) if args.run_dir else None
    manifest = {}
    summary = {}
    if run_dir is not None:
        manifest_path = run_dir / "run_manifest.json"
        manifest = load_json(manifest_path, {}) if manifest_path.exists() else {}
        summary_path = run_dir / "reconstruction_summary.json"
        summary = load_json(summary_path, {}) if summary_path.exists() else {}
        model_dir = resolve_model_dir(run_dir, summary,
                                      get_cfg(config, "colmap.executable", "colmap"))
        out_dir = Path(args.out) if args.out else run_dir
    else:
        if not args.model:
            print("ERROR: pass --run-dir or --model.", file=sys.stderr)
            return 2
        model_dir = Path(args.model)
        out_dir = Path(args.out) if args.out else model_dir.parent

    images_dir = args.images or manifest.get("images_dir") or get_cfg(config, "dataset.images_dir")
    db_path = args.db or manifest.get("database") or summary.get("database")
    if db_path and not Path(db_path).is_absolute() and run_dir is not None:
        db_path = str(run_dir / db_path)
    image_names = manifest.get("image_names")
    input_count = args.input_image_count or summary.get("num_input_images") or (
        len(image_names) if image_names else None
    )

    print(f"Model:  {model_dir}")
    print(f"Output: {out_dir}")
    print(f"DB:     {db_path or 'not provided (pairs.csv will use model covisibility only)'}")
    print(f"Images: {images_dir or 'not provided (blur scores will be NaN)'}")
    if args.dry_run:
        print("\nDry-run: would read the model, write tracks.csv/images.csv/pairs.csv, and "
              "reconstruction_summary.json. No files were written.")
        return 0

    try:
        model_dir = model_io.ensure_readable_model(
            model_dir, get_cfg(config, "colmap.executable", "colmap"),
            work_dir=out_dir / "model_txt",
        )
        rec = model_io.read_model(model_dir)
        analysis = track_analysis.analyze_reconstruction(
            rec, config,
            image_dir=images_dir,
            db_path=db_path,
            all_image_names=image_names,
            input_image_count=input_count,
        )
        track_analysis.write_analysis(analysis, out_dir)
        label = evaluation.label_summary(analysis["summary"], config)
        analysis["summary"].update({
            "success_label": label["label"],
            "success_label_name": label["label_name"],
            "failure_reasons": label["reasons"],
            "label_checks": label["checks"],
            "model_dir": str(model_dir),
            "data_kind": manifest.get("data_kind", analysis["summary"].get("data_kind", "real")),
        })
        if run_dir is not None:
            analysis["summary"].setdefault("run_name", run_dir.name)
            analysis["summary"].setdefault("condition", manifest.get("condition", "unknown"))
        dump_json(analysis["summary"], out_dir / "reconstruction_summary.json")
        if run_dir is not None and not args.no_pre:
            try:
                features.pre_reconstruction_features(run_dir, config, write=True)
            except ActionableError as exc:
                print(f"WARNING: pre-reconstruction features not extracted: {exc}")
        if not args.no_plots:
            visualization.plot_track_length_hist(
                analysis["tracks"], out_dir / "plots" / "track_length.png")
            visualization.plot_reprojection_hist(
                analysis["tracks"], out_dir / "plots" / "reproj_error.png")
            visualization.plot_pointcloud_projection(
                rec, out_dir / "plots" / "sparse_pointcloud.png")
    except ActionableError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1

    s = analysis["summary"]
    print("\nAnalysis complete")
    print(f"  registered:   {s.get('num_registered_images')}/{s.get('num_input_images')}")
    print(f"  points3D:     {s.get('num_points3D')}")
    print(f"  tracks:       {s.get('num_tracks')} (median length {s.get('track_length_median')})")
    print(f"  mean reproj:  {s.get('mean_reproj_error')} px")
    print(f"  label:        {s.get('success_label_name')}")
    print(f"  tracks.csv:   {out_dir / 'tracks.csv'}")
    print(f"  images.csv:   {out_dir / 'images.csv'}")
    print(f"  pairs.csv:    {out_dir / 'pairs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
