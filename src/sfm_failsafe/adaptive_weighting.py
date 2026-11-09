"""Reliability-weighted track selection and post-mapper refinement comparison."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import evaluation, model_io, reliability, track_analysis, visualization
from .config import get_cfg
from .utils import (ActionableError, dump_json, ensure_dir, load_json,
                    safe_float, setup_logger)
from .weighted_ba import (filter_tracks, refine_weighted, weights_from_dataframe)


def resolve_model_dir(run_dir: str | Path, summary: Mapping | None = None,
                      colmap_executable: str = "colmap") -> Path:
    run_dir = Path(run_dir)
    summary = summary or {}
    candidates: list[Path] = []
    configured = summary.get("model_dir")
    if configured:
        candidate = Path(str(configured))
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.extend([run_dir / candidate, Path.cwd() / candidate])
    candidates.extend([
        run_dir / "model",
        run_dir / "selected_model",
        run_dir / "sparse_selected" / "0",
        run_dir / "sparse" / "0",
    ])
    for candidate in candidates:
        if candidate.is_dir() and any((candidate / n).exists()
                                      for n in ("cameras.txt", "cameras.bin")):
            return candidate
    sparse = run_dir / "sparse"
    if sparse.is_dir():
        try:
            best, _ = model_io.select_largest_model(sparse, colmap_executable)
            return best
        except ActionableError:
            pass
    raise ActionableError(
        f"No COLMAP model found for run {run_dir}.\n"
        "Expected model_dir in reconstruction_summary.json, or model/, sparse/0/."
    )


def load_run(run_dir: str | Path, config: Mapping,
             logger=None) -> tuple[model_io.Reconstruction, pd.DataFrame, dict]:
    run_dir = Path(run_dir)
    summary_path = run_dir / "reconstruction_summary.json"
    summary = load_json(summary_path, {}) if summary_path.exists() else {}
    model_dir = resolve_model_dir(run_dir, summary,
                                  get_cfg(config, "colmap.executable", "colmap"))
    try:
        model_dir = model_io.ensure_readable_model(
            model_dir, get_cfg(config, "colmap.executable", "colmap"),
            work_dir=run_dir / "model_txt",
        )
    except ActionableError:
        raise
    rec = model_io.read_model(model_dir)
    tracks_path = run_dir / "tracks.csv"
    if tracks_path.exists():
        tracks = pd.read_csv(tracks_path)
    else:
        if logger:
            logger.info("tracks.csv missing; computing from model")
        tracks = track_analysis.compute_track_table(rec, config)
        tracks.to_csv(run_dir / "tracks.csv", index=False)
    if not summary:
        manifest = load_json(run_dir / "run_manifest.json", {})
        input_count = manifest.get("num_input_images") or rec.num_images
        summary = track_analysis.summarize_reconstruction(
            rec, tracks, input_image_count=input_count
        )
        summary["data_kind"] = manifest.get("data_kind", "real")
        dump_json(summary, summary_path)
    return rec, tracks, summary


def compute_mode_weights(tracks: pd.DataFrame, config: Mapping,
                         mode: str = "full") -> pd.DataFrame:
    if mode == "full":
        weights = reliability.score_tracks(tracks, config, length_only=False)
    elif mode == "length":
        weights = reliability.score_tracks(tracks, config, length_only=True)
    elif mode == "uniform":
        weights = tracks.copy()
        weights["reliability"] = 1.0
        weights["weight"] = 1.0
    else:
        raise ActionableError(f"Unknown weighting mode: {mode} (use full|length|uniform)")
    weights["weight_mode"] = mode
    return weights


def _write_model(rec: model_io.Reconstruction, out_dir: Path) -> Path:
    path = ensure_dir(out_dir)
    model_io.write_model(rec, path)
    return path


def run_adaptive(run_dir: str | Path, out_dir: str | Path, config: Mapping,
                 modes: Sequence[str] = ("uniform", "length", "full"),
                 engine: str = "auto", logger=None) -> dict:
    """Compare unmodified baseline vs weighted post-mapper refinement.

    All variants start from the *same* initial COLMAP model.  No claim is made
    that the weights are integrated into COLMAP's incremental mapper.
    """
    started = time.perf_counter()
    run_dir, out_dir = Path(run_dir), ensure_dir(Path(out_dir))
    logger = logger or setup_logger("sfm_failsafe.adaptive")
    rec, tracks, baseline_summary = load_run(run_dir, config, logger=logger)
    input_count = int(baseline_summary.get("num_input_images") or rec.num_images)
    data_kind = str(baseline_summary.get("data_kind", "real"))

    rows = [evaluation.comparison_row(
        "baseline", baseline_summary, config,
        runtime_seconds=baseline_summary.get("runtime_seconds"),
        extra={"engine": "colmap_baseline", "weighted": False},
    )]
    refinements: dict[str, dict] = {}
    weight_tables: dict[str, pd.DataFrame] = {}
    models: dict[str, str] = {}

    for mode in modes:
        variant_started = time.perf_counter()
        logger.info("Weighting mode: %s", mode)
        weights = compute_mode_weights(tracks, config, mode)
        weight_tables[mode] = weights
        weight_path = out_dir / ("track_weights.csv" if mode == "full"
                                 else f"track_weights_{mode}.csv")
        weights.to_csv(weight_path, index=False)
        weight_map = weights_from_dataframe(weights)

        if engine == "colmap_filtered":
            # Approximate path: filter tracks, then run COLMAP BA (if present).
            weight_threshold = float(get_cfg(config, "weighted_ba.filter_weight_threshold", 0.1))
            filtered, filter_report = filter_tracks(
                rec, weight_map, weight_threshold,
                int(get_cfg(config, "weighting.min_track_length", 2)),
            )
            from .colmap_runner import run_bundle_adjuster

            model_path = out_dir / "models" / f"{mode}_filtered"
            _write_model(filtered, model_path)
            refined, ba_report = run_bundle_adjuster(
                model_path, out_dir / "models" / f"{mode}", config,
                dry_run=False,
            )
            report = {
                "status": "ok" if refined is not None else "unavailable",
                "engine_requested": "colmap",
                "engine_used": "colmap_bundle_adjuster_after_filtering",
                "approximation": True,
                "filter": filter_report,
                "colmap": ba_report,
                "runtime_seconds": time.perf_counter() - variant_started,
            }
            if refined is None:
                refined = filtered
        else:
            refined, report = refine_weighted(rec, weight_map, config, engine=engine)

        models[mode] = str(_write_model(refined, out_dir / "models" / mode))
        refined_tracks = track_analysis.compute_track_table(refined, config)
        refined_tracks.to_csv(out_dir / f"tracks_{mode}.csv", index=False)
        refined_summary = track_analysis.summarize_reconstruction(
            refined, refined_tracks, input_image_count=input_count
        )
        refined_summary["weight_mean"] = safe_float(weights["weight"].mean())
        refined_summary["reliability_mean"] = safe_float(weights["reliability"].mean())
        refined_summary["data_kind"] = data_kind
        report["summary"] = refined_summary
        report["weight_summary"] = reliability.weight_summary(weights)
        refinements[mode] = report
        rows.append(evaluation.comparison_row(
            mode, refined_summary, config,
            runtime_seconds=report.get("runtime_seconds"),
            extra={
                "engine": report.get("engine_used", engine),
                "weighted": mode != "uniform",
                "approximation": bool(report.get("approximation", False)),
            },
        ))
        logger.info(
            "%s: points %s -> %s, mean reproj %.4f -> %.4f, %s",
            mode, baseline_summary.get("num_points3D"),
            refined_summary.get("num_points3D"),
            safe_float(baseline_summary.get("mean_reproj_error")),
            safe_float(refined_summary.get("mean_reproj_error")),
            report.get("status"),
        )
        dump_json(refined_summary, out_dir / f"reconstruction_summary_{mode}.json")

    table = evaluation.add_deltas(evaluation.comparison_table(rows), "baseline")
    table.to_csv(out_dir / "comparison.csv", index=False)
    notes = [
        "Baseline is the unmodified COLMAP sparse model.",
        "Adaptive variants are post-mapper refinements from the same initial model; "
        "they are not integrated into COLMAP's incremental mapper.",
        "Uniform refinement is a control: same BA, all weights = 1.",
        "Length-only is an ablation of the full reliability score.",
        "Selected tracks receive the weighted robust BA; unselected tracks are "
        "re-triangulated against the refined cameras when the point cap excludes any.",
        "errors_before/after in refinement diagnostics cover selected observations "
        "with the robust objective; the comparison table uses full-model track "
        "mean/median reprojection error.",
        f"data_kind={data_kind}",
    ]
    evaluation.write_comparison_markdown(
        table, out_dir / "comparison.md",
        title=f"Baseline vs adaptive track weighting ({run_dir.name})",
        notes=notes,
    )

    # Plots (headless matplotlib).
    try:
        visualization.plot_weight_distribution(
            weight_tables["full"], out_dir / "plots" / "weight_distribution.png",
            title=f"Track weights: {run_dir.name}",
        )
        visualization.plot_comparison_bars(
            table, out_dir / "plots" / "comparison_bars.png"
        )
        visualization.plot_track_length_hist(
            tracks, out_dir / "plots" / "track_length_baseline.png",
            title="Baseline track length distribution",
        )
        if "full" in weight_tables:
            visualization.plot_reprojection_hist(
                tracks, out_dir / "plots" / "reproj_error_baseline.png",
                title="Baseline track mean reprojection error",
            )
        visualization.plot_reliability_scatter(
            weight_tables["full"], out_dir / "plots" / "reliability_scatter.png"
        )
    except Exception as exc:  # plots must never break the experiment
        logger.warning("Plot generation failed: %s", exc)

    payload = {
        "run_name": run_dir.name,
        "data_kind": data_kind,
        "baseline_model": str(resolve_model_dir(
            run_dir, baseline_summary, get_cfg(config, "colmap.executable", "colmap"))),
        "model_dir": str(run_dir),
        "modes": list(modes),
        "engine_requested": engine,
        "limitations": notes,
        "weight_summaries": {
            mode: reliability.weight_summary(table_df)
            for mode, table_df in weight_tables.items()
        },
        "refinements": refinements,
        "variants": table.to_dict(orient="records"),
        "runtime_seconds": time.perf_counter() - started,
        "outputs": {
            "comparison_csv": str(out_dir / "comparison.csv"),
            "comparison_md": str(out_dir / "comparison.md"),
            "models": models,
        },
    }
    dump_json(payload, out_dir / "adaptive_summary.json")
    logger.info("Adaptive comparison written to %s", out_dir)
    return payload
