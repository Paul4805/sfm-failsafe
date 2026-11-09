"""Generate the detailed experiment report from saved artifacts only.

The report never recomputes or invents metrics: every number is read from a
JSON/CSV written by the pipeline.  Synthetic runs are explicitly separated
from real COLMAP/ETH3D runs.
"""

from __future__ import annotations

import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import evaluation
from .config import get_cfg
from .utils import load_json, safe_float


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _link(path: Path, base: Path, label: str | None = None) -> str:
    if not path.exists():
        return "n/a"
    return f"[{label or path.name}]({_rel(path, base)})"


def _float(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _size(path: Path) -> str:
    if not path.exists():
        return "-"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _run_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [p.parent for p in sorted(root.rglob("reconstruction_summary.json"))
            if "adaptive" not in p.parent.parts and "models" not in p.parent.parts]


def _runs_table(run_dirs: Sequence[Path], config: Mapping) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        summary = load_json(run_dir / "reconstruction_summary.json", {})
        manifest = load_json(run_dir / "run_manifest.json", {})
        rows.append({
            "run": run_dir.name,
            "condition": manifest.get("condition", summary.get("condition", "unknown")),
            "data": manifest.get("data_kind", summary.get("data_kind", "real")),
            "status": summary.get("status"),
            "input": summary.get("num_input_images"),
            "registered": summary.get("num_registered_images"),
            "ratio": summary.get("registration_ratio"),
            "points": summary.get("num_points3D"),
            "mean_err": summary.get("mean_reproj_error"),
            "median_err": summary.get("median_reproj_error"),
            "track_med": summary.get("track_length_median"),
            "label": summary.get("success_label_name"),
            "reasons": "; ".join(summary.get("failure_reasons", []) or []),
            "seconds": summary.get("runtime_seconds"),
        })
    return pd.DataFrame(rows)


def _run_detail(run_dir: Path, base: Path) -> list[str]:
    summary = load_json(run_dir / "reconstruction_summary.json", {})
    manifest = load_json(run_dir / "run_manifest.json", {})
    lines = [f"### `{run_dir.name}`", ""]
    data_kind = manifest.get("data_kind", summary.get("data_kind", "real"))
    engine = manifest.get("engine", summary.get("engine", "colmap_cli"))
    lines.append(f"- Condition: `{manifest.get('condition', summary.get('condition', 'unknown'))}`")
    lines.append(f"- Data kind: `{data_kind}`")
    lines.append(f"- Engine: `{engine}`")
    lines.append(f"- Status: `{summary.get('status')}`")
    lines.append(f"- Registered: {summary.get('num_registered_images')}/{summary.get('num_input_images')} "
                 f"({_float(summary.get('registration_ratio'), 3)})")
    lines.append(f"- 3D points: {summary.get('num_points3D')}")
    lines.append(f"- Reprojection error mean/median: "
                 f"{_float(summary.get('mean_reproj_error'))} / "
                 f"{_float(summary.get('median_reproj_error'))} px")
    lines.append(f"- Track length mean/median/p90: "
                 f"{_float(summary.get('track_length_mean'), 2)} / "
                 f"{_float(summary.get('track_length_median'), 2)} / "
                 f"{_float(summary.get('track_length_p90'), 2)}")
    lines.append(f"- Success label: **{summary.get('success_label_name')}**")
    if summary.get("failure_reasons"):
        lines.append(f"- Failure reasons: {'; '.join(summary['failure_reasons'])}")
    if manifest.get("colmap_version"):
        lines.append(f"- COLMAP: {manifest['colmap_version']}")
    if manifest.get("num_input_images"):
        lines.append(f"- Input images: {manifest['num_input_images']}")
    timings = summary.get("timings") or {}
    if timings:
        timing_text = ", ".join(f"{k}={_float(v, 2)}s" for k, v in timings.items())
        lines.append(f"- Timings: {timing_text}")
    if summary.get("runtime_seconds"):
        lines.append(f"- Total runtime: {_float(summary.get('runtime_seconds'), 1)} s")
    if summary.get("error"):
        lines.append(f"- Error: `{summary['error']}`")
    lines.append("")
    artifacts = [
        ("database", run_dir / "database.db"),
        ("model cameras", run_dir / "model" / "cameras.txt"),
        ("model images", run_dir / "model" / "images.txt"),
        ("model points", run_dir / "model" / "points3D.txt"),
        ("tracks.csv", run_dir / "tracks.csv"),
        ("images.csv", run_dir / "images.csv"),
        ("pairs.csv", run_dir / "pairs.csv"),
        ("candidate_tracks.csv", run_dir / "candidate_tracks.csv"),
        ("features_pre.json", run_dir / "features_pre.json"),
        ("summary", run_dir / "reconstruction_summary.json"),
        ("command log", run_dir / "logs" / "commands.log"),
    ]
    rows = []
    for label, path in artifacts:
        rows.append({"artifact": label, "path": _rel(path, base) if path.exists() else "-",
                     "size": _size(path)})
    lines.append(evaluation.table_to_markdown(pd.DataFrame(rows)))
    plot_dir = run_dir / "plots"
    if plot_dir.is_dir():
        plots = sorted(plot_dir.glob("*.png"))
        if plots:
            lines.append("")
            lines.append("Plots: " + ", ".join(_link(p, base) for p in plots))
    lines.append("")
    return lines


def _training_section(metrics_path: Path, base: Path, title: str) -> list[str]:
    lines = [f"### {title}", ""]
    if not metrics_path.exists():
        lines.append("_Not run / no metrics file._")
        lines.append("")
        return lines
    metrics = load_json(metrics_path, {})
    if metrics.get("status") != "ok":
        lines.append(f"**Not trained:** {metrics.get('error') or metrics.get('message')}")
        lines.append("")
        if metrics.get("class_counts"):
            lines.append(f"Class counts: `{metrics['class_counts']}`")
            lines.append("")
        return lines
    lines.append(f"- Mode: `{metrics.get('mode')}`")
    lines.append(f"- Leakage-free: `{metrics.get('leakage_free')}`")
    lines.append(f"- Runs: {metrics.get('n_runs')} (classes {metrics.get('class_counts')})")
    lines.append(f"- Features: {metrics.get('n_features')}")
    split = metrics.get("split", {})
    lines.append(f"- Split: `{split.get('evaluation_mode')}` "
                 f"(train {len(metrics.get('train_runs', []))}, "
                 f"test {len(metrics.get('test_runs', []))})")
    lines.append(f"- Best model: `{metrics.get('best_model')}`")
    lines.append(f"- Threshold (tuned on train): {_float(metrics.get('threshold'), 4)}")
    lines.append(f"- Preliminary: `{metrics.get('preliminary')}`")
    if metrics.get("preliminary_reason"):
        lines.append(f"- Preliminary reason: {metrics['preliminary_reason']}")
    lines.append("")
    rows = []
    for name, payload in (metrics.get("models") or {}).items():
        row = payload.get("metrics", {})
        rows.append({
            "model": name,
            "accuracy": row.get("accuracy"),
            "precision": row.get("precision"),
            "recall": row.get("recall"),
            "f1": row.get("f1"),
            "roc_auc": row.get("roc_auc"),
            "avg_precision": row.get("average_precision"),
            "threshold": row.get("threshold"),
            "tp": row.get("tp"), "tn": row.get("tn"),
            "fp": row.get("fp"), "fn": row.get("fn"),
        })
    lines.append(evaluation.table_to_markdown(pd.DataFrame(rows)))
    lines.append("")
    oof = metrics.get("leave_one_condition_out") or {}
    if oof.get("status") == "ok":
        lines.append("Leave-one-condition-out (random forest):")
        lines.append("")
        lines.append(evaluation.table_to_markdown(pd.DataFrame([{
            k: v for k, v in oof.get("metrics", {}).items()
            if k not in ("confusion_matrix",)
        }])))
        lines.append("")
    importance_path = metrics_path.parent / "feature_importances.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(15)
        lines.append("Top features:")
        lines.append("")
        lines.append(evaluation.table_to_markdown(importance))
        lines.append("")
    plot_dir = metrics_path.parent / "plots"
    if plot_dir.is_dir():
        plots = sorted(plot_dir.glob("*.png"))
        if plots:
            lines.append("Plots: " + ", ".join(_link(p, base) for p in plots))
            lines.append("")
    dataset_file = f"runs_dataset_{metrics.get('mode')}.csv"
    lines.append(f"- Dataset: {_link(metrics_path.parent / dataset_file, base)}")
    lines.append(f"- Model: {_link(metrics_path.parent / 'failure_model.joblib', base)}")
    lines.append("")
    return lines


def _adaptive_section(adaptive_dir: Path, base: Path, heading: str = "##") -> list[str]:
    lines = [f"{heading} Baseline vs adaptive track weighting", ""]
    summary_path = adaptive_dir / "adaptive_summary.json"
    if not summary_path.exists():
        lines.append(f"_Not run. Expected `{_rel(summary_path, base)}`._")
        lines.append("")
        return lines
    summary = load_json(summary_path, {})
    data_kind = summary.get("data_kind", "real")
    if data_kind != "real":
        lines.append(f"> Data kind: **{data_kind}**.")
        lines.append("")
    baseline_model = Path(str(summary.get("baseline_model", "")))
    lines.append(f"- Baseline model: `{_rel(baseline_model, base) if baseline_model.exists() else summary.get('baseline_model')}`")
    lines.append(f"- Engine requested: `{summary.get('engine_requested')}`")
    lines.append(f"- Modes: {', '.join(summary.get('modes', []))}")
    lines.append(f"- Total runtime: {_float(summary.get('runtime_seconds'), 1)} s")
    lines.append("")
    comparison_path = adaptive_dir / "comparison.csv"
    if comparison_path.exists():
        table = pd.read_csv(comparison_path)
        display = table.copy()
        columns = [c for c in [
            "variant", "engine", "num_registered_images", "num_points3D",
            "mean_reproj_error", "median_reproj_error", "track_length_median",
            "weight_mean", "success_label", "runtime_seconds",
            "delta_mean_reproj_error", "delta_num_points3D",
        ] if c in display.columns]
        lines.append(evaluation.table_to_markdown(display[columns]))
        lines.append("")
    weight_summaries = summary.get("weight_summaries") or {}
    if weight_summaries:
        rows = [{"mode": mode, **values} for mode, values in weight_summaries.items()]
        lines.append("Weight distributions:")
        lines.append("")
        lines.append(evaluation.table_to_markdown(pd.DataFrame(rows)))
        lines.append("")
    refinements = summary.get("refinements") or {}
    if refinements:
        rows = []
        for mode, report in refinements.items():
            before = report.get("errors_before", {}) or {}
            after = report.get("errors_after", {}) or {}
            rows.append({
                "mode": mode,
                "status": report.get("status"),
                "engine": report.get("engine_used"),
                "selected_points": report.get("num_selected_points"),
                "observations": report.get("num_observations"),
                "rmse_before": before.get("rmse"),
                "rmse_after": after.get("rmse"),
                "improved": report.get("improved"),
                "retri_points": (report.get("retriangulation") or {}).get("num_updated"),
                "runtime_s": report.get("runtime_seconds"),
                "note": report.get("engine_note") or report.get("error") or "",
            })
        lines.append("Refinement diagnostics:")
        lines.append("")
        lines.append(evaluation.table_to_markdown(pd.DataFrame(rows)))
        lines.append("")
    plot_dir = adaptive_dir / "plots"
    if plot_dir.is_dir():
        plots = sorted(plot_dir.glob("*.png"))
        if plots:
            lines.append("Plots: " + ", ".join(_link(p, base) for p in plots))
            lines.append("")
    lines.append(f"- Comparison table: {_link(comparison_path, base)}")
    lines.append(f"- Track weights: {_link(adaptive_dir / 'track_weights.csv', base)}")
    lines.append(f"- Adaptive summary: {_link(summary_path, base)}")
    lines.append("- Refined models: " + ", ".join(
        f"{mode}: `{_rel(Path(path), base)}`" for mode, path in (summary.get("outputs", {}).get("models") or {}).items()
    ))
    lines.append("")
    for note in summary.get("limitations", []):
        lines.append(f"- {note}")
    lines.append("")
    return lines


def _synthetic_section(synthetic_runs: Path, synthetic_models: Path,
                       synthetic_experiments: Path, base: Path) -> list[str]:
    lines = ["## Synthetic pipeline validation", ""]
    lines.append("> These runs use a generated sphere scene and a **simulated mapper**. "
                 "They validate code paths only and are not COLMAP/ETH3D results.")
    lines.append("")
    demo_path = synthetic_runs / "_demo_summary.json"
    if not demo_path.exists():
        lines.append("_Not run._")
        lines.append("")
        return lines
    demo = load_json(demo_path, {})
    rows = demo.get("runs", [])
    if rows:
        demo_rows = []
        for row in rows:
            row = dict(row)
            if row.get("run_dir"):
                row["run_dir"] = _rel(Path(str(row["run_dir"])), base)
            demo_rows.append(row)
        lines.append(evaluation.table_to_markdown(pd.DataFrame(demo_rows)))
        lines.append("")
    lines.extend(_training_section(
        synthetic_models / "pre" / "metrics.json", base,
        "Synthetic pre-reconstruction model",
    ))
    lines.extend(_training_section(
        synthetic_models / "post" / "metrics.json", base,
        "Synthetic post-reconstruction diagnostic model",
    ))
    adaptive = synthetic_experiments / "clean_s0" / "adaptive"
    if (adaptive / "adaptive_summary.json").exists():
        lines.extend(_adaptive_section(adaptive, base, heading="###"))
    return lines


def _inventory(root: Path, base: Path, limit: int = 500) -> list[str]:
    if not root.is_dir():
        return ["_No outputs directory._"]
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    rows = []
    for path in files[:limit]:
        rows.append({
            "path": _rel(path, base),
            "size": _size(path),
        })
    lines = [evaluation.table_to_markdown(pd.DataFrame(rows))]
    if len(files) > limit:
        lines.append(f"\n_Inventory truncated to {limit} of {len(files)} files._")
    return lines


def generate_report(config: Mapping, runs_root: str | Path,
                    experiments_root: str | Path, out_path: str | Path,
                    synthetic_runs_root: str | Path | None = None,
                    synthetic_models_root: str | Path | None = None,
                    synthetic_experiments_root: str | Path | None = None,
                    title: str = "SfM-FailSafe experiment report",
                    synthetic_only: bool = False) -> Path:
    runs_root, experiments_root = Path(runs_root), Path(experiments_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = out_path.parent
    run_dirs = _run_dirs(runs_root)
    runs = _runs_table(run_dirs, config)
    real_runs = runs[runs["data"] == "real"] if not runs.empty else runs
    synthetic_runs = runs[runs["data"] != "real"] if not runs.empty else runs

    lines: list[str] = [f"# {title}", ""]
    pdf_name = out_path.with_suffix(".pdf").name
    lines.append(f"PDF version (with embedded plots): [{pdf_name}]({pdf_name})")
    lines.append("")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    repo_root = Path(config.get("_meta", {}).get("repo_root", "."))
    config_display = _rel(Path(config.get("_meta", {}).get("config_path", "configs/default.yaml")), repo_root)
    lines.append(f"- Config: `{config_display}`")
    lines.append(f"- Runs root: `{_rel(runs_root, base)}`")
    lines.append(f"- Experiments root: `{_rel(experiments_root, base)}`")
    lines.append(f"- Python: {platform.python_version()} / {platform.platform()}")
    try:
        import numpy, pandas, sklearn, scipy, cv2
        lines.append(f"- Packages: numpy {numpy.__version__}, pandas {pandas.__version__}, "
                     f"scikit-learn {sklearn.__version__}, scipy {scipy.__version__}, "
                     f"opencv {cv2.__version__}")
    except Exception:
        pass
    lines.append("")

    # Executive summary.
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Real COLMAP runs recorded: **{len(real_runs)}**")
    synthetic_count = 0
    if synthetic_runs_root is not None:
        demo_path = Path(synthetic_runs_root) / "_demo_summary.json"
        if demo_path.exists():
            synthetic_count = len(load_json(demo_path, {}).get("runs", []))
    lines.append(f"- Synthetic validation runs recorded: **{synthetic_count}**")
    if not real_runs.empty:
        clean = real_runs[real_runs["condition"] == "clean"]
        if not clean.empty:
            row = clean.iloc[0]
            lines.append(f"- Clean baseline: {row['registered']}/{row['input']} cameras, "
                         f"{row['points']} points, mean reprojection error "
                         f"{_float(row['mean_err'])} px, label `{row['label']}`.")
        successes = int((real_runs["label"] == "success").sum())
        failures = int((real_runs["label"] == "failure").sum())
        lines.append(f"- Real labels: {successes} success / {failures} failure.")
    adaptive_path = experiments_root / "clean" / "adaptive" / "adaptive_summary.json"
    if adaptive_path.exists():
        adaptive = load_json(adaptive_path, {})
        variants = adaptive.get("variants", [])
        if variants:
            best = min(variants, key=lambda r: safe_float(r.get("mean_reproj_error"), float("inf")))
            baseline = next((r for r in variants if r.get("variant") == "baseline"), None)
            lines.append(f"- Adaptive best variant by mean reprojection: `{best.get('variant')}` "
                         f"({_float(best.get('mean_reproj_error'))} px).")
            if baseline:
                lines.append(f"- Baseline mean reprojection: {_float(baseline.get('mean_reproj_error'))} px.")
    pre_metrics = Path(get_cfg(config, "paths.failure_model_dir", "outputs/failure_model")) / "pre" / "metrics.json"
    if not synthetic_only and pre_metrics.exists():
        metrics = load_json(pre_metrics, {})
        if metrics.get("status") == "ok":
            lines.append(f"- Real pre-reconstruction failure model: `{metrics.get('best_model')}`, "
                         f"F1={_float(metrics.get('best_metrics', {}).get('f1'), 3)}, "
                         f"ROC-AUC={_float(metrics.get('best_metrics', {}).get('roc_auc'), 3)}, "
                         f"preliminary={metrics.get('preliminary')}.")
        else:
            lines.append(f"- Real failure model not trained: {metrics.get('error') or metrics.get('message')}")
    lines.append("")

    lines.append("## Data and model provenance")
    lines.append("")
    if synthetic_only:
        lines.append("- Synthetic sphere scene generated by `sfm_failsafe.synthetic`.")
        lines.append("- Simulated mapper; **not ETH3D and not COLMAP**.")
        lines.append("- See `outputs/demo_runs/_demo_summary.json` and each run's `run_manifest.json`.")
        lines.append("")
    else:
        manifest_path = Path(get_cfg(config, "dataset.manifest", "data/courtyard/manifest.json"))
        if not manifest_path.is_absolute():
            manifest_path = Path(config.get("_meta", {}).get("repo_root", ".")) / manifest_path
        if manifest_path.exists():
            manifest = load_json(manifest_path, {})
            lines.append(f"- Dataset manifest: `{_rel(manifest_path, base)}`")
            lines.append(f"- Scene: {manifest.get('scene')}")
            lines.append(f"- Source images: {manifest.get('num_images')}")
            lines.append(f"- Camera model: {manifest.get('camera_model')}")
            calibration = manifest.get("calibration_dir")
            if calibration:
                calibration_path = Path(str(calibration))
                calibration = _rel(calibration_path, base) if calibration_path.exists() else calibration_path.name
            lines.append(f"- Calibration reference: `{calibration}`")
        else:
            lines.append("- Dataset manifest: not found")
        if not real_runs.empty:
            first = next((d for d in run_dirs if load_json(d / "run_manifest.json", {}).get("data_kind") == "real"), None)
            if first is not None:
                run_manifest = load_json(first / "run_manifest.json", {})
                lines.append(f"- COLMAP version: {run_manifest.get('colmap_version', 'unknown')}")
                lines.append(f"- Real run image staging: `{run_manifest.get('images_dir')}`")
        lines.append("")

        lines.append("## Real experiment matrix")
        lines.append("")
        if real_runs.empty:
            lines.append("_No real COLMAP runs completed._")
        else:
            display = real_runs.copy()
            display = display.rename(columns={"input": "images", "registered": "registered",
                                              "mean_err": "mean_err_px", "median_err": "median_err_px",
                                              "track_med": "track_median", "seconds": "runtime_s"})
            lines.append(evaluation.table_to_markdown(display[[
                "run", "condition", "status", "images", "registered", "ratio", "points",
                "mean_err_px", "median_err_px", "track_median", "label", "runtime_s",
            ]]))
        lines.append("")

        lines.append("## Per-run details")
        lines.append("")
        for run_dir in run_dirs:
            summary = load_json(run_dir / "reconstruction_summary.json", {})
            if summary.get("data_kind", "real") == "real":
                lines.extend(_run_detail(run_dir, base))
        if not run_dirs:
            lines.append("_None._")
            lines.append("")

        lines.append("## Failure prediction")
        lines.append("")
        lines.extend(_training_section(pre_metrics, base, "Pre-reconstruction model (primary, leakage-free)"))
        lines.extend(_training_section(
            Path(get_cfg(config, "paths.failure_model_dir", "outputs/failure_model")) / "post" / "metrics.json",
            base, "Post-reconstruction diagnostic model (NOT leakage-free)",
        ))
        lines.append("Failure labels are threshold-based reconstruction outcomes. "
                     "With few runs, metrics validate the pipeline but are not a scientific result.")
        lines.append("")
        lines.append("Threshold-dependent metrics (accuracy/precision/recall/F1) can be unstable "
                     "when the held-out set is tiny or contains a single class; ROC-AUC and "
                     "average precision are threshold-free and should be preferred in that case.")
        lines.append("")

        lines.extend(_adaptive_section(experiments_root / "clean" / "adaptive", base))

    if synthetic_runs_root is not None:
        lines.extend(_synthetic_section(
            Path(synthetic_runs_root),
            Path(synthetic_models_root) if synthetic_models_root else Path("outputs/failure_model_synthetic"),
            Path(synthetic_experiments_root) if synthetic_experiments_root else Path("outputs/experiments_synthetic"),
            base,
        ))

    lines.append("## Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("bash scripts/download_eth3d.sh data/raw")
    lines.append("python scripts/prepare_courtyard.py --extract")
    lines.append("python scripts/run_baseline.py --config configs/courtyard.yaml --run-name clean --condition clean")
    lines.append("python scripts/analyze_tracks.py --config configs/courtyard.yaml --run-dir outputs/runs/clean")
    lines.append("python scripts/make_challenging_subsets.py --config configs/courtyard.yaml")
    lines.append("python scripts/train_failure_model.py --config configs/courtyard.yaml --runs-root outputs/runs --mode pre")
    lines.append("python scripts/run_adaptive.py --config configs/courtyard.yaml --run-dir outputs/runs/clean")
    lines.append("python scripts/run_all_experiments.py --config configs/courtyard.yaml")
    lines.append("python -m pytest -q")
    lines.append("```")
    lines.append("")

    lines.append("## Outputs inventory")
    lines.append("")
    lines.extend(_inventory(Path(config.get("_meta", {}).get("repo_root", ".")) / "outputs", base))

    lines.append("")
    lines.append("## Limitations and not-run items")
    lines.append("")
    limitations = [
        "Adaptive weighting is a **post-mapper refinement** of the same COLMAP model; it is not integrated into COLMAP's incremental mapper.",
        "The scipy weighted BA keeps intrinsics fixed, fixes the first camera as gauge, and does not add/remove tracks.",
        "Unselected tracks are re-triangulated against the refined cameras; this is reported per refinement, not hidden.",
        "pyceres is not installed, so the documented scipy fallback is used. No Ceres run is claimed.",
        "pycolmap is unavailable in this environment, so COLMAP binary models are converted with the COLMAP CLI and parsed from TXT; no pycolmap result is claimed.",
        "ETH3D ground-truth laser-scan accuracy/completeness is not evaluated; only SfM internal metrics and labels are reported.",
        "Failure labels depend on the number and diversity of runs; with few conditions, the prediction metrics are preliminary.",
        "Only the ETH3D Courtyard scene is used; leave-one-scene-out generalization is not evaluated.",
    ]
    if pre_metrics.exists():
        metrics = load_json(pre_metrics, {})
        if metrics.get("preliminary"):
            limitations.append("The real failure model is marked preliminary because of the small run/condition count.")
        if metrics.get("status") != "ok":
            limitations.append("The real failure model could not be trained (see the failure-prediction section).")
    for item in limitations:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Next steps")
    lines.append("")
    lines.append("- Add more ETH3D scenes and severity levels to obtain enough runs per class.")
    lines.append("- Add a pyceres/Ceres adapter when a signed, importable build is available.")
    lines.append("- Integrate weights into a modified mapper / retriangulation stage.")
    lines.append("- Evaluate against ETH3D ground-truth scans rather than only internal SfM metrics.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
