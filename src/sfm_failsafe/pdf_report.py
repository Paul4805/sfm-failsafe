"""Generate a PDF report with tables and embedded plots from saved artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from xml.sax.saxutils import escape

from .config import get_cfg
from .report import _float, _rel, _run_dirs, _runs_table, _size
from .utils import load_json, safe_float

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (Image as RLImage, KeepTogether, LongTable,
                                    PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)
    REPORTLAB_AVAILABLE = True
except Exception:  # pragma: no cover
    REPORTLAB_AVAILABLE = False


PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 0.55 * inch
AVAILABLE_WIDTH = PAGE_WIDTH - 2 * MARGIN


def _ss(text: Any, style) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", parent=styles["Title"],
                              fontSize=22, leading=27, spaceAfter=10))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"],
                              fontSize=10.5, leading=14, textColor=colors.HexColor("#444444")))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"],
                              fontSize=15, leading=19, spaceBefore=10, spaceAfter=7,
                              textColor=colors.HexColor("#16324f")))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"],
                              fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=5,
                              textColor=colors.HexColor("#26547c")))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["Normal"],
                              fontSize=8.5, leading=11.5, spaceAfter=3))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["Normal"],
                              fontSize=7, leading=8.6))
    styles.add(ParagraphStyle(name="Tinyx", parent=styles["Normal"],
                              fontSize=6.2, leading=7.4))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"],
                              fontSize=6.8, leading=8.2, alignment=TA_CENTER,
                              textColor=colors.HexColor("#555555"), spaceAfter=4))
    styles.add(ParagraphStyle(name="Warn", parent=styles["Normal"],
                              fontSize=8.5, leading=11.5, textColor=colors.HexColor("#7a1f1f"),
                              backColor=colors.HexColor("#fbe9e7"), borderPadding=5,
                              spaceBefore=5, spaceAfter=6))
    return styles


STYLES = _styles() if REPORTLAB_AVAILABLE else {}


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "n/a"
        return f"{number:.{digits}f}"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return str(value)


SHORT_COLUMNS = {
    "condition": "cond",
    "mean_reproj_error": "mean_err",
    "median_reproj_error": "med_err",
    "track_length_median": "trk_med",
    "track_length_mean": "trk_mean",
    "track_length_p90": "trk_p90",
    "registration_ratio": "ratio",
    "num_registered_images": "reg",
    "num_input_images": "imgs",
    "num_points3D": "points",
    "runtime_seconds": "sec",
    "delta_mean_reproj_error": "d_mean",
    "delta_num_points3D": "d_points",
    "weight_mean": "w_mean",
    "reliability_mean": "rel_mean",
    "fraction_below_0_1": "w<0.1",
    "fraction_above_0_9": "w>0.9",
    "selected_points": "sel_pts",
    "observations": "obs",
    "rmse_before": "rmse0",
    "rmse_after": "rmse1",
    "retri_points": "retri",
    "average_precision": "avg_prec",
    "predicted_positive": "pred_pos",
    "actual_positive": "actual_pos",
    "num_input_images": "imgs",
    "mean_reproj_error": "mean_err",
}


def _df_table(frame: pd.DataFrame, font_size: float = 5.8,
              max_rows: int | None = None) -> LongTable:
    if frame.empty:
        return LongTable([[_ss("No rows", STYLES["Smallx"])]], colWidths=[AVAILABLE_WIDTH])
    frame = frame.rename(columns={k: v for k, v in SHORT_COLUMNS.items() if k in frame.columns})
    if max_rows is not None:
        frame = frame.head(max_rows)
    data = [[_ss(column, STYLES["Tinyx"]) for column in frame.columns]]
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            text = _fmt(value)
            if len(text) > 80:
                text = text[:77] + "..."
            cells.append(_ss(text, STYLES["Tinyx"]))
        data.append(cells)
    # Width by content length, with a floor; short labels prevent header overlap.
    character_weights = []
    for column in frame.columns:
        values = [len(_fmt(v)) for v in frame[column].head(30)]
        character_weights.append(max(len(str(column)), min(max(values or [0]), 16), 4))
    total = sum(character_weights)
    widths = [max(0.34 * inch, AVAILABLE_WIDTH * w / total) for w in character_weights]
    scale = AVAILABLE_WIDTH / sum(widths)
    widths = [w * scale for w in widths]
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0b8c2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fc")]),
    ]))
    return table


def _img(path: Path, width_in: float = 6.2, caption: str | None = None,
         max_height_in: float = 7.0) -> list:
    if not path.exists():
        return []
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as pil:
            width_px, height_px = pil.size
        height_in = width_in * height_px / max(width_px, 1)
        if height_in > max_height_in:
            width_in *= max_height_in / height_in
            height_in = max_height_in
        image = RLImage(str(path), width=width_in * inch, height=height_in * inch)
        story = [image]
        if caption:
            story.append(_ss(caption, STYLES["Caption"]))
        return [KeepTogether(story)]
    except Exception:
        return []


def _side_by_side(paths: Sequence[Path], captions: Sequence[str]) -> list:
    images = []
    for path, caption in zip(paths, captions):
        if not path.exists():
            continue
        try:
            from PIL import Image as PILImage
            with PILImage.open(path) as pil:
                w, h = pil.size
            width = 3.25
            height = width * h / max(w, 1)
            if height > 3.2:
                height = 3.2
                width = height * w / max(h, 1)
            images.append([RLImage(str(path), width=width * inch, height=height * inch),
                           _ss(caption, STYLES["Caption"])])
        except Exception:
            continue
    if not images:
        return []
    table = Table([[item[0] for item in images], [item[1] for item in images]],
                  colWidths=[AVAILABLE_WIDTH / max(len(images), 1)] * len(images))
    table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return [table, Spacer(1, 4)]


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(MARGIN, 0.35 * inch, "SfM-FailSafe - failure prediction and adaptive track weighting")
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _add_heading(story: list, text: str, level: int = 1) -> None:
    story.append(_ss(text, STYLES["H1x" if level == 1 else "H2x"]))


def _training_tables(metrics_path: Path) -> list:
    story: list = []
    if not metrics_path.exists():
        story.append(_ss("Not run / no metrics file.", STYLES["Bodyx"]))
        return story
    metrics = load_json(metrics_path, {})
    if metrics.get("status") != "ok":
        story.append(_ss(f"Not trained: {metrics.get('error') or metrics.get('message')}",
                         STYLES["Bodyx"]))
        return story
    info = [
        ["Mode", metrics.get("mode")],
        ["Leakage-free", metrics.get("leakage_free")],
        ["Runs", metrics.get("n_runs")],
        ["Class counts", metrics.get("class_counts")],
        ["Features", metrics.get("n_features")],
        ["Split", metrics.get("split", {}).get("evaluation_mode")],
        ["Best model", metrics.get("best_model")],
        ["Threshold", _float(metrics.get("threshold"), 4)],
        ["Preliminary", metrics.get("preliminary")],
    ]
    table = Table([[_ss(k, STYLES["Tinyx"]), _ss(v, STYLES["Tinyx"])] for k, v in info],
                  colWidths=[1.5 * inch, AVAILABLE_WIDTH - 1.5 * inch])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0b8c2")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 4))
    rows = []
    for name, payload in (metrics.get("models") or {}).items():
        m = payload.get("metrics", {})
        rows.append({
            "model": name,
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "roc_auc": m.get("roc_auc"),
            "avg_prec": m.get("average_precision"),
            "tp": m.get("tp"), "tn": m.get("tn"),
            "fp": m.get("fp"), "fn": m.get("fn"),
        })
    story.append(_df_table(pd.DataFrame(rows), font_size=5.5))
    story.append(Spacer(1, 4))
    if metrics.get("test_runs"):
        story.append(_ss("Held-out runs: " + ", ".join(metrics["test_runs"]), STYLES["Smallx"]))
    oof = metrics.get("leave_one_condition_out") or {}
    if oof.get("status") == "ok":
        story.append(_ss("Leave-one-condition-out (random forest):", STYLES["Smallx"]))
        row = {k: v for k, v in (oof.get("metrics") or {}).items() if k != "confusion_matrix"}
        story.append(_df_table(pd.DataFrame([row]), font_size=5.5))
    return story


def _training_plots(metrics_dir: Path) -> list:
    story = []
    plot_dir = metrics_dir / "plots"
    if not plot_dir.is_dir():
        return story
    roc = plot_dir / "roc_pr.png"
    conf = plot_dir / "confusion_matrix.png"
    imp = plot_dir / "feature_importance.png"
    story.extend(_side_by_side([roc, conf], ["ROC / precision-recall", "Confusion matrix"]))
    if imp.exists():
        story.extend(_img(imp, width_in=5.8, caption="Feature importance"))
    return story


def _run_section(run_dir: Path, base: Path) -> list:
    story: list = []
    summary = load_json(run_dir / "reconstruction_summary.json", {})
    manifest = load_json(run_dir / "run_manifest.json", {})
    _add_heading(story, f"Run: {run_dir.name}", 2)
    info = [
        ("Condition", manifest.get("condition", summary.get("condition", "unknown"))),
        ("Data kind", manifest.get("data_kind", summary.get("data_kind", "real"))),
        ("Status", summary.get("status")),
        ("Registered", f"{summary.get('num_registered_images')}/{summary.get('num_input_images')}"),
        ("Registration ratio", _float(summary.get("registration_ratio"), 3)),
        ("3D points", summary.get("num_points3D")),
        ("Mean / median reprojection (px)",
         f"{_float(summary.get('mean_reproj_error'))} / {_float(summary.get('median_reproj_error'))}"),
        ("Track length mean / median / p90",
         f"{_float(summary.get('track_length_mean'), 2)} / "
         f"{_float(summary.get('track_length_median'), 2)} / "
         f"{_float(summary.get('track_length_p90'), 2)}"),
        ("Success label", summary.get("success_label_name")),
        ("Failure reasons", "; ".join(summary.get("failure_reasons", []) or []) or "-"),
        ("Runtime (s)", _float(summary.get("runtime_seconds"), 1)),
    ]
    table = Table([[_ss(k, STYLES["Tinyx"]), _ss(v, STYLES["Tinyx"])] for k, v in info],
                  colWidths=[2.1 * inch, AVAILABLE_WIDTH - 2.1 * inch])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0b8c2")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 5))
    plot_dir = run_dir / "plots"
    if plot_dir.is_dir():
        story.extend(_side_by_side(
            [plot_dir / "track_length.png", plot_dir / "reproj_error.png"],
            ["Track length distribution", "Track mean reprojection error"],
        ))
        pointcloud = plot_dir / "sparse_pointcloud.png"
        if pointcloud.exists() and run_dir.name == "clean":
            story.extend(_img(pointcloud, width_in=4.5,
                              caption="Sparse point cloud projection (matplotlib, not a rendered screenshot)"))
    else:
        story.append(_ss("No analysis plots for this run.", STYLES["Smallx"]))
    story.append(PageBreak())
    return story


def generate_pdf_report(config: Mapping, runs_root: str | Path,
                        experiments_root: str | Path, out_path: str | Path,
                        synthetic_runs_root: str | Path | None = None,
                        synthetic_models_root: str | Path | None = None,
                        synthetic_experiments_root: str | Path | None = None,
                        synthetic_only: bool = False) -> Path:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed; pip install reportlab")
    runs_root, experiments_root = Path(runs_root), Path(experiments_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = out_path.parent
    run_dirs = _run_dirs(runs_root)
    runs = _runs_table(run_dirs, config)
    real_runs = runs[runs["data"] == "real"] if not runs.empty else runs
    story: list = []

    # Cover.
    story.append(Spacer(1, 0.7 * inch))
    story.append(_ss("SfM-FailSafe" if not synthetic_only else "SfM-FailSafe (synthetic validation)",
                     STYLES["TitleBig"]))
    story.append(_ss("Failure prediction and adaptive track weighting for sparse 3D reconstruction",
                     STYLES["Subtitle"]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(_ss(f"Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  |  "
                     f"Python {__import__('sys').version.split()[0]}  |  "
                     f"COLMAP-based pipeline", STYLES["Bodyx"]))
    if synthetic_only:
        story.append(_ss("WARNING: synthetic sphere scene and simulated mapper. "
                         "Not COLMAP or ETH3D results.", STYLES["Warn"]))
    story.append(Spacer(1, 0.25 * inch))
    overview = [
        ("Real COLMAP runs", len(real_runs)),
        ("Synthetic validation runs",
         len(load_json(Path(synthetic_runs_root) / "_demo_summary.json", {}).get("runs", []))
         if synthetic_runs_root and Path(synthetic_runs_root).exists() else 0),
        ("Report artifacts", _rel(out_path, base)),
    ]
    table = Table([[_ss(k, STYLES["Smallx"]), _ss(v, STYLES["Smallx"])] for k, v in overview],
                  colWidths=[2.2 * inch, AVAILABLE_WIDTH - 2.2 * inch])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b8c2")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
    ]))
    story.append(table)
    story.append(PageBreak())

    if not synthetic_only:
        _add_heading(story, "Executive summary")
        clean = real_runs[real_runs["condition"] == "clean"] if not real_runs.empty else pd.DataFrame()
        if not clean.empty:
            row = clean.iloc[0]
            story.append(_ss(
                f"Clean COLMAP baseline registered {row['registered']}/{row['input']} cameras "
                f"with {row['points']} 3D points and mean reprojection error "
                f"{_float(row['mean_err'])} px. Real runs: "
                f"{int((real_runs['label'] == 'success').sum())} success / "
                f"{int((real_runs['label'] == 'failure').sum())} failure.", STYLES["Bodyx"]))
        adaptive_summary_path = experiments_root / "clean" / "adaptive" / "adaptive_summary.json"
        if adaptive_summary_path.exists():
            adaptive = load_json(adaptive_summary_path, {})
            variants = adaptive.get("variants", [])
            if variants:
                baseline = next((v for v in variants if v.get("variant") == "baseline"), None)
                best = min(variants, key=lambda v: safe_float(v.get("mean_reproj_error"), float("inf")))
                if baseline:
                    story.append(_ss(
                        f"Post-mapper refinement: baseline {_float(baseline.get('mean_reproj_error'))} px; "
                        f"best variant {best.get('variant')} {_float(best.get('mean_reproj_error'))} px. "
                        "Uniform BA is the control; weighted gains are not assumed.", STYLES["Bodyx"]))
        story.append(Spacer(1, 6))

        _add_heading(story, "Real run matrix")
        if not real_runs.empty:
            display = real_runs.rename(columns={
                "input": "images", "mean_err": "mean_px", "median_err": "median_px",
                "track_med": "track_med", "seconds": "sec",
            })
            columns = ["run", "condition", "status", "images", "registered", "ratio",
                       "points", "mean_px", "median_px", "track_med", "label", "sec"]
            story.append(_df_table(display[[c for c in columns if c in display.columns]]))
        else:
            story.append(_ss("No real runs.", STYLES["Bodyx"]))
        story.append(PageBreak())

        _add_heading(story, "Per-run analysis and plots")
        for run_dir in run_dirs:
            summary = load_json(run_dir / "reconstruction_summary.json", {})
            if summary.get("data_kind", "real") == "real":
                story.extend(_run_section(run_dir, base))

        _add_heading(story, "Failure prediction")
        pre_dir = Path(get_cfg(config, "paths.failure_model_dir", "outputs/failure_model")) / "pre"
        post_dir = Path(get_cfg(config, "paths.failure_model_dir", "outputs/failure_model")) / "post"
        _add_heading(story, "Pre-reconstruction model (leakage-free)", 2)
        story.extend(_training_tables(pre_dir / "metrics.json"))
        story.extend(_training_plots(pre_dir))
        story.append(PageBreak())
        _add_heading(story, "Post-reconstruction diagnostic model (NOT leakage-free)", 2)
        story.extend(_training_tables(post_dir / "metrics.json"))
        story.extend(_training_plots(post_dir))
        story.append(PageBreak())

        _add_heading(story, "Baseline vs adaptive track weighting")
        adaptive_dir = experiments_root / "clean" / "adaptive"
        comparison_path = adaptive_dir / "comparison.csv"
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)
            columns = [c for c in ["variant", "engine", "num_points3D", "mean_reproj_error",
                                   "median_reproj_error", "track_length_median",
                                   "weight_mean", "success_label", "runtime_seconds",
                                   "delta_mean_reproj_error"] if c in comparison.columns]
            story.append(_df_table(comparison[columns]))
            story.append(Spacer(1, 4))
        adaptive_path = adaptive_dir / "adaptive_summary.json"
        if adaptive_path.exists():
            adaptive = load_json(adaptive_path, {})
            weight_rows = [{"mode": mode, **values}
                           for mode, values in (adaptive.get("weight_summaries") or {}).items()]
            if weight_rows:
                story.append(_ss("Weight distributions", STYLES["Smallx"]))
                story.append(_df_table(pd.DataFrame(weight_rows)))
                story.append(Spacer(1, 4))
            refinement_rows = []
            for mode, report in (adaptive.get("refinements") or {}).items():
                before = report.get("errors_before", {}) or {}
                after = report.get("errors_after", {}) or {}
                refinement_rows.append({
                    "mode": mode,
                    "status": report.get("status"),
                    "engine": report.get("engine_used"),
                    "selected_points": report.get("num_selected_points"),
                    "observations": report.get("num_observations"),
                    "rmse_before": before.get("rmse"),
                    "rmse_after": after.get("rmse"),
                    "retri_points": (report.get("retriangulation") or {}).get("num_updated"),
                    "runtime_s": report.get("runtime_seconds"),
                })
            if refinement_rows:
                story.append(_ss("Refinement diagnostics", STYLES["Smallx"]))
                story.append(_df_table(pd.DataFrame(refinement_rows)))
                story.append(Spacer(1, 4))
        plot_dir = adaptive_dir / "plots"
        if plot_dir.is_dir():
            story.extend(_side_by_side(
                [plot_dir / "weight_distribution.png", plot_dir / "reliability_scatter.png"],
                ["Track weight distribution", "Track length vs weight"],
            ))
            story.extend(_img(plot_dir / "comparison_bars.png", width_in=6.1,
                              caption="Baseline vs refinement metrics (same initial model)"))
        story.append(PageBreak())

    # Synthetic section.
    if synthetic_runs_root is not None:
        synthetic_runs = Path(synthetic_runs_root)
        demo_path = synthetic_runs / "_demo_summary.json"
        if demo_path.exists():
            demo = load_json(demo_path, {})
            _add_heading(story, "Synthetic pipeline validation")
            story.append(_ss("WARNING: generated sphere scene with a simulated mapper. "
                             "Not COLMAP/ETH3D results; validation only.", STYLES["Warn"]))
            rows = demo.get("runs", [])
            if rows:
                frame = pd.DataFrame(rows)
                columns = [c for c in ["run_name", "condition", "status", "label_name",
                                       "num_input_images", "num_registered_images",
                                       "num_points3D", "mean_reproj_error"] if c in frame.columns]
                story.append(_df_table(frame[columns], font_size=5.5))
            story.append(Spacer(1, 5))
            syn_models = Path(synthetic_models_root) if synthetic_models_root else Path("outputs/failure_model_synthetic")
            _add_heading(story, "Synthetic failure model (pre)", 2)
            story.extend(_training_tables(syn_models / "pre" / "metrics.json"))
            story.extend(_training_plots(syn_models / "pre"))
            story.append(PageBreak())
            syn_adaptive = (Path(synthetic_experiments_root) if synthetic_experiments_root
                            else Path("outputs/experiments_synthetic")) / "clean_s0" / "adaptive"
            if (syn_adaptive / "adaptive_summary.json").exists():
                _add_heading(story, "Synthetic adaptive comparison", 2)
                comparison = pd.read_csv(syn_adaptive / "comparison.csv")
                columns = [c for c in ["variant", "engine", "num_points3D", "mean_reproj_error",
                                       "median_reproj_error", "track_length_median",
                                       "success_label", "runtime_seconds"] if c in comparison.columns]
                story.append(_df_table(comparison[columns]))
                plot_dir = syn_adaptive / "plots"
                if plot_dir.is_dir():
                    story.extend(_img(plot_dir / "comparison_bars.png", width_in=6.1,
                                      caption="Synthetic baseline vs refinement"))
                story.append(PageBreak())

    _add_heading(story, "Limitations")
    limitations = [
        "Adaptive weighting is post-mapper refinement, not integrated into COLMAP's incremental mapper.",
        "scipy BA fixes intrinsics and gauge; it is not a Ceres/pyceres run.",
        "Failure labels and metrics are preliminary with few runs/conditions.",
        "Only ETH3D Courtyard and a synthetic scene are used; no cross-scene or ground-truth scan evaluation.",
        "pycolmap/pyceres were unavailable; COLMAP CLI conversion and scipy fallback were used.",
    ]
    for item in limitations:
        story.append(_ss(f"- {item}", STYLES["Bodyx"]))
    story.append(Spacer(1, 8))
    story.append(_ss("Artifacts: outputs/runs, outputs/failure_model, outputs/experiments, "
                     "outputs/REPORT.md, outputs/REPORT_SYNTHETIC.md", STYLES["Smallx"]))

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.55 * inch, bottomMargin=0.55 * inch,
        title="SfM-FailSafe report", author="SfM-FailSafe",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return out_path
