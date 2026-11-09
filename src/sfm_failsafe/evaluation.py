"""Failure labels, classification metrics, and baseline-vs-adaptive tables."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import get_cfg
from .utils import safe_float


def label_summary(summary: Mapping[str, Any], config: Mapping) -> dict[str, Any]:
    """Turn a reconstruction summary into a success/failure label.

    Failure if any configured condition holds.  Missing metrics are treated as
    unavailable, not as failures (except an explicitly failed mapper).
    """
    thresholds = get_cfg(config, "thresholds", {}) or {}
    min_ratio = float(thresholds.get("min_registered_ratio", 0.6))
    min_points = float(thresholds.get("min_num_points3D", 1000))
    max_mean = float(thresholds.get("max_mean_reproj_error", 2.0))
    max_median = float(thresholds.get("max_median_reproj_error", 1.5))
    min_track_median = float(thresholds.get("min_track_length_median", 2.0))
    mapper_failure_is_failure = bool(thresholds.get("mapper_failure_is_failure", True))

    checks: dict[str, bool | None] = {}
    reasons: list[str] = []

    status = str(summary.get("status", "success")).lower()
    if status in ("failed", "failure", "error"):
        checks["mapper_failed"] = True
        if mapper_failure_is_failure:
            reasons.append("mapper_failed")
    else:
        checks["mapper_failed"] = False

    ratio = safe_float(summary.get("registration_ratio"))
    if math.isfinite(ratio):
        checks["low_registration_ratio"] = ratio < min_ratio
        if ratio < min_ratio:
            reasons.append(f"registration_ratio {ratio:.3f} < {min_ratio:.3f}")
    else:
        checks["low_registration_ratio"] = None

    num_points = safe_float(summary.get("num_points3D"))
    if math.isfinite(num_points):
        checks["too_few_points"] = num_points < min_points
        if num_points < min_points:
            reasons.append(f"num_points3D {num_points:.0f} < {min_points:.0f}")
    else:
        checks["too_few_points"] = None

    mean_error = safe_float(summary.get("mean_reproj_error"))
    if math.isfinite(mean_error):
        checks["high_mean_reproj_error"] = mean_error > max_mean
        if mean_error > max_mean:
            reasons.append(f"mean_reproj_error {mean_error:.3f} > {max_mean:.3f}")
    else:
        checks["high_mean_reproj_error"] = None

    median_error = safe_float(summary.get("median_reproj_error"))
    if math.isfinite(median_error):
        checks["high_median_reproj_error"] = median_error > max_median
        if median_error > max_median:
            reasons.append(f"median_reproj_error {median_error:.3f} > {max_median:.3f}")
    else:
        checks["high_median_reproj_error"] = None

    track_median = safe_float(summary.get("track_length_median"))
    if math.isfinite(track_median) and track_median > 0:
        checks["short_tracks"] = track_median < min_track_median
        if track_median < min_track_median:
            reasons.append(f"track_length_median {track_median:.2f} < {min_track_median:.2f}")
    else:
        checks["short_tracks"] = None

    label = int(bool(reasons))
    return {
        "label": label,
        "label_name": "failure" if label else "success",
        "reasons": reasons,
        "checks": checks,
        "thresholds": {
            "min_registered_ratio": min_ratio,
            "min_num_points3D": min_points,
            "max_mean_reproj_error": max_mean,
            "max_median_reproj_error": max_median,
            "min_track_length_median": min_track_median,
        },
    }


def classification_metrics(y_true: Sequence[int], y_prob: Sequence[float],
                           threshold: float = 0.5) -> dict[str, Any]:
    """Binary metrics without requiring sklearn at call time."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    if y.size != p.size:
        raise ValueError("y_true and y_prob must have the same length")
    if y.size == 0:
        return {"n": 0, "error": "no samples"}
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    accuracy = (tp + tn) / y.size
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    roc_auc = float("nan")
    average_precision = float("nan")
    if len(np.unique(y)) == 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            roc_auc = float(roc_auc_score(y, p))
            average_precision = float(average_precision_score(y, p))
        except Exception:
            roc_auc = _auc_fallback(y, p)
    return {
        "n": int(y.size),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "predicted_positive": int(pred.sum()),
        "actual_positive": int(y.sum()),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _auc_fallback(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    positives = y == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def comparison_row(name: str, summary: Mapping[str, Any], config: Mapping,
                   runtime_seconds: float | None = None,
                   extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    label = label_summary(summary, config)
    row = {
        "variant": name,
        "status": summary.get("status", "success"),
        "num_input_images": summary.get("num_input_images"),
        "num_registered_images": summary.get("num_registered_images"),
        "registration_ratio": summary.get("registration_ratio"),
        "num_points3D": summary.get("num_points3D"),
        "num_observations": summary.get("num_observations"),
        "mean_reproj_error": summary.get("mean_reproj_error"),
        "median_reproj_error": summary.get("median_reproj_error"),
        "p95_reproj_error": summary.get("p95_reproj_error"),
        "track_length_mean": summary.get("track_length_mean"),
        "track_length_median": summary.get("track_length_median"),
        "track_length_p90": summary.get("track_length_p90"),
        "weight_mean": summary.get("weight_mean"),
        "success_label": label["label"],
        "failure_reasons": "; ".join(label["reasons"]),
        "runtime_seconds": runtime_seconds,
    }
    if extra:
        row.update(extra)
    return row


def comparison_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([dict(row) for row in rows])


def add_deltas(table: pd.DataFrame, baseline_variant: str = "baseline") -> pd.DataFrame:
    """Add absolute/relative deltas versus a baseline row (no fake values)."""
    table = table.copy()
    baseline = table[table["variant"] == baseline_variant]
    if baseline.empty:
        return table
    base = baseline.iloc[0]
    for metric in ("num_registered_images", "num_points3D", "mean_reproj_error",
                   "median_reproj_error", "track_length_median", "registration_ratio"):
        if metric not in table.columns:
            continue
        table[f"delta_{metric}"] = [
            (safe_float(v) - safe_float(base[metric]))
            if math.isfinite(safe_float(v)) and math.isfinite(safe_float(base[metric]))
            else np.nan
            for v in table[metric]
        ]
    return table


def table_to_markdown(table: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Small markdown writer so tables do not need the optional tabulate dep."""
    if table.empty:
        return "_No rows._\n"
    columns = list(table.columns)
    lines = ["| " + " | ".join(str(c) for c in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in table.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                if math.isnan(float(value)):
                    cells.append("n/a")
                else:
                    cells.append(float_format.format(float(value)))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_comparison_markdown(table: pd.DataFrame, path: str | Path,
                              title: str = "Baseline vs adaptive refinement",
                              notes: Sequence[str] = ()) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if notes:
        lines.extend([f"- {note}" for note in notes])
        lines.append("")
    lines.append(table_to_markdown(table))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summarize_runs(runs: pd.DataFrame, group_columns: Sequence[str] = ("condition",)) -> pd.DataFrame:
    if runs.empty:
        return runs
    columns = [c for c in group_columns if c in runs.columns]
    if not columns:
        return runs.describe(include="all").transpose()
    numeric = [c for c in runs.columns if pd.api.types.is_numeric_dtype(runs[c])]
    return runs.groupby(columns, dropna=False)[numeric].agg(["count", "mean", "std"])
