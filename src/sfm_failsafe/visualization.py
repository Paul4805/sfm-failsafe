"""Headless plots for track statistics, weights, and failure-model diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _prepare(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _placeholder(path: Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_track_length_hist(tracks: pd.DataFrame, path: str | Path,
                           title: str = "Track length distribution",
                           column: str = "track_length") -> Path:
    path = _prepare(path)
    if tracks.empty or column not in tracks:
        return _placeholder(path, "No tracks available")
    values = pd.to_numeric(tracks[column], errors="coerce").dropna()
    if values.empty:
        return _placeholder(path, "No valid track lengths")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(0.5, max(values.max() + 1.5, 3.5), 1.0)
    ax.hist(values, bins=bins, color="#3b6ea8", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Track length (observations)")
    ax.set_ylabel("Number of tracks")
    ax.set_title(f"{title}\nmedian={values.median():.1f}, p90={values.quantile(0.9):.1f}, n={len(values)}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_reprojection_hist(tracks: pd.DataFrame, path: str | Path,
                           title: str = "Track mean reprojection error",
                           column: str = "mean_reproj_error") -> Path:
    path = _prepare(path)
    if tracks.empty or column not in tracks:
        return _placeholder(path, "No reprojection errors available")
    values = pd.to_numeric(tracks[column], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return _placeholder(path, "No valid reprojection errors")
    upper = float(values.quantile(0.99))
    upper = max(upper, float(values.min()) + 1e-6)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values.clip(upper=upper), bins=40, color="#c26b3a", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Mean reprojection error (px, clipped at p99)")
    ax.set_ylabel("Number of tracks")
    ax.set_title(f"{title}\nmedian={values.median():.3f}px, p95={values.quantile(0.95):.3f}px")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_weight_distribution(weights: pd.DataFrame, path: str | Path,
                             title: str = "Track weight distribution") -> Path:
    path = _prepare(path)
    if weights.empty or "weight" not in weights:
        return _placeholder(path, "No weights available")
    values = pd.to_numeric(weights["weight"], errors="coerce").dropna()
    if values.empty:
        return _placeholder(path, "No valid weights")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(values, bins=40, color="#2f7d6b", edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Weight")
    axes[0].set_ylabel("Tracks")
    axes[0].set_title(f"weight median={values.median():.3f}, min={values.min():.3f}")
    if "reliability" in weights:
        rel = pd.to_numeric(weights["reliability"], errors="coerce").dropna()
        axes[1].hist(rel, bins=40, color="#7a5aa6", edgecolor="white", linewidth=0.4)
        axes[1].set_xlabel("Reliability")
        axes[1].set_title(f"reliability median={rel.median():.3f}")
    else:
        axes[1].axis("off")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_reliability_scatter(weights: pd.DataFrame, path: str | Path) -> Path:
    path = _prepare(path)
    required = {"track_length", "weight"}
    if weights.empty or not required.issubset(weights.columns):
        return _placeholder(path, "Need track_length and weight columns")
    frame = weights.dropna(subset=["track_length", "weight"]).copy()
    if frame.empty:
        return _placeholder(path, "No valid tracks")
    color = pd.to_numeric(frame.get("mean_reproj_error", pd.Series(0, index=frame.index)),
                          errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    scatter = ax.scatter(
        np.maximum(frame["track_length"].to_numpy(dtype=float), 1.0),
        frame["weight"].to_numpy(dtype=float),
        c=np.clip(color.to_numpy(dtype=float), 0, np.quantile(color, 0.99) + 1e-9),
        s=10, alpha=0.65, cmap="viridis",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Track length (log scale)")
    ax.set_ylabel("Weight")
    ax.set_title("Reliability weighting: length vs weight")
    fig.colorbar(scatter, ax=ax, label="Mean reprojection error (px)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_comparison_bars(table: pd.DataFrame, path: str | Path) -> Path:
    path = _prepare(path)
    if table.empty:
        return _placeholder(path, "No comparison rows")
    metrics = [
        ("num_points3D", "3D points"),
        ("mean_reproj_error", "Mean reproj. error (px)"),
        ("median_reproj_error", "Median reproj. error (px)"),
        ("track_length_median", "Median track length"),
    ]
    available = [(column, label) for column, label in metrics if column in table.columns]
    if not available:
        return _placeholder(path, "Comparison table lacks metrics")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    for ax, (column, label) in zip(axes, available):
        values = pd.to_numeric(table[column], errors="coerce")
        bars = ax.bar(table["variant"].astype(str), values, color="#4d7ea8")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}",
                        ha="center", va="bottom", fontsize=8)
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.suptitle("Baseline vs adaptive refinement (same initial model)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_pr(y_true: Sequence[int], y_prob: Sequence[float], path: str | Path,
                title: str = "Failure prediction") -> Path:
    path = _prepare(path)
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    if y.size == 0 or len(np.unique(y)) < 2:
        return _placeholder(path, "ROC/PR need both classes; current evaluation has "
                                  f"{len(np.unique(y))} class(es) (n={y.size})")
    from sklearn.metrics import (auc, precision_recall_curve, roc_curve)

    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y, p)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(fpr, tpr, color="#26547c", lw=2, label=f"AUC={roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(loc="lower right")
    axes[1].plot(recall, precision, color="#a23e48", lw=2, label=f"AUC={pr_auc:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall")
    axes[1].legend(loc="lower left")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(matrix: Sequence[Sequence[int]], path: str | Path,
                   labels: Sequence[str] = ("success", "failure")) -> Path:
    path = _prepare(path)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2):
        return _placeholder(path, "Confusion matrix must be 2x2")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() * 0.5 else "black",
                    fontsize=14)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(names: Sequence[str], values: Sequence[float],
                            path: str | Path, top: int = 20) -> Path:
    path = _prepare(path)
    names = list(names)
    values = np.asarray(values, dtype=float)
    if len(names) == 0 or values.size == 0 or not np.any(np.isfinite(values)):
        return _placeholder(path, "No feature importance available")
    order = np.argsort(np.nan_to_num(values, nan=-np.inf))[::-1][:top]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.28 * len(order) + 1.5)))
    ax.barh(np.arange(len(order))[::-1], values[order], color="#4c956c")
    ax.set_yticks(np.arange(len(order))[::-1], [names[i] for i in order])
    ax.set_xlabel("Importance (model-specific)")
    ax.set_title(f"Top {len(order)} features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_class_balance(labels: Sequence[int], path: str | Path) -> Path:
    path = _prepare(path)
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return _placeholder(path, "No labels")
    counts = np.bincount(labels, minlength=2)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(["success", "failure"], counts, color=["#4c956c", "#a23e48"])
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(int(value)),
                ha="center", va="bottom")
    ax.set_ylabel("Runs")
    ax.set_title("Label balance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pointcloud_projection(rec, path: str | Path, max_points: int = 20000) -> Path:
    """Project COLMAP's sparse points with matplotlib.

    This is explicitly **not** a COLMAP rendered screenshot; it is a qualitative
    view of the sparse point cloud that works headlessly.
    """
    path = _prepare(path)
    if rec.num_points == 0:
        return _placeholder(path, "No 3D points")
    xyz, ids = rec.point_array()
    if len(xyz) > max_points:
        positions = np.linspace(0, len(xyz) - 1, max_points).round().astype(int)
        xyz = xyz[positions]
        ids = ids[positions]
    colors = np.stack([rec.points3D[int(i)].rgb for i in ids]).astype(float) / 255.0
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=2, alpha=0.7, depthshade=False)
    centers = np.stack([img.camera_center for img in rec.images.values()]) if rec.images else np.zeros((0, 3))
    if len(centers):
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="black", marker="^",
                   s=25, label="cameras")
        ax.legend(loc="upper right")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("COLMAP sparse point cloud\n(matplotlib projection, not a rendered screenshot)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
