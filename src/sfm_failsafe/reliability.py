"""Reliability scoring for feature tracks and mapping to BA weights.

A track is reliable when it is long, has a low reprojection error, spans a
useful baseline / viewing angle, has a high verified-inlier ratio, and comes
from sharp images.  Missing features are excluded per row and the remaining
weights are renormalised, so the score works on real COLMAP tracks (which may
not have blur or inlier columns) as well as on enriched synthetic fixtures.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .config import get_cfg

DEFAULT_FEATURE_WEIGHTS = {
    "length": 0.25,
    "reproj_error": 0.25,
    "angle": 0.15,
    "spread": 0.15,
    "consistency": 0.10,
    "blur": 0.10,
}

DEFAULT_REFS = {
    "track_length": 6.0,
    "reproj_error": 1.5,
    "angle_deg": 8.0,
    "spread_norm": 0.15,
    "blur_laplacian": 120.0,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _column(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _rowwise_weighted(components: dict[str, pd.Series],
                      weights: Mapping[str, float]) -> pd.Series:
    numerator = pd.Series(0.0, index=next(iter(components.values())).index)
    denominator = pd.Series(0.0, index=numerator.index)
    for name, values in components.items():
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        numerator = numerator.add(values.fillna(0.0) * w, fill_value=0.0)
        denominator = denominator.add(values.notna().astype(float) * w, fill_value=0.0)
    composite = numerator / denominator.where(denominator > 0, np.nan)
    return composite.fillna(0.5)


def track_feature_components(tracks: pd.DataFrame,
                             refs: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Return normalised [0, 1] component scores (NaN when input is missing)."""
    refs = {**DEFAULT_REFS, **(refs or {})}
    n = len(tracks)
    length = _column(tracks, "track_length", "track_len")
    error = _column(tracks, "mean_reproj_error", "reproj_error", "error")
    angle = _column(tracks, "triangulation_angle_deg", "triangulation_angle")
    spread = _column(tracks, "image_spread_norm", "spread_norm")
    consistency = _column(tracks, "inlier_consistency", "match_inlier_ratio")
    blur = _column(tracks, "blur_score", "mean_blur", "laplacian_variance")

    length_ref = max(float(refs["track_length"]), 1.0)
    length_score = np.clip(np.log1p(length.clip(lower=0)) / np.log1p(length_ref), 0.0, 1.0)

    error_ref = max(float(refs["reproj_error"]), 1e-6)
    error_score = np.exp(-np.clip(error.clip(lower=0), 0.0, None) / error_ref)

    angle_ref = max(float(refs["angle_deg"]), 1e-6)
    angle_score = np.clip(angle / angle_ref, 0.0, 1.0)

    spread_ref = max(float(refs["spread_norm"]), 1e-6)
    spread_score = np.clip(spread / spread_ref, 0.0, 1.0)

    consistency_score = np.clip(consistency, 0.0, 1.0)

    blur_ref = max(float(refs["blur_laplacian"]), 1e-6)
    blur_score = np.clip(blur / blur_ref, 0.0, 1.0)

    return pd.DataFrame({
        "score_length": length_score if n else pd.Series(dtype=float),
        "score_reproj_error": error_score if n else pd.Series(dtype=float),
        "score_angle": angle_score if n else pd.Series(dtype=float),
        "score_spread": spread_score if n else pd.Series(dtype=float),
        "score_consistency": consistency_score if n else pd.Series(dtype=float),
        "score_blur": blur_score if n else pd.Series(dtype=float),
    }, index=tracks.index)


def score_tracks(tracks: pd.DataFrame, config: Mapping,
                 length_only: bool = False) -> pd.DataFrame:
    """Add ``reliability`` and ``weight`` columns to a track table.

    The mapping is deterministic and does not use reconstruction success labels,
    so it is safe to evaluate as a heuristic.  ``length_only`` is an ablation.
    """
    if tracks.empty:
        result = tracks.copy()
        result["reliability"] = pd.Series(dtype=float)
        result["weight"] = pd.Series(dtype=float)
        return result

    weighting = get_cfg(config, "weighting", {}) or {}
    w_min = float(weighting.get("w_min", 0.05))
    w_min = float(np.clip(w_min, 0.0, 1.0))
    map_mode = str(weighting.get("map_mode", "sigmoid"))
    steepness = float(weighting.get("sigmoid_steepness", 6.0))
    refs = {**DEFAULT_REFS, **(weighting.get("refs", {}) or {})}
    feature_weights = dict(DEFAULT_FEATURE_WEIGHTS)
    feature_weights.update(weighting.get("feature_weights", {}) or {})

    if length_only:
        components = track_feature_components(tracks, refs)[["score_length"]]
        composite = components["score_length"].fillna(0.0)
    else:
        components = track_feature_components(tracks, refs)
        composite = _rowwise_weighted(
            {
                "length": components["score_length"],
                "reproj_error": components["score_reproj_error"],
                "angle": components["score_angle"],
                "spread": components["score_spread"],
                "consistency": components["score_consistency"],
                "blur": components["score_blur"],
            },
            feature_weights,
        )

    if map_mode == "sigmoid":
        reliability = _sigmoid((composite.to_numpy(dtype=float) - 0.5) * steepness)
    elif map_mode == "quantile":
        reliability = composite.rank(pct=True, method="average").to_numpy(dtype=float)
        if len(reliability) and np.allclose(np.nanmin(reliability), np.nanmax(reliability)):
            reliability = np.full_like(reliability, 0.5)
    elif map_mode == "linear":
        reliability = np.clip(composite.to_numpy(dtype=float), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown weighting.map_mode: {map_mode} (use sigmoid|quantile|linear)")

    result = tracks.copy()
    result = pd.concat([result, components], axis=1)
    result["composite_score"] = composite.to_numpy(dtype=float)
    result["reliability"] = np.asarray(reliability, dtype=float)
    result["weight"] = w_min + (1.0 - w_min) * result["reliability"].to_numpy(dtype=float)
    return result


def weight_summary(weights: pd.DataFrame) -> dict[str, float]:
    if weights.empty:
        return {"count": 0.0}
    values = pd.to_numeric(weights["weight"], errors="coerce").dropna()
    reliability = pd.to_numeric(weights["reliability"], errors="coerce").dropna()
    return {
        "count": int(len(values)),
        "weight_mean": float(values.mean()),
        "weight_std": float(values.std(ddof=0)),
        "weight_min": float(values.min()),
        "weight_p10": float(values.quantile(0.10)),
        "weight_median": float(values.median()),
        "weight_max": float(values.max()),
        "reliability_mean": float(reliability.mean()) if len(reliability) else float("nan"),
        "fraction_below_0_1": float((values < 0.1).mean()),
        "fraction_above_0_9": float((values > 0.9).mean()),
    }
