"""Run-level features, including leakage-free pre-reconstruction candidate tracks.

Two feature modes exist:

``pre``
    Uses only the COLMAP database (keypoints, putative matches, verified
    two-view geometries), image statistics, and match-graph candidate tracks.
    No mapper output is touched.  This is the primary failure-prediction mode.

``post``
    Uses the reconstruction/track statistics.  These features are diagnostic
    only and are **not leakage-free** if the goal is to predict mapper failure.

Candidate tracks are connected components of the keypoint match graph.  They
are intentionally computed before reconstruction: they approximate the tracks
COLMAP *would* build, without using any 3D point / pose output.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import evaluation
from .database import ColmapDatabase, decode_pair_id
from .config import get_cfg
from .utils import (ActionableError, dump_json, load_json, safe_float,
                    setup_logger)

NODE_SHIFT = 32
NODE_MASK = (1 << NODE_SHIFT) - 1

PRE_METADATA_COLUMNS = {
    "run_name", "condition", "scene", "subset_id", "data_kind", "mode",
    "label", "label_name", "label_reasons", "generated_at",
}
PRE_FEATURE_COLUMNS = [
    "num_input_images",
    "num_cameras",
    "num_keypoints",
    "keypoints_per_image",
    "image_width",
    "image_height",
    "aspect_ratio",
    "num_pairs",
    "verified_pair_fraction",
    "pair_match_mean",
    "pair_match_median",
    "pair_match_p90",
    "pair_inlier_mean",
    "pair_inlier_median",
    "pair_inlier_p10",
    "candidate_track_count",
    "candidate_track_mean_len",
    "candidate_track_median_len",
    "candidate_track_p90_len",
    "candidate_track_max_len",
    "candidate_frac_len_ge3",
    "candidate_frac_len_ge5",
    "candidate_mean_image_diversity",
    "candidate_mean_inlier_consistency",
    "candidate_mean_kp_spread_norm",
    "candidate_mean_image_span_norm",
    "candidate_largest_component_frac",
    "candidate_edges_per_node",
    "candidate_nodes_per_image",
    "blur_mean",
    "blur_median",
    "blur_p10",
    "blur_coverage",
]


def _image_arrays(db: ColmapDatabase, image_ids: Sequence[int]) -> dict[int, np.ndarray]:
    return {int(image_id): db.keypoints(int(image_id)) for image_id in image_ids}


def _union_find_components(nodes1: np.ndarray, nodes2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (unique_nodes, component_ids, edge_endpoint_indices)."""
    if nodes1.size == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, np.zeros((2, 0), dtype=np.int64)
    all_nodes = np.concatenate([nodes1, nodes2])
    unique_nodes, inverse = np.unique(all_nodes, return_inverse=True)
    parent = np.arange(unique_nodes.size, dtype=np.int64)
    rank = np.zeros(unique_nodes.size, dtype=np.int8)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    idx1 = inverse[:nodes1.size]
    idx2 = inverse[nodes1.size:]
    for a, b in zip(idx1.tolist(), idx2.tolist()):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
    roots = np.fromiter((find(i) for i in range(unique_nodes.size)),
                        dtype=np.int64, count=unique_nodes.size)
    unique_roots, component_ids = np.unique(roots, return_inverse=True)
    return unique_nodes, component_ids, np.stack([idx1, idx2])


def candidate_tracks_from_database(
    db_path: str | Path,
    min_length: int = 2,
    use_verified: bool = True,
    max_pairs: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build candidate tracks from the match graph (pre-reconstruction)."""
    with ColmapDatabase(db_path) as db:
        images = db.images()
        cameras = db.cameras()
        pair_records = db.pair_records()
        stats = db.stats()
        image_ids = images["image_id"].astype(int).tolist()
        keypoints = _image_arrays(db, image_ids)

        if pair_records.empty:
            return _empty_candidate_result(stats)

        if max_pairs is not None and len(pair_records) > int(max_pairs):
            # Deterministic, evenly spaced subset; record the truncation.
            positions = np.linspace(0, len(pair_records) - 1, int(max_pairs)).round().astype(int)
            pair_records = pair_records.iloc[np.unique(positions)].reset_index(drop=True)
            truncated = True
        else:
            truncated = False

        edge_nodes1: list[np.ndarray] = []
        edge_nodes2: list[np.ndarray] = []
        edge_inlier_ratio: list[np.ndarray] = []
        pair_match_counts: list[int] = []
        pair_inlier_counts: list[int] = []
        for row in pair_records.itertuples(index=False):
            pair_id = int(row.pair_id)
            matches = db.matches(pair_id)
            pair_match_counts.append(int(matches.shape[0]))
            inlier_ratio = float(row.inlier_ratio) if np.isfinite(row.inlier_ratio) else np.nan
            verified = db.two_view_geometry(pair_id) if use_verified else None
            if verified is not None and verified.shape[0] > 0:
                edges = verified
                pair_inlier_counts.append(int(verified.shape[0]))
            else:
                edges = matches
                pair_inlier_counts.append(0)
                inlier_ratio = np.nan
            if edges.size == 0:
                continue
            id1, id2 = decode_pair_id(pair_id)
            nodes1 = id1 * (1 << NODE_SHIFT) + edges[:, 0].astype(np.int64)
            nodes2 = id2 * (1 << NODE_SHIFT) + edges[:, 1].astype(np.int64)
            edge_nodes1.append(nodes1)
            edge_nodes2.append(nodes2)
            edge_inlier_ratio.append(np.full(nodes1.size, inlier_ratio))

    if not edge_nodes1:
        return _empty_candidate_result(stats)

    nodes1 = np.concatenate(edge_nodes1)
    nodes2 = np.concatenate(edge_nodes2)
    edge_ratio = np.concatenate(edge_inlier_ratio)
    unique_nodes, component_ids, edge_indices = _union_find_components(nodes1, nodes2)
    num_components = int(component_ids.max()) + 1 if component_ids.size else 0
    if num_components == 0:
        return _empty_candidate_result(stats)

    lengths = np.bincount(component_ids, minlength=num_components)
    image_ids_of_nodes = (unique_nodes >> NODE_SHIFT).astype(np.int64)
    kp_indices = (unique_nodes & NODE_MASK).astype(np.int64)

    # Unique images per component.
    unique_image_ids = {int(v): i for i, v in enumerate(sorted(set(image_ids_of_nodes.tolist())))}
    image_token = np.fromiter(
        (unique_image_ids[int(v)] for v in image_ids_of_nodes), dtype=np.int64,
        count=len(image_ids_of_nodes),
    )
    pairs = np.unique(np.column_stack([component_ids, image_token]), axis=0)
    images_per_component = np.bincount(pairs[:, 0], minlength=num_components)

    # Keypoint coordinates and per-component bbox spread in pixels.
    xs = np.full(unique_nodes.size, np.nan, dtype=np.float64)
    ys = np.full(unique_nodes.size, np.nan, dtype=np.float64)
    for image_id, points in keypoints.items():
        mask = image_ids_of_nodes == int(image_id)
        if not mask.any() or points.size == 0:
            continue
        idx = kp_indices[mask]
        valid = idx < points.shape[0]
        local = np.flatnonzero(mask)
        xs[local[valid]] = points[idx[valid], 0]
        ys[local[valid]] = points[idx[valid], 1]
    min_x = np.full(num_components, np.inf)
    max_x = np.full(num_components, -np.inf)
    min_y = np.full(num_components, np.inf)
    max_y = np.full(num_components, -np.inf)
    valid_xy = np.isfinite(xs) & np.isfinite(ys)
    np.minimum.at(min_x, component_ids[valid_xy], xs[valid_xy])
    np.maximum.at(max_x, component_ids[valid_xy], xs[valid_xy])
    np.minimum.at(min_y, component_ids[valid_xy], ys[valid_xy])
    np.maximum.at(max_y, component_ids[valid_xy], ys[valid_xy])
    spread_px = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    spread_px[~np.isfinite(spread_px)] = 0.0

    image_min = np.full(num_components, np.iinfo(np.int64).max, dtype=np.int64)
    image_max = np.full(num_components, -1, dtype=np.int64)
    np.minimum.at(image_min, component_ids, image_ids_of_nodes)
    np.maximum.at(image_max, component_ids, image_ids_of_nodes)
    image_span = np.where(image_max >= 0, image_max - image_min, 0).astype(np.float64)

    # Match support and inlier consistency per component.
    edge_components = component_ids[edge_indices[0]] if edge_indices.shape[1] else np.zeros((0,), dtype=np.int64)
    match_support = np.bincount(edge_components, minlength=num_components)
    valid_ratio = np.isfinite(edge_ratio)
    ratio_sum = np.bincount(edge_components[valid_ratio], weights=edge_ratio[valid_ratio],
                            minlength=num_components)
    ratio_count = np.bincount(edge_components[valid_ratio], minlength=num_components)
    consistency = np.divide(ratio_sum, ratio_count, out=np.full(num_components, np.nan),
                            where=ratio_count > 0)

    # Normalisation references.
    if not cameras.empty:
        width = float(np.median(cameras["width"].astype(float)))
        height = float(np.median(cameras["height"].astype(float)))
    else:
        width, height = 640.0, 480.0
    image_diagonal = max(float(np.hypot(width, height)), 1.0)
    num_image_values = max(len(image_ids), 1)
    spread_norm = spread_px / image_diagonal
    span_norm = image_span / max(num_image_values - 1, 1)

    tracks = pd.DataFrame({
        "candidate_track_id": np.arange(num_components, dtype=np.int64),
        "track_length": lengths,
        "num_images": images_per_component,
        "image_diversity": images_per_component / np.maximum(lengths, 1),
        "match_support": match_support,
        "inlier_consistency": consistency,
        "kp_spread_px": spread_px,
        "kp_spread_norm": spread_norm,
        "image_id_span": image_span,
        "image_span_norm": span_norm,
    })
    total_components_all = len(tracks)
    tracks = tracks[tracks["track_length"] >= int(min_length)].reset_index(drop=True)

    total_nodes = int(unique_nodes.size)
    summary = {
        "candidate_total_nodes": total_nodes,
        "candidate_total_components": int(total_components_all),
        "candidate_total_edges": int(nodes1.size),
        "candidate_pair_count_used": int(len(pair_match_counts)),
        "candidate_pairs_truncated": bool(truncated),
        "candidate_track_count": int(len(tracks)),
        "candidate_track_mean_len": safe_float(tracks["track_length"].mean()) if len(tracks) else 0.0,
        "candidate_track_median_len": safe_float(tracks["track_length"].median()) if len(tracks) else 0.0,
        "candidate_track_p90_len": safe_float(tracks["track_length"].quantile(0.90)) if len(tracks) else 0.0,
        "candidate_track_max_len": int(tracks["track_length"].max()) if len(tracks) else 0,
        "candidate_frac_len_ge3": safe_float((tracks["track_length"] >= 3).mean()) if len(tracks) else 0.0,
        "candidate_frac_len_ge5": safe_float((tracks["track_length"] >= 5).mean()) if len(tracks) else 0.0,
        "candidate_mean_image_diversity": safe_float(tracks["image_diversity"].mean()) if len(tracks) else 0.0,
        "candidate_mean_inlier_consistency": safe_float(tracks["inlier_consistency"].mean()) if len(tracks) else 0.0,
        "candidate_mean_kp_spread_norm": safe_float(tracks["kp_spread_norm"].mean()) if len(tracks) else 0.0,
        "candidate_mean_image_span_norm": safe_float(tracks["image_span_norm"].mean()) if len(tracks) else 0.0,
        "candidate_largest_component_frac": safe_float(
            (tracks["track_length"].max() / total_nodes) if len(tracks) and total_nodes else 0.0
        ),
        "candidate_edges_per_node": safe_float(nodes1.size / max(total_nodes, 1)),
        "candidate_nodes_per_image": safe_float(total_nodes / max(len(image_ids), 1)),
        "num_keypoints": int(stats.get("num_keypoints", 0)),
        "keypoints_per_image": safe_float(stats.get("num_keypoints", 0) / max(stats.get("num_images", 1), 1)),
        "image_width": width,
        "image_height": height,
        "aspect_ratio": width / max(height, 1.0),
        "num_pairs": int(len(pair_match_counts)),
        "verified_pair_fraction": safe_float(
            np.mean(np.asarray(pair_inlier_counts) > 0) if pair_inlier_counts else 0.0
        ),
    }
    if pair_match_counts:
        counts = np.asarray(pair_match_counts, dtype=np.float64)
        summary.update({
            "pair_match_mean": float(np.mean(counts)),
            "pair_match_median": float(np.median(counts)),
            "pair_match_p90": float(np.quantile(counts, 0.90)),
        })
        ratios = np.asarray([r for r in edge_ratio if np.isfinite(r)], dtype=np.float64)
        if ratios.size:
            summary.update({
                "pair_inlier_mean": float(np.mean(ratios)),
                "pair_inlier_median": float(np.median(ratios)),
                "pair_inlier_p10": float(np.quantile(ratios, 0.10)),
            })
    return tracks, summary


def _empty_candidate_result(stats: Mapping) -> tuple[pd.DataFrame, dict[str, Any]]:
    tracks = pd.DataFrame(columns=[
        "candidate_track_id", "track_length", "num_images", "image_diversity",
        "match_support", "inlier_consistency", "kp_spread_px", "kp_spread_norm",
        "image_id_span", "image_span_norm",
    ])
    summary = {
        "candidate_total_nodes": 0, "candidate_total_components": 0,
        "candidate_total_edges": 0, "candidate_track_count": 0,
        "candidate_track_mean_len": 0.0, "candidate_track_median_len": 0.0,
        "candidate_track_p90_len": 0.0, "candidate_track_max_len": 0,
        "candidate_frac_len_ge3": 0.0, "candidate_frac_len_ge5": 0.0,
        "candidate_mean_image_diversity": 0.0, "candidate_mean_inlier_consistency": 0.0,
        "candidate_mean_kp_spread_norm": 0.0, "candidate_mean_image_span_norm": 0.0,
        "candidate_largest_component_frac": 0.0, "candidate_edges_per_node": 0.0,
        "candidate_nodes_per_image": 0.0,
        "num_keypoints": int(stats.get("num_keypoints", 0)),
        "keypoints_per_image": 0.0,
        "num_pairs": 0, "verified_pair_fraction": 0.0,
        "pair_match_mean": 0.0, "pair_match_median": 0.0, "pair_match_p90": 0.0,
        "pair_inlier_mean": np.nan, "pair_inlier_median": np.nan, "pair_inlier_p10": np.nan,
    }
    return tracks, summary


def image_blur_stats(image_dir: str | Path | None, names: Sequence[str] = (),
                     max_images: int = 200, max_width: int = 1024) -> dict[str, float]:
    result = {"blur_mean": np.nan, "blur_median": np.nan, "blur_p10": np.nan,
              "blur_coverage": 0.0}
    if image_dir is None or not names:
        return result
    image_dir = Path(image_dir)
    try:
        import cv2
    except ImportError:
        return result
    values: list[float] = []
    for name in list(names)[:max_images]:
        path = image_dir / name
        if not path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if image.shape[1] > max_width:
            scale = max_width / float(image.shape[1])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        values.append(float(cv2.Laplacian(image, cv2.CV_64F).var()))
    if values:
        arr = np.asarray(values, dtype=np.float64)
        result.update({
            "blur_mean": float(np.mean(arr)),
            "blur_median": float(np.median(arr)),
            "blur_p10": float(np.quantile(arr, 0.10)),
            "blur_coverage": float(len(values) / max(len(names), 1)),
        })
    return result


def _run_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if path.exists():
        try:
            return load_json(path, {})
        except Exception:
            return {}
    return {}


def pre_reconstruction_features(run_dir: str | Path, config: Mapping,
                                write: bool = True) -> dict[str, Any]:
    """Extract leakage-free pre-reconstruction features for one run dir."""
    run_dir = Path(run_dir)
    manifest = _run_manifest(run_dir)
    db_path = Path(manifest.get("database", run_dir / "database.db"))
    if not db_path.is_absolute():
        db_path = run_dir / db_path
    if not db_path.exists():
        raise ActionableError(
            f"No COLMAP database for run {run_dir.name}: {db_path}\n"
            "Pre-reconstruction features require database.db from feature extraction + matching."
        )

    min_length = int(get_cfg(config, "features.min_candidate_track_length", 2))
    max_pairs = get_cfg(config, "features.max_db_pairs", None)
    with ColmapDatabase(db_path) as db:
        db_stats = db.stats()
        db_images = db.images()
        cameras = db.cameras()
    tracks, candidate_summary = candidate_tracks_from_database(
        db_path, min_length=min_length,
        max_pairs=int(max_pairs) if max_pairs else None,
    )

    feature_values: dict[str, Any] = {
        "num_input_images": int(db_stats.get("num_images", 0)),
        "num_cameras": int(db_stats.get("num_cameras", 0)),
        **{k: v for k, v in candidate_summary.items()},
    }
    if not cameras.empty:
        feature_values["image_width"] = float(np.median(cameras["width"].astype(float)))
        feature_values["image_height"] = float(np.median(cameras["height"].astype(float)))
        feature_values["aspect_ratio"] = feature_values["image_width"] / max(
            feature_values["image_height"], 1.0
        )

    image_dir = manifest.get("images_dir")
    if image_dir:
        image_dir = Path(image_dir)
        if not image_dir.is_absolute():
            image_dir = run_dir / image_dir
    else:
        configured = get_cfg(config, "dataset.images_dir", None)
        image_dir = Path(configured) if configured else None
        if image_dir is not None and not image_dir.is_absolute():
            image_dir = Path(get_cfg(config, "_meta.repo_root", ".")) / image_dir
    names = manifest.get("image_names") or db_images["name"].tolist()
    feature_values.update(image_blur_stats(
        image_dir, names,
        int(get_cfg(config, "analysis.blur_max_images", 200)),
        int(get_cfg(config, "analysis.blur_max_width", 1024)),
    ))

    for column in PRE_FEATURE_COLUMNS:
        feature_values.setdefault(column, np.nan)

    meta = {
        "run_name": str(manifest.get("run_name", run_dir.name)),
        "condition": str(manifest.get("condition", "unknown")),
        "scene": str(manifest.get("scene", get_cfg(config, "dataset.name", "unknown"))),
        "subset_id": manifest.get("subset_id"),
        "data_kind": str(manifest.get("data_kind", "real")),
        "mode": "pre",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "database": str(db_path),
        "num_candidate_tracks_saved": int(len(tracks)),
    }
    payload = {"meta": meta, "features": feature_values}
    tracks.to_csv(run_dir / "candidate_tracks.csv", index=False)
    if write:
        dump_json(payload, run_dir / "features_pre.json")
    return payload


def post_reconstruction_features(run_dir: str | Path, config: Mapping,
                                 write: bool = True) -> dict[str, Any]:
    """Diagnostic post-reconstruction features (not leakage-free)."""
    run_dir = Path(run_dir)
    manifest = _run_manifest(run_dir)
    summary_path = run_dir / "reconstruction_summary.json"
    if not summary_path.exists():
        raise ActionableError(
            f"No reconstruction_summary.json in {run_dir}; run baseline/analysis first."
        )
    summary = load_json(summary_path, {})
    features: dict[str, Any] = {
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
        "num_tracks_ge3": summary.get("num_tracks_ge3"),
        "frac_tracks_ge3": summary.get("frac_tracks_ge3"),
        "mean_observations_per_registered_image": summary.get("mean_observations_per_registered_image"),
        "num_covis_pairs": summary.get("num_covis_pairs"),
        "weight_mean": summary.get("weight_mean"),
        "reliability_mean": summary.get("reliability_mean"),
    }
    tracks_path = run_dir / "tracks.csv"
    if tracks_path.exists():
        tracks = pd.read_csv(tracks_path)
        if not tracks.empty:
            features.update({
                "track_weight_p10": safe_float(tracks["weight"].quantile(0.10)) if "weight" in tracks else np.nan,
                "track_weight_median": safe_float(tracks["weight"].median()) if "weight" in tracks else np.nan,
                "track_error_mean": safe_float(tracks["mean_reproj_error"].mean()),
                "track_length_max": safe_float(tracks["track_length"].max()),
            })
    images_path = run_dir / "images.csv"
    if images_path.exists():
        images = pd.read_csv(images_path)
        if "blur_score" in images:
            features["blur_mean"] = safe_float(images["blur_score"].mean())
            features["blur_p10"] = safe_float(images["blur_score"].quantile(0.10))
    meta = {
        "run_name": str(manifest.get("run_name", run_dir.name)),
        "condition": str(manifest.get("condition", summary.get("condition", "unknown"))),
        "scene": str(manifest.get("scene", summary.get("scene", "unknown"))),
        "data_kind": str(manifest.get("data_kind", summary.get("data_kind", "real"))),
        "mode": "post",
        "leakage_free": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    payload = {"meta": meta, "features": features}
    if write:
        dump_json(payload, run_dir / "features_post.json")
    return payload


def iter_run_dirs(runs_root: str | Path) -> list[Path]:
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    run_dirs: list[Path] = []
    for summary in sorted(runs_root.rglob("reconstruction_summary.json")):
        run_dir = summary.parent
        relative = run_dir.relative_to(runs_root)
        if any(part in {"adaptive", "models", "analysis", "refined"} for part in relative.parts):
            continue
        if len(relative.parts) > 3:
            continue
        run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def collect_runs(runs_root: str | Path, config: Mapping, mode: str = "pre",
                 logger=None) -> tuple[pd.DataFrame, list[str]]:
    """Collect run-level labels and features from a runs directory."""
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    run_dirs = iter_run_dirs(runs_root)
    if not run_dirs:
        return pd.DataFrame(), [f"No run directories with reconstruction_summary.json under {runs_root}"]
    for run_dir in run_dirs:
        summary_path = run_dir / "reconstruction_summary.json"
        try:
            summary = load_json(summary_path, {})
        except Exception as exc:
            warnings.append(f"{run_dir.name}: cannot read summary ({exc})")
            continue
        label_info = evaluation.label_summary(summary, config)
        manifest = _run_manifest(run_dir)
        base_row = {
            "run_name": str(manifest.get("run_name", run_dir.name)),
            "condition": str(manifest.get("condition", summary.get("condition", "unknown"))),
            "scene": str(manifest.get("scene", summary.get("scene", "unknown"))),
            "subset_id": manifest.get("subset_id"),
            "data_kind": str(manifest.get("data_kind", summary.get("data_kind", "real"))),
            "mode": mode,
            "label": label_info["label"],
            "label_name": label_info["label_name"],
            "label_reasons": "; ".join(label_info["reasons"]),
            "run_dir": str(run_dir),
        }
        try:
            if mode == "pre":
                feature_path = run_dir / "features_pre.json"
                payload = load_json(feature_path, {}) if feature_path.exists() else \
                    pre_reconstruction_features(run_dir, config, write=True)
            elif mode == "post":
                feature_path = run_dir / "features_post.json"
                payload = load_json(feature_path, {}) if feature_path.exists() else \
                    post_reconstruction_features(run_dir, config, write=True)
            else:
                raise ActionableError(f"Unknown feature mode: {mode}")
            features = payload.get("features", payload)
            row = {**base_row, **{k: v for k, v in features.items() if not isinstance(v, (dict, list))}}
            rows.append(row)
        except Exception as exc:
            warnings.append(f"{run_dir.name}: skipped ({exc})")
            if logger:
                logger.warning("Skipping %s: %s", run_dir, exc)
    frame = pd.DataFrame(rows)
    return frame, warnings


def feature_columns(df: pd.DataFrame, mode: str = "pre") -> list[str]:
    if df.empty:
        return []
    excluded = set(PRE_METADATA_COLUMNS) | {"run_dir", "label_reasons"}
    numeric: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric.append(column)
    return numeric
