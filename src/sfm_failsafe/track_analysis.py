"""Track, image, and image-pair analysis for COLMAP reconstructions.

The output tables are the shared currency of the project:

* ``tracks.csv``  - one row per 3D point / feature track
* ``images.csv``  - one row per input image (registered or not)
* ``pairs.csv``   - one row per covisible (and optionally matched) image pair
* ``reconstruction_summary.json`` - run-level statistics used for labels

All statistics are derived from the COLMAP model, never invented.  The
triangulation angle and track reprojection errors are recomputed from the
model's camera poses and 2D observations; COLMAP's stored point error is kept
as ``stored_error`` for comparison.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import reliability
from .database import ColmapDatabase, decode_pair_id
from .model_io import Camera, Reconstruction
from .config import get_cfg
from .utils import safe_float


def _observation_table(rec: Reconstruction) -> pd.DataFrame:
    obs = rec.observation_array()
    return pd.DataFrame({
        "point3D_id": obs["point3D_id"],
        "image_id": obs["image_id"],
        "point2D_idx": obs["point2D_idx"],
        "u": obs["uv"][:, 0],
        "v": obs["uv"][:, 1],
        "xyz_x": obs["xyz"][:, 0],
        "xyz_y": obs["xyz"][:, 1],
        "xyz_z": obs["xyz"][:, 2],
    })


def compute_observation_errors(rec: Reconstruction, obs: pd.DataFrame) -> np.ndarray:
    """Recompute reprojection error for every observation (NaN if invalid)."""
    errors = np.full(len(obs), np.nan, dtype=np.float64)
    if obs.empty:
        return errors
    xyz_all = obs[["xyz_x", "xyz_y", "xyz_z"]].to_numpy(dtype=np.float64)
    uv_all = obs[["u", "v"]].to_numpy(dtype=np.float64)
    image_ids = obs["image_id"].to_numpy()
    for image_id in np.unique(image_ids):
        mask = image_ids == image_id
        image = rec.images.get(int(image_id))
        camera = rec.cameras.get(int(image.camera_id)) if image is not None else None
        if image is None or camera is None:
            continue
        uv_pred, valid = image.project_world(xyz_all[mask])
        diff = uv_pred - uv_all[mask]
        dist = np.linalg.norm(diff, axis=1)
        dist[~valid] = np.nan
        errors[mask] = dist
    return errors


def _pairwise_triangulation_angles(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """All pairwise viewing-ray angles (degrees) for one track."""
    rays = centers - points
    norms = np.linalg.norm(rays, axis=1)
    good = norms > 1e-9
    if good.sum() < 2:
        return np.zeros((0,), dtype=np.float64)
    rays = rays[good] / norms[good, None]
    dots = np.clip(rays @ rays.T, -1.0, 1.0)
    iu = np.triu_indices(rays.shape[0], k=1)
    return np.degrees(np.arccos(dots[iu]))


def _subsample(indices: np.ndarray, max_count: int) -> np.ndarray:
    if len(indices) <= max_count:
        return indices
    positions = np.linspace(0, len(indices) - 1, max_count).round().astype(int)
    return indices[positions]


def compute_track_table(rec: Reconstruction, config: Mapping | None = None) -> pd.DataFrame:
    config = config or {}
    max_points = int(get_cfg(config, "analysis.max_points", 200_000))
    angle_sample = int(get_cfg(config, "analysis.angle_sample", 8))
    min_length = int(get_cfg(config, "analysis.min_track_length", 2))

    obs = _observation_table(rec)
    centers = rec.camera_centers()
    if obs.empty:
        empty = pd.DataFrame(columns=[
            "point3D_id", "track_length", "num_images", "mean_reproj_error",
            "median_reproj_error", "max_reproj_error", "stored_error",
            "min_image_id", "max_image_id", "image_spread", "image_spread_norm",
            "image_diversity", "triangulation_angle_deg", "mean_depth", "std_depth",
            "min_depth", "max_depth", "depth_ratio", "reliability", "weight",
        ])
        return empty

    errors = compute_observation_errors(rec, obs)
    obs = obs.assign(reproj_error=errors)
    point_xyz = obs[["point3D_id", "xyz_x", "xyz_y", "xyz_z"]].drop_duplicates("point3D_id")
    xyz_by_id = {
        int(row.point3D_id): np.array([row.xyz_x, row.xyz_y, row.xyz_z], dtype=np.float64)
        for row in point_xyz.itertuples(index=False)
    }
    depth = np.full(len(obs), np.nan)
    for image_id in np.unique(obs["image_id"].to_numpy()):
        mask = obs["image_id"].to_numpy() == image_id
        if int(image_id) in centers:
            depth[mask] = np.linalg.norm(
                obs.loc[mask, ["xyz_x", "xyz_y", "xyz_z"]].to_numpy() - centers[int(image_id)],
                axis=1,
            )
    obs = obs.assign(depth=depth)

    agg = obs.groupby("point3D_id", sort=True).agg(
        track_length=("image_id", "size"),
        num_images=("image_id", "nunique"),
        mean_reproj_error=("reproj_error", "mean"),
        median_reproj_error=("reproj_error", "median"),
        max_reproj_error=("reproj_error", "max"),
        min_image_id=("image_id", "min"),
        max_image_id=("image_id", "max"),
        mean_depth=("depth", "mean"),
        std_depth=("depth", "std"),
        min_depth=("depth", "min"),
        max_depth=("depth", "max"),
        num_valid_errors=("reproj_error", "count"),
    )
    agg["stored_error"] = [float(rec.points3D[int(pid)].error) for pid in agg.index]
    # If 2D observations were unavailable, fall back to COLMAP's stored error.
    fallback = agg["mean_reproj_error"].isna() | (agg["num_valid_errors"] == 0)
    agg.loc[fallback, "mean_reproj_error"] = agg.loc[fallback, "stored_error"]
    agg.loc[fallback, "median_reproj_error"] = agg.loc[fallback, "stored_error"]
    agg.loc[fallback, "max_reproj_error"] = agg.loc[fallback, "stored_error"]

    spread_rows: list[dict[str, float | int]] = []
    grouped = obs.groupby("point3D_id", sort=True)
    for point_id, group in grouped:
        point_id = int(point_id)
        image_ids = group["image_id"].to_numpy(dtype=np.int64)
        point = xyz_by_id[point_id]
        unique_images = np.unique(image_ids)
        if len(unique_images) >= 2:
            cam_centers = np.stack([centers[int(i)] for i in unique_images if int(i) in centers])
            if len(cam_centers) >= 2:
                spread = float(np.linalg.norm(cam_centers.max(axis=0) - cam_centers.min(axis=0)))
            else:
                spread = 0.0
            sample_indices = _subsample(np.arange(len(unique_images)), angle_sample)
            sample_centers = np.stack([
                centers[int(unique_images[i])] for i in sample_indices
                if int(unique_images[i]) in centers
            ]) if any(int(unique_images[i]) in centers for i in sample_indices) else np.zeros((0, 3))
            angles = _pairwise_triangulation_angles(point, sample_centers)
            angle_deg = float(np.max(angles)) if angles.size else 0.0
        else:
            spread = 0.0
            angle_deg = 0.0
        depth_vals = group["depth"].dropna().to_numpy(dtype=np.float64)
        median_depth = float(np.median(depth_vals)) if depth_vals.size else np.nan
        spread_norm = (spread / median_depth) if (median_depth and np.isfinite(median_depth)
                                                  and median_depth > 1e-9) else 0.0
        spread_rows.append({
            "point3D_id": point_id,
            "image_spread": spread,
            "image_spread_norm": float(spread_norm),
            "triangulation_angle_deg": angle_deg,
        })
    spread_df = pd.DataFrame(spread_rows).set_index("point3D_id")
    tracks = agg.join(spread_df, how="left")
    tracks["image_diversity"] = tracks["num_images"] / tracks["track_length"].clip(lower=1)
    tracks["depth_ratio"] = tracks["max_depth"] / tracks["min_depth"].replace(0, np.nan)
    tracks["mean_reproj_error"] = tracks["mean_reproj_error"].fillna(tracks["stored_error"])
    tracks["point3D_id"] = tracks.index.astype(np.int64)
    tracks = tracks.reset_index(drop=True)
    tracks = tracks.sort_values("point3D_id").reset_index(drop=True)
    tracks["image_id_span"] = tracks["max_image_id"] - tracks["min_image_id"]

    tracks = reliability.score_tracks(tracks, config)
    if len(tracks) > max_points:
        # Deterministic cap: keep the strongest tracks, but never silently
        # mis-report the reconstruction size (summary uses len(points3D)).
        tracks = tracks.sort_values(
            ["track_length", "weight", "point3D_id"], ascending=[False, False, True]
        ).head(max_points).sort_values("point3D_id").reset_index(drop=True)
    return tracks


def _blur_score(path: Path, max_width: int = 1024) -> float:
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return float("nan")
        if image.shape[1] > max_width:
            scale = max_width / float(image.shape[1])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(image, cv2.CV_64F).var())
    except Exception:
        return float("nan")


def compute_image_table(rec: Reconstruction, tracks: pd.DataFrame,
                        config: Mapping | None = None,
                        image_dir: str | Path | None = None,
                        all_image_names: Sequence[str] | None = None) -> pd.DataFrame:
    config = config or {}
    blur_max_images = int(get_cfg(config, "analysis.blur_max_images", 200))
    blur_max_width = int(get_cfg(config, "analysis.blur_max_width", 1024))
    image_dir = Path(image_dir) if image_dir else None

    name_to_id = {image.name: image.image_id for image in rec.images.values()}
    names: list[str] = list(all_image_names) if all_image_names else sorted(name_to_id)
    for name in sorted(name_to_id):
        if name not in names:
            names.append(name)
    names = sorted(set(names))

    point_lengths = dict(zip(tracks["point3D_id"], tracks["track_length"])) if not tracks.empty else {}
    known_point_ids = np.fromiter(point_lengths.keys(), dtype=np.int64) if point_lengths else np.zeros((0,), dtype=np.int64)
    obs_records: list[dict] = []
    for image_id, image in rec.images.items():
        valid = image.point3D_ids >= 0
        point_ids = image.point3D_ids[valid]
        if known_point_ids.size:
            point_ids = point_ids[np.isin(point_ids, known_point_ids)]
        lengths = [point_lengths.get(int(p), np.nan) for p in point_ids]
        obs_records.append({
            "image_id": int(image_id),
            "name": image.name,
            "camera_id": int(image.camera_id),
            "num_observations": int(valid.sum()),
            "mean_track_length": float(np.nanmean(lengths)) if lengths else np.nan,
            "camera_center_x": float(image.camera_center[0]),
            "camera_center_y": float(image.camera_center[1]),
            "camera_center_z": float(image.camera_center[2]),
            "registered": True,
        })
    image_stats = pd.DataFrame(obs_records)
    if not image_stats.empty:
        image_stats = image_stats.set_index("image_id")

    # Reprojection error per image from the observation table.
    obs = _observation_table(rec)
    if not obs.empty:
        obs["reproj_error"] = compute_observation_errors(rec, obs)
        per_image = obs.groupby("image_id")["reproj_error"].agg(["mean", "median", "count"])
        per_image = per_image.rename(columns={
            "mean": "mean_reproj_error", "median": "median_reproj_error",
            "count": "num_valid_observations",
        })
        image_stats = image_stats.join(per_image, how="left") if not image_stats.empty else per_image

    rows: list[dict] = []
    for index, name in enumerate(names):
        image_id = int(name_to_id.get(name, -1))
        if not image_stats.empty and image_id in image_stats.index:
            row = image_stats.loc[image_id].to_dict()
            row["image_id"] = image_id
        else:
            row = {
                "image_id": image_id, "camera_id": -1,
                "num_observations": 0, "mean_track_length": 0.0,
                "camera_center_x": np.nan, "camera_center_y": np.nan,
                "camera_center_z": np.nan, "registered": False,
                "mean_reproj_error": np.nan, "median_reproj_error": np.nan,
                "num_valid_observations": 0,
            }
        row["name"] = name
        row["registered"] = bool(row.get("registered", False))
        blur = float("nan")
        if image_dir is not None and index < blur_max_images:
            candidate = image_dir / name
            if candidate.exists():
                blur = _blur_score(candidate, blur_max_width)
        row["blur_score"] = blur
        rows.append(row)
    images = pd.DataFrame(rows)

    # Image overlap = number of other registered images sharing >=1 track.
    overlap: dict[int, set[int]] = {int(i): set() for i in rec.images}
    for image_id, image in rec.images.items():
        valid = image.point3D_ids[image.point3D_ids >= 0]
        for point_id in valid:
            point = rec.points3D.get(int(point_id))
            if point is None:
                continue
            for other in point.image_ids:
                if int(other) != int(image_id):
                    overlap[int(image_id)].add(int(other))
    images["overlap"] = [
        len(overlap.get(int(i), set())) if int(i) >= 0 else 0
        for i in images["image_id"].to_numpy()
    ]
    if not images.empty:
        images = images.sort_values(["registered", "name"], ascending=[False, True]).reset_index(drop=True)
    return images


def compute_pair_table(rec: Reconstruction, tracks: pd.DataFrame,
                       db_path: str | Path | None = None) -> pd.DataFrame:
    centers = rec.camera_centers()
    pair_counts: dict[tuple[int, int], dict[str, float]] = {}
    angles: dict[tuple[int, int], list[float]] = {}
    for point_id, point in rec.points3D.items():
        image_ids = np.unique(point.image_ids)
        if len(image_ids) < 2:
            continue
        sample = _subsample(np.arange(len(image_ids)), 30)
        image_ids = image_ids[sample]
        subset_centers = np.stack([centers[int(i)] for i in image_ids if int(i) in centers])
        if len(subset_centers) != len(image_ids):
            continue
        angles_for_point = _pairwise_triangulation_angles(point.xyz, subset_centers)
        if angles_for_point.size:
            upper = np.triu_indices(len(image_ids), k=1)
            for k, (i, j) in enumerate(zip(*upper)):
                pair = (int(min(image_ids[i], image_ids[j])), int(max(image_ids[i], image_ids[j])))
                entry = pair_counts.setdefault(pair, {
                    "num_shared_points": 0, "num_shared_points_ge3": 0,
                    "angle_sum": 0.0, "angle_count": 0,
                })
                entry["num_shared_points"] += 1
                if point.track_length >= 3:
                    entry["num_shared_points_ge3"] += 1
                entry["angle_sum"] += float(angles_for_point[k])
                entry["angle_count"] += 1
    rows = []
    for (id1, id2), entry in sorted(pair_counts.items()):
        center1, center2 = centers.get(id1), centers.get(id2)
        baseline = float(np.linalg.norm(center1 - center2)) if center1 is not None and center2 is not None else np.nan
        rows.append({
            "image_id1": id1,
            "image_id2": id2,
            "image1": rec.image_name(id1),
            "image2": rec.image_name(id2),
            "num_shared_points": int(entry["num_shared_points"]),
            "num_shared_points_ge3": int(entry["num_shared_points_ge3"]),
            "baseline": baseline,
            "median_pair_angle_deg": (entry["angle_sum"] / entry["angle_count"]
                                      if entry["angle_count"] else np.nan),
        })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        pairs = pd.DataFrame(columns=[
            "image_id1", "image_id2", "image1", "image2", "num_shared_points",
            "num_shared_points_ge3", "baseline", "median_pair_angle_deg",
        ])

    if db_path is not None and Path(db_path).exists():
        try:
            with ColmapDatabase(db_path) as db:
                db_pairs = db.pair_records()
            if not db_pairs.empty:
                pairs = pairs.merge(
                    db_pairs[["image_id1", "image_id2", "num_matches", "num_inliers",
                              "inlier_ratio"]],
                    on=["image_id1", "image_id2"], how="outer",
                )
                if "image1" in pairs.columns:
                    id_to_name = {img.image_id: img.name for img in rec.images.values()}
                    pairs["image1"] = pairs["image1"].fillna(
                        pairs["image_id1"].map(id_to_name)
                    )
                    pairs["image2"] = pairs["image2"].fillna(
                        pairs["image_id2"].map(id_to_name)
                    )
        except Exception:
            pass
    return pairs.sort_values(["image_id1", "image_id2"]).reset_index(drop=True)


def summarize_reconstruction(rec: Reconstruction, tracks: pd.DataFrame | None = None,
                             input_image_count: int | None = None,
                             max_points: int | None = None) -> dict:
    """Aggregate metrics used for success/failure labels and comparisons."""
    if tracks is None:
        tracks = compute_track_table(rec, {})
    num_registered = rec.num_images
    num_input = int(input_image_count) if input_image_count else num_registered
    summary: dict = {
        "num_input_images": num_input,
        "num_registered_images": int(num_registered),
        "registration_ratio": float(num_registered / num_input) if num_input else 0.0,
        "num_cameras": len(rec.cameras),
        "num_points3D": int(rec.num_points),
        "num_observations": int(sum(p.track_length for p in rec.points3D.values())),
        "mean_reproj_error": float("nan"),
        "median_reproj_error": float("nan"),
        "max_reproj_error": float("nan"),
        "p95_reproj_error": float("nan"),
        "track_length_mean": float("nan"),
        "track_length_median": float("nan"),
        "track_length_p90": float("nan"),
        "track_length_max": float("nan"),
        "num_tracks": int(len(tracks)),
        "num_tracks_ge3": int((tracks["track_length"] >= 3).sum()) if not tracks.empty else 0,
        "frac_tracks_ge3": float((tracks["track_length"] >= 3).mean()) if not tracks.empty else 0.0,
        "mean_observations_per_registered_image": (
            float(np.mean([img.point3D_ids[img.point3D_ids >= 0].size
                           for img in rec.images.values()]))
            if rec.images else 0.0
        ),
    }
    if not tracks.empty:
        summary.update({
            "mean_reproj_error": safe_float(tracks["mean_reproj_error"].mean()),
            "median_reproj_error": safe_float(tracks["mean_reproj_error"].median()),
            "max_reproj_error": safe_float(tracks["mean_reproj_error"].max()),
            "p95_reproj_error": safe_float(tracks["mean_reproj_error"].quantile(0.95)),
            "track_length_mean": safe_float(tracks["track_length"].mean()),
            "track_length_median": safe_float(tracks["track_length"].median()),
            "track_length_p90": safe_float(tracks["track_length"].quantile(0.90)),
            "track_length_max": safe_float(tracks["track_length"].max()),
        })
        centers = np.stack(list(rec.camera_centers().values())) if rec.images else np.zeros((0, 3))
        if len(centers) >= 2:
            summary["camera_bbox_diagonal"] = float(
                np.linalg.norm(centers.max(axis=0) - centers.min(axis=0))
            )
        xyz, _ = rec.point_array()
        if len(xyz):
            summary["point_bbox_diagonal"] = float(
                np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
            )
    return summary


def analyze_reconstruction(rec: Reconstruction,
                           config: Mapping | None = None,
                           image_dir: str | Path | None = None,
                           db_path: str | Path | None = None,
                           all_image_names: Sequence[str] | None = None,
                           input_image_count: int | None = None) -> dict:
    config = config or {}
    tracks = compute_track_table(rec, config)
    images = compute_image_table(rec, tracks, config, image_dir, all_image_names)
    pairs = compute_pair_table(rec, tracks, db_path=db_path)
    summary = summarize_reconstruction(rec, tracks, input_image_count=input_image_count)
    summary["num_covis_pairs"] = int(len(pairs))
    if not tracks.empty:
        summary["weight_mean"] = safe_float(tracks["weight"].mean())
        summary["reliability_mean"] = safe_float(tracks["reliability"].mean())
    return {
        "tracks": tracks,
        "images": images,
        "pairs": pairs,
        "summary": summary,
    }


def write_analysis(analysis: Mapping, out_dir: str | Path,
                   write_pairs: bool = True) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    analysis["tracks"].to_csv(out_dir / "tracks.csv", index=False)
    paths["tracks"] = out_dir / "tracks.csv"
    analysis["images"].to_csv(out_dir / "images.csv", index=False)
    paths["images"] = out_dir / "images.csv"
    if write_pairs:
        analysis["pairs"].to_csv(out_dir / "pairs.csv", index=False)
        paths["pairs"] = out_dir / "pairs.csv"
    from .utils import dump_json

    dump_json(analysis["summary"], out_dir / "reconstruction_summary.json")
    paths["summary"] = out_dir / "reconstruction_summary.json"
    return paths


def load_tracks(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir) / "tracks.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"tracks.csv not found in {run_dir}; run scripts/analyze_tracks.py first"
        )
    return pd.read_csv(path)
