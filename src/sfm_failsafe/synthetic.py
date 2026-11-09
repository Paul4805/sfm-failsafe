"""Synthetic multi-view fixtures for tests and clearly-labelled demos.

The synthetic scene is a textured sphere viewed by cameras on a ring.  It is
*not* a replacement for ETH3D/COLMAP: it exists so unit tests and the pipeline
can run on machines without a dataset.  Every artifact produced by this module
is marked ``data_kind: synthetic`` and any simulated reconstruction is marked
``engine: synthetic_simulated_mapper``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import track_analysis
from .database import encode_pair_id, write_database
from .model_io import (Camera, Image, Point3D, Reconstruction, rotmat_to_qvec,
                       write_model)
from .utils import dump_json, ensure_dir, load_json


def make_scene(num_points: int = 500, radius: float = 1.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_points, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / num_points)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * indices
    points = np.column_stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ]) * radius
    points *= rng.uniform(0.95, 1.05, size=(num_points, 1))
    normalized = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-9)
    colors = np.clip((normalized * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return points, colors


def make_cameras(num_images: int = 12, width: int = 640, height: int = 480,
                 radius: float = 3.4, fov_deg: float = 55.0,
                 elevation_deg: float = 12.0, arc_deg: float = 360.0,
                 seed: int = 0) -> tuple[Camera, list[Image]]:
    rng = np.random.default_rng(seed + 1000)
    focal = 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2.0)
    camera = Camera(1, "PINHOLE", width, height,
                    np.array([focal, focal, width / 2.0, height / 2.0]))
    elevation = np.deg2rad(elevation_deg)
    images: list[Image] = []
    for i in range(num_images):
        if num_images == 1:
            azimuth = 0.0
        elif arc_deg >= 359.9:
            azimuth = 2.0 * np.pi * i / num_images
        else:
            azimuth = np.deg2rad(-arc_deg / 2.0 + arc_deg * i / max(num_images - 1, 1))
        center = radius * np.array([
            np.cos(azimuth) * np.cos(elevation),
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation),
        ])
        center += rng.normal(scale=0.01, size=3)
        forward = -center / np.linalg.norm(center)
        up_world = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])
        tvec = -rotation @ center
        images.append(Image(i + 1, rotmat_to_qvec(rotation), tvec, 1,
                            f"image_{i + 1:04d}.png", camera=camera))
    return camera, images


def compute_visibility(points: np.ndarray, camera: Camera, images: Sequence[Image],
                       margin: int = 4) -> dict[int, np.ndarray]:
    visibility: dict[int, np.ndarray] = {}
    normals = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-9)
    for image in images:
        uv, valid = image.project_world(points)
        center = image.camera_center
        facing = np.einsum("ij,ij->i", normals, center[None, :] - points) > 0.0
        in_bounds = (
            (uv[:, 0] >= margin) & (uv[:, 0] < camera.width - margin) &
            (uv[:, 1] >= margin) & (uv[:, 1] < camera.height - margin)
        )
        visibility[image.image_id] = np.flatnonzero(valid & facing & in_bounds)
    return visibility


def render_images(points: np.ndarray, colors: np.ndarray, camera: Camera,
                  images: Sequence[Image], visibility: Mapping[int, np.ndarray],
                  out_dir: str | Path, blur_sigma: float = 0.0, seed: int = 0) -> list[Path]:
    import cv2

    rng = np.random.default_rng(seed + 2000)
    out_dir = ensure_dir(out_dir)
    paths: list[Path] = []
    for image in images:
        gradient = np.linspace(35, 90, camera.height, dtype=np.float32)[:, None]
        background = np.repeat(gradient, camera.width, axis=1)
        canvas = np.stack([background * 0.75, background * 0.85, background], axis=2)
        canvas += rng.normal(0.0, 7.0, size=canvas.shape)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        visible = visibility.get(image.image_id, np.zeros((0,), dtype=int))
        uv, valid = image.project_world(points[visible])
        depths = (points[visible] - image.camera_center) @ image.rotation[2]
        order = np.argsort(depths)[::-1]  # far first
        for index in order:
            if not valid[index]:
                continue
            u, v = uv[index]
            color = colors[visible[index]]
            cv2.circle(canvas, (int(round(u)), int(round(v))), 2,
                       (int(color[2]), int(color[1]), int(color[0])), -1,
                       lineType=cv2.LINE_AA)
        if blur_sigma and blur_sigma > 0:
            canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=float(blur_sigma))
        path = out_dir / image.name
        ensure_dir(path.parent)
        cv2.imwrite(str(path), canvas)
        paths.append(path)
    return paths


def build_reconstruction(points: np.ndarray, colors: np.ndarray, camera: Camera,
                          images: Sequence[Image], visibility: Mapping[int, np.ndarray],
                          noise_px: float = 0.4, outlier_fraction: float = 0.02,
                          pose_noise_deg: float = 0.0,
                          registered_image_ids: Sequence[int] | None = None,
                          seed: int = 0) -> Reconstruction:
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed + 3000)
    keep = list(registered_image_ids) if registered_image_ids is not None else [im.image_id for im in images]
    keep = sorted(set(int(i) for i in keep))
    image_by_id = {image.image_id: image for image in images}
    kept_images = [image_by_id[i] for i in keep if i in image_by_id]
    if not kept_images:
        raise ValueError("No registered images selected")

    # Count observations among kept images; keep tracks with >= 2 observations.
    obs_count = np.zeros(len(points), dtype=np.int64)
    for image in kept_images:
        obs_count[visibility[image.image_id]] += 1
    retained_points = np.flatnonzero(obs_count >= 2)
    point_remap = {int(pid): i + 1 for i, pid in enumerate(retained_points)}

    rec = Reconstruction(cameras={camera.camera_id: camera})
    point_obs: dict[int, list[tuple[int, int, float]]] = {pid: [] for pid in retained_points}
    for image in kept_images:
        rotation = image.rotation
        tvec = image.tvec
        if pose_noise_deg > 0:
            delta = Rotation.from_rotvec(rng.normal(0.0, np.deg2rad(pose_noise_deg), 3)).as_matrix()
            rotation = delta @ rotation
            tvec = delta @ tvec
        noisy_image = Image(image.image_id, rotmat_to_qvec(rotation), tvec,
                            image.camera_id, image.name, camera=camera)
        visible = visibility[image.image_id]
        uv, valid = noisy_image.project_world(points[visible])
        xys = []
        point_ids = []
        for local_index, point_index in enumerate(visible):
            if not valid[local_index]:
                continue
            pid = int(point_index)
            u, v = uv[local_index] + rng.normal(0.0, max(noise_px, 1e-6), 2)
            if pid in point_remap and rng.random() < outlier_fraction:
                u += rng.normal(0.0, max(noise_px, 0.1) * 15.0)
                v += rng.normal(0.0, max(noise_px, 0.1) * 15.0)
            xys.append([u, v])
            point_ids.append(point_remap.get(pid, -1))
            if pid in point_remap:
                point_obs[pid].append((image.image_id, len(xys) - 1,
                                       float(np.hypot(u - uv[local_index][0],
                                                      v - uv[local_index][1]))))
        noisy_image.xys = np.asarray(xys, dtype=np.float64).reshape(-1, 2)
        noisy_image.point3D_ids = np.asarray(point_ids, dtype=np.int64)
        rec.images[image.image_id] = noisy_image

    for pid in retained_points:
        observations = point_obs[int(pid)]
        if len(observations) < 2:
            continue
        point_id = point_remap[int(pid)]
        error = float(np.sqrt(np.mean([err ** 2 for _, _, err in observations])))
        rec.points3D[point_id] = Point3D(
            point_id,
            points[int(pid)].copy(),
            colors[int(pid)].copy(),
            error,
            np.asarray([image_id for image_id, _, _ in observations], dtype=np.int64),
            np.asarray([idx for _, idx, _ in observations], dtype=np.int64),
        )
    return rec.link_cameras()


def _database_from_reconstruction(rec: Reconstruction, db_path: str | Path,
                                  outlier_ratio: float = 0.15,
                                  min_shared: int = 3, extra_keypoints: int = 10,
                                  seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 4000)
    keypoints: dict[int, np.ndarray] = {}
    point_index: dict[int, dict[int, int]] = {}
    for image_id, image in rec.images.items():
        point_index[image_id] = {}
        for idx, point_id in enumerate(image.point3D_ids):
            if int(point_id) >= 0:
                point_index[image_id][int(point_id)] = idx
        extra = rng.uniform([0, 0], [rec.cameras[image.camera_id].width,
                                    rec.cameras[image.camera_id].height],
                            size=(extra_keypoints, 2))
        keypoints[image_id] = np.vstack([image.xys, extra]) if image.xys.size else extra

    images = list(rec.images.values())
    matches: dict[int, np.ndarray] = {}
    inliers: dict[int, np.ndarray] = {}
    pair_info: list[dict[str, Any]] = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            id1, id2 = images[i].image_id, images[j].image_id
            common = sorted(set(point_index[id1]) & set(point_index[id2]))
            if len(common) < min_shared:
                continue
            true_pairs = np.asarray([[point_index[id1][p], point_index[id2][p]]
                                     for p in common], dtype=np.uint32)
            n_outliers = int(round(outlier_ratio * len(true_pairs)))
            if n_outliers:
                out1 = rng.integers(0, keypoints[id1].shape[0], size=n_outliers)
                out2 = rng.integers(0, keypoints[id2].shape[0], size=n_outliers)
                outliers = np.column_stack([out1, out2]).astype(np.uint32)
                all_matches = np.vstack([true_pairs, outliers])
            else:
                all_matches = true_pairs
            pair_id = encode_pair_id(id1, id2)
            matches[pair_id] = all_matches
            inliers[pair_id] = true_pairs
            pair_info.append({
                "image_id1": id1, "image_id2": id2, "pair_id": pair_id,
                "num_matches": int(all_matches.shape[0]),
                "num_inliers": int(true_pairs.shape[0]),
                "inlier_ratio": float(true_pairs.shape[0] / max(all_matches.shape[0], 1)),
            })
    write_database(db_path, rec.cameras.values(), images, keypoints, matches, inliers)
    return {"pairs": pair_info}


def generate_synthetic_dataset(out_dir: str | Path, num_images: int = 12,
                               num_points: int = 500, width: int = 640,
                               height: int = 480, noise_px: float = 0.4,
                               outlier_fraction: float = 0.02,
                               pose_noise_deg: float = 0.0,
                               blur_sigma: float = 0.0,
                               image_subset: Sequence[int] | None = None,
                               outlier_ratio_matches: float = 0.15,
                               min_shared: int = 3, seed: int = 0,
                               write_images: bool = True) -> dict[str, Any]:
    """Create a complete synthetic dataset directory (images/db/model/manifest)."""
    out_dir = ensure_dir(Path(out_dir))
    points, colors = make_scene(num_points, seed=seed)
    camera, all_images = make_cameras(num_images, width, height, seed=seed)
    if image_subset is not None:
        images = [all_images[i] for i in image_subset if 0 <= i < len(all_images)]
        # Re-number images/cameras deterministically.
        for new_id, image in enumerate(images, start=1):
            image.image_id = new_id
            image.name = f"image_{new_id:04d}.png"
    else:
        images = all_images
    visibility = compute_visibility(points, camera, images)
    if write_images:
        render_images(points, colors, camera, images, visibility,
                      out_dir / "images", blur_sigma=blur_sigma, seed=seed)
    rec = build_reconstruction(points, colors, camera, images, visibility,
                               noise_px=noise_px, outlier_fraction=outlier_fraction,
                               pose_noise_deg=pose_noise_deg, seed=seed)
    write_model(rec, out_dir / "model")
    db_info = _database_from_reconstruction(
        rec, out_dir / "database.db", outlier_ratio=outlier_ratio_matches,
        min_shared=min_shared, seed=seed,
    )
    manifest = {
        "dataset": "synthetic_sphere",
        "data_kind": "synthetic",
        "condition": "clean",
        "engine": "synthetic_generator",
        "num_images": len(images),
        "num_points": int(rec.num_points),
        "image_names": [image.name for image in images],
        "images_dir": str((out_dir / "images").resolve()),
        "camera": {
            "model": camera.model, "width": camera.width, "height": camera.height,
            "params": camera.params.tolist(),
        },
        "noise_px": float(noise_px),
        "blur_sigma": float(blur_sigma),
        "outlier_fraction": float(outlier_fraction),
        "pose_noise_deg": float(pose_noise_deg),
        "num_pairs": len(db_info["pairs"]),
        "seed": int(seed),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    dump_json(manifest, out_dir / "manifest.json")
    dump_json({"pairs": db_info["pairs"]}, out_dir / "database_pairs.json")
    return {
        "reconstruction": rec,
        "images_dir": out_dir / "images",
        "database": out_dir / "database.db",
        "model_dir": out_dir / "model",
        "manifest": manifest,
        "scene": {"points": points, "colors": colors, "camera": camera},
    }


def subset_reconstruction(rec: Reconstruction,
                          keep_image_ids: Iterable[int]) -> Reconstruction:
    keep = sorted(set(int(i) for i in keep_image_ids if int(i) in rec.images))
    result = rec.copy()
    for image_id in list(result.images):
        if image_id not in keep:
            del result.images[image_id]
    for point_id in list(result.points3D):
        point = result.points3D[point_id]
        mask = np.isin(point.image_ids, keep)
        if mask.sum() < 2:
            del result.points3D[point_id]
            continue
        point.image_ids = point.image_ids[mask]
        point.point2D_idxs = point.point2D_idxs[mask]
    kept_points = set(result.points3D)
    for image in result.images.values():
        image.point3D_ids = np.asarray([
            pid if int(pid) in kept_points else -1 for pid in image.point3D_ids
        ], dtype=np.int64)
    return result


def simulate_registration(rec: Reconstruction, database_path: str | Path,
                          min_links: int = 6) -> list[int]:
    """Greedy registration simulation from verified two-view inlier counts.

    This is a *simulation* for synthetic demos, not COLMAP.  It starts from the
    image with the most distinct links and grows the registered set.
    """
    from .database import ColmapDatabase, decode_pair_id

    with ColmapDatabase(database_path) as db:
        pair_records = db.pair_records()
    name_to_id = {image.name: image.image_id for image in rec.images.values()}
    all_ids = set(rec.images)
    adjacency: dict[int, dict[int, int]] = {i: {} for i in all_ids}
    for row in pair_records.itertuples(index=False):
        id1, id2 = int(row.image_id1), int(row.image_id2)
        if id1 not in all_ids or id2 not in all_ids:
            continue
        adjacency[id1][id2] = int(row.num_inliers)
        adjacency[id2][id1] = int(row.num_inliers)
    if not adjacency:
        return []
    start = max(adjacency, key=lambda i: (len(adjacency[i]), i))
    registered = {start}
    remaining = all_ids - registered
    while remaining:
        best, best_links = None, 0
        for candidate in sorted(remaining):
            links = max((weight for other, weight in adjacency[candidate].items()
                         if other in registered), default=0)
            if links > best_links:
                best, best_links = candidate, links
        if best is None or best_links < min_links:
            break
        registered.add(best)
        remaining.remove(best)
    return sorted(registered)


DEMO_VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("clean", {"blur": 0.0, "drop": 0.0, "subset": None, "noise": 0.4}),
    ("blur_mild", {"blur": 2.0, "drop": 0.10, "subset": None, "noise": 0.7}),
    ("blur_severe", {"blur": 5.0, "drop": 0.50, "subset": None, "noise": 1.4}),
    ("limited_views", {"blur": 0.0, "drop": 0.0, "subset": [0, 2, 4, 6, 8, 10], "noise": 0.6}),
    ("limited_views_severe", {"blur": 1.0, "drop": 0.50, "subset": [0, 3, 6, 9], "noise": 1.0}),
    ("degraded_matches_mild", {"blur": 0.0, "drop": 0.45, "subset": None, "noise": 0.7}),
    ("degraded_matches_severe", {"blur": 0.0, "drop": 0.85, "subset": None, "noise": 1.5}),
    ("combined_severe", {"blur": 3.0, "drop": 0.70, "subset": [0, 2, 4, 6, 8, 10], "noise": 1.2}),
]


def generate_demo_runs(out_root: str | Path, config: Mapping,
                       seeds: Sequence[int] = (0, 1),
                       num_images: int = 12, num_points: int = 3000,
                       width: int = 640, height: int = 480,
                       min_links: int = 100, overwrite: bool = False,
                       logger=None) -> list[dict[str, Any]]:
    """Create **synthetic** summary/database runs for pipeline validation.

    A simulated mapper decides registration from the verified two-view match
    graph.  These runs are not COLMAP results and are written with
    ``data_kind=synthetic`` and ``engine=synthetic_simulated_mapper``.
    """
    import os

    from . import evaluation, features
    from .database import drop_match_fraction

    out_root = ensure_dir(Path(out_root))
    created: list[dict[str, Any]] = []
    for variant, params in DEMO_VARIANTS:
        for seed in seeds:
            run_name = f"{variant}_s{seed}"
            run_dir = out_root / run_name
            if run_dir.exists() and any(run_dir.iterdir()):
                if overwrite:
                    import shutil
                    shutil.rmtree(run_dir)
                else:
                    raise ValueError(
                        f"Demo run directory exists: {run_dir}. Use overwrite=True."
                    )
            ensure_dir(run_dir)
            started = time.perf_counter()
            if logger:
                logger.info("Generating synthetic run %s", run_name)
            dataset = generate_synthetic_dataset(
                run_dir,
                num_images=num_images,
                num_points=num_points,
                width=width,
                height=height,
                noise_px=float(params["noise"]),
                outlier_fraction=0.02,
                pose_noise_deg=0.15,
                blur_sigma=float(params["blur"]),
                image_subset=params["subset"],
                outlier_ratio_matches=0.20,
                min_shared=3,
                seed=seed,
            )
            full_rec = dataset["reconstruction"]
            database = Path(dataset["database"])
            if float(params["drop"]) > 0:
                dropped = run_dir / "database_dropped.db"
                drop_match_fraction(database, dropped, float(params["drop"]),
                                    seed=seed, drop_pairs=True)
                os.replace(dropped, database)
            registered = simulate_registration(full_rec, database, min_links=min_links)
            if registered:
                sub_rec = subset_reconstruction(full_rec, registered)
                status = "success"
            else:
                sub_rec = Reconstruction()
                status = "failed"
            write_model(sub_rec, run_dir / "model")
            image_names = dataset["manifest"]["image_names"]
            analysis = track_analysis.analyze_reconstruction(
                sub_rec, config,
                image_dir=run_dir / "images",
                db_path=database,
                all_image_names=image_names,
                input_image_count=len(image_names),
            )
            track_analysis.write_analysis(analysis, run_dir)
            summary = analysis["summary"]
            summary.update({
                "run_name": run_name,
                "condition": variant,
                "data_kind": "synthetic",
                "engine": "synthetic_simulated_mapper",
                "status": status,
                "model_dir": str((run_dir / "model").resolve()),
                "database": str(database.resolve()),
                "num_input_images": len(image_names),
                "registered_image_names": [full_rec.image_name(i) for i in registered
                                           if i in full_rec.images],
                "synthetic_simulation": True,
                "simulation_parameters": {
                    "blur_sigma": float(params["blur"]),
                    "match_drop_fraction": float(params["drop"]),
                    "image_subset": params["subset"],
                    "noise_px": float(params["noise"]),
                    "min_links": int(min_links),
                    "seed": int(seed),
                },
                "runtime_seconds": time.perf_counter() - started,
            })
            label = evaluation.label_summary(summary, config)
            summary["success_label"] = label["label"]
            summary["success_label_name"] = label["label_name"]
            summary["failure_reasons"] = label["reasons"]
            summary["label_checks"] = label["checks"]
            dump_json(summary, run_dir / "reconstruction_summary.json")
            dump_json({"label": label}, run_dir / "label.json")
            run_manifest = {
                "run_name": run_name,
                "condition": variant,
                "data_kind": "synthetic",
                "engine": "synthetic_simulated_mapper",
                "synthetic_simulation": True,
                "images_dir": str((run_dir / "images").resolve()),
                "image_names": image_names,
                "num_input_images": len(image_names),
                "database": str(database.resolve()),
                "simulation_parameters": summary["simulation_parameters"],
            }
            dump_json(run_manifest, run_dir / "run_manifest.json")
            features.pre_reconstruction_features(run_dir, config, write=True)
            created.append({
                "run_name": run_name,
                "run_dir": str(run_dir),
                "condition": variant,
                "status": status,
                "label": label["label"],
                "label_name": label["label_name"],
                "reasons": label["reasons"],
                "num_input_images": len(image_names),
                "num_registered_images": summary["num_registered_images"],
                "num_points3D": summary["num_points3D"],
                "mean_reproj_error": summary["mean_reproj_error"],
            })
    return created

