"""Weighted bundle adjustment / refinement of an existing COLMAP model.

This is a **post-mapper refinement**, not an integration into COLMAP's
incremental mapper.  It refines camera poses and 3D points of an already
reconstructed model with per-track weights while keeping intrinsics fixed.

Engine choices
--------------
``scipy``
    Implemented and tested here: ``scipy.optimize.least_squares`` with an
    analytic sparse Jacobian, local SE(3) camera updates, fixed gauge camera,
    and a robust loss.  This is the documented fallback required by the
    assignment when pyceres is unavailable.
``pyceres``
    Detected at runtime.  A native Ceres problem builder is intentionally not
    faked: if pyceres is present but the adapter is unavailable, the run
    records that fact and uses scipy instead.  See ``engine_used`` in the
    report.
``colmap``
    Approximate path implemented in ``adaptive_weighting``: filter low-weight
    tracks, then call COLMAP ``bundle_adjuster``.  It is clearly labelled an
    approximation because COLMAP does not consume the continuous weights.
"""

from __future__ import annotations

import time
from typing import Iterable, Mapping

import numpy as np

from .model_io import Image, Reconstruction, rotmat_to_qvec
from .config import get_cfg
from .utils import ActionableError, format_seconds, safe_float


def _camera_arrays(rec: Reconstruction, point_ids: list[int],
                   weight_map: Mapping[int, float], sigma_px: float):
    """Flatten selected observations into aligned arrays for BA."""
    image_ids: list[int] = []
    point_indices: list[int] = []
    uv: list[np.ndarray] = []
    point_slots: dict[int, int] = {}
    image_ids_all: list[int] = []
    outliers = 0
    for point_id in point_ids:
        point = rec.points3D[point_id]
        for image_id, idx in zip(point.image_ids, point.point2D_idxs):
            image = rec.images.get(int(image_id))
            if image is None or int(idx) >= image.xys.shape[0]:
                outliers += 1
                continue
            image_ids.append(int(image_id))
            uv.append(image.xys[int(idx)])
            if point_id not in point_slots:
                point_slots[point_id] = len(point_slots)
            point_indices.append(point_slots[point_id])
    if not uv:
        return None
    image_ids_arr = np.asarray(image_ids, dtype=np.int64)
    point_idx_arr = np.asarray(point_indices, dtype=np.int64)
    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    point_ids_ordered = [pid for pid, _ in sorted(point_slots.items(), key=lambda kv: kv[1])]
    xyz = np.stack([rec.points3D[pid].xyz for pid in point_ids_ordered])

    # Undistort observations once (BA is pinhole; intrinsics stay fixed).
    uv_pinhole = np.empty_like(uv_arr)
    fx = np.empty(len(uv_arr))
    fy = np.empty(len(uv_arr))
    cx = np.empty(len(uv_arr))
    cy = np.empty(len(uv_arr))
    image_to_index = {int(image_id): i for i, image_id in enumerate(
        sorted(rec.images)
    )}
    free_image_ids = sorted(rec.images)[1:]  # first camera is the gauge
    free_slots = {int(image_id): i for i, image_id in enumerate(free_image_ids)}
    for image_id in np.unique(image_ids_arr):
        mask = image_ids_arr == image_id
        image = rec.images[int(image_id)]
        camera = rec.cameras[int(image.camera_id)]
        uv_pinhole[mask] = camera.undistort_to_pinhole(uv_arr[mask])
        fx[mask], fy[mask] = camera.focal
        cx[mask], cy[mask] = camera.principal
    free_slot = np.array([
        free_slots.get(int(image_id), -1) for image_id in image_ids_arr
    ], dtype=np.int64)
    weights = np.array([max(0.0, float(weight_map.get(pid, 1.0)))
                        for pid in point_ids_ordered], dtype=np.float64)
    sqrt_w_obs = np.sqrt(weights[point_idx_arr]) / max(float(sigma_px), 1e-9)
    scale = sqrt_w_obs[:, None]
    return {
        "image_ids": image_ids_arr,
        "point_idx": point_idx_arr,
        "uv": uv_pinhole,
        "xyz": xyz,
        "point_ids": point_ids_ordered,
        "free_slot": free_slot,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "scale": scale,
        "weights": weights,
        "n_observations": int(len(uv_arr)),
        "n_outliers": outliers,
        "image_to_index": image_to_index,
        "free_image_ids": free_image_ids,
    }


def _pose_bases(rec: Reconstruction, free_image_ids: list[int]):
    rotations = np.stack([rec.images[int(i)].rotation for i in free_image_ids]) \
        if free_image_ids else np.zeros((0, 3, 3))
    translations = np.stack([rec.images[int(i)].tvec for i in free_image_ids]) \
        if free_image_ids else np.zeros((0, 3))
    return rotations, translations


def _project_residual_and_jac(x: np.ndarray, data: dict, rot0: np.ndarray, t0: np.ndarray,
                              free_image_ids: list[int], need_jac: bool = True):
    from scipy.spatial.transform import Rotation

    n_free = len(free_image_ids)
    camera_params = x[:6 * n_free].reshape(-1, 6) if n_free else np.zeros((0, 6))
    drot = camera_params[:, :3]
    dt = camera_params[:, 3:6]
    dp = x[6 * n_free:].reshape(-1, 3)
    point_idx = data["point_idx"]
    free_slot = data["free_slot"]

    points = data["xyz"] + dp
    p_cam = np.empty((len(point_idx), 3), dtype=np.float64)
    r_new_obs = np.empty((len(point_idx), 3, 3), dtype=np.float64)
    for slot in range(n_free):
        mask = free_slot == slot
        if not mask.any():
            continue
        rotation = Rotation.from_rotvec(drot[slot]).as_matrix() if np.any(drot[slot]) else np.eye(3)
        r_new = rotation @ rot0[slot]
        t_new = rotation @ t0[slot] + dt[slot]
        p_cam[mask] = points[point_idx[mask]] @ r_new.T + t_new
        r_new_obs[mask] = r_new
    fixed_mask = free_slot < 0
    if fixed_mask.any():
        fixed_image = rec_image = data.get("fixed_image")
        if fixed_image is not None:
            r_new = fixed_image.rotation
            t_new = fixed_image.tvec
            p_cam[fixed_mask] = points[point_idx[fixed_mask]] @ r_new.T + t_new
            r_new_obs[fixed_mask] = r_new

    z = p_cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-8, np.sign(z) * 1e-8 + (z == 0) * 1e-8, z)
    inv_z = 1.0 / safe_z
    u = data["fx"] * p_cam[:, 0] * inv_z + data["cx"]
    v = data["fy"] * p_cam[:, 1] * inv_z + data["cy"]
    residual = np.stack([u - data["uv"][:, 0], v - data["uv"][:, 1]], axis=1) * data["scale"]
    flat_residual = residual.reshape(-1)
    if not need_jac:
        return flat_residual, None

    n_obs = len(point_idx)
    n_points = len(data["xyz"])
    n_params = 6 * n_free + 3 * n_points
    from scipy import sparse

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    for k in range(n_obs):
        x_c, y_c, z_c = p_cam[k]
        inv_z_k = inv_z[k]
        jp = np.array([
            [data["fx"][k] * inv_z_k, 0.0, -data["fx"][k] * x_c * inv_z_k * inv_z_k],
            [0.0, data["fy"][k] * inv_z_k, -data["fy"][k] * y_c * inv_z_k * inv_z_k],
        ])
        s = float(data["scale"][k, 0])
        jp = jp * s
        rows = (2 * k, 2 * k + 1)
        slot = int(free_slot[k])
        if slot >= 0:
            p_for_skew = p_cam[k] - dt[slot]
            skew = np.array([
                [0.0, -p_for_skew[2], p_for_skew[1]],
                [p_for_skew[2], 0.0, -p_for_skew[0]],
                [-p_for_skew[1], p_for_skew[0], 0.0],
            ])
            j_cam = jp @ (-skew)  # rotation update (2x3)
            j_trans = jp  # translation update (2x3)
            base = 6 * slot
            for a in range(3):
                row_indices.extend(rows)
                col_indices.extend((base + a, base + a))
                values.extend((j_cam[0, a], j_cam[1, a]))
            base_t = 6 * slot + 3
            for a in range(3):
                row_indices.extend(rows)
                col_indices.extend((base_t + a, base_t + a))
                values.extend((j_trans[0, a], j_trans[1, a]))
        j_point = jp @ r_new_obs[k]
        base_p = 6 * n_free + 3 * n_points  # placeholder for clarity
        base_p = 6 * n_free + 3 * data["point_idx"][k]
        for a in range(3):
            row_indices.extend(rows)
            col_indices.extend((base_p + a, base_p + a))
            values.extend((j_point[0, a], j_point[1, a]))
    jac = sparse.coo_matrix(
        (values, (row_indices, col_indices)), shape=(2 * n_obs, n_params)
    ).tocsr()
    return flat_residual, jac


def _point_errors(points_xyz: np.ndarray, data: dict, rec: Reconstruction,
                  free_image_ids: list[int], drot: np.ndarray, dt: np.ndarray,
                  dp: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    n_free = len(free_image_ids)
    points = points_xyz + dp
    pcam = np.empty((len(data["point_idx"]), 3))
    slot = data["free_slot"]
    r0, t0 = data["rot0"], data["t0"]
    for s in range(n_free):
        mask = slot == s
        if not mask.any():
            continue
        rotation = Rotation.from_rotvec(drot[s]).as_matrix()
        pcam[mask] = points[data["point_idx"][mask]] @ (rotation @ r0[s]).T + rotation @ t0[s] + dt[s]
    fixed = slot < 0
    if fixed.any():
        image = data["fixed_image"]
        pcam[fixed] = points[data["point_idx"][fixed]] @ image.rotation.T + image.tvec
    z = np.where(np.abs(pcam[:, 2]) < 1e-8, 1e-8, pcam[:, 2])
    u = data["fx"] * pcam[:, 0] / z + data["cx"]
    v = data["fy"] * pcam[:, 1] / z + data["cy"]
    return np.linalg.norm(np.stack([u - data["uv"][:, 0], v - data["uv"][:, 1]], axis=1), axis=1)


def _error_stats(errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {"rmse": float("nan"), "mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.quantile(errors, 0.95)),
    }


def retriangulate_points(rec: Reconstruction, point_ids: Iterable[int] | None = None,
                         iterations: int = 3) -> tuple[Reconstruction, dict]:
    """Re-triangulate points against fixed (already refined) camera poses.

    After weighted pose refinement, tracks outside the optimized subset can be
    inconsistent with the new cameras.  This pass solves a robust-ish
    Gauss-Newton triangulation per point with cameras held fixed.  It does not
    add or remove tracks and does not change intrinsics/poses.
    """
    refined = rec.copy()
    if point_ids is None:
        targets = list(refined.points3D)
    else:
        targets = [int(pid) for pid in point_ids if int(pid) in refined.points3D]
    updated = 0
    before_errors: list[float] = []
    after_errors: list[float] = []
    for point_id in targets:
        point = refined.points3D[point_id]
        obs = []
        for image_id, idx in zip(point.image_ids, point.point2D_idxs):
            image = refined.images.get(int(image_id))
            if image is None or int(idx) >= image.xys.shape[0]:
                continue
            camera = refined.cameras.get(int(image.camera_id))
            if camera is None:
                continue
            uv = camera.undistort_to_pinhole(image.xys[int(idx)][None, :])[0]
            if not np.all(np.isfinite(uv)):
                continue
            obs.append((image, camera, uv))
        if len(obs) < 2:
            continue
        # Initial error for reporting.
        initial = np.zeros(len(obs))
        for k, (image, camera, uv) in enumerate(obs):
            pcam = image.rotation @ point.xyz + image.tvec
            if pcam[2] <= 1e-8:
                initial[k] = np.nan
            else:
                pred = camera.project(pcam[None, :])[0]
                initial[k] = np.linalg.norm(pred - uv)
        if not np.all(np.isfinite(initial)):
            continue

        xyz = point.xyz.copy()
        converged = False
        for _ in range(max(int(iterations), 1)):
            rows, rhs = [], []
            for image, camera, uv in obs:
                rotation = image.rotation
                tvec = image.tvec
                fx, fy = camera.focal
                cx, cy = camera.principal
                u_n = (uv[0] - cx) / fx
                v_n = (uv[1] - cy) / fy
                rows.append(u_n * rotation[2] - rotation[0])
                rhs.append(tvec[0] - u_n * tvec[2])
                rows.append(v_n * rotation[2] - rotation[1])
                rhs.append(tvec[1] - v_n * tvec[2])
            matrix = np.asarray(rows, dtype=np.float64)
            vector = np.asarray(rhs, dtype=np.float64)
            try:
                solution, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(solution)):
                break
            xyz = solution
            converged = True
            break  # linear triangulation is followed by Gauss-Newton below

        if not converged:
            continue

        # Gauss-Newton on the geometric reprojection error.
        ok = True
        for _ in range(max(int(iterations), 1)):
            residuals = []
            jacobians = []
            valid = True
            for image, camera, uv in obs:
                rotation = image.rotation
                pcam = rotation @ xyz + image.tvec
                if pcam[2] <= 1e-8:
                    valid = False
                    break
                fx, fy = camera.focal
                cx, cy = camera.principal
                inv_z = 1.0 / pcam[2]
                pred = np.array([fx * pcam[0] * inv_z + cx,
                                 fy * pcam[1] * inv_z + cy])
                jp = np.array([[fx * inv_z, 0.0, -fx * pcam[0] * inv_z * inv_z],
                               [0.0, fy * inv_z, -fy * pcam[1] * inv_z * inv_z]])
                residuals.append(pred - uv)
                jacobians.append(jp @ rotation)
            if not valid:
                ok = False
                break
            r = np.concatenate(residuals)
            j = np.vstack(jacobians)
            try:
                delta, *_ = np.linalg.lstsq(j, -r, rcond=None)
            except np.linalg.LinAlgError:
                ok = False
                break
            if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) < 1e-10:
                break
            xyz = xyz + delta
        if not ok or not np.all(np.isfinite(xyz)):
            continue

        final = np.zeros(len(obs))
        for k, (image, camera, uv) in enumerate(obs):
            pcam = image.rotation @ xyz + image.tvec
            if pcam[2] <= 1e-8:
                final[k] = np.nan
            else:
                pred = camera.project(pcam[None, :])[0]
                final[k] = np.linalg.norm(pred - uv)
        if not np.all(np.isfinite(final)):
            continue
        refined.points3D[point_id].xyz = xyz
        refined.points3D[point_id].error = float(np.mean(final))
        updated += 1
        before_errors.append(float(np.sqrt(np.mean(initial ** 2))))
        after_errors.append(float(np.sqrt(np.mean(final ** 2))))
    report = {
        "num_candidates": len(targets),
        "num_updated": int(updated),
        "iterations": int(iterations),
        "rmse_before": float(np.mean(before_errors)) if before_errors else float("nan"),
        "rmse_after": float(np.mean(after_errors)) if after_errors else float("nan"),
    }
    return refined, report


def refine_weighted(rec: Reconstruction, weights: Mapping[int, float],
                    config: Mapping, engine: str = "auto") -> tuple[Reconstruction, dict]:
    """Refine an existing reconstruction with per-track weights.

    The gauge is fixed by holding the first camera (lowest image id) constant.
    Intrinsics are held fixed.  Returns a copy of the model and a report.
    """
    started = time.perf_counter()
    ba_cfg = get_cfg(config, "weighted_ba", {}) or {}
    max_points = int(ba_cfg.get("max_points", 1500))
    max_iterations = int(ba_cfg.get("max_iterations", 40))
    loss = str(ba_cfg.get("loss", "soft_l1"))
    loss_scale = float(ba_cfg.get("loss_scale", 2.0))
    sigma_px = float(ba_cfg.get("sigma_px", 1.0))
    min_weight = float(ba_cfg.get("min_weight", 0.0))
    min_observations = int(ba_cfg.get("min_observations", 20))
    requested_engine = engine if engine != "auto" else str(ba_cfg.get("engine", "auto"))

    candidates = [
        pid for pid, point in rec.points3D.items()
        if point.track_length >= 2 and float(weights.get(pid, 1.0)) >= min_weight
    ]
    # Deterministic selection independent of the weighting mode so that
    # baseline / length-only / full comparisons use the same subset.
    candidates.sort(key=lambda pid: (-rec.points3D[pid].track_length, pid))
    selected = sorted(candidates[:max_points])

    report: dict = {
        "engine_requested": requested_engine,
        "engine_used": None,
        "num_total_points": rec.num_points,
        "num_selected_points": len(selected),
        "fixed_camera_id": min(rec.images) if rec.images else None,
        "weighted_ba": True,
        "loss": loss,
        "loss_scale": loss_scale,
        "iteration_limit": max_iterations,
    }
    if not selected:
        report.update({"status": "skipped", "reason": "no eligible tracks",
                       "runtime_seconds": time.perf_counter() - started})
        return rec.copy(), report

    data = _camera_arrays(rec, selected, weights, sigma_px)
    if data is None:
        report.update({"status": "skipped", "reason": "no valid observations",
                       "runtime_seconds": time.perf_counter() - started})
        return rec.copy(), report
    data["fixed_image"] = rec.images[min(rec.images)] if rec.images else None
    if data["n_observations"] < min_observations:
        report.update({
            "status": "skipped",
            "reason": f"only {data['n_observations']} observations < min_observations={min_observations}",
            "runtime_seconds": time.perf_counter() - started,
        })
        return rec.copy(), report

    free_image_ids = data["free_image_ids"]
    rot0, t0 = _pose_bases(rec, free_image_ids)
    data["rot0"], data["t0"] = rot0, t0
    n_points = len(data["xyz"])
    n_free = len(free_image_ids)
    n_params = 6 * n_free + 3 * n_points
    x0 = np.zeros(n_params, dtype=np.float64)

    before_errors = _point_errors(data["xyz"], data, rec, free_image_ids,
                                  np.zeros((n_free, 3)), np.zeros((n_free, 3)),
                                  np.zeros((n_points, 3)))
    report.update({
        "num_observations": int(data["n_observations"]),
        "num_free_cameras": int(n_free),
        "num_parameters": int(n_params),
        "errors_before": _error_stats(before_errors),
    })

    # pyceres is not installed in this environment.  Detect it and record the
    # fallback rather than pretending a Ceres run happened.
    pyceres_available = False
    try:
        import pyceres  # noqa: F401
        pyceres_available = True
    except ImportError:
        pyceres_available = False
    report["pyceres_available"] = pyceres_available
    if requested_engine == "pyceres" and not pyceres_available:
        report["engine_note"] = "pyceres not installed; using scipy_least_squares fallback"
    if pyceres_available and requested_engine in ("auto", "pyceres"):
        report["engine_note"] = (
            "pyceres detected but this repository's Ceres adapter is not enabled; "
            "using scipy_least_squares fallback"
        )

    try:
        from scipy.optimize import least_squares

        def fun(x: np.ndarray) -> np.ndarray:
            values, _ = _project_residual_and_jac(x, data, rot0, t0, free_image_ids,
                                                  need_jac=False)
            return values

        def jac(x: np.ndarray):
            _, matrix = _project_residual_and_jac(x, data, rot0, t0, free_image_ids,
                                                  need_jac=True)
            return matrix

        scipy_loss = "linear" if loss in ("none", "linear") else loss
        result = least_squares(
            fun, x0, jac=jac, method="trf", loss=scipy_loss,
            f_scale=max(loss_scale, 1e-6), max_nfev=max(10, max_iterations),
            verbose=0, x_scale="jac",
        )
        cost_after = float(result.cost)
        converged = bool(result.success)
        message = str(result.message)
        nfev = int(result.nfev)
        optimality = safe_float(result.optimality)
    except Exception as exc:
        report.update({
            "status": "failed",
            "engine_used": "scipy_least_squares",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_seconds": time.perf_counter() - started,
        })
        return rec.copy(), report

    x = result.x
    camera_params = x[:6 * n_free].reshape(-1, 6) if n_free else np.zeros((0, 6))
    drot = camera_params[:, :3]
    dt = camera_params[:, 3:6]
    dp = x[6 * n_free:].reshape(-1, 3)

    after_errors = _point_errors(data["xyz"], data, rec, free_image_ids, drot, dt, dp)
    after_stats = _error_stats(after_errors)
    before_stats = report["errors_before"]
    catastrophic = (
        not np.isfinite(after_stats["rmse"])
        or after_stats["rmse"] > before_stats["rmse"] * 10.0 + 100.0
    )
    if catastrophic:
        report.update({
            "status": "reverted",
            "engine_used": "scipy_least_squares",
            "converged": converged,
            "message": message,
            "nfev": nfev,
            "optimality": optimality,
            "cost_after": cost_after,
            "errors_after": after_stats,
            "reason": "refinement was numerically unstable and was reverted",
            "runtime_seconds": time.perf_counter() - started,
        })
        return rec.copy(), report

    from scipy.spatial.transform import Rotation

    refined = rec.copy()
    for slot, image_id in enumerate(free_image_ids):
        rotation = Rotation.from_rotvec(drot[slot]).as_matrix()
        r_new = rotation @ rot0[slot]
        t_new = rotation @ t0[slot] + dt[slot]
        refined.images[int(image_id)].qvec = rotmat_to_qvec(r_new)
        refined.images[int(image_id)].tvec = t_new
    for index, point_id in enumerate(data["point_ids"]):
        refined.points3D[point_id].xyz = data["xyz"][index] + dp[index]
    # Update stored point errors for the refined points only.
    for index, point_id in enumerate(data["point_ids"]):
        mask = data["point_idx"] == index
        if mask.any():
            refined.points3D[point_id].error = float(np.mean(after_errors[mask]))

    report.update({
        "status": "ok",
        "engine_used": "scipy_least_squares",
        "converged": converged,
        "message": message,
        "nfev": int(nfev),
        "optimality": optimality,
        "cost_after": cost_after,
        "errors_after": after_stats,
        "rmse_improvement": before_stats["rmse"] - after_stats["rmse"],
        "improved": bool(after_stats["rmse"] < before_stats["rmse"]),
        "runtime_seconds": time.perf_counter() - started,
    })

    # Weighted BA only touched the selected subset.  Move cameras can hurt
    # unselected tracks, so re-triangulate those points with cameras fixed.
    if bool(ba_cfg.get("retriangulate_unselected", True)):
        selected_set = set(int(pid) for pid in data["point_ids"])
        unselected = [pid for pid in rec.points3D if int(pid) not in selected_set]
        if unselected:
            refined, retri_report = retriangulate_points(
                refined, unselected,
                iterations=int(ba_cfg.get("retriangulation_iterations", 3)),
            )
            report["retriangulation"] = retri_report
            report["runtime_seconds"] = time.perf_counter() - started
    return refined, report


def filter_tracks(rec: Reconstruction, weights: Mapping[int, float],
                  weight_threshold: float = 0.1,
                  min_track_length: int = 2) -> tuple[Reconstruction, dict]:
    """Remove low-weight / short tracks (used by the approximate COLMAP path)."""
    filtered = rec.copy()
    removed = 0
    for point_id in list(filtered.points3D):
        point = filtered.points3D[point_id]
        if (float(weights.get(point_id, 1.0)) < weight_threshold
                or point.track_length < min_track_length):
            del filtered.points3D[point_id]
            removed += 1
    for image in filtered.images.values():
        valid = np.isin(image.point3D_ids, list(filtered.points3D))
        image.point3D_ids = np.where(valid, image.point3D_ids, -1)
    return filtered, {
        "removed_points": int(removed),
        "remaining_points": int(filtered.num_points),
        "weight_threshold": float(weight_threshold),
        "min_track_length": int(min_track_length),
    }


def weights_from_dataframe(tracks) -> dict[int, float]:
    return {
        int(row.point3D_id): float(row.weight)
        for row in tracks.itertuples(index=False)
    }


def refine_from_dataframe(rec: Reconstruction, tracks, config: Mapping,
                          engine: str = "auto") -> tuple[Reconstruction, dict]:
    return refine_weighted(rec, weights_from_dataframe(tracks), config, engine)
