"""COLMAP sparse model I/O and camera geometry.

The project treats COLMAP as the reconstruction backend.  This module reads
COLMAP's native TXT model format directly (no compilation needed), uses
``pycolmap`` for binary models when it is installed, and writes TXT models that
COLMAP's ``bundle_adjuster`` / ``point_triangulator`` can consume.

COLMAP convention used throughout:
    x_camera = R(qvec) @ x_world + tvec
    qvec = (qw, qx, qy, qz)                 # world-to-camera rotation
    camera_center = -R.T @ tvec
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .utils import ActionableError, ensure_dir

# COLMAP camera model ids.  Only the distortion models needed for ETH3D and
# the synthetic fixtures are fully supported for projection; the rest are
# readable and can be passed through unchanged.
CAMERA_MODEL_IDS: dict[str, int] = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
    "RAD_TAN_THIN_PRISM_FISHEYE": 11,
}
ID_TO_CAMERA_MODEL = {v: k for k, v in CAMERA_MODEL_IDS.items()}


def qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    """Convert COLMAP (qw, qx, qy, qz) to a 3x3 rotation matrix."""
    q = np.asarray(qvec, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Zero quaternion is not a valid rotation")
    q = q / norm
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(rotmat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to COLMAP (qw, qx, qy, qz)."""
    r = np.asarray(rotmat, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def __post_init__(self) -> None:
        self.params = np.asarray(self.params, dtype=np.float64)
        self.model = str(self.model).upper()

    @property
    def model_id(self) -> int:
        return CAMERA_MODEL_IDS.get(self.model, -1)

    @property
    def is_pinhole(self) -> bool:
        return self.model in ("SIMPLE_PINHOLE", "PINHOLE")

    @property
    def focal(self) -> tuple[float, float]:
        if self.model == "SIMPLE_PINHOLE":
            return float(self.params[0]), float(self.params[0])
        if self.model in ("PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL",
                          "FULL_OPENCV"):
            return float(self.params[0]), float(self.params[1])
        raise ActionableError(
            f"Camera model {self.model} is not supported by the pure-Python "
            "projection code; use a PINHOLE/OPENCV model or pycolmap."
        )

    @property
    def principal(self) -> tuple[float, float]:
        if self.model == "SIMPLE_PINHOLE":
            return float(self.params[1]), float(self.params[2])
        return float(self.params[2]), float(self.params[3])

    @property
    def distortion(self) -> tuple[float, ...]:
        if self.model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return ()
        if self.model == "SIMPLE_RADIAL":
            return (float(self.params[3]), 0.0, 0.0, 0.0)
        if self.model == "RADIAL":
            return (float(self.params[3]), float(self.params[4]), 0.0, 0.0)
        if self.model == "OPENCV":
            return (float(self.params[4]), float(self.params[5]),
                    float(self.params[6]), float(self.params[7]))
        if self.model == "FULL_OPENCV":
            return (float(self.params[4]), float(self.params[5]),
                    float(self.params[8]), float(self.params[9]))
        return ()

    def _distort_normalized(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self.distortion
        if not values:
            return x, y
        k1, k2, p1, p2 = values
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return xd, yd

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Project Nx3 camera-frame points to Nx2 pixel coordinates."""
        points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = points[:, 2]
        safe_z = np.where(np.abs(z) < 1e-12, np.nan, z)
        x = points[:, 0] / safe_z
        y = points[:, 1] / safe_z
        xd, yd = self._distort_normalized(x, y)
        fx, fy = self.focal
        cx, cy = self.principal
        return np.column_stack([fx * xd + cx, fy * yd + cy])

    def project_valid(self, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        valid = points[:, 2] > 1e-8
        uv = self.project(points)
        valid &= np.isfinite(uv).all(axis=1)
        return uv, valid

    def undistort_to_pinhole(self, uv: np.ndarray) -> np.ndarray:
        """Map distorted pixel coordinates to undistorted pinhole pixels.

        Weighted BA works in a pinhole model.  For distorted COLMAP cameras we
        invert the distortion once for every observation, which keeps the
        residual Jacobian simple and exact for the pinhole camera.
        """
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if self.is_pinhole or not self.distortion:
            return uv.copy()
        fx, fy = self.focal
        cx, cy = self.principal
        xd = (uv[:, 0] - cx) / fx
        yd = (uv[:, 1] - cy) / fy
        k1, k2, p1, p2 = self.distortion
        x = xd.copy()
        y = yd.copy()
        for _ in range(12):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            radial = np.where(np.abs(radial) < 1e-9, 1e-9, radial)
            x = (xd - dx) / radial
            y = (yd - dy) / radial
        return np.column_stack([fx * x + cx, fy * y + cy])


@dataclass
class Image:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    camera: "Camera | None" = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.qvec = np.asarray(self.qvec, dtype=np.float64).reshape(4)
        self.tvec = np.asarray(self.tvec, dtype=np.float64).reshape(3)
        self.xys = np.asarray(self.xys, dtype=np.float64).reshape(-1, 2)
        self.point3D_ids = np.asarray(self.point3D_ids, dtype=np.int64).reshape(-1)
        if self.xys.shape[0] != self.point3D_ids.shape[0]:
            raise ValueError(f"Image {self.image_id}: xys and point3D_ids length mismatch")

    @property
    def rotation(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)

    @property
    def camera_center(self) -> np.ndarray:
        return -self.rotation.T @ self.tvec

    def project_world(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.camera is None:
            raise ActionableError(
                f"Image {self.image_id} has no linked camera; call Reconstruction.link_cameras()"
            )
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        points_cam = points @ self.rotation.T + self.tvec
        return self.camera.project(points_cam), points_cam[:, 2] > 1e-8


@dataclass
class Point3D:
    point3D_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(3)
        self.rgb = np.asarray(self.rgb, dtype=np.uint8).reshape(3)
        self.image_ids = np.asarray(self.image_ids, dtype=np.int64).reshape(-1)
        self.point2D_idxs = np.asarray(self.point2D_idxs, dtype=np.int64).reshape(-1)
        if self.image_ids.shape != self.point2D_idxs.shape:
            raise ValueError(f"Point {self.point3D_id}: inconsistent track arrays")

    @property
    def track_length(self) -> int:
        return int(self.image_ids.size)


@dataclass
class Reconstruction:
    cameras: dict[int, Camera] = field(default_factory=dict)
    images: dict[int, Image] = field(default_factory=dict)
    points3D: dict[int, Point3D] = field(default_factory=dict)

    def copy(self) -> "Reconstruction":
        return copy.deepcopy(self)

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_points(self) -> int:
        return len(self.points3D)

    def image_name(self, image_id: int) -> str:
        image = self.images.get(image_id)
        return image.name if image is not None else str(image_id)

    def link_cameras(self) -> "Reconstruction":
        for image in self.images.values():
            image.camera = self.cameras.get(int(image.camera_id))
        return self

    def camera_centers(self) -> dict[int, np.ndarray]:
        return {image_id: image.camera_center for image_id, image in self.images.items()}

    def point_array(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.points3D:
            return np.zeros((0, 3)), np.zeros((0,), dtype=np.int64)
        ids = np.array(sorted(self.points3D), dtype=np.int64)
        xyz = np.stack([self.points3D[i].xyz for i in ids])
        return xyz, ids

    def observation_array(self) -> dict[str, np.ndarray]:
        """Flatten all observations into aligned numpy arrays."""
        point_ids: list[int] = []
        image_ids: list[int] = []
        point2D_idxs: list[int] = []
        uv: list[np.ndarray] = []
        xyz: list[np.ndarray] = []
        for point_id, point in self.points3D.items():
            for image_id, idx in zip(point.image_ids, point.point2D_idxs):
                image = self.images.get(int(image_id))
                if image is None or int(idx) >= image.xys.shape[0]:
                    continue
                point_ids.append(int(point_id))
                image_ids.append(int(image_id))
                point2D_idxs.append(int(idx))
                uv.append(image.xys[int(idx)])
                xyz.append(point.xyz)
        return {
            "point3D_id": np.asarray(point_ids, dtype=np.int64),
            "image_id": np.asarray(image_ids, dtype=np.int64),
            "point2D_idx": np.asarray(point2D_idxs, dtype=np.int64),
            "uv": np.asarray(uv, dtype=np.float64).reshape(-1, 2),
            "xyz": np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        }


# ---------------------------------------------------------------------------
# TXT model parsing / writing
# ---------------------------------------------------------------------------

def _content_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line.lstrip().startswith("#"):
                continue
            lines.append(line)
    return lines


def read_cameras_text(path: str | Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _content_lines(Path(path)):
        parts = line.split()
        if not parts:
            continue
        camera_id = int(parts[0])
        raw_model = parts[1]
        try:
            model = ID_TO_CAMERA_MODEL.get(int(raw_model), str(raw_model))
        except ValueError:
            model = str(raw_model).upper()
        width, height = int(parts[2]), int(parts[3])
        params = np.asarray([float(v) for v in parts[4:]], dtype=np.float64)
        cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def _obs_triplets(line: str) -> tuple[np.ndarray, np.ndarray]:
    parts = line.split()
    if len(parts) % 3 != 0:
        return np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)
    values = np.asarray([float(v) for v in parts], dtype=np.float64).reshape(-1, 3)
    return values[:, :2], values[:, 2].astype(np.int64)


def read_images_text(path: str | Path) -> dict[int, Image]:
    lines = _content_lines(Path(path))
    images: dict[int, Image] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            raise ActionableError(
                f"Malformed images.txt line (expected 10 fields): {path}: {line[:120]}"
            )
        image_id = int(parts[0])
        qvec = np.asarray([float(v) for v in parts[1:5]])
        tvec = np.asarray([float(v) for v in parts[5:8]])
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        xys = np.zeros((0, 2), dtype=np.float64)
        point3D_ids = np.zeros((0,), dtype=np.int64)
        if i < len(lines):
            candidate = lines[i]
            stripped = candidate.strip()
            # COLMAP writes a (possibly empty) second line per image.  Be
            # tolerant of files that omit the empty line for zero observations.
            if not stripped:
                i += 1
            elif len(candidate.split()) % 3 == 0 and len(candidate.split()) > 0:
                xys, point3D_ids = _obs_triplets(candidate)
                i += 1
        images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)
    return images


def read_points3D_text(path: str | Path) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    for line in _content_lines(Path(path)):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 8:
            raise ActionableError(f"Malformed points3D.txt line: {line[:120]}")
        point_id = int(parts[0])
        xyz = np.asarray([float(v) for v in parts[1:4]])
        rgb = np.asarray([int(float(v)) for v in parts[4:7]], dtype=np.uint8)
        error = float(parts[7])
        track = np.asarray([int(float(v)) for v in parts[8:]], dtype=np.int64)
        if track.size % 2 != 0:
            raise ActionableError(f"Malformed track in points3D.txt: {line[:120]}")
        track = track.reshape(-1, 2)
        points[point_id] = Point3D(
            point_id, xyz, rgb, error,
            track[:, 0] if track.size else np.zeros((0,), dtype=np.int64),
            track[:, 1] if track.size else np.zeros((0,), dtype=np.int64),
        )
    return points


def model_files_exist(path: str | Path) -> tuple[bool, bool]:
    path = Path(path)
    return (path / "cameras.txt").exists(), (path / "cameras.bin").exists()


def read_model(path: str | Path, prefer_pycolmap: bool = False,
               allow_binary: bool = True) -> Reconstruction:
    """Read a COLMAP model directory (TXT preferred, binary via pycolmap)."""
    path = Path(path)
    txt_exists, bin_exists = model_files_exist(path)
    if not txt_exists and not bin_exists:
        raise ActionableError(
            f"No COLMAP model found in {path}\n"
            "Expected cameras.txt/cameras.bin, images.txt/images.bin, points3D.txt/points3D.bin.\n"
            "If this is a sub-model, check sparse/0, sparse/1, ..."
        )
    if prefer_pycolmap and bin_exists:
        return _read_with_pycolmap(path)
    if txt_exists:
        return Reconstruction(
            cameras=read_cameras_text(path / "cameras.txt"),
            images=read_images_text(path / "images.txt"),
            points3D=read_points3D_text(path / "points3D.txt"),
        ).link_cameras()
    if bin_exists and allow_binary:
        return _read_with_pycolmap(path)
    raise ActionableError(
        f"Binary COLMAP model at {path} cannot be read without pycolmap.\n"
        "Install it with:  python -m pip install pycolmap\n"
        "or convert once with:\n"
        f"  colmap model_converter --input_path \"{path}\" "
        f"--output_path \"{path.parent / 'txt'}\" --output_type TXT"
    )


def _read_with_pycolmap(path: Path) -> Reconstruction:
    try:
        import pycolmap  # type: ignore
    except Exception as exc:
        raise ActionableError(
            "pycolmap is required to read binary COLMAP models.\n"
            "Install it with: python -m pip install pycolmap\n"
            f"Or convert the model: colmap model_converter --input_path \"{path}\" "
            f"--output_path \"{path.parent / 'txt'}\" --output_type TXT"
        ) from exc

    rec = pycolmap.Reconstruction(str(path))
    cameras: dict[int, Camera] = {}
    for camera_id, camera in rec.cameras.items():
        model = getattr(camera, "model", None)
        model_name = getattr(model, "name", None) or str(model)
        cameras[int(camera_id)] = Camera(
            int(camera_id),
            str(model_name).upper(),
            int(camera.width),
            int(camera.height),
            np.asarray(camera.params, dtype=np.float64),
        )
    images: dict[int, Image] = {}
    for image_id, image in rec.images.items():
        rigid = getattr(image, "cam_from_world", None)
        if rigid is not None:
            rot = np.asarray(rigid.rotation.matrix(), dtype=np.float64)
            tvec = np.asarray(rigid.translation, dtype=np.float64)
            qvec = rotmat_to_qvec(rot)
        else:  # Older pycolmap
            qvec = np.asarray(image.qvec, dtype=np.float64)
            tvec = np.asarray(image.tvec, dtype=np.float64)
        xys_list, ids_list = [], []
        for point2D in image.points2D:
            xys_list.append(np.asarray(point2D.xy, dtype=np.float64))
            ids_list.append(int(point2D.point3D_id))
        images[int(image_id)] = Image(
            int(image_id), qvec, tvec, int(image.camera_id), str(image.name),
            np.asarray(xys_list, dtype=np.float64).reshape(-1, 2),
            np.asarray(ids_list, dtype=np.int64),
        )
    points: dict[int, Point3D] = {}
    for point_id, point in rec.points3D.items():
        image_ids = [int(el.image_id) for el in point.track.elements]
        idxs = [int(el.point2D_idx) for el in point.track.elements]
        points[int(point_id)] = Point3D(
            int(point_id),
            np.asarray(point.xyz, dtype=np.float64),
            np.asarray(point.color, dtype=np.uint8),
            float(point.error),
            np.asarray(image_ids, dtype=np.int64),
            np.asarray(idxs, dtype=np.int64),
        )
    return Reconstruction(cameras=cameras, images=images, points3D=points).link_cameras()


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def write_model(rec: Reconstruction, path: str | Path) -> Path:
    """Write a COLMAP TXT model (consumable by COLMAP CLI tools)."""
    path = ensure_dir(Path(path))
    with open(path / "cameras.txt", "w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for camera_id in sorted(rec.cameras):
            camera = rec.cameras[camera_id]
            params = " ".join(_fmt(p) for p in camera.params)
            handle.write(
                f"{camera.camera_id} {camera.model} {camera.width} {camera.height} {params}\n"
            )
    with open(path / "images.txt", "w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id in sorted(rec.images):
            image = rec.images[image_id]
            q = " ".join(_fmt(v) for v in image.qvec)
            t = " ".join(_fmt(v) for v in image.tvec)
            handle.write(
                f"{image.image_id} {q} {t} {image.camera_id} {image.name}\n"
            )
            triples = []
            for xy, point_id in zip(image.xys, image.point3D_ids):
                triples.append(f"{_fmt(xy[0])} {_fmt(xy[1])} {int(point_id)}")
            handle.write(" ".join(triples) + ("\n" if triples else "\n"))
    with open(path / "points3D.txt", "w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for point_id in sorted(rec.points3D):
            point = rec.points3D[point_id]
            track = " ".join(
                f"{int(image_id)} {int(idx)}"
                for image_id, idx in zip(point.image_ids, point.point2D_idxs)
            )
            handle.write(
                f"{point.point3D_id} {_fmt(point.xyz[0])} {_fmt(point.xyz[1])} "
                f"{_fmt(point.xyz[2])} {int(point.rgb[0])} {int(point.rgb[1])} "
                f"{int(point.rgb[2])} {_fmt(point.error)} {track}\n"
            )
    return path


# ---------------------------------------------------------------------------
# Model selection and conversion helpers
# ---------------------------------------------------------------------------

def model_stats(path: str | Path) -> dict[str, int]:
    rec = read_model(path)
    return {
        "num_images": rec.num_images,
        "num_points3D": rec.num_points,
        "num_observations": int(sum(p.track_length for p in rec.points3D.values())),
    }


def select_largest_model(sparse_dir: str | Path,
                         colmap_executable: str = "colmap") -> tuple[Path, dict[str, int]]:
    """Pick the COLMAP sub-model with the most registered images.

    Ties are broken by point count, then by directory name.  COLMAP may emit
    several disconnected sub-models (sparse/0, sparse/1, ...); the largest is
    the intended reconstruction.  Binary sub-models are converted to TXT when
    pycolmap is unavailable.
    """
    sparse_dir = Path(sparse_dir)
    candidates = sorted(
        [p for p in sparse_dir.iterdir() if p.is_dir()]
    ) if sparse_dir.is_dir() else []
    candidates = [p for p in candidates if any((p / n).exists() for n in ("cameras.txt", "cameras.bin"))]
    if not candidates:
        raise ActionableError(
            f"No COLMAP sparse sub-models found under {sparse_dir}\n"
            "Run the mapper first, or pass --model pointing at a sparse/0 directory."
        )
    ranked: list[tuple[tuple[int, int, str], Path, dict[str, int]]] = []
    for candidate in candidates:
        readable = candidate
        if not (candidate / "cameras.txt").exists() and (candidate / "cameras.bin").exists():
            try:
                import pycolmap  # noqa: F401
            except Exception:
                readable = convert_binary_model_to_txt(
                    candidate, sparse_dir.parent / "sparse_txt" / candidate.name,
                    colmap_executable=colmap_executable,
                )
        stats = model_stats(readable)
        score = (stats["num_images"], stats["num_points3D"], candidate.name)
        ranked.append((score, candidate, stats))
    ranked.sort(key=lambda item: item[0], reverse=True)
    _, best, stats = ranked[0]
    return best, stats


def _colmap_environment(colmap_executable: str) -> tuple[str, dict[str, str]]:
    """Resolve bin/colmap.exe and add the release archive's lib/ to PATH."""
    import os

    executable = Path(colmap_executable)
    if executable.suffix.lower() in (".bat", ".cmd"):
        candidate = executable.parent / "bin" / "colmap.exe"
        if candidate.exists():
            executable = candidate
    env = os.environ.copy()
    lib_dir = executable.parent.parent / "lib"
    if lib_dir.is_dir():
        env["PATH"] = str(lib_dir) + os.pathsep + env.get("PATH", "")
        plugins = lib_dir / "plugins"
        if plugins.is_dir():
            env["QT_PLUGIN_PATH"] = str(plugins) + os.pathsep + env.get("QT_PLUGIN_PATH", "")
    return str(executable), env


def convert_binary_model_to_txt(model_dir: str | Path, output_dir: str | Path,
                                colmap_executable: str = "colmap",
                                dry_run: bool = False) -> Path:
    """Convert a binary model with the COLMAP CLI (used only when pycolmap is absent)."""
    import subprocess

    model_dir, output_dir = Path(model_dir), Path(output_dir)
    executable, env = _colmap_environment(colmap_executable)
    cmd = [
        executable, "model_converter",
        "--input_path", str(model_dir),
        "--output_path", str(output_dir),
        "--output_type", "TXT",
    ]
    if dry_run:
        print("DRY-RUN:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return output_dir
    ensure_dir(output_dir)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise ActionableError(
            "COLMAP model conversion failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n"
            "Install pycolmap or add COLMAP to PATH."
        )
    return output_dir


def ensure_readable_model(model_dir: str | Path, colmap_executable: str = "colmap",
                          work_dir: str | Path | None = None,
                          dry_run: bool = False) -> Path:
    """Return a directory with TXT model files, converting if necessary."""
    model_dir = Path(model_dir)
    if (model_dir / "cameras.txt").exists() and (model_dir / "points3D.txt").exists():
        return model_dir
    if (model_dir / "cameras.bin").exists():
        try:
            import pycolmap  # noqa: F401
            return model_dir  # read_model handles binary via pycolmap
        except Exception:
            pass
        output_dir = Path(work_dir) if work_dir else model_dir.parent / (model_dir.name + "_txt")
        return convert_binary_model_to_txt(model_dir, output_dir, colmap_executable, dry_run)
    raise ActionableError(f"Unrecognised COLMAP model layout: {model_dir}")


def image_names(rec: Reconstruction) -> set[str]:
    return {img.name for img in rec.images.values()}
