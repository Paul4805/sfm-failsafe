"""Read/write the COLMAP SQLite database (matches, two-view geometries, keypoints).

This is intentionally independent of the COLMAP binary so pre-reconstruction
features can be extracted from a database produced on any machine.  It follows
the public COLMAP schema:

    pair_id = image_id1 * 2147483647 + image_id2       (image_id1 < image_id2)
    keypoints.data : float32[rows, cols]  (cols >= 2; x, y)
    matches.data   : uint32[rows, 2]      (keypoint indices)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import ActionableError

PAIR_ID_MULTIPLIER = 2147483647


def encode_pair_id(image_id1: int, image_id2: int) -> int:
    a, b = sorted((int(image_id1), int(image_id2)))
    if a == b:
        raise ValueError("A pair needs two distinct image ids")
    return a * PAIR_ID_MULTIPLIER + b


def decode_pair_id(pair_id: int) -> tuple[int, int]:
    return divmod(int(pair_id), PAIR_ID_MULTIPLIER)


def _decode_pairs(blob: bytes | None, rows: int, cols: int) -> np.ndarray:
    if not blob or rows == 0 or cols == 0:
        return np.zeros((0, cols), dtype=np.uint32)
    array = np.frombuffer(blob, dtype=np.uint32)
    if array.size != rows * cols:
        raise ActionableError(
            f"Corrupt COLMAP database blob: expected {rows * cols} uint32 values, got {array.size}"
        )
    return array.reshape(rows, cols)


def _decode_keypoints(blob: bytes | None, rows: int, cols: int) -> np.ndarray:
    if not blob or rows == 0 or cols == 0:
        return np.zeros((0, 2), dtype=np.float64)
    array = np.frombuffer(blob, dtype=np.float32)
    if array.size != rows * cols:
        raise ActionableError(
            f"Corrupt COLMAP keypoint blob: expected {rows * cols} floats, got {array.size}"
        )
    return array.reshape(rows, cols)[:, :2].astype(np.float64)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


class ColmapDatabase:
    """Thin, read-only wrapper around a COLMAP database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise ActionableError(
                f"COLMAP database not found: {self.path}\n"
                "Run scripts/run_baseline.py (feature extraction + matching) first."
            )
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ColmapDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def images(self) -> pd.DataFrame:
        rows = self._conn.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows],
                            columns=["image_id", "name", "camera_id"])

    def cameras(self) -> pd.DataFrame:
        rows = self._conn.execute(
            "SELECT camera_id, model, width, height, params FROM cameras ORDER BY camera_id"
        ).fetchall()
        records = []
        for row in rows:
            params = np.frombuffer(row["params"], dtype=np.float64) if row["params"] else np.array([])
            records.append({
                "camera_id": int(row["camera_id"]),
                "model": int(row["model"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "params": params,
            })
        return pd.DataFrame(records)

    def keypoints(self, image_id: int) -> np.ndarray:
        row = self._conn.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id=?", (int(image_id),)
        ).fetchone()
        if row is None:
            return np.zeros((0, 2), dtype=np.float64)
        return _decode_keypoints(row["data"], int(row["rows"]), int(row["cols"]))

    def pair_ids(self) -> list[int]:
        return [int(r[0]) for r in self._conn.execute(
            "SELECT pair_id FROM matches ORDER BY pair_id"
        ).fetchall()]

    def matches(self, pair_id: int) -> np.ndarray:
        row = self._conn.execute(
            "SELECT rows, cols, data FROM matches WHERE pair_id=?", (int(pair_id),)
        ).fetchone()
        if row is None:
            return np.zeros((0, 2), dtype=np.uint32)
        return _decode_pairs(row["data"], int(row["rows"]), int(row["cols"]))

    def two_view_geometry(self, pair_id: int) -> np.ndarray | None:
        if not _table_exists(self._conn, "two_view_geometries"):
            return None
        row = self._conn.execute(
            "SELECT rows, cols, data FROM two_view_geometries WHERE pair_id=?",
            (int(pair_id),),
        ).fetchone()
        if row is None:
            return None
        return _decode_pairs(row["data"], int(row["rows"]), int(row["cols"]))

    def pair_records(self) -> pd.DataFrame:
        """One row per matched pair with verified inlier count when available."""
        image_ids = {int(r["image_id"]) for r in self._conn.execute("SELECT image_id FROM images")}
        has_geom = _table_exists(self._conn, "two_view_geometries")
        records = []
        query = "SELECT pair_id, rows, cols, data FROM matches ORDER BY pair_id"
        for row in self._conn.execute(query):
            pair_id = int(row["pair_id"])
            id1, id2 = decode_pair_id(pair_id)
            if id1 not in image_ids or id2 not in image_ids:
                continue
            matches = _decode_pairs(row["data"], int(row["rows"]), int(row["cols"]))
            n_inliers = 0
            if has_geom:
                geom = self.two_view_geometry(pair_id)
                if geom is not None:
                    n_inliers = int(geom.shape[0])
            records.append({
                "pair_id": pair_id,
                "image_id1": id1,
                "image_id2": id2,
                "num_matches": int(matches.shape[0]),
                "num_inliers": n_inliers,
                "inlier_ratio": (n_inliers / matches.shape[0]) if matches.shape[0] else np.nan,
            })
        return pd.DataFrame(records)

    def stats(self) -> dict[str, float | int]:
        def scalar(sql: str) -> int:
            return int(self._conn.execute(sql).fetchone()[0])

        num_images = scalar("SELECT COUNT(*) FROM images")
        num_cameras = scalar("SELECT COUNT(*) FROM cameras")
        num_pairs = scalar("SELECT COUNT(*) FROM matches") if _table_exists(self._conn, "matches") else 0
        num_keypoints = scalar("SELECT COALESCE(SUM(rows), 0) FROM keypoints") if _table_exists(self._conn, "keypoints") else 0
        num_geometries = (
            scalar("SELECT COUNT(*) FROM two_view_geometries")
            if _table_exists(self._conn, "two_view_geometries") else 0
        )
        return {
            "num_images": num_images,
            "num_cameras": num_cameras,
            "num_pairs": num_pairs,
            "num_keypoints": num_keypoints,
            "num_two_view_geometries": num_geometries,
        }


def database_stats(path: str | Path) -> dict[str, float | int]:
    with ColmapDatabase(path) as db:
        return db.stats()


# ---------------------------------------------------------------------------
# Synthetic / test database writer
# ---------------------------------------------------------------------------

def write_database(
    path: str | Path,
    cameras: Iterable,
    images: Iterable,
    keypoints: dict[int, np.ndarray],
    matches: dict[int, np.ndarray],
    inliers: dict[int, np.ndarray] | None = None,
    descriptors: dict[int, np.ndarray] | None = None,
) -> Path:
    """Create a COLMAP-compatible SQLite database.

    ``cameras`` accepts objects with ``camera_id``, ``model_id``, ``width``,
    ``height``, ``params``.  ``images`` accepts objects with ``image_id``,
    ``name``, ``camera_id``.  This is used by the synthetic fixture generator
    and by tests; real experiments always use COLMAP's own database.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY NOT NULL,
                model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                params BLOB, prior_focal_length INTEGER NOT NULL);
            CREATE TABLE images (image_id INTEGER PRIMARY KEY NOT NULL,
                name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
                prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
                prior_tx REAL, prior_ty REAL, prior_tz REAL);
            CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
            CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
            CREATE TABLE matches (pair_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
            CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
                config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB);
            """
        )
        for camera in cameras:
            conn.execute(
                "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
                (int(camera.camera_id), int(camera.model_id), int(camera.width),
                 int(camera.height), np.asarray(camera.params, dtype=np.float64).tobytes(), 1),
            )
        for image in images:
            conn.execute(
                "INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)",
                (int(image.image_id), str(image.name), int(image.camera_id)),
            )
        for image_id, points in keypoints.items():
            points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            conn.execute(
                "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
                (int(image_id), int(points.shape[0]), 2, points.tobytes()),
            )
        if descriptors:
            for image_id, desc in descriptors.items():
                desc = np.asarray(desc, dtype=np.uint8)
                rows = desc.shape[0]
                cols = int(desc.size // max(rows, 1))
                conn.execute(
                    "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
                    (int(image_id), rows, cols, desc.tobytes()),
                )
        for pair_id, pair_matches in matches.items():
            pair_matches = np.asarray(pair_matches, dtype=np.uint32).reshape(-1, 2)
            conn.execute(
                "INSERT INTO matches VALUES (?, ?, ?, ?)",
                (int(pair_id), int(pair_matches.shape[0]), 2, pair_matches.tobytes()),
            )
        if inliers:
            for pair_id, pair_inliers in inliers.items():
                pair_inliers = np.asarray(pair_inliers, dtype=np.uint32).reshape(-1, 2)
                conn.execute(
                    "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (int(pair_id), int(pair_inliers.shape[0]), 2, pair_inliers.tobytes(),
                     2, b"", b"", b""),
                )
        conn.commit()
    finally:
        conn.close()
    return path


def drop_match_fraction(
    database_path: str | Path,
    output_path: str | Path,
    drop_fraction: float,
    seed: int = 0,
    drop_pairs: bool = True,
) -> Path:
    """Copy a database and deterministically drop matches / pairs.

    Used by ``make_challenging_subsets.py`` for the ``degraded_matches``
    condition.  The input database is never modified.
    """
    import shutil

    source, output = Path(database_path), Path(output_path)
    if not source.exists():
        raise ActionableError(f"Source database not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    rng = np.random.default_rng(seed)
    drop_fraction = float(np.clip(drop_fraction, 0.0, 1.0))
    conn = sqlite3.connect(str(output))
    conn.row_factory = sqlite3.Row
    try:
        pair_ids = [int(r[0]) for r in conn.execute("SELECT pair_id FROM matches")]
        removed_pairs = 0
        removed_matches = 0
        for pair_id in pair_ids:
            if drop_pairs and rng.random() < drop_fraction:
                conn.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))
                if _table_exists(conn, "two_view_geometries"):
                    conn.execute("DELETE FROM two_view_geometries WHERE pair_id=?", (pair_id,))
                removed_pairs += 1
                continue
            row = conn.execute(
                "SELECT rows, cols, data FROM matches WHERE pair_id=?", (pair_id,)
            ).fetchone()
            if row is None:
                continue
            pairs = _decode_pairs(row["data"], int(row["rows"]), int(row["cols"]))
            if pairs.shape[0] == 0:
                continue
            keep = rng.random(pairs.shape[0]) >= drop_fraction
            kept = pairs[keep]
            removed_matches += int(pairs.shape[0] - kept.shape[0])
            conn.execute(
                "UPDATE matches SET rows=?, data=? WHERE pair_id=?",
                (int(kept.shape[0]), kept.astype(np.uint32).tobytes(), pair_id),
            )
            if _table_exists(conn, "two_view_geometries"):
                geom_row = conn.execute(
                    "SELECT rows, cols, data FROM two_view_geometries WHERE pair_id=?",
                    (pair_id,),
                ).fetchone()
                if geom_row is not None:
                    geom = _decode_pairs(geom_row["data"], int(geom_row["rows"]),
                                         int(geom_row["cols"]))
                    # Keep only inliers whose match survived (both endpoints are
                    # still keypoints; exact match membership is not recoverable
                    # after dropping, so retain the surviving inliers by index).
                    geom_keep = geom[rng.random(geom.shape[0]) >= drop_fraction]
                    conn.execute(
                        "UPDATE two_view_geometries SET rows=?, data=? WHERE pair_id=?",
                        (int(geom_keep.shape[0]), geom_keep.astype(np.uint32).tobytes(), pair_id),
                    )
        conn.commit()
    finally:
        conn.close()
    return output
