"""COLMAP CLI orchestration (the real baseline backend).

Every COLMAP invocation is logged with its full command, stdout/stderr, return
code, and wall-clock time.  Input databases are copied into the run directory;
the source database is never modified.  Unsupported CLI flags are skipped after
probing ``colmap <command> -h`` so the runner works across COLMAP 3.x/4.x.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import evaluation, features, model_io, track_analysis
from .config import get_cfg
from .utils import (ActionableError, Timer, dump_json, ensure_dir, format_seconds,
                    load_json, setup_logger, slugify)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class ColmapCommandError(ActionableError):
    def __init__(self, message: str, command: Sequence[str], returncode: int,
                 stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.command = list(command)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass
class CommandReport:
    command: list[str]
    returncode: int
    runtime_seconds: float
    stdout: str = ""
    stderr: str = ""
    dry_run: bool = False
    step: str = ""


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return "...[truncated]...\n" + text[-limit:]


def list_images(images_dir: str | Path) -> list[Path]:
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return []
    return sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_images(paths: Sequence[Path], max_images: int | None) -> list[Path]:
    paths = sorted(paths)
    if not max_images or max_images >= len(paths):
        return list(paths)
    if max_images <= 0:
        return []
    positions = np.linspace(0, len(paths) - 1, int(max_images)).round().astype(int)
    return [paths[i] for i in sorted(set(positions.tolist()))]


def stage_images(images_dir: str | Path, stage_dir: str | Path,
                 max_images: int | None = None, downscale: float = 1.0,
                 names: Sequence[str] | None = None) -> tuple[Path, list[str], dict]:
    """Create a run-local image view without modifying source images.

    Hardlinks are used when possible (fast, no extra disk); otherwise files are
    copied.  ``downscale < 1`` writes resized copies.  Returns
    ``(image_dir, relative_names, info)``.
    """
    images_dir, stage_dir = Path(images_dir), Path(stage_dir)
    if not images_dir.is_dir():
        raise ActionableError(
            f"Image directory not found: {images_dir}\n"
            "Run scripts/prepare_courtyard.py or set dataset.images_dir in the config."
        )
    if names:
        selected = [images_dir / name for name in names]
        missing = [p for p in selected if not p.exists()]
        if missing:
            raise ActionableError(f"Missing images from manifest: {missing[:3]}")
        selected = [p for p in selected if p.exists()]
    else:
        selected = list_images(images_dir)
    selected = [p for p in selected if p.is_file()]
    selected = select_images(selected, max_images)
    if not selected:
        raise ActionableError(f"No images found under {images_dir}")

    needs_stage = names is not None or \
        bool(max_images and max_images < len(list_images(images_dir))) or \
        float(downscale) != 1.0
    if not needs_stage:
        relative = [str(path.relative_to(images_dir)).replace("\\", "/") for path in selected]
        return images_dir, relative, {
            "staged": False,
            "source_images_dir": str(images_dir),
            "num_source_images": len(selected),
            "downscale": 1.0,
        }

    stage_dir = ensure_dir(stage_dir)
    relative_names: list[str] = []
    for path in selected:
        relative = path.relative_to(images_dir)
        target = stage_dir / relative
        ensure_dir(target.parent)
        if float(downscale) == 1.0:
            if target.exists():
                target.unlink()
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
        else:
            try:
                import cv2
            except ImportError as exc:
                raise ActionableError(
                    "opencv-python is required for --downscale staging: pip install opencv-python"
                ) from exc
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ActionableError(f"Cannot read image for downscaling: {path}")
            new_size = (max(1, int(round(image.shape[1] * downscale))),
                        max(1, int(round(image.shape[0] * downscale))))
            resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if target.suffix.lower() in (".jpg", ".jpeg") else []
            cv2.imwrite(str(target), resized, params)
        relative_names.append(str(relative).replace("\\", "/"))
    return stage_dir, sorted(relative_names), {
        "staged": True,
        "source_images_dir": str(images_dir),
        "num_source_images": len(selected),
        "downscale": float(downscale),
    }


class ColmapRunner:
    def __init__(self, config: Mapping, logger=None, dry_run: bool = False) -> None:
        self.config = config
        executable = os.environ.get("COLMAP_EXECUTABLE") or str(
            get_cfg(config, "colmap.executable", "colmap")
        )
        executable_path = Path(executable)
        if not executable_path.is_absolute() and any(sep in executable for sep in ("/", "\\")):
            root = Path(get_cfg(config, "_meta.repo_root", "."))
            candidate = root / executable_path
            if candidate.exists():
                executable = str(candidate)
        self.executable = executable
        self.dry_run = bool(dry_run)
        self.logger = logger or setup_logger("sfm_failsafe.colmap")
        self.use_gpu = bool(get_cfg(config, "colmap.use_gpu", True))
        self.timeout = int(get_cfg(config, "colmap.timeout_seconds", 14400))
        self._flag_cache: dict[str, set[str] | None] = {}
        self.reports: list[CommandReport] = []
        self._version: str | None = None
        self._is_batch = Path(self.executable).suffix.lower() in (".bat", ".cmd")
        self.env = self._build_env()

    def _build_env(self) -> dict[str, str]:
        """Make portable COLMAP layouts (bin/ + lib/) work without COLMAP.bat.

        Release archives are statically linked except for the DLLs in ``lib``;
        setting ``PATH``/``QT_PLUGIN_PATH`` is equivalent to what COLMAP.bat
        does, but avoids fragile cmd.exe quoting for paths with spaces.
        """
        env = os.environ.copy()
        executable = Path(self.executable)
        if executable.suffix.lower() in (".bat", ".cmd"):
            candidate = executable.parent / "bin" / "colmap.exe"
            if candidate.exists():
                self.executable = str(candidate)
                executable = candidate
                self._is_batch = False
        install_root = executable.parent.parent
        lib_dir = install_root / "lib"
        if lib_dir.is_dir():
            env["PATH"] = str(lib_dir) + os.pathsep + env.get("PATH", "")
            plugin_dir = lib_dir / "plugins"
            if plugin_dir.is_dir():
                env["QT_PLUGIN_PATH"] = str(plugin_dir) + os.pathsep + env.get("QT_PLUGIN_PATH", "")
        return env

    def _wrap(self, command: Sequence[str]) -> list[str]:
        """Run unresolved .bat/.cmd launchers through cmd.exe (fallback only)."""
        command = [str(c) for c in command]
        if command and Path(command[0]).suffix.lower() in (".bat", ".cmd"):
            return ["cmd.exe", "/c", "call", *command]
        return command


    # -- executable / flag probing -----------------------------------------
    def check_available(self) -> str:
        if self.dry_run:
            return "dry-run (COLMAP availability not checked)"
        candidate = shutil.which(self.executable)
        if candidate is None:
            path = Path(self.executable)
            if path.is_file():
                candidate = str(path)
            else:
                raise ActionableError(
                    f"COLMAP executable not found: {self.executable}\n\n"
                    "Install COLMAP and either add it to PATH or set:\n"
                    "  colmap.executable: /absolute/path/to/colmap\n"
                    "or environment variable COLMAP_EXECUTABLE.\n"
                    "Windows portable build: https://github.com/colmap/colmap/releases\n"
                    "Linux: sudo apt install colmap   |   macOS: brew install colmap\n"
                    "Then verify with: colmap -h"
                )
        self.executable = candidate
        if self._version is None:
            try:
                proc = subprocess.run(self._wrap([candidate, "-h"]),
                                      capture_output=True, text=True, timeout=60,
                                      env=self.env)
                first = (proc.stdout or proc.stderr).strip().splitlines()
                self._version = first[0] if first else "unknown"
            except Exception:
                self._version = "unknown"
        self.logger.info("Using COLMAP: %s (%s)", candidate, self._version)
        return self._version

    def _flags(self, command: str) -> set[str] | None:
        if command in self._flag_cache:
            return self._flag_cache[command]
        if self.dry_run:
            self._flag_cache[command] = None
            return None
        try:
            proc = subprocess.run(
                self._wrap([self.executable, command, "-h"]),
                capture_output=True, text=True, timeout=60, env=self.env
            )
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            flags = set(re.findall(r"(--[A-Za-z0-9_.]+)", text))
            self._flag_cache[command] = flags if flags else None
        except Exception:
            self._flag_cache[command] = None
        return self._flag_cache[command]

    def _supported(self, command: str, flag: str) -> bool:
        flags = self._flags(command)
        return True if flags is None else f"--{flag}" in flags

    def _option(self, command: str, flag: str, value: Any) -> list[str]:
        if value is None:
            return []
        if not self._supported(command, flag):
            self.logger.debug("Skipping unsupported COLMAP flag --%s for %s", flag, command)
            return []
        return [f"--{flag}", str(value)]

    # -- command execution -------------------------------------------------
    def _run(self, command: Sequence[str], log_path: str | Path, step: str,
             check: bool = True) -> CommandReport:
        command = self._wrap([str(c) for c in command])
        log_path = Path(log_path)
        ensure_dir(log_path.parent)
        printable = " ".join(f'"{c}"' if " " in c else c for c in command)
        started = time.perf_counter()
        if self.dry_run:
            self.logger.info("DRY-RUN %s", printable)
            report = CommandReport(command, 0, 0.0, dry_run=True, step=step)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"DRY-RUN: {printable}\n")
            self.reports.append(report)
            return report
        self.logger.info("Running %s: %s", step, printable)
        try:
            proc = subprocess.run(command, capture_output=True, text=True,
                                  timeout=self.timeout, env=self.env)
        except subprocess.TimeoutExpired as exc:
            message = (f"COLMAP {step} timed out after {self.timeout}s\n"
                       f"Command: {printable}")
            raise ColmapCommandError(message, command, -1, str(exc)) from exc
        runtime = time.perf_counter() - started
        report = CommandReport(
            command=command, returncode=int(proc.returncode), runtime_seconds=runtime,
            stdout=proc.stdout or "", stderr=proc.stderr or "", step=step,
        )
        self.reports.append(report)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"\n$ {printable}\n")
            handle.write(f"[returncode={proc.returncode} runtime={runtime:.2f}s]\n")
            handle.write("--- stdout ---\n" + (proc.stdout or "") + "\n")
            handle.write("--- stderr ---\n" + (proc.stderr or "") + "\n")
        if check and proc.returncode != 0:
            message = (
                f"COLMAP {step} failed with exit code {proc.returncode}.\n"
                f"Command: {printable}\n"
                f"Log: {log_path}\n"
                f"stdout:\n{_tail(proc.stdout or '')}\n"
                f"stderr:\n{_tail(proc.stderr or '')}"
            )
            raise ColmapCommandError(message, command, int(proc.returncode),
                                     proc.stdout or "", proc.stderr or "")
        return report

    # -- command builders --------------------------------------------------
    def feature_command(self, image_dir: Path, database: Path) -> list[str]:
        command = "feature_extractor"
        args = [self.executable, command,
                "--database_path", str(database),
                "--image_path", str(image_dir)]
        camera_model = str(get_cfg(self.config, "colmap.camera_model", "OPENCV"))
        args += self._option(command, "ImageReader.camera_model", camera_model)
        args += self._option(command, "ImageReader.single_camera",
                             int(bool(get_cfg(self.config, "colmap.single_camera", True))))
        args += self._option(command, "SiftExtraction.max_image_size",
                             get_cfg(self.config, "colmap.max_image_size", 2000))
        args += self._option(command, "SiftExtraction.max_num_features",
                             get_cfg(self.config, "colmap.sift_max_features", 8192))
        args += self._option(command, "SiftExtraction.peak_threshold",
                             get_cfg(self.config, "colmap.sift_peak_threshold", 0.004))
        args += self._option(command, "SiftExtraction.edge_threshold",
                             get_cfg(self.config, "colmap.sift_edge_threshold", 10))
        gpu_flag = "SiftExtraction.use_gpu"
        if not self._supported(command, gpu_flag):
            gpu_flag = "FeatureExtraction.use_gpu"
        args += self._option(command, gpu_flag, int(self.use_gpu))
        if self.use_gpu:
            args += self._option(command, "SiftExtraction.gpu_index",
                                 get_cfg(self.config, "colmap.gpu_index", "0"))
        args += [str(v) for v in (get_cfg(self.config, "colmap.extra_feature_args", []) or [])]
        return args

    def matcher_command(self, database: Path) -> list[str]:
        matcher = str(get_cfg(self.config, "colmap.matcher", "exhaustive")).lower()
        if matcher in ("exhaustive", "exhaustive_matcher"):
            command = "exhaustive_matcher"
            args = [self.executable, command, "--database_path", str(database)]
            args += self._option(command, "ExhaustiveMatching.block_size",
                                 get_cfg(self.config, "colmap.exhaustive_block_size", 50))
        elif matcher in ("sequential", "sequential_matcher"):
            command = "sequential_matcher"
            args = [self.executable, command, "--database_path", str(database)]
            args += self._option(command, "SequentialMatching.overlap",
                                 get_cfg(self.config, "colmap.sequential_overlap", 10))
        elif matcher in ("vocab_tree", "vocabtree", "vocab_tree_matcher"):
            command = "vocab_tree_matcher"
            args = [self.executable, command, "--database_path", str(database)]
            vocab = get_cfg(self.config, "colmap.vocab_tree_path", None)
            if not vocab:
                raise ActionableError(
                    "colmap.matcher=vocab_tree requires colmap.vocab_tree_path in the config."
                )
            args += ["--VocabTreeMatching.vocab_tree_path", str(vocab)]
        else:
            raise ActionableError(f"Unsupported matcher: {matcher}")
        gpu_flag = "SiftMatching.use_gpu"
        if not self._supported(command, gpu_flag):
            gpu_flag = "FeatureMatching.use_gpu"
        args += self._option(command, gpu_flag, int(self.use_gpu))
        if self.use_gpu:
            args += self._option(command, "SiftMatching.gpu_index",
                                 get_cfg(self.config, "colmap.gpu_index", "0"))
        args += [str(v) for v in (get_cfg(self.config, "colmap.extra_matcher_args", []) or [])]
        return args

    def mapper_command(self, image_dir: Path, database: Path, sparse_dir: Path) -> list[str]:
        command = "mapper"
        args = [self.executable, command,
                "--database_path", str(database),
                "--image_path", str(image_dir),
                "--output_path", str(sparse_dir)]
        args += self._option(command, "Mapper.min_num_matches",
                             get_cfg(self.config, "colmap.mapper_min_num_matches", 15))
        args += self._option(command, "Mapper.ba_global_images_ratio",
                             get_cfg(self.config, "colmap.mapper_ba_global_images_ratio", 1.1))
        args += self._option(command, "Mapper.ba_global_points_ratio",
                             get_cfg(self.config, "colmap.mapper_ba_global_points_ratio", 1.1))
        args += self._option(command, "Mapper.ba_local_max_refinements",
                             get_cfg(self.config, "colmap.mapper_ba_local_max_refinements", 3))
        args += self._option(command, "Mapper.max_num_models",
                             get_cfg(self.config, "colmap.mapper_max_num_models", 10))
        args += [str(v) for v in (get_cfg(self.config, "colmap.extra_mapper_args", []) or [])]
        return args

    # -- pipeline ----------------------------------------------------------
    def run_baseline(self, images_dir: str | Path, run_dir: str | Path,
                     max_images: int | None = None, downscale: float = 1.0,
                     input_database: str | Path | None = None,
                     reuse_database: bool = False,
                     condition: str = "clean", data_kind: str = "real",
                     run_name: str | None = None,
                     all_image_names: Sequence[str] | None = None,
                     overwrite: bool = False) -> dict:
        run_dir = Path(run_dir)
        run_name = run_name or run_dir.name
        if not self.dry_run and run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
            raise ActionableError(
                f"Run directory is not empty: {run_dir}\n"
                "Use --overwrite or choose a different --run-name."
            )
        ensure_dir(run_dir)
        if not self.dry_run and overwrite:
            import shutil as _shutil
            _shutil.rmtree(run_dir)
            ensure_dir(run_dir)
        logs_dir = ensure_dir(run_dir / "logs")
        command_log = logs_dir / "commands.log"
        started = time.perf_counter()
        version = self.check_available()

        if self.dry_run:
            image_dir = Path(images_dir)
            db = run_dir / "database.db"
            sparse = run_dir / "sparse"
            commands = []
            if not reuse_database:
                commands.append(self.feature_command(image_dir, db))
                commands.append(self.matcher_command(db))
            commands.append(self.mapper_command(image_dir, db, sparse))
            print("DRY-RUN baseline plan:")
            for command in commands:
                print("  " + " ".join(f'"{c}"' if " " in c else c for c in command))
            dump_json({
                "dry_run": True,
                "run_name": run_name,
                "images_dir": str(image_dir),
                "run_dir": str(run_dir),
                "commands": commands,
                "note": "No COLMAP process was executed and no outputs were written by COLMAP.",
            }, run_dir / "dry_run_plan.json")
            return {"dry_run": True, "commands": commands, "run_dir": str(run_dir)}

        staged_dir, image_names, stage_info = stage_images(
            images_dir, run_dir / "images", max_images=max_images,
            downscale=downscale, names=all_image_names,
        )
        manifest = {
            "run_name": run_name,
            "condition": condition,
            "data_kind": data_kind,
            "images_dir": str(staged_dir.resolve()),
            "source_images_dir": str(Path(images_dir).resolve()),
            "image_names": image_names,
            "num_input_images": len(image_names),
            "downscale": float(downscale),
            "staged": stage_info["staged"],
            "colmap_version": version,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        dump_json(manifest, run_dir / "run_manifest.json")
        dump_json({"stage": stage_info, "image_names": image_names},
                  logs_dir / "image_staging.json")

        database = run_dir / "database.db"
        if input_database is not None:
            source = Path(input_database)
            if not source.exists():
                raise ActionableError(f"Input database not found: {source}")
            if source.resolve() != database.resolve():
                shutil.copy2(source, database)
                self.logger.info("Copied input database -> %s (source untouched)", database)
        sparse_dir = ensure_dir(run_dir / "sparse")
        timings: dict[str, float] = {}
        status = "success"
        failure_message = ""

        try:
            if reuse_database:
                if not database.exists():
                    raise ActionableError(
                        f"reuse_database=True but no database at {database}. "
                        "Pass --input-database or run feature extraction/matching first."
                    )
                self.logger.info("Reusing existing COLMAP database; skipping extraction/matching")
                timings["feature_extraction"] = 0.0
                timings["matching"] = 0.0
            else:
                with Timer() as timer:
                    self._run(self.feature_command(staged_dir, database),
                              command_log, "feature_extractor")
                timings["feature_extraction"] = timer.elapsed
                with Timer() as timer:
                    self._run(self.matcher_command(database), command_log, "matcher")
                timings["matching"] = timer.elapsed
            with Timer() as timer:
                self._run(self.mapper_command(staged_dir, database, sparse_dir),
                          command_log, "mapper")
            timings["mapping"] = timer.elapsed
        except ColmapCommandError as exc:
            status = "failed"
            failure_message = str(exc)
            self.logger.error("COLMAP pipeline failed: %s", exc)
            dump_json({"status": "failed", "error": failure_message,
                       "command": exc.command, "returncode": exc.returncode},
                      logs_dir / "pipeline_failure.json")

        summary: dict[str, Any] = {
            "run_name": run_name,
            "condition": condition,
            "data_kind": data_kind,
            "status": status,
            "num_input_images": len(image_names),
            "num_registered_images": 0,
            "registration_ratio": 0.0,
            "num_points3D": 0,
            "mean_reproj_error": float("nan"),
            "median_reproj_error": float("nan"),
            "track_length_median": float("nan"),
            "model_dir": None,
            "sparse_dir": str(sparse_dir),
            "database": str(database),
            "command_log": str(command_log),
            "colmap_version": version,
            "timings": timings,
            "startup_runtime_seconds": 0.0,
        }

        if status == "success":
            try:
                selected, stats = model_io.select_largest_model(sparse_dir, self.executable)
                readable = model_io.ensure_readable_model(
                    selected, self.executable, work_dir=run_dir / "sparse_txt"
                )
                rec = model_io.read_model(readable)
                model_dir = model_io.write_model(rec, run_dir / "model")
                analysis = track_analysis.analyze_reconstruction(
                    rec, self.config,
                    image_dir=staged_dir,
                    db_path=database,
                    all_image_names=image_names,
                    input_image_count=len(image_names),
                )
                track_analysis.write_analysis(analysis, run_dir)
                summary.update(analysis["summary"])
                summary["selected_sparse_model"] = str(selected)
                summary["selected_sparse_stats"] = stats
                summary["model_dir"] = str(model_dir)
                summary["status"] = "success"
                try:
                    from . import visualization
                    plot_dir = run_dir / "plots"
                    visualization.plot_track_length_hist(
                        analysis["tracks"], plot_dir / "track_length.png")
                    visualization.plot_reprojection_hist(
                        analysis["tracks"], plot_dir / "reproj_error.png")
                    visualization.plot_pointcloud_projection(
                        rec, plot_dir / "sparse_pointcloud.png")
                except Exception as exc:
                    self.logger.warning("Baseline plots failed: %s", exc)
            except Exception as exc:
                status = "failed"
                failure_message = f"{type(exc).__name__}: {exc}"
                summary["status"] = "failed"
                summary["error"] = failure_message
                self.logger.error("Post-mapper analysis failed: %s", exc)

        label = evaluation.label_summary(summary, self.config)
        summary["success_label"] = label["label"]
        summary["success_label_name"] = label["label_name"]
        summary["failure_reasons"] = label["reasons"]
        summary["label_checks"] = label["checks"]
        summary["runtime_seconds"] = time.perf_counter() - started
        if failure_message:
            summary["error"] = failure_message
        dump_json(summary, run_dir / "reconstruction_summary.json")
        dump_json({"label": label, "summary_path": str(run_dir / "reconstruction_summary.json")},
                  run_dir / "label.json")
        if status == "success":
            try:
                features.pre_reconstruction_features(run_dir, self.config, write=True)
            except Exception as exc:
                self.logger.warning("Pre-reconstruction feature extraction failed: %s", exc)

        dump_json({
            "run_name": run_name,
            "status": status,
            "timings": timings,
            "reports": [
                {"step": r.step, "command": r.command, "returncode": r.returncode,
                 "runtime_seconds": r.runtime_seconds}
                for r in self.reports
            ],
            "total_runtime_seconds": time.perf_counter() - started,
        }, logs_dir / "timing.json")
        self.logger.info(
            "Baseline %s: status=%s, registered=%s/%s, points=%s, mean_reproj=%.4f, label=%s",
            run_name, status, summary.get("num_registered_images"),
            summary.get("num_input_images"), summary.get("num_points3D"),
            float(summary.get("mean_reproj_error") or float("nan")),
            summary.get("success_label_name"),
        )
        return summary


def run_bundle_adjuster(model_dir: str | Path, output_dir: str | Path,
                        config: Mapping, dry_run: bool = False,
                        refine_intrinsics: bool = False) -> tuple[model_io.Reconstruction | None, dict]:
    """Run COLMAP bundle adjustment on an existing model (unweighted)."""
    runner = ColmapRunner(config, dry_run=dry_run)
    runner.check_available()
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    command = "bundle_adjuster"
    args = [runner.executable, command,
            "--input_path", str(model_dir),
            "--output_path", str(output_dir),
            "--BundleAdjustment.max_num_iterations",
            str(get_cfg(config, "colmap.ba_max_iterations", 100)),
            "--BundleAdjustment.refine_focal_length", str(int(refine_intrinsics)),
            "--BundleAdjustment.refine_principal_point", "0",
            "--BundleAdjustment.refine_extra_params", str(int(refine_intrinsics))]
    log = model_dir.parent / "logs" / "bundle_adjuster.log"
    report = runner._run(args, log, "bundle_adjuster")
    if dry_run:
        return None, {"dry_run": True, "command": args, "runtime_seconds": 0.0}
    readable = model_io.ensure_readable_model(
        output_dir, runner.executable, work_dir=output_dir.parent / (output_dir.name + "_txt")
    )
    rec = model_io.read_model(readable)
    return rec, {"command": args, "runtime_seconds": report.runtime_seconds,
                 "status": "ok", "approximation": True}


def run_point_triangulator(database: str | Path, image_dir: str | Path,
                           model_dir: str | Path, output_dir: str | Path,
                           config: Mapping, dry_run: bool = False) -> dict:
    """Optional approximate path: re-triangulate a filtered model."""
    runner = ColmapRunner(config, dry_run=dry_run)
    runner.check_available()
    args = [runner.executable, "point_triangulator",
            "--database_path", str(database),
            "--image_path", str(image_dir),
            "--input_path", str(model_dir),
            "--output_path", str(output_dir)]
    log = Path(output_dir).parent / "logs" / "point_triangulator.log"
    report = runner._run(args, log, "point_triangulator")
    return {"command": args, "runtime_seconds": report.runtime_seconds,
            "dry_run": dry_run}
