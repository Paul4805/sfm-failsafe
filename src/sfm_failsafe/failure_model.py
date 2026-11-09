"""Failure-prediction training and evaluation.

Primary mode ``pre`` uses only pre-reconstruction features (database, matches,
two-view geometry, blur, candidate tracks).  ``post`` uses reconstruction
statistics and is explicitly marked as not leakage-free.

The trainer is deliberately conservative: with few runs it flags results as
preliminary, uses a group-aware split when conditions allow, and never invents
cross-validation folds that contain a single class.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from . import evaluation, features, visualization
from .config import get_cfg
from .utils import (ActionableError, dump_json, ensure_dir, load_json,
                    safe_float, setup_logger)

METADATA_COLUMNS = {
    "run_name", "condition", "scene", "subset_id", "data_kind", "mode",
    "label", "label_name", "label_reasons", "run_dir",
}


def _choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    """Pick a probability threshold on *training* predictions by F1."""
    if len(np.unique(y)) < 2 or len(y) == 0:
        return 0.5
    candidates = np.unique(np.concatenate([probability, [0.0, 0.5, 1.0]]))
    best_threshold, best_score = 0.5, (-1.0, -1.0, -1.0)
    for threshold in candidates:
        metrics = evaluation.classification_metrics(y, probability, float(threshold))
        score = (metrics["f1"], metrics["accuracy"], -abs(float(threshold) - 0.5))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def build_pipeline(model_name: str, config: Mapping):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    seed = int(get_cfg(config, "failure_model.random_state", 0))
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=seed, C=1.0,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=250, max_depth=4, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", classifier),
        ]
    elif model_name == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=120, max_depth=2, learning_rate=0.05, random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", classifier),
        ]
    else:
        raise ActionableError(f"Unknown model: {model_name}")
    return Pipeline(steps)


def _prepare_matrix(df: pd.DataFrame, feature_names: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    matrix = df[list(feature_names)].copy()
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    usable = [column for column in matrix.columns if matrix[column].notna().any()]
    dropped = [column for column in matrix.columns if column not in usable]
    return matrix[usable].astype(float), dropped


def _can_stratify(y: np.ndarray) -> bool:
    values, counts = np.unique(y, return_counts=True)
    return len(values) == 2 and counts.min() >= 2


def split_dataset(df: pd.DataFrame, config: Mapping) -> dict[str, Any]:
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    fm = get_cfg(config, "failure_model", {}) or {}
    test_size = float(fm.get("test_size", 0.3))
    seed = int(fm.get("random_state", 0))
    group_column = str(fm.get("group_column", "condition"))
    y = df["label"].to_numpy(dtype=int)
    info: dict[str, Any] = {
        "test_size": test_size,
        "random_state": seed,
        "group_column": group_column,
        "n_runs": int(len(df)),
        "n_groups": int(df[group_column].nunique()) if group_column in df else 0,
        "evaluation_mode": "unknown",
    }
    if len(df) < 4 or not _can_stratify(y):
        info.update({
            "evaluation_mode": "in_sample_preliminary",
            "train_indices": list(range(len(df))),
            "test_indices": list(range(len(df))),
            "reason": "too few runs or too few samples per class for a held-out split",
        })
        return info

    # Group-aware split when there are enough distinct conditions.
    if group_column in df and df[group_column].nunique() >= 4:
        groups = df[group_column].astype(str).to_numpy()
        for attempt in range(30):
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                         random_state=seed + attempt)
            train_idx, test_idx = next(splitter.split(df, y, groups))
            if _can_stratify(y[train_idx]) and _can_stratify(y[test_idx]):
                info.update({
                    "evaluation_mode": "group_held_out",
                    "train_indices": train_idx.tolist(),
                    "test_indices": test_idx.tolist(),
                    "test_groups": sorted(set(groups[test_idx].tolist())),
                    "train_groups": sorted(set(groups[train_idx].tolist())),
                    "attempt": attempt,
                })
                return info
        info["group_split_warning"] = "no group split contained both classes; falling back to stratified run split"

    if _can_stratify(y):
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=test_size, random_state=seed, stratify=y
        )
        info.update({
            "evaluation_mode": "stratified_holdout",
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
        })
    else:
        info.update({
            "evaluation_mode": "in_sample_preliminary",
            "train_indices": list(range(len(df))),
            "test_indices": list(range(len(df))),
            "reason": "no valid held-out split",
        })
    return info


def _importance(pipeline, feature_names: Sequence[str]) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_, dtype=float)).reshape(-1)
    elif hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
    else:
        return np.zeros(len(feature_names))
    if len(values) != len(feature_names):
        return np.zeros(len(feature_names))
    total = float(np.sum(np.abs(values)))
    return values / total if total > 0 else values


def leave_one_condition_out(df: pd.DataFrame, feature_names: Sequence[str],
                            config: Mapping) -> dict[str, Any]:
    """Leave-one-condition-out predictions (only where training has both classes)."""
    group_column = str(get_cfg(config, "failure_model.group_column", "condition"))
    if group_column not in df or df[group_column].nunique() < 3:
        return {"status": "skipped", "reason": "fewer than 3 conditions"}
    records: list[dict[str, Any]] = []
    for group in sorted(df[group_column].astype(str).unique()):
        test_mask = df[group_column].astype(str) == group
        train = df[~test_mask]
        test = df[test_mask]
        y_train = train["label"].to_numpy(dtype=int)
        if not _can_stratify(y_train):
            continue
        matrix_train, _ = _prepare_matrix(train, feature_names)
        matrix_test, _ = _prepare_matrix(test, feature_names)
        # Use the globally usable feature set for every fold; the imputer
        # handles a column that happens to be empty within one training fold.
        matrix_train = matrix_train.reindex(columns=feature_names)
        matrix_test = matrix_test.reindex(columns=feature_names)
        pipeline = build_pipeline("random_forest", config)
        pipeline.fit(matrix_train.to_numpy(float), y_train)
        train_probability = pipeline.predict_proba(matrix_train.to_numpy(float))[:, 1]
        fold_threshold = _choose_threshold(y_train, train_probability)
        probability = pipeline.predict_proba(matrix_test.to_numpy(float))[:, 1]
        for (_, row), prob in zip(test.iterrows(), probability):
            records.append({
                "run_name": row.get("run_name"),
                "condition": group,
                "label": int(row["label"]),
                "probability": float(prob),
                "predicted": int(prob >= fold_threshold),
                "threshold": float(fold_threshold),
                "model": "random_forest",
            })
    if not records:
        return {"status": "skipped", "reason": "no condition fold had both classes in training"}
    frame = pd.DataFrame(records)
    y = frame["label"].to_numpy(dtype=int)
    p = frame["probability"].to_numpy(dtype=float)
    return {
        "status": "ok",
        "folds": int(frame["condition"].nunique()),
        "predictions": frame.to_dict(orient="records"),
        "metrics": evaluation.classification_metrics(
            y, p, float(get_cfg(config, "failure_model.threshold", 0.5))
        ),
    }


def train_failure_model(runs_root: str | Path, out_dir: str | Path,
                        config: Mapping, mode: str = "pre",
                        logger=None) -> dict[str, Any]:
    logger = logger or setup_logger("sfm_failsafe.failure")
    out_dir = ensure_dir(Path(out_dir))
    runs_root = Path(runs_root)
    df, warnings = features.collect_runs(runs_root, config, mode=mode, logger=logger)
    if df.empty:
        raise ActionableError(
            f"No runs with usable features found under {runs_root}.\n"
            "Run scripts/run_baseline.py for each condition, then retry.\n"
            "Warnings:\n  " + "\n  ".join(warnings or ["none"])
        )
    dataset_path = out_dir / f"runs_dataset_{mode}.csv"
    df.to_csv(dataset_path, index=False)
    feature_names = features.feature_columns(df, mode=mode)
    if not feature_names:
        raise ActionableError(
            "No numeric feature columns found. Check database.db / features_pre.json."
        )
    class_counts = df["label"].value_counts().to_dict()
    min_class = int(get_cfg(config, "failure_model.min_class_samples", 2))
    if len(class_counts) < 2 or min(class_counts.values()) < min_class:
        payload = {
            "status": "insufficient_data",
            "mode": mode,
            "n_runs": int(len(df)),
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "warnings": warnings,
            "message": (
                "At least two classes and "
                f"{min_class} runs per class are required to train a classifier. "
                "Generate more subsets/runs (see scripts/make_challenging_subsets.py)."
            ),
            "feature_names": feature_names,
        }
        dump_json(payload, out_dir / "metrics.json")
        logger.error(payload["message"])
        raise ActionableError(payload["message"])

    matrix, dropped = _prepare_matrix(df, feature_names)
    feature_names = list(matrix.columns)
    y = df["label"].to_numpy(dtype=int)
    split = split_dataset(df, config)
    train_idx = np.asarray(split["train_indices"], dtype=int)
    test_idx = np.asarray(split["test_indices"], dtype=int)
    X_train = matrix.iloc[train_idx].to_numpy(dtype=float)
    X_test = matrix.iloc[test_idx].to_numpy(dtype=float)
    y_train, y_test = y[train_idx], y[test_idx]
    if len(np.unique(y_train)) < 2:
        raise ActionableError(
            "The training split contains a single class. Add more diverse runs or "
            "reduce failure_model.test_size / use a different split."
        )

    model_names = list(get_cfg(config, "failure_model.models", [
        "logistic_regression", "random_forest", "gradient_boosting"
    ]))
    threshold = float(get_cfg(config, "failure_model.threshold", 0.5))
    model_results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    base_fitted: dict[str, Any] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for name in model_names:
        base_pipeline = build_pipeline(name, config)
        base_pipeline.fit(X_train, y_train)
        predictor = base_pipeline
        if bool(get_cfg(config, "failure_model.calibrate", True)):
            try:
                from sklearn.calibration import CalibratedClassifierCV
                minority = int(np.bincount(y_train, minlength=2).min())
                folds = int(min(3, max(2, minority)))
                calibrated = CalibratedClassifierCV(
                    estimator=base_pipeline, method="sigmoid", cv=folds,
                )
                calibrated.fit(X_train, y_train)
                predictor = calibrated
            except Exception:
                predictor = base_pipeline
        train_probability = predictor.predict_proba(X_train)[:, 1]
        tuned_threshold = _choose_threshold(y_train, train_probability)
        probability = predictor.predict_proba(X_test)[:, 1]
        metrics = evaluation.classification_metrics(y_test, probability, tuned_threshold)
        metrics_at_default = evaluation.classification_metrics(y_test, probability, threshold)
        model_results[name] = {
            "metrics": metrics,
            "metrics_at_default_threshold": metrics_at_default,
            "threshold_tuned_on_train": tuned_threshold,
            "calibrated": predictor is not base_pipeline,
            "importance": dict(zip(feature_names, _importance(base_pipeline, feature_names).tolist())),
        }
        fitted[name] = predictor
        base_fitted[name] = base_pipeline
        test_probabilities[name] = probability

    selection_metric = str(get_cfg(config, "failure_model.selection_metric", "roc_auc"))
    def score(name: str) -> tuple[float, float, float]:
        metrics = model_results[name]["metrics"]
        primary = safe_float(metrics.get(selection_metric))
        if not np.isfinite(primary):
            primary = safe_float(metrics.get("f1"), -1.0)
        secondary = safe_float(metrics.get("f1"), -1.0)
        tertiary = safe_float(metrics.get("accuracy"), -1.0)
        return (primary, secondary, tertiary)

    best_name = max(fitted, key=score)
    best_pipeline = fitted[best_name]
    best_metrics = model_results[best_name]["metrics"]
    best_probability = test_probabilities[best_name]
    best_threshold = float(model_results[best_name]["threshold_tuned_on_train"])
    best_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": _importance(base_fitted[best_name], feature_names),
    }).sort_values("importance", ascending=False)

    preliminary = (
        split["evaluation_mode"] == "in_sample_preliminary"
        or len(df) < 10
        or int(min(class_counts.values())) < 3
        or split.get("n_groups", 0) < 3
    )
    data_kinds = sorted(set(df["data_kind"].astype(str))) if "data_kind" in df else ["unknown"]
    data_kind = data_kinds[0] if len(data_kinds) == 1 else "mixed"

    oof = leave_one_condition_out(df, feature_names, config) if len(df) >= 8 else {
        "status": "skipped", "reason": "fewer than 8 runs"
    }
    oof_path = None
    if oof.get("status") == "ok":
        oof_path = out_dir / "oof_predictions.csv"
        pd.DataFrame(oof["predictions"]).to_csv(oof_path, index=False)

    label_config = evaluation.label_summary({}, config)["thresholds"]
    bundle = {
        "mode": mode,
        "leakage_free": mode == "pre",
        "feature_names": feature_names,
        "pipeline": best_pipeline,
        "threshold": best_threshold,
        "default_threshold": threshold,
        "label_config": label_config,
        "model_name": best_name,
        "metrics": best_metrics,
        "split": split,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_kind": data_kind,
        "preliminary": preliminary,
        "n_runs": int(len(df)),
    }
    joblib.dump(bundle, out_dir / "failure_model.joblib")
    best_importance.to_csv(out_dir / "feature_importances.csv", index=False)

    metrics_payload = {
        "status": "ok",
        "mode": mode,
        "leakage_free": mode == "pre",
        "leakage_note": (
            "Pre-reconstruction features only (database/matches/candidate tracks)."
            if mode == "pre" else
            "Post-reconstruction features use mapper output and are NOT leakage-free."
        ),
        "data_kind": data_kind,
        "n_runs": int(len(df)),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "dropped_all_nan_features": dropped,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "split": {k: v for k, v in split.items() if "indices" not in k},
        "train_runs": df.iloc[train_idx]["run_name"].astype(str).tolist(),
        "test_runs": df.iloc[test_idx]["run_name"].astype(str).tolist(),
        "models": model_results,
        "best_model": best_name,
        "best_metrics": best_metrics,
        "threshold": best_threshold,
        "default_threshold": threshold,
        "preliminary": bool(preliminary),
        "preliminary_reason": (
            "Small number of runs / conditions; treat metrics as pipeline validation, "
            "not a scientific result." if preliminary else None
        ),
        "warnings": warnings,
        "leave_one_condition_out": {
            k: v for k, v in oof.items() if k != "predictions"
        },
        "oof_predictions": str(oof_path) if oof_path else None,
        "label_config": label_config,
        "trained_at": bundle["trained_at"],
        "dataset_csv": str(dataset_path),
        "model_path": str(out_dir / "failure_model.joblib"),
    }
    dump_json(metrics_payload, out_dir / "metrics.json")
    dump_json(split, out_dir / "split.json")

    plots = out_dir / "plots"
    visualization.plot_class_balance(y, plots / "class_balance.png")
    visualization.plot_roc_pr(y_test, best_probability, plots / "roc_pr.png",
                              title=f"{mode} mode - {best_name}")
    visualization.plot_confusion(best_metrics["confusion_matrix"], plots / "confusion_matrix.png")
    visualization.plot_feature_importance(
        best_importance["feature"].tolist(), best_importance["importance"].tolist(),
        plots / "feature_importance.png",
    )
    if oof.get("status") == "ok":
        oof_frame = pd.DataFrame(oof["predictions"])
        visualization.plot_roc_pr(
            oof_frame["label"].to_numpy(), oof_frame["probability"].to_numpy(),
            plots / "roc_pr_leave_one_condition_out.png",
            title="Leave-one-condition-out (random forest)",
        )
    logger.info(
        "Trained %s mode on %d runs (%d features); best=%s, %s=%.3f, preliminary=%s",
        mode, len(df), len(feature_names), best_name, selection_metric,
        safe_float(best_metrics.get(selection_metric)), preliminary,
    )
    return metrics_payload


def predict_bundle(bundle_path: str | Path, rows: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(bundle_path)
    matrix, _ = _prepare_matrix(rows, bundle["feature_names"])
    matrix = matrix.reindex(columns=bundle["feature_names"])
    return bundle["pipeline"].predict_proba(matrix.to_numpy(dtype=float))[:, 1]
