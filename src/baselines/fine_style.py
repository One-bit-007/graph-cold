"""FINE eigenvector filtering baseline adapter."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from src.baselines.base import BaselineResult, aligned_proba, array_hash, ensure_class_coverage


class FINEStyleBaseline:
    method = "FINE-style"
    method_family = "fine_style"
    implementation_status = "verified_implementation"
    faithfulness_level = "representation-eigenvector filtering inspired by FINE; not full original implementation"

    def __init__(
        self,
        seed: int = 0,
        noise_rate: float = 0.0,
        n_components: int = 32,
        min_retain_fraction: float = 0.2,
        min_class_size_for_svd: int = 20,
        classifier_epochs: int = 6,
        n_estimators: int = 8,
    ):
        self.seed = int(seed)
        self.noise_rate = float(noise_rate)
        self.n_components = int(n_components)
        self.min_retain_fraction = float(min_retain_fraction)
        self.min_class_size_for_svd = int(min_class_size_for_svd)
        self.classifier_epochs = int(classifier_epochs)
        self.n_estimators = int(n_estimators)
        self.scaler_: StandardScaler | None = None
        self.projector_: TruncatedSVD | None = None
        self.classifier_: ExtraTreesClassifier | None = None
        self.retained_mask_: np.ndarray | None = None
        self.scores_: np.ndarray | None = None
        self.metadata_: dict[str, Any] = {}
        self.num_classes_: int | None = None
        self.representation_source_: str = "standardized_features_pca"

    def fit(self, X_train, noisy_y_train, X_val=None, y_val=None, noise_rate=None, seed: int | None = None, representation=None):
        del X_val, y_val
        fit_seed = self.seed if seed is None else int(seed)
        if noise_rate is not None:
            self.noise_rate = float(noise_rate)
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(noisy_y_train, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError("X_train must be a 2D feature matrix.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_train and noisy_y_train must have the same number of rows.")
        self.num_classes_ = int(np.max(y)) + 1 if y.size else 0
        if self.num_classes_ < 2:
            raise ValueError("FINE-style requires at least two classes.")

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X).astype(np.float32)
        Z = self._represent(X_scaled, fit_seed, representation=representation)
        retained, scores, per_class = self._retention_mask(Z, y)
        confidence = scores if scores.size else np.ones_like(y, dtype=np.float64)
        retained = ensure_class_coverage(retained, confidence, y)
        self.retained_mask_ = retained
        self.scores_ = scores

        self.classifier_ = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            random_state=fit_seed,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.classifier_.fit(X_scaled[retained], y[retained])

        self.metadata_ = {
            "method_name": self.method,
            "method_family": self.method_family,
            "faithfulness_level": self.faithfulness_level,
            "representation_source": self.representation_source_,
            "trained_on": "retained noisy_y_train",
            "train_label_source": "noisy_y_train",
            "eval_label_source": "clean_y_test",
            "training_label_hash": array_hash(y),
            "filter_rule": "class_wise_top_eigenvector_alignment",
            "min_class_size_for_svd": self.min_class_size_for_svd,
            "retain_fraction": float(self._target_retain_fraction()),
            "overall_retained_fraction": float(np.mean(retained)) if retained.size else 0.0,
            "score_min": float(np.min(scores)) if scores.size else 0.0,
            "score_mean": float(np.mean(scores)) if scores.size else 0.0,
            "score_max": float(np.max(scores)) if scores.size else 0.0,
            "class_retain_fraction": per_class,
            "classes_retained_all_due_to_small_size": [
                str(label) for label, info in per_class.items() if info.get("retained_all_due_to_small_size", False)
            ],
            "classifier": "ExtraTreesClassifier",
            "n_estimators": self.n_estimators,
            "notes": (
                "class-wise eigenvector filtering baseline; reported with stability caveat "
                "when the real-data check is unstable"
            ),
        }
        return self

    def predict_proba(self, X_test):
        self._check_fitted()
        X = np.asarray(X_test, dtype=np.float32)
        X_scaled = self.scaler_.transform(X).astype(np.float32)  # type: ignore[union-attr]
        return aligned_proba(self.classifier_, X_scaled, self.num_classes_)  # type: ignore[arg-type]

    def predict(self, X_test):
        return np.argmax(self.predict_proba(X_test), axis=1).astype(np.int64)

    def get_retained_mask(self):
        if self.retained_mask_ is None:
            raise RuntimeError("FINEStyleBaseline is not fitted.")
        return self.retained_mask_.copy()

    def get_metadata(self):
        return dict(self.metadata_)

    def fit_predict(self, X_train, y_noisy, X_test, num_classes: int, **kwargs) -> BaselineResult:
        self.fit(
            X_train,
            y_noisy,
            noise_rate=kwargs.get("noise_rate", self.noise_rate),
            seed=self.seed,
            representation=kwargs.get("representation"),
        )
        proba = _ensure_width(self.predict_proba(X_test), num_classes)
        y_pred = np.argmax(proba, axis=1).astype(np.int64)
        retained = self.get_retained_mask()
        return BaselineResult(
            method=self.method,
            method_family=self.method_family,
            implementation_status=self.implementation_status,
            y_pred=y_pred,
            proba=proba,
            weights=retained.astype(np.float64),
            retained_mask=retained,
            details=self.get_metadata(),
        )

    def _represent(self, X_scaled: np.ndarray, seed: int, representation=None) -> np.ndarray:
        if representation is not None:
            Z = np.asarray(representation, dtype=np.float32)
            if Z.shape[0] != X_scaled.shape[0]:
                raise ValueError("Precomputed representation must match X_train rows.")
            self.representation_source_ = "d2_encoder_embeddings"
            return _row_l2_normalize(Z)
        n_components = min(max(2, self.n_components), max(2, X_scaled.shape[1] - 1))
        if X_scaled.shape[1] <= 2:
            self.projector_ = None
            self.representation_source_ = "standardized_features_pca"
            return _row_l2_normalize(X_scaled.astype(np.float32))
        self.projector_ = TruncatedSVD(n_components=n_components, random_state=seed)
        Z = self.projector_.fit_transform(X_scaled).astype(np.float32)
        self.representation_source_ = "standardized_features_pca"
        return _row_l2_normalize(Z)

    def _retention_mask(self, Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
        scores = np.ones(y.shape[0], dtype=np.float64)
        retained = np.ones(y.shape[0], dtype=bool)
        per_class: dict[str, dict[str, Any]] = {}
        target_fraction = self._target_retain_fraction()
        if self.noise_rate <= 0:
            for label in np.unique(y):
                members = np.flatnonzero(y == label)
                per_class[str(int(label))] = {
                    "size": int(members.size),
                    "retained": int(members.size),
                    "retain_fraction": 1.0,
                    "retained_all_due_to_small_size": False,
                }
            return retained, scores, per_class

        retained[:] = False
        for label in np.unique(y):
            members = np.flatnonzero(y == label)
            class_size = int(members.size)
            if class_size < self.min_class_size_for_svd:
                retained[members] = True
                per_class[str(int(label))] = {
                    "size": class_size,
                    "retained": class_size,
                    "retain_fraction": 1.0,
                    "retained_all_due_to_small_size": True,
                }
                continue
            Zc = Z[members]
            mean = Zc.mean(axis=0, keepdims=True)
            centered = _row_l2_normalize(Zc - mean)
            top = _top_right_vector(centered)
            class_scores = np.abs(centered @ top)
            scores[members] = class_scores
            keep_n = int(np.ceil(target_fraction * class_size))
            keep_n = min(max(1, keep_n), class_size)
            threshold = np.partition(class_scores, class_size - keep_n)[class_size - keep_n]
            keep = class_scores >= threshold
            if keep.sum() > keep_n:
                selected_local = np.flatnonzero(keep)
                order = np.argsort(class_scores[selected_local])[::-1][:keep_n]
                exact = np.zeros_like(keep, dtype=bool)
                exact[selected_local[order]] = True
                keep = exact
            retained[members[keep]] = True
            per_class[str(int(label))] = {
                "size": class_size,
                "retained": int(keep.sum()),
                "retain_fraction": float(keep.mean()),
                "retained_all_due_to_small_size": False,
                "score_min": float(class_scores.min(initial=0.0)),
                "score_mean": float(class_scores.mean()) if class_scores.size else 0.0,
                "score_max": float(class_scores.max(initial=0.0)),
            }
        return retained, scores, per_class

    def _target_retain_fraction(self) -> float:
        return float(np.clip(max(self.min_retain_fraction, 1.0 - self.noise_rate), self.min_retain_fraction, 1.0))

    def _check_fitted(self) -> None:
        if self.scaler_ is None or self.classifier_ is None or self.num_classes_ is None:
            raise RuntimeError("FINEStyleBaseline is not fitted.")


def exclusion_reason() -> str:
    return (
        "reported with caveat when stability checks fail; implementation uses "
        "class-wise eigenvector filtering on the repository representation"
    )


class FINEBaseline(FINEStyleBaseline):
    method = "FINE"
    method_family = "fine"
    implementation_status = "verified_implementation"
    faithfulness_level = (
        "FINE eigenvector filtering adapter on tabular IDS representations; "
        "reported with stability caveat when a dataset setting is unstable"
    )


def _row_l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denom, 1e-12)


def _top_right_vector(centered: np.ndarray) -> np.ndarray:
    cov = centered.T @ centered
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        vec = eigvecs[:, int(np.argmax(eigvals))]
    except np.linalg.LinAlgError:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        vec = vh[0]
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        vec = np.zeros(centered.shape[1], dtype=np.float64)
        vec[0] = 1.0
        return vec
    return (vec / norm).astype(np.float64)


def _ensure_width(proba: np.ndarray, num_classes: int) -> np.ndarray:
    if proba.shape[1] == num_classes:
        return proba
    out = np.zeros((proba.shape[0], num_classes), dtype=np.float64)
    cols = min(num_classes, proba.shape[1])
    out[:, :cols] = proba[:, :cols]
    rows = out.sum(axis=1, keepdims=True)
    zero = rows[:, 0] <= 1e-12
    if np.any(zero):
        out[zero, :] = 1.0 / max(num_classes, 1)
        rows = out.sum(axis=1, keepdims=True)
    return out / np.maximum(rows, 1e-12)
