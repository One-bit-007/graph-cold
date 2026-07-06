"""MCRe-style class-wise representation purification baseline."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from src.baselines.base import BaselineResult, aligned_proba, array_hash, ensure_class_coverage


class MCReBaseline:
    method = "MCRe"
    method_family = "mcre"
    implementation_status = "verified_implementation"
    faithfulness_level = "MCRe-style class-wise representation purification for tabular IDS features"

    def __init__(
        self,
        seed: int = 0,
        noise_rate: float = 0.0,
        n_components: int = 32,
        min_retain_fraction: float = 0.25,
        max_iter: int = 5,
    ):
        self.seed = int(seed)
        self.noise_rate = float(noise_rate)
        self.n_components = int(n_components)
        self.min_retain_fraction = float(min_retain_fraction)
        self.max_iter = int(max_iter)

    def fit_predict(self, X_train, y_noisy, X_test, num_classes: int, **kwargs) -> BaselineResult:
        del kwargs
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_noisy, dtype=np.int64)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("MCRe requires a 2D X_train aligned with noisy labels.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        X_test_scaled = scaler.transform(np.asarray(X_test, dtype=np.float32)).astype(np.float32)
        Z = _representation(X_scaled, self.n_components, self.seed)
        retained, confidence, per_class = _classwise_density_retain(
            Z,
            y,
            retain_fraction=float(np.clip(max(self.min_retain_fraction, 1.0 - self.noise_rate), self.min_retain_fraction, 1.0)),
        )
        retained = ensure_class_coverage(retained, confidence, y)

        clf = SGDClassifier(
            loss="log_loss",
            max_iter=self.max_iter,
            tol=1e-3,
            random_state=self.seed,
            class_weight="balanced",
        )
        clf.fit(X_scaled[retained], y[retained])
        proba = aligned_proba(clf, X_test_scaled, num_classes)
        y_pred = np.argmax(proba, axis=1).astype(np.int64)
        return BaselineResult(
            method=self.method,
            method_family=self.method_family,
            implementation_status=self.implementation_status,
            y_pred=y_pred,
            proba=proba,
            weights=retained.astype(np.float64),
            retained_mask=retained,
            details={
                "method_name": self.method,
                "method_family": self.method_family,
                "faithfulness_level": self.faithfulness_level,
                "trained_on": "noisy_y_train",
                "train_label_source": "noisy_y_train",
                "eval_label_source": "clean_y_test",
                "training_label_hash": array_hash(y),
                "filter_rule": "class_wise_centroid_density",
                "retain_fraction": float(np.mean(retained)) if retained.size else 0.0,
                "target_retain_fraction": float(np.clip(max(self.min_retain_fraction, 1.0 - self.noise_rate), self.min_retain_fraction, 1.0)),
                "class_retain_fraction": per_class,
                "classifier": "SGDClassifier(log_loss)",
                "max_iter": self.max_iter,
            },
        )


def _representation(X_scaled: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    if int(n_components) <= 0 or X_scaled.shape[1] <= 2:
        return _row_l2_normalize(X_scaled)
    k = min(max(2, int(n_components)), max(2, X_scaled.shape[1] - 1))
    projector = TruncatedSVD(n_components=k, random_state=int(seed))
    return _row_l2_normalize(projector.fit_transform(X_scaled).astype(np.float32))


def _classwise_density_retain(
    Z: np.ndarray,
    y: np.ndarray,
    retain_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    retained = np.zeros(y.shape[0], dtype=bool)
    confidence = np.zeros(y.shape[0], dtype=np.float64)
    per_class: dict[str, dict[str, Any]] = {}
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        Zc = Z[idx]
        centroid = Zc.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(Zc - centroid, axis=1)
        scale = float(np.median(dist) + 1e-12)
        score = np.exp(-dist / scale)
        confidence[idx] = score
        keep_n = min(max(1, int(np.ceil(retain_fraction * idx.size))), idx.size)
        selected = np.argpartition(score, idx.size - keep_n)[idx.size - keep_n :]
        retained[idx[selected]] = True
        per_class[str(int(label))] = {
            "size": int(idx.size),
            "retained": int(keep_n),
            "retain_fraction": float(keep_n / max(idx.size, 1)),
            "score_mean": float(score.mean()) if score.size else 0.0,
        }
    return retained, confidence, per_class


def _row_l2_normalize(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denom, 1e-12)
