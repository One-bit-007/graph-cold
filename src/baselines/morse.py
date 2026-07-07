"""MORSE-style semi-supervised purification baseline."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from src.baselines.base import BaselineResult, aligned_proba, array_hash, class_balance_weights, ensure_class_coverage


class MORSEBaseline:
    method = "MORSE"
    method_family = "morse"
    implementation_status = "verified_implementation"
    faithfulness_level = "MORSE-style split of suspected noisy labels into unlabeled pseudo-label candidates"

    def __init__(
        self,
        seed: int = 0,
        noise_rate: float = 0.0,
        n_components: int = 32,
        pseudo_threshold: float = 0.6,
        min_clean_fraction: float = 0.25,
        max_iter: int = 5,
        n_estimators: int = 32,
    ):
        self.seed = int(seed)
        self.noise_rate = float(noise_rate)
        self.n_components = int(n_components)
        self.pseudo_threshold = float(pseudo_threshold)
        self.min_clean_fraction = float(min_clean_fraction)
        self.max_iter = int(max_iter)
        self.n_estimators = int(n_estimators)

    def fit_predict(self, X_train, y_noisy, X_test, num_classes: int, **kwargs) -> BaselineResult:
        del kwargs
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_noisy, dtype=np.int64)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("MORSE requires a 2D X_train aligned with noisy labels.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        X_test_scaled = scaler.transform(np.asarray(X_test, dtype=np.float32)).astype(np.float32)
        Z = _representation(X_scaled, self.n_components, self.seed)
        confidence = _centroid_confidence(Z, y)
        clean_fraction = float(np.clip(1.0 - self.noise_rate, self.min_clean_fraction, 1.0))
        clean_mask = _top_fraction_by_class(confidence, y, clean_fraction)
        clean_mask = ensure_class_coverage(clean_mask, confidence, y)
        unlabeled_mask = ~clean_mask

        teacher = SGDClassifier(
            loss="log_loss",
            max_iter=self.max_iter,
            tol=1e-3,
            random_state=self.seed,
            class_weight="balanced",
        )
        teacher.fit(X_scaled[clean_mask], y[clean_mask])
        teacher_proba = aligned_proba(teacher, X_scaled, num_classes)
        pseudo_labels = np.argmax(teacher_proba, axis=1).astype(np.int64)
        pseudo_conf = np.max(teacher_proba, axis=1)
        pseudo_mask = unlabeled_mask & (pseudo_conf >= self.pseudo_threshold)

        train_labels = y.copy()
        train_labels[pseudo_mask] = pseudo_labels[pseudo_mask]
        sample_weight = np.zeros(y.shape[0], dtype=np.float64)
        sample_weight[clean_mask] = class_balance_weights(y)[clean_mask]
        sample_weight[pseudo_mask] = 0.35 * class_balance_weights(train_labels)[pseudo_mask]
        final_mask = clean_mask | pseudo_mask
        final_mask = ensure_class_coverage(final_mask, confidence, y)
        sample_weight[final_mask & (sample_weight <= 0)] = 0.25

        clf = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed + 17,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X_scaled[final_mask], train_labels[final_mask], sample_weight=sample_weight[final_mask])
        proba = aligned_proba(clf, X_test_scaled, num_classes)
        y_pred = np.argmax(proba, axis=1).astype(np.int64)
        return BaselineResult(
            method=self.method,
            method_family=self.method_family,
            implementation_status=self.implementation_status,
            y_pred=y_pred,
            proba=proba,
            weights=final_mask.astype(np.float64),
            retained_mask=final_mask,
            details={
                "method_name": self.method,
                "method_family": self.method_family,
                "faithfulness_level": self.faithfulness_level,
                "trained_on": "noisy_y_train_with_pseudo_labels_for_suspected_noisy_samples",
                "train_label_source": "noisy_y_train",
                "eval_label_source": "clean_y_test",
                "training_label_hash": array_hash(y),
                "split_rule": "actual_noise_rate_as_unlabeled_split_ratio",
                "target_clean_fraction": clean_fraction,
                "clean_fraction": float(np.mean(clean_mask)) if clean_mask.size else 0.0,
                "pseudo_labeled_fraction": float(np.mean(pseudo_mask)) if pseudo_mask.size else 0.0,
                "retained_fraction": float(np.mean(final_mask)) if final_mask.size else 0.0,
                "pseudo_threshold": self.pseudo_threshold,
                "classifier": "teacher_SGDClassifier_final_ExtraTreesClassifier",
                "max_iter": self.max_iter,
                "n_estimators": self.n_estimators,
            },
        )


def _representation(X_scaled: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    if int(n_components) <= 0 or X_scaled.shape[1] <= 2:
        return _row_l2_normalize(X_scaled)
    k = min(max(2, int(n_components)), max(2, X_scaled.shape[1] - 1))
    projector = TruncatedSVD(n_components=k, random_state=int(seed))
    return _row_l2_normalize(projector.fit_transform(X_scaled).astype(np.float32))


def _centroid_confidence(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    confidence = np.zeros(y.shape[0], dtype=np.float64)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        Zc = Z[idx]
        centroid = Zc.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(Zc - centroid, axis=1)
        scale = float(np.median(dist) + 1e-12)
        confidence[idx] = np.exp(-dist / scale)
    return confidence


def _top_fraction_by_class(confidence: np.ndarray, y: np.ndarray, fraction: float) -> np.ndarray:
    mask = np.zeros(y.shape[0], dtype=bool)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        keep_n = min(max(1, int(np.ceil(fraction * idx.size))), idx.size)
        selected = np.argpartition(confidence[idx], idx.size - keep_n)[idx.size - keep_n :]
        mask[idx[selected]] = True
    return mask


def _row_l2_normalize(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denom, 1e-12)
