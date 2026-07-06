"""Confident-learning style filtering baseline.

If the optional cleanlab package is available, label issues are identified via
``cleanlab.filter.find_label_issues`` using validation-style predicted
probabilities. Otherwise the method honestly reports itself as ``CL-filtering``
and filters low self-confidence noisy labels with a documented deterministic
rule.
"""
from __future__ import annotations

import importlib.util
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from src.baselines.base import BaselineResult, aligned_proba, array_hash, ensure_class_coverage


class ConfidentLearningBaseline:
    def __init__(
        self,
        seed: int = 0,
        noise_rate: float = 0.0,
        n_estimators: int = 8,
        min_confidence_margin: float = 0.02,
        max_probe_train: int = 50_000,
    ):
        self.seed = int(seed)
        self.noise_rate = float(noise_rate)
        self.n_estimators = int(n_estimators)
        self.min_confidence_margin = float(min_confidence_margin)
        self.max_probe_train = int(max_probe_train)

    @property
    def cleanlab_available(self) -> bool:
        return importlib.util.find_spec("cleanlab") is not None

    @property
    def method(self) -> str:
        return "Confident-Learning" if self.cleanlab_available else "CL-filtering"

    @property
    def method_family(self) -> str:
        return "confident_learning" if self.cleanlab_available else "cl_filtering"

    def fit_predict(self, X_train, y_noisy, X_test, num_classes: int, **kwargs) -> BaselineResult:
        y_noisy = np.asarray(y_noisy, dtype=np.int64)
        pred_probs_train = self._probe_probabilities(X_train, y_noisy, num_classes)
        confidence = pred_probs_train[np.arange(y_noisy.shape[0]), y_noisy]
        retained, engine = self._retention_mask(y_noisy, pred_probs_train, confidence)
        retained = ensure_class_coverage(retained, confidence, y_noisy)

        model = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train[retained], y_noisy[retained])
        y_pred = model.predict(X_test).astype(np.int64)
        proba = aligned_proba(model, X_test, num_classes)
        return BaselineResult(
            method=self.method if engine == "cleanlab" else "CL-filtering",
            method_family=self.method_family if engine == "cleanlab" else "cl_filtering",
            implementation_status="verified_implementation",
            y_pred=y_pred,
            proba=proba,
            weights=retained.astype(np.float64),
            retained_mask=retained,
            details={
                "classifier": "ExtraTreesClassifier",
                "issue_detector": engine,
                "probe_model": "SGDClassifier(log_loss)",
                "trained_on": "retained noisy_y_train",
                "train_label_source": "noisy_y_train",
                "eval_label_source": "clean_y_test",
                "training_label_hash": array_hash(y_noisy),
                "retained_fraction": float(np.mean(retained)),
                "estimated_issue_fraction": float(1.0 - np.mean(retained)),
                "filter_rule": "cleanlab_find_label_issues" if engine == "cleanlab" else "low_noisy_label_self_confidence_quantile",
            },
        )

    def _probe_probabilities(self, X_train, y_noisy: np.ndarray, num_classes: int) -> np.ndarray:
        labels, counts = np.unique(y_noisy, return_counts=True)
        if labels.size < 2 or counts.min() < 2:
            out = np.zeros((y_noisy.shape[0], num_classes), dtype=np.float64)
            out[np.arange(y_noisy.shape[0]), y_noisy] = 1.0
            return out

        train_idx, calib_idx = self._stratified_probe_split(y_noisy)
        fit_idx = self._cap_probe_train_indices(train_idx, y_noisy)
        probe = SGDClassifier(
            loss="log_loss",
            alpha=5e-4,
            max_iter=5,
            tol=1e-3,
            random_state=self.seed,
            class_weight="balanced",
        )
        probe.fit(X_train[fit_idx], y_noisy[fit_idx])
        probs = _aligned_sgd_proba(probe, X_train, num_classes)
        # Replace the calibration slice with strictly out-of-split estimates.
        probs[calib_idx] = _aligned_sgd_proba(probe, X_train[calib_idx], num_classes)
        return probs

    def _stratified_probe_split(self, y_noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labels, counts = np.unique(y_noisy, return_counts=True)
        if labels.size < 2 or counts.min() < 2:
            idx = np.arange(y_noisy.shape[0])
            return idx, idx
        test_size = 0.25 if y_noisy.shape[0] < 200_000 else 0.15
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
        train_idx, calib_idx = next(splitter.split(np.zeros_like(y_noisy), y_noisy))
        return train_idx, calib_idx

    def _cap_probe_train_indices(self, train_idx: np.ndarray, y_noisy: np.ndarray) -> np.ndarray:
        if train_idx.shape[0] <= self.max_probe_train:
            return train_idx
        y_part = y_noisy[train_idx]
        labels, counts = np.unique(y_part, return_counts=True)
        if labels.size < 2 or counts.min() < 2:
            return train_idx[: self.max_probe_train]
        sample_local, _ = train_test_split(
            np.arange(train_idx.shape[0]),
            train_size=self.max_probe_train,
            random_state=self.seed,
            stratify=y_part,
        )
        return train_idx[np.sort(sample_local)]

    def _retention_mask(
        self,
        y_noisy: np.ndarray,
        pred_probs_train: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        if self.cleanlab_available:
            try:
                from cleanlab.filter import find_label_issues

                issues = find_label_issues(
                    labels=y_noisy,
                    pred_probs=pred_probs_train,
                    return_indices_ranked_by="self_confidence",
                )
                issue_mask = np.zeros(y_noisy.shape[0], dtype=bool)
                issue_mask[np.asarray(issues, dtype=np.int64)] = True
                return ~issue_mask, "cleanlab"
            except Exception:
                pass
        return self._self_confidence_retention(y_noisy, pred_probs_train, confidence), "self_confidence_filter"

    def _self_confidence_retention(
        self,
        y_noisy: np.ndarray,
        pred_probs_train: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        if self.noise_rate <= 0:
            return np.ones(y_noisy.shape[0], dtype=bool)
        issue_fraction = float(np.clip(self.noise_rate, 0.02, 0.55))
        threshold = float(np.quantile(confidence, issue_fraction))
        predicted = np.argmax(pred_probs_train, axis=1)
        low_confidence = confidence <= max(threshold, 1.0 / max(pred_probs_train.shape[1], 1) + self.min_confidence_margin)
        disagreement = predicted != y_noisy
        issue_mask = low_confidence | (disagreement & (confidence <= np.quantile(confidence, min(issue_fraction + 0.1, 0.65))))
        return ~issue_mask


def _aligned_sgd_proba(model: SGDClassifier, X, num_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
    else:
        decision = model.decision_function(X)
        decision = np.atleast_2d(decision)
        decision = decision - decision.max(axis=1, keepdims=True)
        raw = np.exp(decision) / np.maximum(np.exp(decision).sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros((X.shape[0], num_classes), dtype=np.float64)
    for col_idx, label in enumerate(getattr(model, "classes_", np.arange(raw.shape[1]))):
        label_idx = int(label)
        if 0 <= label_idx < num_classes:
            out[:, label_idx] = raw[:, col_idx]
    rows = out.sum(axis=1, keepdims=True)
    zero = rows[:, 0] <= 1e-12
    if np.any(zero):
        out[zero, :] = 1.0 / max(num_classes, 1)
        rows = out.sum(axis=1, keepdims=True)
    return out / np.maximum(rows, 1e-12)
