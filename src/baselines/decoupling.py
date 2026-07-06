"""Decoupling disagreement-update baseline for noisy-label experiments.

This is a deterministic tabular implementation of the core Decoupling idea:
two independently initialized classifiers are updated only on examples where
their predictions disagree after a warmup pass. It trains only on
``noisy_y_train``; clean labels are accepted by runners only for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from src.baselines.base import BaselineResult, array_hash, class_balance_weights


@dataclass
class DecouplingMetadata:
    method: str = "Decoupling"
    method_family: str = "decoupling"
    warmup_epochs: int = 1
    epochs: int = 6
    batch_size: int = 4096
    disagreement_fraction: float = 0.0
    update_fraction: float = 0.0
    retained_for_update_after_warmup: float = 0.0
    warmup_retained_all: bool = True
    trained_on: str = "noisy_y_train"
    train_label_source: str = "noisy_y_train"
    eval_label_source: str = "clean_y_test"
    classifier: str = "two SGDClassifier(log_loss)"
    notes: str = "standard tabular implementation of disagreement-update Decoupling"
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "method": self.method,
            "method_family": self.method_family,
            "warmup_epochs": self.warmup_epochs,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "disagreement_fraction": self.disagreement_fraction,
            "update_fraction": self.update_fraction,
            "retained_for_update_after_warmup": self.retained_for_update_after_warmup,
            "warmup_retained_all": self.warmup_retained_all,
            "trained_on": self.trained_on,
            "train_label_source": self.train_label_source,
            "eval_label_source": self.eval_label_source,
            "classifier": self.classifier,
            "notes": self.notes,
        }
        out.update(self.extra)
        return out


class DecouplingBaseline:
    method = "Decoupling"
    method_family = "decoupling"
    implementation_status = "verified_implementation"
    faithfulness_level = "standard tabular implementation of disagreement-update Decoupling"

    def __init__(
        self,
        seed: int = 0,
        epochs: int = 6,
        warmup_epochs: int = 1,
        batch_size: int = 4096,
        alpha: float = 1e-4,
    ):
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.batch_size = int(batch_size)
        self.alpha = float(alpha)
        self.scaler_: StandardScaler | None = None
        self.clf_a_: SGDClassifier | None = None
        self.clf_b_: SGDClassifier | None = None
        self.retained_mask_: np.ndarray | None = None
        self.metadata_: dict[str, Any] = {}
        self.num_classes_: int | None = None

    def fit(self, X_train, noisy_y_train, X_val=None, y_val=None, noise_rate=None, seed: int | None = None):
        del X_val, y_val, noise_rate
        fit_seed = self.seed if seed is None else int(seed)
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(noisy_y_train, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError("X_train must be a 2D feature matrix.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_train and noisy_y_train must have the same number of rows.")
        self.num_classes_ = int(np.max(y)) + 1 if y.size else 0
        if self.num_classes_ < 2:
            raise ValueError("Decoupling requires at least two classes.")

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X).astype(np.float32)
        classes = np.arange(self.num_classes_, dtype=np.int64)
        self.clf_a_ = _sgd(fit_seed, self.alpha)
        self.clf_b_ = _sgd(fit_seed + 10_007, self.alpha)
        balance = class_balance_weights(y)
        rng = np.random.default_rng(fit_seed)
        retained = np.zeros(y.shape[0], dtype=bool)
        disagreement_total = 0
        post_warmup_seen = 0
        update_total = 0

        for epoch in range(max(self.epochs, 1)):
            order = rng.permutation(y.shape[0])
            is_warmup = epoch < max(self.warmup_epochs, 0)
            for start in range(0, order.shape[0], self.batch_size):
                idx = order[start : start + self.batch_size]
                if idx.size == 0:
                    continue
                if is_warmup or not _is_fitted(self.clf_a_) or not _is_fitted(self.clf_b_):
                    update_idx = idx
                else:
                    pred_a = self.clf_a_.predict(X_scaled[idx])
                    pred_b = self.clf_b_.predict(X_scaled[idx])
                    disagree = pred_a != pred_b
                    update_idx = idx[disagree]
                    post_warmup_seen += int(idx.size)
                    disagreement_total += int(disagree.sum())
                    if update_idx.size:
                        retained[update_idx] = True
                if update_idx.size == 0:
                    continue
                self.clf_a_.partial_fit(X_scaled[update_idx], y[update_idx], classes=classes, sample_weight=balance[update_idx])
                self.clf_b_.partial_fit(X_scaled[update_idx], y[update_idx], classes=classes, sample_weight=balance[update_idx])
                if not is_warmup:
                    update_total += int(update_idx.size)

        self.retained_mask_ = retained
        denominator = max(post_warmup_seen, 1)
        meta = DecouplingMetadata(
            warmup_epochs=max(self.warmup_epochs, 0),
            epochs=max(self.epochs, 1),
            batch_size=self.batch_size,
            disagreement_fraction=float(disagreement_total / denominator),
            update_fraction=float(update_total / denominator),
            retained_for_update_after_warmup=float(np.mean(retained)) if retained.size else 0.0,
            extra={
                "training_label_hash": array_hash(y),
                "retained_fraction": float(np.mean(retained)) if retained.size else 0.0,
                "post_warmup_samples_seen": int(post_warmup_seen),
                "post_warmup_disagreements": int(disagreement_total),
                "post_warmup_updates": int(update_total),
            },
        ).as_dict()
        self.metadata_ = meta
        return self

    def predict_proba(self, X_test):
        self._check_fitted()
        X = np.asarray(X_test, dtype=np.float32)
        X_scaled = self.scaler_.transform(X).astype(np.float32)  # type: ignore[union-attr]
        proba = 0.5 * (
            _aligned_proba(self.clf_a_, X_scaled, self.num_classes_)  # type: ignore[arg-type]
            + _aligned_proba(self.clf_b_, X_scaled, self.num_classes_)  # type: ignore[arg-type]
        )
        return proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X_test):
        return np.argmax(self.predict_proba(X_test), axis=1).astype(np.int64)

    def get_retained_mask(self):
        if self.retained_mask_ is None:
            raise RuntimeError("DecouplingBaseline is not fitted.")
        return self.retained_mask_.copy()

    def get_metadata(self):
        return dict(self.metadata_)

    def fit_predict(self, X_train, y_noisy, X_test, num_classes: int, **kwargs) -> BaselineResult:
        del kwargs
        self.fit(X_train, y_noisy, seed=self.seed)
        proba = self.predict_proba(X_test)
        y_pred = np.argmax(proba, axis=1).astype(np.int64)
        retained = self.get_retained_mask()
        return BaselineResult(
            method=self.method,
            method_family=self.method_family,
            implementation_status=self.implementation_status,
            y_pred=y_pred,
            proba=_ensure_width(proba, num_classes),
            weights=retained.astype(np.float64),
            retained_mask=retained,
            details=self.get_metadata(),
        )

    def _check_fitted(self) -> None:
        if self.scaler_ is None or self.clf_a_ is None or self.clf_b_ is None or self.num_classes_ is None:
            raise RuntimeError("DecouplingBaseline is not fitted.")


def _sgd(seed: int, alpha: float) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        alpha=float(alpha),
        max_iter=1,
        tol=None,
        random_state=int(seed),
        learning_rate="optimal",
        class_weight=None,
    )


def _is_fitted(model: SGDClassifier) -> bool:
    return hasattr(model, "classes_")


def _aligned_proba(model: SGDClassifier, X: np.ndarray, num_classes: int | None) -> np.ndarray:
    if num_classes is None:
        raise RuntimeError("num_classes_ is not set.")
    raw = model.predict_proba(X)
    out = np.zeros((X.shape[0], num_classes), dtype=np.float64)
    for col_idx, label in enumerate(getattr(model, "classes_", np.arange(raw.shape[1]))):
        label_idx = int(label)
        if 0 <= label_idx < num_classes:
            out[:, label_idx] = raw[:, col_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 1e-12
    if np.any(zero):
        out[zero, :] = 1.0 / max(num_classes, 1)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row_sum, 1e-12)


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
