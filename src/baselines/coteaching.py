"""Co-Teaching baseline for tabular noisy-label IDS experiments.

The implementation follows the Co-Teaching small-loss exchange protocol with
two independently initialized classifiers. It uses deterministic tabular SGD
classifiers because the project evaluates fixed tabular flow features.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from src.baselines.base import BaselineResult, array_hash, class_balance_weights, ensure_class_coverage


class CoTeachingBaseline:
    method = "Co-Teaching"
    method_family = "co_teaching"
    implementation_status = "verified_implementation"

    def __init__(
        self,
        seed: int = 0,
        noise_rate: float = 0.0,
        epochs: int = 6,
        batch_size: int = 4096,
        n_estimators: int = 32,
    ):
        self.seed = int(seed)
        self.noise_rate = float(noise_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.n_estimators = int(n_estimators)

    def fit_predict(self, X_train, y_noisy, X_test, num_classes: int, **kwargs) -> BaselineResult:
        del kwargs
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_noisy = np.asarray(y_noisy, dtype=np.int64)
        classes = np.arange(num_classes, dtype=np.int64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
        clf_a = _sgd(self.seed)
        clf_b = _sgd(self.seed + 997)
        mask_a = np.ones(y_noisy.shape[0], dtype=bool)
        mask_b = np.ones(y_noisy.shape[0], dtype=bool)
        confidence = np.ones(y_noisy.shape[0], dtype=np.float64)
        balance = class_balance_weights(y_noisy)
        rng = np.random.default_rng(self.seed)
        exchanged_counts: list[int] = []

        for epoch in range(max(self.epochs, 1)):
            clf_a.partial_fit(X_scaled[mask_b], y_noisy[mask_b], classes=classes, sample_weight=balance[mask_b])
            clf_b.partial_fit(X_scaled[mask_a], y_noisy[mask_a], classes=classes, sample_weight=balance[mask_a])
            proba_a = _aligned_proba(clf_a, X_scaled, num_classes)
            proba_b = _aligned_proba(clf_b, X_scaled, num_classes)
            loss_a = -np.log(np.maximum(proba_a[np.arange(y_noisy.shape[0]), y_noisy], 1e-12))
            loss_b = -np.log(np.maximum(proba_b[np.arange(y_noisy.shape[0]), y_noisy], 1e-12))
            confidence = 0.5 * (
                proba_a[np.arange(y_noisy.shape[0]), y_noisy]
                + proba_b[np.arange(y_noisy.shape[0]), y_noisy]
            )
            remember_rate = self._remember_rate(epoch)
            mask_a = np.zeros(y_noisy.shape[0], dtype=bool)
            mask_b = np.zeros(y_noisy.shape[0], dtype=bool)
            order = rng.permutation(y_noisy.shape[0])
            for start in range(0, order.shape[0], max(1, self.batch_size)):
                idx = order[start : start + max(1, self.batch_size)]
                if idx.size == 0:
                    continue
                keep_n = max(num_classes, int(np.ceil(remember_rate * idx.size)))
                keep_n = min(keep_n, idx.size)
                mask_a[idx] = _small_loss_mask(loss_b[idx], keep_n)
                mask_b[idx] = _small_loss_mask(loss_a[idx], keep_n)
            mask_a = ensure_class_coverage(mask_a, confidence, y_noisy)
            mask_b = ensure_class_coverage(mask_b, confidence, y_noisy)
            exchanged_counts.append(int((mask_a & mask_b).sum()))

        retained = ensure_class_coverage(mask_a & mask_b, confidence, y_noisy)
        final_a = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed + 20_003,
            class_weight="balanced",
            n_jobs=-1,
        )
        final_b = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed + 30_007,
            class_weight="balanced",
            n_jobs=-1,
        )
        train_a = ensure_class_coverage(mask_b, confidence, y_noisy)
        train_b = ensure_class_coverage(mask_a, confidence, y_noisy)
        final_a.fit(X_scaled[train_a], y_noisy[train_a])
        final_b.fit(X_scaled[train_b], y_noisy[train_b])
        test_proba = 0.5 * (
            _aligned_tree_proba(final_a, X_test_scaled, num_classes)
            + _aligned_tree_proba(final_b, X_test_scaled, num_classes)
        )
        y_pred = np.argmax(test_proba, axis=1).astype(np.int64)
        return BaselineResult(
            method=self.method,
            method_family=self.method_family,
            implementation_status=self.implementation_status,
            y_pred=y_pred,
            proba=test_proba,
            weights=retained.astype(np.float64),
            retained_mask=retained,
            details={
                "classifier": "two SGDClassifier(log_loss) selectors + two ExtraTreesClassifier heads",
                "n_estimators": self.n_estimators,
                "trained_on": "noisy_y_train",
                "train_label_source": "noisy_y_train",
                "eval_label_source": "clean_y_test",
                "training_label_hash": array_hash(y_noisy),
                "small_loss_exchange": True,
                "mini_batch_exchange": True,
                "forget_rate": float(self._target_forget_rate()),
                "retained_fraction": float(np.mean(retained)),
                "epochs": max(self.epochs, 1),
                "batch_size": max(1, self.batch_size),
                "mean_exchanged_fraction": float(np.mean(exchanged_counts) / max(y_noisy.shape[0], 1))
                if exchanged_counts
                else 0.0,
            },
        )

    def _target_forget_rate(self) -> float:
        return float(np.clip(self.noise_rate, 0.0, 0.55))

    def _remember_rate(self, epoch: int) -> float:
        if self.noise_rate <= 0:
            return 1.0
        progress = min(1.0, float(epoch + 1) / float(max(self.epochs, 1)))
        return float(np.clip(1.0 - self._target_forget_rate() * progress, 0.35, 1.0))


def _sgd(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=1,
        tol=None,
        random_state=int(seed),
        learning_rate="optimal",
        class_weight=None,
    )


def _small_loss_mask(loss: np.ndarray, keep_n: int) -> np.ndarray:
    if keep_n >= loss.shape[0]:
        return np.ones(loss.shape[0], dtype=bool)
    selected = np.argpartition(loss, keep_n - 1)[:keep_n]
    mask = np.zeros(loss.shape[0], dtype=bool)
    mask[selected] = True
    return mask


def _aligned_proba(model: SGDClassifier, X, num_classes: int) -> np.ndarray:
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


def _aligned_tree_proba(model: ExtraTreesClassifier, X, num_classes: int) -> np.ndarray:
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


CoTeachingLiteBaseline = CoTeachingBaseline
