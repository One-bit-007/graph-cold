"""Noisy-supervised real baseline.

The model is trained on the injected noisy labels and evaluated on the clean
test split by the experiment runner. It performs no denoising and therefore
retains every training sample for ERR accounting.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from src.baselines.base import BaselineResult, aligned_proba, array_hash


class NoisySupervisedBaseline:
    method = "Noisy-Supervised"
    method_family = "noisy_supervised"
    implementation_status = "implemented_smoke_passed"

    def __init__(self, seed: int = 0, n_estimators: int = 8):
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)

    def fit_predict(self, X_train, y_noisy, X_test, num_classes: int, **kwargs) -> BaselineResult:
        y_noisy = np.asarray(y_noisy, dtype=np.int64)
        model = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train, y_noisy)
        y_pred = model.predict(X_test).astype(np.int64)
        proba = aligned_proba(model, X_test, num_classes)
        retained = np.ones(y_noisy.shape[0], dtype=bool)
        return BaselineResult(
            method=self.method,
            method_family=self.method_family,
            implementation_status=self.implementation_status,
            y_pred=y_pred,
            proba=proba,
            weights=np.ones(y_noisy.shape[0], dtype=np.float64),
            retained_mask=retained,
            details={
                "classifier": "ExtraTreesClassifier",
                "n_estimators": self.n_estimators,
                "trained_on": "noisy_y_train",
                "training_label_hash": array_hash(y_noisy),
                "filtering": "none",
            },
        )
