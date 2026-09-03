"""Paper-Algorithm-1 FINE detector with an explicitly static tabular encoder.

Not an end-to-end reproduction of the original deep FINE training pipeline.
See reports/p9_fine_fidelity.md for pre-evaluation settings and adaptations.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.baselines.base import BaselineResult, aligned_proba, array_hash

METHOD = "FINE detector (tabular adaptation)"
FAITHFULNESS = ("Gram-eigenvector + per-class GMM selection as in Kim et al.; "
                "representation is a tabular adaptation, not the original deep features")


def squared_gram_alignment(Z):
    """Uncentered within-class Gram leading eigenvector and squared scores."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or not len(Z) or not np.isfinite(Z).all():
        raise ValueError("Expected finite, nonempty 2D class representation")
    _, vectors = np.linalg.eigh(Z.T @ Z)
    vector = vectors[:, -1]
    return np.square(Z @ vector), vector


class FINETabularBaseline:
    method = METHOD
    method_family = "fine_tabular"
    implementation_status = "verified_tabular_adaptation"
    faithfulness_level = FAITHFULNESS
    CONFIG = {"representation_components": 32, "posterior_threshold": 0.5,
              "min_class_size": 4, "score_range_epsilon": 1e-12,
              "gmm_components": 2, "gmm_covariance_type": "full", "gmm_tol": 1e-6,
              "gmm_max_iter": 100, "gmm_reg_covar": 1e-6, "gmm_n_init": 1,
              "n_estimators": 24, "classifier_seed_offset": 7, "n_jobs": 1}

    def __init__(self, seed=0):
        self.seed = int(seed)

    def select(self, Z, noisy_y_train):
        """Selection accepts only representations and observed training labels."""
        y = np.asarray(noisy_y_train, dtype=np.int64)
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2 or y.ndim != 1 or len(Z) != len(y) or not np.isfinite(Z).all():
            raise ValueError("Invalid training representation or labels")
        keep = np.zeros(len(y), dtype=bool)
        scores = np.zeros(len(y), dtype=float)
        details = {}
        for label in np.unique(y):
            idx = np.flatnonzero(y == label)
            score, _ = squared_gram_alignment(Z[idx])
            scores[idx] = score
            reason = None
            record = {"n": int(len(idx)), "score_min": float(score.min()),
                      "score_max": float(score.max()), "warnings": []}
            selected = np.ones(len(idx), dtype=bool)
            if len(idx) < self.CONFIG["min_class_size"]:
                reason = "small_class_fewer_than_4"
            elif np.ptp(score) <= self.CONFIG["score_range_epsilon"]:
                reason = "degenerate_alignment_scores"
            else:
                mixture = GaussianMixture(n_components=2, covariance_type="full", tol=1e-6,
                                          max_iter=100, reg_covar=1e-6, n_init=1,
                                          init_params="kmeans", random_state=self.seed)
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        mixture.fit(score[:, None])
                    record["warnings"] = [str(w.message) for w in caught]
                    record["gmm_converged"] = bool(mixture.converged_)
                    record["gmm_means"] = mixture.means_.ravel().tolist()
                    record["gmm_n_iter"] = int(mixture.n_iter_)
                    if not mixture.converged_:
                        reason = "gmm_not_converged"
                    elif np.ptp(mixture.means_) <= self.CONFIG["score_range_epsilon"]:
                        reason = "indistinguishable_gmm_means"
                    else:
                        posterior = mixture.predict_proba(score[:, None])[:, np.argmax(mixture.means_)]
                        if not np.isfinite(posterior).all():
                            reason = "nonfinite_gmm_posterior"
                        else:
                            selected = posterior > self.CONFIG["posterior_threshold"]
                            if not selected.any():
                                reason = "empty_gmm_selection"
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    reason = "gmm_numeric_failure"
                    record["error"] = f"{type(exc).__name__}: {exc}"
            if reason is not None:
                selected[:] = True
            keep[idx] = selected
            record.update({"n_retained": int(selected.sum()), "fallback_reason": reason,
                           "fallback_action": "retain_all_in_class" if reason else None})
            details[str(int(label))] = record
        return keep, scores, details

    def fit(self, X_train, noisy_y_train):
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(noisy_y_train, dtype=np.int64)
        if X.ndim != 2 or not X.shape[1] or y.ndim != 1 or len(X) != len(y):
            raise ValueError("Expected aligned 2D features and 1D labels")
        if not np.isfinite(X).all() or np.unique(y).size < 2 or np.any(y < 0):
            raise ValueError("Expected finite features and at least two nonnegative classes")
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(X)
        width = min(self.CONFIG["representation_components"], X.shape[1], len(X)-1)
        self.projector_ = None
        if X.shape[1] > 1:
            self.projector_ = TruncatedSVD(n_components=width, random_state=self.seed)
            Z = self.projector_.fit_transform(scaled)
        else:
            Z = scaled
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = Z / np.maximum(norms, 1e-12)
        self.retained_mask_, self.scores_, per_class = self.select(Z, y)
        self.classifier_ = ExtraTreesClassifier(n_estimators=24, random_state=self.seed+7,
                                               class_weight="balanced", n_jobs=1)
        self.classifier_.fit(X[self.retained_mask_], y[self.retained_mask_])
        self.metadata_ = {
            "faithfulness_level": FAITHFULNESS, "config": dict(self.CONFIG),
            "representation": "train-only StandardScaler + TruncatedSVD + row L2",
            "representation_width": int(Z.shape[1]), "training_label_hash": array_hash(y),
            "train_label_source": "noisy_y_train", "eval_label_source": "clean_y_test_only",
            "noise_rate_source": "not_used", "selection_rule": "uncentered_gram_squared_alignment_per_class_gmm",
            "retained_fraction": float(self.retained_mask_.mean()), "class_selection": per_class,
            "fallback_class_count": sum(v["fallback_reason"] is not None for v in per_class.values()),
            "fallback_sample_count": sum(v["n"] for v in per_class.values() if v["fallback_reason"]),
            "retained_mask_sha256": array_hash(self.retained_mask_),
        }
        return self

    def predict(self, X_test):
        return self.classifier_.predict(np.asarray(X_test, dtype=np.float64)).astype(np.int64)

    def fit_predict(self, X_train, noisy_y_train, X_test, num_classes):
        self.fit(X_train, noisy_y_train)
        return BaselineResult(method=METHOD, method_family=self.method_family,
                              implementation_status=self.implementation_status,
                              y_pred=self.predict(X_test),
                              proba=aligned_proba(self.classifier_, np.asarray(X_test, dtype=np.float64), num_classes),
                              weights=self.retained_mask_.astype(float),
                              retained_mask=self.retained_mask_.copy(), details=self.metadata_)
