"""P6b / F-1.2 scale validation: CICIDS at 100k-scale window.

Frozen definition (Phase-0 F-1.2): one dataset (CICIDS-2017), train 100k /
test 20k, key configs {CoLD, ablation_hard, Graph-CoLD-soft, CL,
CL+Graph-CDM}, 3 seeds x asymmetric rates {0.4, 0.6, 0.8}. Criterion: the
Tail-F1 direction (soft > hard; CL+Graph vs CL) does not flip with scale.

Runs under the frozen CICIDS primary protocol (tier1 mask + temporal split +
train dedup) via p4's scenario machinery with overridden windows. Descriptive
+ paired deltas with bootstrap CI; no new inferential family (F-6).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments import d5
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence
from src.paper import p2e_salvage as p2e
from src.paper import p2f_tighten as p2f
from src.paper import p3_proxy_audit as p3
from src.paper import p4_cl_graph_combo as p4
from src.baselines.confident_learning import ConfidentLearningBaseline
from src.metrics import macro_f1


DATASET = "cicids2017"
SEEDS = (0, 1, 2)
RATES = (0.4, 0.6, 0.8)
TRAIN_WINDOW = 100_000
TEST_WINDOW = 20_000
METHODS = ("CoLD", "ablation_hard", "Graph-CoLD-soft", "Confident-Learning", "CL+Graph-soft")


def _scenario_rows(ds, seed: int, rate: float) -> list[dict[str, Any]]:
    # Reuse p4._sample but with 100k/20k windows.
    old_train, old_test = p4.TRAIN_WINDOW, p4.TEST_WINDOW
    p4.TRAIN_WINDOW, p4.TEST_WINDOW = TRAIN_WINDOW, TEST_WINDOW
    try:
        sample = p4._sample(ds, DATASET, seed)
    finally:
        p4.TRAIN_WINDOW, p4.TEST_WINDOW = old_train, old_test

    noisy, flip = p2e._inject_tail_asymmetric(sample.y_train, sample.tail, sample.noise_target, rate, seed=seed)
    evidence = compute_evidence(
        noisy, {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}}, anomaly=sample.anomaly
    )
    cdm = d5._cdm_from_observed_labels(noisy, evidence, sample.graph, sample.num_classes)
    soft = graph_cdm.soft_weights(cdm, evidence, p2e.SOFT_CFG)
    hard = d5._weights_for_hard(cdm, evidence)
    cold = d5._weights_for_cold(cdm, evidence)

    cl = ConfidentLearningBaseline(seed=seed, noise_rate=float(np.mean(flip)), n_estimators=24)
    cl_result = cl.fit_predict(sample.X_train, noisy, sample.X_test, sample.num_classes)
    cl_keep = np.asarray(cl_result.retained_mask, dtype=bool)

    predictions = {
        "CoLD": p4._predict_weighted(sample, noisy, cold, seed, hard=True),
        "ablation_hard": p4._predict_weighted(sample, noisy, hard, seed, hard=True),
        "Graph-CoLD-soft": p4._predict_weighted(sample, noisy, soft, seed, hard=False),
        "Confident-Learning": np.asarray(cl_result.y_pred, dtype=np.int64),
        "CL+Graph-soft": p4._predict_weighted(sample, noisy, cl_keep.astype(float) * soft, seed, hard=False),
    }
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        pred = predictions[method]
        rows.append({
            "dataset": DATASET,
            "seed": int(seed),
            "tail_flip_rate": float(rate),
            "method": method,
            "macro_f1": macro_f1(sample.y_test, pred),
            "tail_macro_f1": p2e._tail_macro_f1(sample.y_test, pred, sample.tail),
            "n_train": int(sample.X_train.shape[0]),
            "n_test": int(sample.X_test.shape[0]),
        })
    return rows


def _delta(frame: pd.DataFrame, method: str, reference: str) -> dict[str, Any]:
    per_rate: list[dict[str, Any]] = []
    for rate in sorted(frame.tail_flip_rate.unique()):
        part = frame[np.isclose(frame.tail_flip_rate, rate)]
        piv = part[part.method.isin([method, reference])].pivot_table(
            index="seed", columns="method", values="tail_macro_f1"
        ).dropna()
        if piv.empty or method not in piv or reference not in piv:
            continue
        diff = (piv[method] - piv[reference]).to_numpy(dtype=float)
        lo, hi = p2f._bootstrap_ci(diff)
        per_rate.append({
            "tail_flip_rate": float(rate),
            "paired_seeds": int(len(diff)),
            "mean_delta": float(np.mean(diff)),
            "ref_mean": float(piv[reference].mean()),
            "method_mean": float(piv[method].mean()),
            "bootstrap_ci95_low": float(lo),
            "bootstrap_ci95_high": float(hi),
            "direction_positive": bool(np.mean(diff) > 0),
        })
    return {"method": method, "reference": reference, "per_rate": per_rate,
            "direction_consistent": bool(per_rate) and all(r["direction_positive"] for r in per_rate)}


def run(
    configs_dir: str | Path = "configs",
    out_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    out, reports = Path(out_dir), Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    partial = out / "p6_scale_validation_cicids.partial.csv"
    rows: list[dict[str, Any]] = []
    done: set[tuple[int, float]] = set()
    if partial.exists():
        existing = pd.read_csv(partial)
        rows = existing.to_dict(orient="records")
        counts = existing.groupby(["seed", "tail_flip_rate"])["method"].nunique()
        done = {(int(k[0]), float(k[1])) for k, c in counts.items() if int(c) == len(METHODS)}

    ds = p4._load_primary_dataset(DATASET, Path(configs_dir))
    for seed in SEEDS:
        for rate in RATES:
            if (seed, rate) in done:
                continue
            t0 = time.time()
            rows.extend(_scenario_rows(ds, seed, rate))
            print(f"[p6b] scale seed={seed} rate={rate} done in {time.time()-t0:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(partial, index=False)

    frame = pd.DataFrame(rows)
    final_csv = out / "p6_scale_validation_cicids.csv"
    frame.to_csv(final_csv, index=False)

    comparisons = [
        _delta(frame, "Graph-CoLD-soft", "ablation_hard"),   # family A direction
        _delta(frame, "CL+Graph-soft", "Confident-Learning"),  # G2 direction
    ]
    report = {
        "stage": "P6b-F-1.2-scale-validation",
        "dataset": DATASET,
        "train_window": TRAIN_WINDOW,
        "test_window": TEST_WINDOW,
        "seeds": list(SEEDS),
        "rates": list(RATES),
        "methods": list(METHODS),
        "row_count": int(len(frame)),
        "expected_row_count": len(SEEDS) * len(RATES) * len(METHODS),
        "completed": int(len(frame)) == len(SEEDS) * len(RATES) * len(METHODS),
        "comparisons": comparisons,
        "verdict": "direction_holds" if all(c["direction_consistent"] for c in comparisons) else "direction_flips",
        "outputs": {"results": str(final_csv).replace("\\", "/")},
    }
    (reports / "p6_scale_validation_cicids.json").write_text(
        json.dumps(p4._jsonable(report) if hasattr(p4, "_jsonable") else report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="P6b/F-1.2 CICIDS 100k scale validation")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    report = run(args.configs, args.out, args.reports)
    print(json.dumps({"completed": report["completed"], "row_count": report["row_count"],
                      "verdict": report["verdict"]}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
