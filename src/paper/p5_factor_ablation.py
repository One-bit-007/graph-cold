"""P5 / P0-3 factor ablation (advisor-scoped scope, WP-2).

Scope frozen by the 2026-08-27 advisor decision (Q2) and plan v3:

- Full tier (inferential): cells #3 graph+CDM+hard, #4 graph+CDM+soft,
  #5 graph+CDM+freq-floor, #6 no-graph class-balanced CE.
  10 seeds x 3 rates x 3 datasets under each dataset's frozen primary
  protocol (masked + group-disjoint, same PROTOCOLS as WP-1).
- Screening tier (descriptive only): #1 no-graph CE, #2 graph without CDM,
  #7 CDM single-term variants (Dneigh-only / Devid-only / full).
  3 seeds x rate 0.6, no inference.

Frozen statistical families (F-6):
- Family A (9 tests): {3 datasets} x Tail-F1 x (#4 vs #3) x {3 rates}, Holm.
  Confirms F-E (soft retention over hard deletion). robust >= 2 datasets PASS.
  Macro-F1 non-inferiority (TOST delta=0.01) is reported alongside.
- Family C (18 tests): {3 datasets} x Tail-F1 x (#4 vs #5, #4 vs #6) x
  {3 rates}, Holm. The "evidence-rescue beats plain class reweighting" claim
  stands only if #4 beats BOTH #5 and #6; otherwise the claim is deleted.

Pipeline note (documented degeneracy): in the frozen final-protocol pipeline
the graph enters training only through the Graph-CDM diagnostic (the shared
classifier is ExtraTrees; there is no graph encoder in the training loop).
Cells #1 and #2 are therefore identical by construction; the guard test
asserts bit-identical predictions so this collapse is explicit, not silent.

Cell #5 frozen definition (for Section 4.4 disclosure):
    freq(v) = min(1, median_class_count / count(observed noisy label of v))
    w_5(v)  = max(tau_ret, freq(v))
i.e. a non-zero minimum-weight floor driven purely by observed class
frequency - the simplest possible "rescue by reweighting" baseline.

Cell #6 frozen definition: class-balanced CE with effective-number weights
(Cui et al., CVPR 2019): w_c proportional to (1 - beta) / (1 - beta^{n_c}),
beta = 0.9999, L1-normalized to mean 1 over the training sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from src.experiments import d5
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence
from src.paper import p2d_clean_rerun as p2d
from src.paper import p2e_salvage as p2e
from src.paper import p2f_tighten as p2f
from src.paper import p3_proxy_audit as p3
from src.paper import p4_cl_graph_combo as p4


DATASETS = p3.DATASETS
FULL_SEEDS = tuple(range(10))
SCREEN_SEEDS = (0, 1, 2)
FULL_RATES = (0.4, 0.6, 0.8)
SCREEN_RATES = (0.6,)
TRAIN_WINDOW = 10_000
TEST_WINDOW = 5_000
MACRO_NI_DELTA = 0.01
TAU_RET = float(p2e.RETENTION_THRESHOLD)  # 0.1
CBCE_BETA = 0.9999

CELL_FULL = ("3_graph_cdm_hard", "4_graph_cdm_soft", "5_graph_cdm_freqfloor", "6_nograph_cbce")
CELL_SCREEN = ("1_nograph_ce", "2_graph_no_cdm", "7a_cdm_neigh_only", "7b_cdm_evid_only", "7c_cdm_full")
ALL_CELLS = CELL_FULL + CELL_SCREEN

FAMILY_A_SIZE = len(DATASETS) * len(FULL_RATES)          # 9
FAMILY_C_SIZE = len(DATASETS) * 2 * len(FULL_RATES)      # 18


# ---------------------------------------------------------------------------
# Scenario preparation (shared across all cells of the same scenario)
# ---------------------------------------------------------------------------

def _prepare_scenario(ds, dataset_name: str, seed: int, rate: float) -> dict[str, Any]:
    """One (dataset, seed, rate) scenario. Every ablation cell of this
    scenario consumes exactly these arrays - the guard test asserts that
    repeated calls are bit-identical so inter-cell differences can only come
    from the factor being varied."""
    y_train_all = np.asarray(ds.y_train)
    train_idx = p2d._stratified_sample(y_train_all, min(TRAIN_WINDOW, len(y_train_all)), seed)
    test_idx = p2d._stratified_sample(ds.y_test, min(TEST_WINDOW, len(ds.y_test)), seed + 1000)
    X_train = np.asarray(ds.X_train)[train_idx]
    y_train = y_train_all[train_idx]
    X_test = np.asarray(ds.X_test)[test_idx]
    y_test = np.asarray(ds.y_test)[test_idx]

    graph = p2d._feature_graph(X_train, active_views=list(ds.meta.get("active_views", ["ip"])))
    noise_target, _ = p3._noise_target(dataset_name, y_train, ds.meta)
    tail = p3._tail_labels(y_train)
    noisy, flip = p2e._inject_tail_asymmetric(y_train, tail, noise_target, rate, seed=seed)
    evidence = compute_evidence(noisy, {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}}, anomaly=d5._unsupervised_feature_anomaly(X_train))
    cdm = d5._cdm_from_observed_labels(noisy, evidence, graph, ds.num_classes)
    return {
        "seed": int(seed), "rate": float(rate),
        "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
        "train_idx": train_idx, "test_idx": test_idx,
        "num_classes": ds.num_classes, "noise_target": int(noise_target),
        "tail": tail, "noisy": noisy, "flip": flip,
        "evidence": evidence, "cdm": cdm, "graph": graph,
        "class_names": list(ds.meta.get("class_names", [])),
    }


# ---------------------------------------------------------------------------
# Cell weighting rules
# ---------------------------------------------------------------------------

def _freq_floor_weights(noisy: np.ndarray) -> np.ndarray:
    """Cell #5: w(v) = max(tau_ret, min(1, median_count / count(y_observed)))."""
    labels, counts = np.unique(noisy, return_counts=True)
    median_count = float(np.median(counts))
    count_of = {int(label): float(count) for label, count in zip(labels, counts)}
    freq = np.array([min(1.0, median_count / max(1.0, count_of[int(label)])) for label in noisy])
    return np.maximum(TAU_RET, freq).astype(np.float64)


def _cbce_weights(y: np.ndarray, beta: float = CBCE_BETA) -> np.ndarray:
    """Cell #6: effective-number class-balanced weights (Cui et al. 2019)."""
    labels, counts = np.unique(y, return_counts=True)
    eff = {int(label): (1.0 - beta) / (1.0 - beta ** float(count)) for label, count in zip(labels, counts)}
    w = np.array([eff[int(label)] for label in y], dtype=np.float64)
    return w / max(w.mean(), 1e-12)


def _cell_weights(cell: str, scenario: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (weights, train_labels, fit_mode). fit_mode in {mask, weight, plain}."""
    cdm = scenario["cdm"]
    evidence = scenario["evidence"]
    noisy = scenario["noisy"]
    if cell == "3_graph_cdm_hard":
        return d5._weights_for_hard(cdm, evidence), noisy.copy(), "mask"
    if cell == "4_graph_cdm_soft":
        return graph_cdm.soft_weights(cdm, evidence, p2e.SOFT_CFG), noisy.copy(), "weight_classbalanced"
    if cell == "5_graph_cdm_freqfloor":
        return _freq_floor_weights(noisy), noisy.copy(), "weight_plain"
    if cell == "6_nograph_cbce":
        return _cbce_weights(noisy), noisy.copy(), "weight_plain"
    if cell in ("1_nograph_ce", "2_graph_no_cdm"):
        # #2 builds the same graph but never uses it: identical by construction.
        return np.ones(noisy.shape[0], dtype=np.float64), noisy.copy(), "plain"
    if cell.startswith("7"):
        variant = {"7a_cdm_neigh_only": "neigh", "7b_cdm_evid_only": "evid", "7c_cdm_full": "full"}[cell]
        cdm_variant = _cdm_variant(variant, scenario)
        return graph_cdm.soft_weights(cdm_variant, evidence, p2e.SOFT_CFG), noisy.copy(), "weight_classbalanced"
    raise ValueError(f"Unknown cell: {cell}")


def _cdm_variant(variant: str, scenario: dict[str, Any]) -> np.ndarray:
    """Screening-tier single-term CDM variants (#7)."""
    if variant == "full":
        return scenario["cdm"]
    noisy = scenario["noisy"]
    evidence = scenario["evidence"]
    graph = scenario["graph"]
    num_classes = scenario["num_classes"]
    probs = np.full((noisy.shape[0], int(num_classes)), 1e-6, dtype=np.float64)
    valid = (noisy >= 0) & (noisy < int(num_classes))
    probs[np.arange(noisy.shape[0])[valid], noisy[valid]] = 1.0
    probs = probs / probs.sum(axis=1, keepdims=True)
    neigh = graph_cdm._neighbor_soft_mean(probs, graph)
    d_neigh = graph_cdm._d_neigh(probs, neigh)
    if variant == "neigh":
        return np.clip(d_neigh, 0.0, 1.0)
    return np.clip(d5._minmax(evidence), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def _fit_predict_cell(X_train, train_labels, X_test, weights, fit_mode: str, seed: int):
    model = ExtraTreesClassifier(n_estimators=24, random_state=seed, n_jobs=-1)
    if fit_mode == "mask":
        keep = weights >= 0.5
        model.set_params(class_weight="balanced")
        if keep.sum() >= 2 and np.unique(train_labels[keep]).size >= 2:
            model.fit(X_train[keep], train_labels[keep])
        else:
            model.fit(X_train, train_labels)
    elif fit_mode == "weight_classbalanced":
        # Matches the frozen p2e soft path: retention weight x class balance.
        retained = np.where(weights >= TAU_RET, weights, 0.0)
        sample_weight = np.clip(retained, 0.0, 1.0) * p2d._class_balance_weights(train_labels)
        if float(np.sum(sample_weight)) <= 1e-12:
            sample_weight = p2d._class_balance_weights(train_labels)
        model.fit(X_train, train_labels, sample_weight=sample_weight)
    elif fit_mode == "weight_plain":
        model.fit(X_train, train_labels, sample_weight=np.asarray(weights, dtype=np.float64))
    else:  # plain
        model.fit(X_train, train_labels)
    return model.predict(X_test)


def _cell_row(dataset: str, scenario: dict[str, Any], cell: str) -> dict[str, Any]:
    weights, train_labels, fit_mode = _cell_weights(cell, scenario)
    y_pred = _fit_predict_cell(scenario["X_train"], train_labels, scenario["X_test"], weights, fit_mode, scenario["seed"] + 7)
    y_test = scenario["y_test"]
    seen_mask = np.isin(y_test, np.unique(scenario["y_train"]))
    return {
        "dataset": dataset,
        "cell": cell,
        "seed": scenario["seed"],
        "tail_flip_rate": scenario["rate"],
        "macro_f1": p3._macro_f1(y_test, y_pred),
        "macro_f1_seen_classes": p3._macro_f1(y_test[seen_mask], y_pred[seen_mask]) if seen_mask.any() else float("nan"),
        "tail_macro_f1": p3._tail_f1(y_test, y_pred, scenario["tail"]),
        "unseen_test_class_fraction": float(1.0 - seen_mask.mean()),
        "weight_min": float(np.min(weights)),
        "weight_mean": float(np.mean(weights)),
    }


# ---------------------------------------------------------------------------
# Statistics: families A and C
# ---------------------------------------------------------------------------

def _as_method_frame(part: pd.DataFrame) -> pd.DataFrame:
    """p2f._paired_stat / p4._noninferiority expect a `method` column."""
    return part.rename(columns={"cell": "method"})


def _family_tests(frame: pd.DataFrame, comparisons: list[tuple[str, str]], size: int) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    for dataset in sorted(frame.dataset.unique()):
        part = frame[frame.dataset == dataset]
        for rate in sorted(part.tail_flip_rate.unique()):
            rate_part = _as_method_frame(part[np.isclose(part.tail_flip_rate, rate)])
            for method, reference in comparisons:
                item = p2f._paired_stat(rate_part, method, reference, "tail_macro_f1")
                item.update({"dataset": dataset, "tail_flip_rate": float(rate), "method": method, "reference": reference})
                tests.append(item)
    if len(tests) > size:
        raise AssertionError(f"family overflow: {len(tests)} > {size}")
    padded = [float(t["p_value_greater"]) for t in tests] + [1.0] * (size - len(tests))
    adjusted = p2f._holm(padded)[: len(tests)]
    for item, p_holm in zip(tests, adjusted):
        item["p_holm_greater"] = float(p_holm)
    return tests


def _family_verdict(frame: pd.DataFrame, tests: list[dict[str, Any]], method: str, reference: str) -> dict[str, Any]:
    per_dataset: dict[str, Any] = {}
    for dataset in sorted(frame.dataset.unique()):
        part = frame[frame.dataset == dataset]
        rate_rows = []
        for rate in sorted(part.tail_flip_rate.unique()):
            rate_part = part[np.isclose(part.tail_flip_rate, rate)]
            test = next(
                t for t in tests
                if t["dataset"] == dataset and np.isclose(t["tail_flip_rate"], rate)
                and t["method"] == method and t["reference"] == reference
            )
            ni = p4._noninferiority(_as_method_frame(rate_part), method, reference, MACRO_NI_DELTA)
            passed = bool(
                test["mean_delta"] > 0 and test["p_holm_greater"] < 0.05
                and test["cohens_dz"] >= 0.3 and test["bootstrap_ci95_low"] > 0
            )
            rate_rows.append({
                "tail_flip_rate": float(rate), "test": test,
                "macro_noninferiority": ni, "passed": passed,
            })
        per_dataset[dataset] = {"rates": rate_rows, "passed": any(r["passed"] for r in rate_rows)}
    count = sum(bool(v["passed"]) for v in per_dataset.values())
    verdict = "robust" if count >= 2 else "narrow" if count == 1 else "null"
    return {"per_dataset": per_dataset, "pass_datasets": int(count), "verdict": verdict}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    tier: str,
    configs_dir: str | Path = "configs",
    out_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
    tables_dir: str | Path = "tables",
    datasets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    seeds = SCREEN_SEEDS if tier == "screening" else FULL_SEEDS
    rates = SCREEN_RATES if tier == "screening" else FULL_RATES
    cells = CELL_SCREEN if tier == "screening" else CELL_FULL
    selected = tuple(DATASETS if datasets is None else datasets)
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    configs = Path(configs_dir)
    out = Path(out_dir)
    reports = Path(reports_dir)
    tables = Path(tables_dir)
    for directory in (out, reports, tables):
        directory.mkdir(parents=True, exist_ok=True)

    suffix = tier if selected == tuple(DATASETS) else f"{tier}_{'_'.join(selected)}"
    partial = out / f"p5_factor_ablation_{suffix}.partial.csv"
    rows: list[dict[str, Any]] = []
    done: set[tuple[str, int, float]] = set()
    if partial.exists():
        existing = pd.read_csv(partial)
        rows = existing.to_dict(orient="records")
        counts = existing.groupby(["dataset", "seed", "tail_flip_rate"])["cell"].nunique()
        done = {(str(k[0]), int(k[1]), float(k[2])) for k, count in counts.items() if int(count) == len(cells)}

    for dataset in selected:
        ds = p4._load_primary_dataset(dataset, configs)
        for seed in seeds:
            for rate in rates:
                if (dataset, int(seed), float(rate)) in done:
                    continue
                print(f"[p5] {tier} {dataset} seed={seed} rate={rate}", flush=True)
                scenario = _prepare_scenario(ds, dataset, seed, rate)
                for cell in cells:
                    rows.append(_cell_row(dataset, scenario, cell))
                pd.DataFrame(rows).to_csv(partial, index=False)

    frame = pd.DataFrame(rows)
    final_csv = out / f"p5_factor_ablation_{suffix}.csv"
    frame.to_csv(final_csv, index=False)
    expected = len(selected) * len(seeds) * len(rates) * len(cells)

    report: dict[str, Any] = {
        "stage": "P5-P0-3-factor-ablation",
        "tier": tier,
        "completed": int(len(frame)) == expected,
        "row_count": int(len(frame)),
        "expected_row_count": int(expected),
        "seeds": list(seeds),
        "rates": list(rates),
        "cells": list(cells),
        "datasets": list(selected),
        "protocols": p4.PROTOCOLS,
        "outputs": {"results": str(final_csv).replace("\\", "/"), "partial": str(partial).replace("\\", "/")},
    }

    if tier == "full":
        family_a_tests = _family_tests(frame, [("4_graph_cdm_soft", "3_graph_cdm_hard")], FAMILY_A_SIZE)
        family_c_tests = _family_tests(
            frame,
            [("4_graph_cdm_soft", "5_graph_cdm_freqfloor"), ("4_graph_cdm_soft", "6_nograph_cbce")],
            FAMILY_C_SIZE,
        )
        family_a = _family_verdict(frame, family_a_tests, "4_graph_cdm_soft", "3_graph_cdm_hard")
        report["family_a"] = {"tests": family_a_tests, **family_a}
        report["family_c"] = {
            "tests": family_c_tests,
            "vs_freqfloor": _family_verdict(frame, family_c_tests, "4_graph_cdm_soft", "5_graph_cdm_freqfloor"),
            "vs_cbce": _family_verdict(frame, family_c_tests, "4_graph_cdm_soft", "6_nograph_cbce"),
        }
        pd.DataFrame(_flatten_tests(family_a_tests)).to_csv(tables / "table_p5_family_a.csv", index=False)
        pd.DataFrame(_flatten_tests(family_c_tests)).to_csv(tables / "table_p5_family_c.csv", index=False)
    else:
        report["descriptive_means"] = (
            frame.groupby(["dataset", "cell"])[["macro_f1", "tail_macro_f1"]].mean().reset_index().to_dict(orient="records")
        )
        frame.groupby(["dataset", "cell"])[["macro_f1", "tail_macro_f1"]].mean().reset_index().to_csv(
            tables / "table_p5_screening_descriptive.csv", index=False
        )

    (reports / f"p5_factor_ablation_{suffix}.json").write_text(
        json.dumps(_jsonable(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def _flatten_tests(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("dataset", "tail_flip_rate", "method", "reference", "mean_delta", "p_value_greater", "p_holm_greater", "cohens_dz", "bootstrap_ci95_low", "bootstrap_ci95_high")
    return [{k: t.get(k) for k in keys} for t in tests]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="P5/P0-3 factor ablation (WP-2)")
    parser.add_argument("--tier", choices=("screening", "full"), required=True)
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--datasets", default=None, help="Comma-separated; default: all")
    args = parser.parse_args()
    selected = None if args.datasets is None else tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    report = run(args.tier, args.configs, args.out, args.reports, args.tables, selected)
    out: dict[str, Any] = {"completed": report["completed"], "row_count": report["row_count"]}
    if args.tier == "full":
        out["family_a_verdict"] = report["family_a"]["verdict"]
        out["family_c_vs_freqfloor"] = report["family_c"]["vs_freqfloor"]["verdict"]
        out["family_c_vs_cbce"] = report["family_c"]["vs_cbce"]["verdict"]
    print(json.dumps(out, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
