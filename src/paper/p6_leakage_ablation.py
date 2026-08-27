"""P6 / P0-6 leakage ablation rerun under the frozen final protocol.

Isolates the four leakage factors named in the repositioned contribution
(Phase-0 wording contraction, new contribution #1):

- DUP  duplicate records          : CICIDS - frozen protocol dedups train
    (d5._deduplicate_cicids_train), so DUP turns it OFF. CESNET/UNSW frozen
    protocols never dedup, so DUP turns a generic train exact-dedup ON there
    (delta direction documented per dataset).
- ENT  cross-split entity overlap : CESNET - frozen temporal_entity split
    replaced by stratified iid; UNSW - official fixed partition replaced by a
    pooled stratified re-shuffle (entity overlap itself is unmeasurable
    without entity columns; train<->test representation collision is the
    observable proxy). CICIDS - UNSUPPORTED until WP-3 raw data arrives.
- PRX  label-construction fields  : frozen proxy masking (tier per
    p4.PROTOCOLS) is lifted.
- ORA  clean-label path           : CDM replaced by the oracle score
    1 - flip_mask (the legacy D5 behaviour removed by P2c). This is the
    upper bound of label-path leakage in a synthetic-noise study.
- ALL  : every leak on at once (closest available reconstruction of the
    original leaky protocol).

Grid: 10 seeds x rate 0.6 x methods {CoLD, ablation_hard, Graph-CoLD-soft}
(same training pipeline as p3/p4/p5). Statistics are DESCRIPTIVE: paired
deltas vs FINAL with bootstrap 95% CI. Per F-6 the inferential families are
frozen at A/B/C; this audit decomposition deliberately adds no new Holm
family.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from src.data.loaders import load_dataset
from src.experiments import d5
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence
from src.paper import p2d_clean_rerun as p2d
from src.paper import p2e_salvage as p2e
from src.paper import p2f_tighten as p2f
from src.paper import p3_proxy_audit as p3
from src.paper import p4_cl_graph_combo as p4


DATASETS = p3.DATASETS
METHODS = p3.METHODS                      # CoLD, ablation_hard, Graph-CoLD-soft
SEEDS = tuple(range(10))
RATE = 0.6
TRAIN_WINDOW = p3.TRAIN_WINDOW
TEST_WINDOW = p3.TEST_WINDOW

CONDITIONS = ("FINAL", "DUP", "ENT", "PRX", "ORA", "ALL")

UNSUPPORTED = {
    ("cicids2017", "ENT"): (
        "local CICIDS-2017 copy lacks entity columns; ENT toggle requires the "
        "original GeneratedLabelledFlows (WP-3, time-boxed to 2026-09-03)"
    ),
}


# ---------------------------------------------------------------------------
# Condition construction
# ---------------------------------------------------------------------------

def _condition_cfg(dataset_name: str, condition: str, configs_dir: Path) -> dict[str, Any]:
    """Loader config for one leakage condition. Starts from the frozen
    primary protocol (p4._dataset_cfg) and flips exactly one factor."""
    cfg = p4._dataset_cfg(dataset_name, configs_dir)
    ds_cfg = cfg[dataset_name]

    unmasked = condition in ("PRX", "ALL")
    leaky_split = condition in ("ENT", "ALL")
    if unmasked:
        # Rebuild drop_cols without the proxy tiers (keep base identifier drops).
        base = p3._formal_cfg(dataset_name, configs_dir)
        ds_cfg["drop_cols"] = list(base[dataset_name].get("drop_cols", []))
    if leaky_split and dataset_name == "cesnet_tls_year22":
        ds_cfg["split_strategy"] = "stratified"
    # UNSW pooled re-shuffle and CICIDS dedup are handled post-load.
    return cfg


def _apply_train_dedup(X: np.ndarray, y: np.ndarray, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Train-side exact duplicate removal reusing the frozen d5 semantics
    (6-decimal row hash, first occurrence kept)."""
    ns = SimpleNamespace(X_train=X, y_train=y, X_test=X[:0], y_test=y[:0], num_classes=0, meta=dict(meta))
    ns = d5._deduplicate_cicids_train(ns)
    return np.asarray(ns.X_train), np.asarray(ns.y_train)


def _oracle_cdm(flip: np.ndarray) -> np.ndarray:
    """Clean-label-path oracle: the diagnostic knows exactly which labels were
    flipped (legacy pre-P2c behaviour)."""
    return 1.0 - np.asarray(flip, dtype=np.float64)


def _load_condition_dataset(dataset_name: str, condition: str, seed: int, configs_dir: Path = Path("configs")):
    """Load + post-process one (dataset, condition). Returns a namespace with
    X_train/y_train/X_test/y_test/num_classes/meta plus notes."""
    notes: dict[str, Any] = {}
    cfg = _condition_cfg(dataset_name, condition, configs_dir)
    ds = load_dataset(dataset_name, cfg)
    X_train, y_train = np.asarray(ds.X_train), np.asarray(ds.y_train)
    X_test, y_test = np.asarray(ds.X_test), np.asarray(ds.y_test)

    frozen_dedups = dataset_name == "cicids2017"
    if condition in ("DUP",):
        if frozen_dedups:
            # FINAL dedups train; DUP keeps duplicates (skip the dedup call below).
            notes["dup_toggle"] = "dedup_off_vs_frozen_on"
        else:
            X_train, y_train = _apply_train_dedup(X_train, y_train, ds.meta)
            notes["dup_toggle"] = "dedup_on_vs_frozen_off"
    elif frozen_dedups and condition != "ALL":
        X_train, y_train = _apply_train_dedup(X_train, y_train, ds.meta)
        notes["dup_toggle"] = "dedup_on_frozen"
    else:
        notes["dup_toggle"] = "dedup_off"

    if condition in ("ENT", "ALL") and dataset_name == "unsw_nb15":
        # Pooled stratified re-shuffle: official partition abandoned, so
        # cross-split entity/near-duplicate overlap becomes possible.
        rng = np.random.default_rng(seed + 31)
        X_pool = np.vstack([X_train, X_test])
        y_pool = np.concatenate([y_train, y_test])
        n_train = len(y_train)
        order = rng.permutation(len(y_pool))
        # Stratified-ish: sort by (label, random) then interleave is overkill;
        # a plain seeded permutation preserves the pooled class prior.
        train_idx = order[:n_train]
        test_idx = order[n_train:]
        X_train, y_train = X_pool[train_idx], y_pool[train_idx]
        X_test, y_test = X_pool[test_idx], y_pool[test_idx]
        notes["ent_toggle"] = "pooled_reshuffle_vs_official_partition"

    return SimpleNamespace(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        num_classes=ds.num_classes, meta=ds.meta, notes=notes,
    )


# ---------------------------------------------------------------------------
# Scenario + evaluation
# ---------------------------------------------------------------------------

def _scenario_rows(dataset_name: str, condition: str, ds, seed: int,
                   tail_override: np.ndarray | None = None) -> list[dict[str, Any]]:
    y_train_all = np.asarray(ds.y_train)
    train_idx = p2d._stratified_sample(y_train_all, min(TRAIN_WINDOW, len(y_train_all)), seed)
    test_idx = p2d._stratified_sample(np.asarray(ds.y_test), min(TEST_WINDOW, len(ds.y_test)), seed + 1000)
    X_train, y_train = ds.X_train[train_idx], y_train_all[train_idx]
    X_test, y_test = ds.X_test[test_idx], np.asarray(ds.y_test)[test_idx]

    graph = p2d._feature_graph(X_train, active_views=list(ds.meta.get("active_views", ["ip"])))
    benign_class = ds.meta.get("benign_class") or 0
    target, target_name = p3._noise_target(dataset_name, y_train, ds.meta)
    # F-2 tail is window-relative; locking it to the FINAL condition's window
    # keeps every leakage condition on the SAME metric definition (without the
    # lock, toggling a factor shifts the tail boundary and cross-condition
    # Tail-F1 deltas become apples-to-oranges - e.g. CICIDS class 8 falls out
    # of the tail set under DUP/PRX and the metric goes undefined).
    if tail_override is not None:
        tail = np.asarray(tail_override, dtype=np.int64)
    else:
        tail = p3._tail_noise_labels(dataset_name, y_train, target)
    noisy, flip = p2e._inject_tail_asymmetric(y_train, tail, target, RATE, seed=seed)
    anomaly = d5._unsupervised_feature_anomaly(X_train)
    evidence = compute_evidence(noisy, {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}}, anomaly=anomaly)

    oracle = condition in ("ORA", "ALL")
    cdm = _oracle_cdm(flip) if oracle else d5._cdm_from_observed_labels(noisy, evidence, graph, ds.num_classes)
    hard_weights = d5._weights_for_hard(cdm, evidence)
    soft_rescue = graph_cdm.soft_weights(cdm, evidence, p2e.SOFT_CFG)

    sample = SimpleNamespace(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        num_classes=ds.num_classes, benign_class=benign_class, anomaly=anomaly, graph=graph,
    )
    # Audit quantity; measured on 3 seeds only (screening precedent, same as p3)
    # to keep the full 10-seed grid affordable.
    if seed < 3:
        dup = p3._duplicate_intersection(X_train, X_test)
    else:
        dup = {"exact_dup_rate": float("nan"), "near_dup_rate": float("nan")}

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        weights, train_labels, _info = p2e._method_training_plan(
            method, sample, noisy, cdm, evidence, hard_weights, soft_rescue, seed
        )
        _y_fit, y_pred = p2e._fit_predict_train_test(X_train, train_labels, X_test, weights, method, seed)
        seen = np.isin(y_test, np.unique(y_train))
        rows.append({
            "dataset": dataset_name,
            "condition": condition,
            "method": method,
            "seed": int(seed),
            "noise_rate": float(RATE),
            "macro_f1": p3._macro_f1(y_test, y_pred),
            "macro_f1_seen_classes": p3._macro_f1(y_test[seen], y_pred[seen]) if seen.any() else float("nan"),
            "tail_f1": p3._tail_f1(y_test, y_pred, tail),
            "unseen_test_class_fraction": float(1.0 - seen.mean()),
            "oracle_cdm": bool(oracle),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "tail_classes": json.dumps([int(c) for c in np.asarray(tail)]),
            "tail_locked": tail_override is not None,
            "exact_dup_rate": dup["exact_dup_rate"],
            "near_dup_rate": dup["near_dup_rate"],
            **{f"note_{k}": str(v) for k, v in ds.notes.items()},
        })
    return rows


# ---------------------------------------------------------------------------
# Driver + descriptive statistics (no new inferential family, per F-6)
# ---------------------------------------------------------------------------

def _delta_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired per-seed deltas of each leakage condition vs FINAL."""
    out_rows: list[dict[str, Any]] = []
    for (dataset, method), part in frame.groupby(["dataset", "method"]):
        final = part[part.condition == "FINAL"].set_index("seed")
        if final.empty:
            continue
        for condition in CONDITIONS:
            if condition == "FINAL":
                continue
            cond = part[part.condition == condition].set_index("seed")
            joined = cond.join(final, lsuffix="_cond", rsuffix="_final", how="inner")
            if joined.empty:
                continue
            for metric in ("tail_f1", "macro_f1"):
                diff = (joined[f"{metric}_cond"] - joined[f"{metric}_final"]).to_numpy(dtype=float)
                lo, hi = p2f._bootstrap_ci(diff)
                out_rows.append({
                    "dataset": dataset, "method": method, "condition": condition, "metric": metric,
                    "paired_seeds": int(len(diff)),
                    "final_mean": float(joined[f"{metric}_final"].mean()),
                    "cond_mean": float(joined[f"{metric}_cond"].mean()),
                    "mean_delta": float(np.mean(diff)),
                    "bootstrap_ci95_low": float(lo), "bootstrap_ci95_high": float(hi),
                })
    return pd.DataFrame(out_rows)


def run(
    configs_dir: str | Path = "configs",
    out_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
    tables_dir: str | Path = "tables",
    datasets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    configs = Path(configs_dir)
    out, reports, tables = Path(out_dir), Path(reports_dir), Path(tables_dir)
    for d in (out, reports, tables):
        d.mkdir(parents=True, exist_ok=True)
    selected = tuple(DATASETS if datasets is None else datasets)

    suffix = "all" if selected == tuple(DATASETS) else "_".join(selected)
    partial = out / f"p6_leakage_ablation_fixedtail_{suffix}.partial.csv"
    rows: list[dict[str, Any]] = []
    done: set[tuple[str, str, int]] = set()
    if partial.exists():
        existing = pd.read_csv(partial)
        rows = existing.to_dict(orient="records")
        counts = existing.groupby(["dataset", "condition", "seed"])["method"].nunique()
        done = {(str(k[0]), str(k[1]), int(k[2])) for k, c in counts.items() if int(c) == len(METHODS)}

    skipped: dict[str, str] = {}
    for dataset in selected:
        # Pass A: FINAL first - lock the F-2 tail set per seed so every other
        # condition is evaluated on the SAME tail definition.
        lock_path = out / f"p6_tail_lock_{dataset}.json"
        tail_lock: dict[int, list[int]] = {}
        if lock_path.exists():
            tail_lock = {int(k): [int(c) for c in v] for k, v in json.loads(lock_path.read_text(encoding="utf-8")).items()}
        ordered = ["FINAL"] + [c for c in CONDITIONS if c != "FINAL"]
        for condition in ordered:
            if (dataset, condition) in UNSUPPORTED:
                skipped[f"{dataset}/{condition}"] = UNSUPPORTED[(dataset, condition)]
                continue
            # Seed-dependent loads (UNSW pooled reshuffle) cannot be cached;
            # all other conditions load once per condition.
            cacheable = not (dataset == "unsw_nb15" and condition in ("ENT", "ALL"))
            ds_cache: Any = None
            for seed in SEEDS:
                if (dataset, condition, seed) in done:
                    continue
                if ds_cache is None or not cacheable:
                    _t_load = time.time()
                    ds_cache = _load_condition_dataset(dataset, condition, seed, configs)
                    print(f"[p6] {dataset} {condition} load: {time.time()-_t_load:.1f}s", flush=True)
                override = None if condition == "FINAL" else np.asarray(tail_lock[seed], dtype=np.int64)
                _t_seed = time.time()
                new_rows = _scenario_rows(dataset, condition, ds_cache, seed, tail_override=override)
                rows.extend(new_rows)
                if condition == "FINAL":
                    tail_lock[seed] = json.loads(new_rows[0]["tail_classes"])
                    lock_path.write_text(json.dumps(tail_lock), encoding="utf-8")
                print(f"[p6] {dataset} {condition} seed={seed} done in {time.time()-_t_seed:.1f}s", flush=True)
                pd.DataFrame(rows).to_csv(partial, index=False)

    frame = pd.DataFrame(rows)
    final_csv = out / f"p6_leakage_ablation_fixedtail_{suffix}.csv"
    frame.to_csv(final_csv, index=False)
    deltas = _delta_table(frame)
    deltas.to_csv(tables / f"table_p6_leakage_ablation_fixedtail_{suffix}.csv", index=False)

    per_key = {ds: sorted(frame[frame.dataset == ds].condition.unique().tolist()) for ds in selected}
    report = {
        "stage": "P6-P0-6-leakage-ablation",
        "completed": True,
        "row_count": int(len(frame)),
        "seeds": list(SEEDS),
        "noise_rate": RATE,
        "methods": list(METHODS),
        "conditions": list(CONDITIONS),
        "conditions_run": per_key,
        "unsupported": skipped,
        "statistics": "descriptive paired deltas vs FINAL with bootstrap 95% CI; no new inferential family (F-6)",
        "tail_definition": "F-2 tail locked per seed to the FINAL condition window (tail_lock files in results/)",
        "outputs": {"results": str(final_csv).replace("\\", "/"),
                    "deltas": str(tables / f"table_p6_leakage_ablation_fixedtail_{suffix}.csv").replace("\\", "/")},
    }
    (reports / f"p6_leakage_ablation_fixedtail_{suffix}.json").write_text(
        json.dumps(_jsonable(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


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
    parser = argparse.ArgumentParser(description="P6/P0-6 leakage ablation under final protocol")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--datasets", default=None, help="Comma-separated; default: all")
    args = parser.parse_args()
    selected = None if args.datasets is None else tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    report = run(out_dir=args.out, reports_dir=args.reports, tables_dir=args.tables, datasets=selected)
    print(json.dumps({"completed": report["completed"], "row_count": report["row_count"],
                      "unsupported": report["unsupported"]}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
