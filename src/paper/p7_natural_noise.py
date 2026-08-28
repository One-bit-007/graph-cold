"""P1-4: real-noise experiment on CICIDS-2017 raw copy with Engelen-corrected labels.

Design (descriptive, no new inferential family per F-6):
  * Pool: cicids2017_raw, temporal split, FINAL parity = tier1 mask (Destination
    Port dropped) + train exact-dedup (d5 6-decimal row-hash semantics).
  * Training labels: ORIGINAL released labels (contain real annotation errors).
  * Evaluation reference: Engelen-corrected labels (Attempted -> BENIGN applied).
    Test rows whose corrected label falls outside the frozen 11-class space are
    excluded from metrics and counted (unpredictable-by-construction).
  * No noise injection - the original-vs-corrected diff IS the noise.
  * Windows: train 10k / test 5k stratified on ORIGINAL labels (trainer's view),
    seeds 0-9; all four methods share each seed's window (paired).
  * Tail classes: F-2 window-relative on original train labels (frozen metric).
  * Methods: CoLD / ablation_hard / Graph-CoLD-soft (p2e plan, CDM from observed
    original labels) + Confident-Learning (noise_rate=0.0 -> cleanlab
    get_noise_indices on out-of-fold probabilities; no true rate anywhere, F-5).

Outputs: results/p7_natural_noise.csv (partial-resume), reports/p7_natural_noise.json
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
import yaml

from src.baselines.confident_learning import ConfidentLearningBaseline
from src.data.loaders import load_dataset
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence
from src.experiments import d5
from src.paper import p2d_clean_rerun as p2d
from src.paper import p2e_salvage as p2e
from src.paper import p2f_tighten as p2f
from src.paper import p3_proxy_audit as p3

DATASET = "cicids2017_raw"
METHODS = ("CoLD", "ablation_hard", "Graph-CoLD-soft", "Confident-Learning")
SEEDS = tuple(range(10))
TRAIN_WINDOW = p3.TRAIN_WINDOW
TEST_WINDOW = p3.TEST_WINDOW
CORRECTED_PATH = Path("results/p7_corrected_labels.csv.gz")
OUT_CSV = Path("results/p7_natural_noise.csv")
OUT_JSON = Path("reports/p7_natural_noise.json")


def _load_corrected_map(path: Path = CORRECTED_PATH) -> dict[tuple[str, int], str]:
    df = pd.read_csv(path)
    return {
        (str(f), int(r)): str(c)
        for f, r, c in zip(df["source_file"], df["row_in_file"], df["corrected_canon"])
    }


def _final_parity_cfg(configs_dir: Path, split: str = "temporal") -> dict[str, Any]:
    cfg = yaml.safe_load(open(configs_dir / "datasets.yaml", encoding="utf-8"))
    ds_cfg = dict(cfg[DATASET])
    ds_cfg["split_strategy"] = split
    if split == "temporal":
        # FINAL parity: CICIDS FINAL splits temporal by source-file weekday (the ML
        # copy has no timestamp column). Mirror that on the raw copy; a strict
        # timestamp cut would put only Friday-afternoon classes (DDoS/PortScan) in
        # test and make Tail-F1 undefined.
        ds_cfg["temporal_order"] = "source_file"
    # FINAL parity: CICIDS tier1 mask drops Destination Port (p3.PROXY_TIERS).
    ds_cfg["drop_cols"] = list(ds_cfg.get("drop_cols", [])) + ["Destination Port"]
    return {**cfg, DATASET: ds_cfg}


def _dedup_indices(X: np.ndarray) -> np.ndarray:
    """d5._deduplicate_cicids_train semantics at index level (keep first)."""
    row_hash = d5._feature_row_hashes(np.asarray(X))
    _, first_idx = np.unique(row_hash, return_index=True)
    return np.sort(first_idx)


def _map_clean(labels: list[str], label_mapping: dict[str, int]) -> np.ndarray:
    """Corrected labels -> class ids; -1 when outside the frozen class space."""
    return np.array([label_mapping.get(v, -1) for v in labels], dtype=np.int64)


def _scenario_rows(ds, clean_train: np.ndarray, clean_test: np.ndarray, seed: int) -> list[dict[str, Any]]:
    y_train_all = np.asarray(ds.y_train)
    y_test_all = np.asarray(ds.y_test)
    train_idx = p2d._stratified_sample(y_train_all, min(TRAIN_WINDOW, len(y_train_all)), seed)
    test_idx = p2d._stratified_sample(y_test_all, min(TEST_WINDOW, len(y_test_all)), seed + 1000)
    X_train, y_train = ds.X_train[train_idx], y_train_all[train_idx]
    X_test, y_test = ds.X_test[test_idx], y_test_all[test_idx]
    y_test_clean = clean_test[test_idx]
    y_train_clean = clean_train[train_idx]

    graph = p2d._feature_graph(X_train, active_views=list(ds.meta.get("active_views", ["ip"])))
    benign_class = int(ds.meta.get("benign_class") or 0)
    # F-2 tail on ORIGINAL labels (trainer's view); benign excluded as target.
    tail = p3._tail_noise_labels("cicids2017", y_train, benign_class)
    real_flip = y_train_clean != y_train  # training rows whose released label was wrong
    real_flip = real_flip & (y_train_clean != -1)

    anomaly = d5._unsupervised_feature_anomaly(X_train)
    evidence = compute_evidence(y_train, {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}}, anomaly=anomaly)
    cdm = d5._cdm_from_observed_labels(y_train, evidence, graph, ds.num_classes)
    hard_weights = d5._weights_for_hard(cdm, evidence)
    soft_rescue = graph_cdm.soft_weights(cdm, evidence, p2e.SOFT_CFG)

    sample = SimpleNamespace(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                             num_classes=ds.num_classes, benign_class=benign_class, anomaly=anomaly, graph=graph)

    eval_mask = y_test_clean != -1  # corrected label inside frozen class space
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    extra: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        if method == "Confident-Learning":
            cl = ConfidentLearningBaseline(seed=seed, noise_rate=0.0, n_estimators=24)
            cl_result = cl.fit_predict(X_train, y_train, X_test, ds.num_classes)
            predictions[method] = np.asarray(cl_result.y_pred, dtype=np.int64)
            extra[method] = {
                "retained_fraction": float(np.mean(cl_result.retained_mask)),
                "engine": str(cl_result.details.get("issue_detector", "")),
                "probability_protocol": str(cl_result.details.get("probability_protocol", "")),
            }
        else:
            weights, train_labels, _info = p2e._method_training_plan(
                method, sample, y_train, cdm, evidence, hard_weights, soft_rescue, seed
            )
            _y_fit, y_pred = p2e._fit_predict_train_test(X_train, train_labels, X_test, weights, method, seed)
            predictions[method] = np.asarray(y_pred, dtype=np.int64)
            extra[method] = {"retained_fraction": float(np.mean(np.asarray(weights) > 0)), "engine": "", "probability_protocol": ""}

    for method in METHODS:
        pred = predictions[method]
        rows.append({
            "dataset": "cicids2017",
            "setting": "real_noise_engelen",
            "method": method,
            "seed": int(seed),
            "macro_f1_cleanref": p3._macro_f1(y_test_clean[eval_mask], pred[eval_mask]) if eval_mask.any() else float("nan"),
            "tail_f1_cleanref": p3._tail_f1(y_test_clean[eval_mask], pred[eval_mask], tail) if eval_mask.any() else float("nan"),
            "macro_f1_origref": p3._macro_f1(y_test, pred),
            "tail_f1_origref": p3._tail_f1(y_test, pred, tail),
            "tail_classes": json.dumps([int(c) for c in np.asarray(tail)]),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_test_evaluable": int(eval_mask.sum()),
            "train_real_flip_rate": float(np.mean(real_flip)),
            **extra[method],
        })
    return rows


def _delta(frame: pd.DataFrame, a: str, b: str, metric: str) -> dict[str, Any]:
    pa = frame[frame.method == a].set_index("seed")[metric]
    pb = frame[frame.method == b].set_index("seed")[metric]
    joined = pd.concat([pa, pb], axis=1, keys=[a, b]).dropna()
    if joined.empty:
        return {"pair": f"{a}-{b}", "metric": metric, "paired_seeds": 0}
    diff = (joined[a] - joined[b]).to_numpy(dtype=float)
    lo, hi = p2f._bootstrap_ci(diff)
    return {
        "pair": f"{a}-{b}", "metric": metric, "paired_seeds": int(len(diff)),
        "mean_delta": float(np.mean(diff)),
        "bootstrap_ci95_low": float(lo), "bootstrap_ci95_high": float(hi),
        "mean_a": float(joined[a].mean()), "mean_b": float(joined[b].mean()),
    }


def run(seeds: tuple[int, ...] = SEEDS, split: str = "temporal",
        out_csv: Path = OUT_CSV, out_json: Path = OUT_JSON) -> pd.DataFrame:
    corrected = _load_corrected_map()
    cfg = _final_parity_cfg(Path("configs"), split=split)
    t0 = time.perf_counter()
    ds = load_dataset(DATASET, cfg)
    print(f"load: {time.perf_counter() - t0:.1f}s; train {len(ds.y_train)} / test {len(ds.y_test)}")

    # Class ids follow the loader's original-label mapping; corrected labels are
    # in the Engelen namespace, so map through the canonical form.
    from src.paper.p7_engelen_labels import canonicalize_original

    label_mapping = {canonicalize_original(str(k)): int(v) for k, v in ds.meta["label_mapping"].items()}
    sf_train = ds.meta["source_files"]["train"]
    sf_test = ds.meta["source_files"]["test"]
    ri_train = ds.meta["row_in_file"]["train"]
    ri_test = ds.meta["row_in_file"]["test"]
    clean_train = _map_clean([corrected.get((str(f), int(r)), "<missing>") for f, r in zip(sf_train, ri_train)], label_mapping)
    clean_test = _map_clean([corrected.get((str(f), int(r)), "<missing>") for f, r in zip(sf_test, ri_test)], label_mapping)
    missing = int(np.sum(clean_train == -1) + np.sum(clean_test == -1))
    # NOTE: -1 covers both out-of-class-space corrected labels and (unexpected)
    # unmatched row keys; the derivation covers every raw row, so unmatched = 0
    # expected. Verified in the guard tests via the diff summary instead.
    print(f"clean labels mapped; out-of-space/missing train+test: {missing}")

    keep = _dedup_indices(ds.X_train)
    print(f"train dedup: {len(ds.y_train)} -> {len(keep)}")
    ds.X_train = np.asarray(ds.X_train)[keep]
    ds.y_train = np.asarray(ds.y_train)[keep]
    clean_train = clean_train[keep]

    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["method"], prev["seed"]))
        rows = prev.to_dict("records")
    else:
        rows = []
    for seed in seeds:
        if all((m, seed) in done for m in METHODS):
            continue
        t1 = time.perf_counter()
        new_rows = [r for r in _scenario_rows(ds, clean_train, clean_test, seed) if (r["method"], r["seed"]) not in done]
        rows.extend(new_rows)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"seed {seed}: +{len(new_rows)} rows in {time.perf_counter() - t1:.1f}s")
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)

    summary = {
        "design": "train on original released labels (real errors), evaluate on Engelen-corrected labels; descriptive only",
        "split": split,
        "n_rows": int(len(frame)),
        "per_method_means": frame.groupby("method")[["macro_f1_cleanref", "tail_f1_cleanref", "macro_f1_origref", "tail_f1_origref", "train_real_flip_rate"]].mean().round(4).to_dict("index"),
        "deltas_cleanref": [
            _delta(frame, "Graph-CoLD-soft", "ablation_hard", "tail_f1_cleanref"),
            _delta(frame, "Graph-CoLD-soft", "CoLD", "tail_f1_cleanref"),
            _delta(frame, "Graph-CoLD-soft", "Confident-Learning", "tail_f1_cleanref"),
            _delta(frame, "Graph-CoLD-soft", "ablation_hard", "macro_f1_cleanref"),
            _delta(frame, "Confident-Learning", "ablation_hard", "tail_f1_cleanref"),
        ],
        "deltas_origref": [
            _delta(frame, "Graph-CoLD-soft", "ablation_hard", "tail_f1_origref"),
            _delta(frame, "Graph-CoLD-soft", "Confident-Learning", "tail_f1_origref"),
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["per_method_means"], indent=2))
    for d in summary["deltas_cleanref"]:
        print(d)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--split", default="temporal", choices=["temporal", "stratified"])
    args = parser.parse_args()
    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = tuple(range(int(lo), int(hi) + 1))
    else:
        seeds = tuple(int(s) for s in args.seeds.split(","))
    if args.split == "temporal":
        run(seeds)
    else:
        # Supplementary descriptive sensitivity arm (NOT the frozen protocol):
        # the FINAL temporal window keeps real errors almost entirely in the
        # training period; a stratified pool exposes them at eval time.
        run(seeds, split="stratified",
            out_csv=Path("results/p7_natural_noise_stratified.csv"),
            out_json=Path("reports/p7_natural_noise_stratified.json"))


if __name__ == "__main__":
    main()
