"""P3 / P0-2 label-construction proxy leakage audit (screening protocol).

Implements the Phase-1 screening matrix from the revision plan:

- C0  S0 baseline          : stratified iid split, unmasked features
- C1  S0 proxy-masked      : stratified split + label-construction proxy
                             fields removed BEFORE scaling / graph building
- C2  S1 temporal          : time-ordered group split (train before test)
- C3  S2 entity            : entity-disjoint split (source IP groups)
- C4  exploratory holdout  : S3 block/scenario on CICIDS; class holdout on
                             CESNET/UNSW is out-of-protocol and used only as
                             an unseen-class stress test
- C5  transductive check   : inductive ExtraTrees vs transductive label
                             propagation on the joint train+test graph
- C6  S1+S2 joint split    : temporal and entity-disjoint (CESNET); unsupported
                             where the local data lacks required metadata

Screening protocol: 3 seeds x 1 noise rate (tail-asymmetric 0.6) x
{CoLD, ablation_hard, Graph-CoLD-soft}. Surviving configurations are
candidates for the full 10 seeds x 3 rates run.

Also reports, per condition, the train<->test exact and near-duplicate
intersection inside the audit window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import LabelPropagation

from src.data.loaders import load_dataset
from src.experiments import d5
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence
from src.paper import p2d_clean_rerun as p2d
from src.paper import p2e_salvage as p2e


DATASETS = ("cicids2017", "cesnet_tls_year22", "unsw_nb15")
METHODS = ("CoLD", "ablation_hard", "Graph-CoLD-soft")
SCREEN_SEEDS = (0, 1, 2)
SCREEN_RATE = 0.6
TRAIN_WINDOW = 10_000
TEST_WINDOW = 5_000
NEAR_DUP_COSINE = 0.999

# Label-construction proxy fields per dataset, on top of the identifiers
# already excluded by configs/datasets.yaml (IPs, timestamps, flow IDs).
# tier1 = direct label-construction fields; tier2 = strong derived proxies.
PROXY_TIERS: dict[str, dict[str, list[str]]] = {
    "cicids2017": {
        "tier1": ["Destination Port"],
        "tier2": [],
    },
    "cesnet_tls_year22": {
        # CATEGORY is the parent category of the APP label (label proxy);
        # DST_PORT is a service-identity field.
        "tier1": ["CATEGORY", "DST_PORT"],
        "tier2": ["DST_ASN"],
    },
    "unsw_nb15": {
        # is_sm_ips_ports is derived from source/destination IP+port equality;
        # ct_* counters are computed over 5-tuple windows (IP/port proxies).
        "tier1": ["is_sm_ips_ports"],
        "tier2": [
            "ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
            "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst", "ct_state_ttl",
        ],
    },
}

ENTITY_COLS: dict[str, list[str]] = {
    "cicids2017": ["Src IP", "Source IP"],
    "cesnet_tls_year22": ["SRC_IP"],
    "unsw_nb15": [],  # fixed partition carries no entity columns
}

CONDITIONS = ("C0_S0_baseline", "C1_S0_masked_t1", "C1b_S0_masked_t2", "C2_S1_temporal", "C3_S2_entity", "C4_S3_block", "C5_S0_transductive", "C6_S1S2_temporal_entity")

# The UNSW-NB15 fixed train/test partition carries no timestamps or entity
# columns, so S1/S2 cannot be constructed for it. C4 uses an attack-type
# holdout post-processing instead (see _unsw_s3_holdout).
UNSUPPORTED = {
    ("unsw_nb15", "C2_S1_temporal"): "UNSW-NB15 partition layout has no timestamp columns",
    ("unsw_nb15", "C3_S2_entity"): "UNSW-NB15 partition layout has no entity (IP) columns",
    ("cicids2017", "C3_S2_entity"): (
        "local CICIDS-2017 CSVs are pre-stripped (79 cols, no Src/Dst IP); "
        "entity split requires the original MachineLearningCVE files - recorded as data limitation"
    ),
    ("cicids2017", "C6_S1S2_temporal_entity"): (
        "local CICIDS-2017 copy has no entity columns required for joint S1+S2"
    ),
    ("unsw_nb15", "C6_S1S2_temporal_entity"): (
        "UNSW-NB15 partition layout has neither timestamp nor entity columns"
    ),
}


def _formal_cfg(dataset_name: str, configs_dir: Path) -> dict[str, Any]:
    """Replicate d5._load_formal_dataset's config overrides without loading."""
    cfg = yaml.safe_load((configs_dir / "datasets.yaml").read_text(encoding="utf-8"))
    reports_dir = configs_dir.parent / "reports"
    if dataset_name == "cesnet_tls_year22":
        audit = d5._read_json(reports_dir / "cesnet_audit_report.json")
        cfg[dataset_name]["path"] = d5._resolve_recorded_data_path(
            dataset_name, audit.get("actual_data_path"), cfg[dataset_name]["path"], configs_dir
        )
        cfg[dataset_name]["sample_rows"] = d5.CESNET_D5_SAMPLE_ROWS
        cfg[dataset_name]["dataset_hash"] = audit.get("dataset_hash")
    elif dataset_name == "cicids2017":
        protocol = d5._read_json(reports_dir / "cicids_final_protocol.json")
        cfg[dataset_name]["dataset_hash"] = protocol.get("dataset_hash")
    elif dataset_name == "unsw_nb15":
        audit = d5._read_json(reports_dir / "unsw_ingest.json")
        cfg[dataset_name]["path"] = d5._resolve_recorded_data_path(
            dataset_name, audit.get("actual_data_path"), cfg[dataset_name]["path"], configs_dir
        )
        cfg[dataset_name]["dataset_hash"] = audit.get("dataset_hash")
    return cfg


def _condition_cfg(dataset_name: str, condition: str, base_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = {**base_cfg, dataset_name: dict(base_cfg[dataset_name])}
    ds_cfg = cfg[dataset_name]
    ds_cfg["entity_cols"] = list(ENTITY_COLS.get(dataset_name, []))
    ds_cfg["split_strategy"] = "stratified"

    if condition.startswith("C1"):
        tier = "tier1" if condition == "C1_S0_masked_t1" else "tier2"
        proxies = PROXY_TIERS[dataset_name]["tier1"] + (PROXY_TIERS[dataset_name]["tier2"] if tier == "tier2" else [])
        ds_cfg["drop_cols"] = list(ds_cfg.get("drop_cols", [])) + list(proxies)
    elif condition == "C2_S1_temporal":
        ds_cfg["split_strategy"] = "temporal"
    elif condition == "C3_S2_entity":
        ds_cfg["split_strategy"] = "entity"
    elif condition == "C4_S3_block":
        ds_cfg["split_strategy"] = "block_holdout"
        ds_cfg["holdout_blocks"] = 1
    elif condition == "C6_S1S2_temporal_entity":
        ds_cfg["split_strategy"] = "temporal_entity"
    return cfg


def _tail_labels(y_train: np.ndarray) -> np.ndarray:
    """Frozen F-2 tail definition: classes with count <= median class count."""
    labels, counts = np.unique(y_train, return_counts=True)
    threshold = float(np.median(counts))
    return labels[counts <= threshold]


def _noise_target(dataset_name: str, y_train: np.ndarray, meta: dict[str, Any]) -> tuple[int, str]:
    """Return the frozen F-3 asymmetric-noise target.

    IDS datasets map tail attack classes to their declared benign class.
    CESNET has no benign semantics, so it maps to the most frequent service
    class in the frozen audit window; ties resolve to the lowest encoded label.
    """
    if dataset_name == "cesnet_tls_year22":
        target = 11  # microsoft-diagnostic; frozen in F-3 / noise_map.json
        if target not in set(np.unique(y_train).tolist()):
            raise ValueError("Frozen CESNET noise target class 11 is absent from this training window")
        names = list(meta.get("class_names", []))
        name = names[target] if 0 <= target < len(names) else str(target)
        return target, name
    benign = meta.get("benign_class")
    if benign is None:
        raise ValueError(f"{dataset_name} has no declared benign class for F-3 mapping")
    target = int(benign)
    names = list(meta.get("class_names", []))
    name = names[target] if 0 <= target < len(names) else str(target)
    return target, name


def _tail_noise_labels(dataset_name: str, y_train: np.ndarray, target: int) -> np.ndarray:
    tail = _tail_labels(y_train)
    if dataset_name != "cesnet_tls_year22":
        tail = tail[tail != target]
    return tail


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _tail_f1(y_true: np.ndarray, y_pred: np.ndarray, tail: np.ndarray) -> float:
    per_class = f1_score(y_true, y_pred, average=None, labels=np.unique(y_true), zero_division=0)
    classes = np.unique(y_true)
    mask = np.isin(classes, tail)
    if not mask.any():
        return float("nan")
    return float(np.mean(per_class[mask]))


def _duplicate_intersection(X_train: np.ndarray, X_test: np.ndarray) -> dict[str, Any]:
    train_hash = pd.util.hash_pandas_object(pd.DataFrame(X_train), index=False).to_numpy()
    test_hash = pd.util.hash_pandas_object(pd.DataFrame(X_test), index=False).to_numpy()
    exact = int(np.isin(test_hash, np.unique(train_hash)).sum())
    near = 0
    if X_train.shape[0] > 1 and X_test.shape[0] > 0:
        radius = max(1e-6, 1.0 - NEAR_DUP_COSINE)
        nn = NearestNeighbors(radius=radius, metric="cosine", n_jobs=-1)
        nn.fit(X_train)
        hits = nn.radius_neighbors(X_test, return_distance=False)
        near = int(sum(1 for h in hits if len(h) > 0))
    return {
        "test_rows": int(X_test.shape[0]),
        "exact_dup_test_rows": exact,
        "exact_dup_rate": exact / max(1, X_test.shape[0]),
        "near_dup_test_rows": near,
        "near_dup_rate": near / max(1, X_test.shape[0]),
        "near_dup_cosine_threshold": NEAR_DUP_COSINE,
    }


def _transductive_contrast(
    X_train: np.ndarray,
    noisy: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tail: np.ndarray,
    seed: int,
) -> dict[str, float]:
    X_joint = np.vstack([X_train, X_test])
    y_joint = np.concatenate([noisy, np.full(X_test.shape[0], -1, dtype=noisy.dtype)])
    lp = LabelPropagation(kernel="knn", n_neighbors=10, max_iter=100)
    lp.fit(X_joint, y_joint)
    pred = lp.predict(X_joint)[X_train.shape[0]:]
    return {
        "transductive_lp_macro_f1": _macro_f1(y_test, pred),
        "transductive_lp_tail_f1": _tail_f1(y_test, pred, tail),
    }


def _unsw_s3_holdout(ds, rng: np.random.Generator):
    """S3 fallback via class holdout: hold out one attack/service type from
    TRAIN only (unseen-class stress test). Used for UNSW (fixed partition)
    and for CESNET when the time window is too short for block holdout.
    Returns (X_train, y_train, info)."""
    y_train = np.asarray(ds.y_train)
    y_test = np.asarray(ds.y_test)
    benign = ds.meta.get("benign_class")
    candidates = [c for c in np.unique(y_train) if c != benign and c in set(np.unique(y_test))]
    counts = {c: int(np.sum(y_train == c)) for c in candidates}
    held = min(counts, key=counts.get)  # smallest shared class
    keep = y_train != held
    info = {"held_out_class": int(held), "held_out_class_name": ds.meta["class_names"][held],
            "unseen_test_rows": int(np.sum(y_test == held)),
            "s3_variant": "class_holdout"}
    return ds.X_train[keep], y_train[keep], info


def run_condition(
    dataset_name: str,
    condition: str,
    configs_dir: str | Path = "configs",
    out_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
    seeds: tuple[int, ...] = SCREEN_SEEDS,
    rate: float = SCREEN_RATE,
) -> dict[str, Any]:
    configs_dir = Path(configs_dir)
    out_dir = Path(out_dir)
    reports_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if (dataset_name, condition) in UNSUPPORTED:
        summary = {
            "stage": "P3-P0-2-screening",
            "dataset": dataset_name,
            "condition": condition,
            "status": "unsupported",
            "reason": UNSUPPORTED[(dataset_name, condition)],
        }
        (reports_dir / f"p3_proxy_audit_{dataset_name}_{condition}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    base_cfg = _formal_cfg(dataset_name, configs_dir)
    cfg = _condition_cfg(dataset_name, condition, base_cfg)
    ds = load_dataset(dataset_name, cfg)
    if dataset_name == "cicids2017":
        ds = d5._deduplicate_cicids_train(ds)

    split_notes = dict(ds.meta.get("split_notes", {}))
    feature_names = list(ds.meta["feature_names"])
    leaked = [c for c in ("CATEGORY", "DST_PORT", "Destination Port", "is_sm_ips_ports")
              if any(name == c or name.startswith(f"{c}_") for name in feature_names)]

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X_train_all, y_train_all = ds.X_train, np.asarray(ds.y_train)
        s3_info: dict[str, Any] = {}
        if condition == "C4_S3_block":
            fell_back = "fallback" in split_notes or split_notes.get("applied_strategy") == "stratified"
            if dataset_name == "unsw_nb15" or (dataset_name == "cesnet_tls_year22" and fell_back):
                X_train_all, y_train_all, s3_info = _unsw_s3_holdout(ds, rng)
                if dataset_name == "cesnet_tls_year22":
                    s3_info["block_holdout_note"] = (
                        "time window spans <1 day in the sampled audit frame; "
                        "block holdout degenerate, used class holdout instead"
                    )

        train_idx = p2d._stratified_sample(y_train_all, min(TRAIN_WINDOW, len(y_train_all)), seed)
        test_idx = p2d._stratified_sample(ds.y_test, min(TEST_WINDOW, len(ds.y_test)), seed + 1000)
        X_train, y_train = X_train_all[train_idx], y_train_all[train_idx]
        X_test, y_test = ds.X_test[test_idx], np.asarray(ds.y_test[test_idx])

        graph = p2d._feature_graph(X_train, active_views=list(ds.meta.get("active_views", ["ip"])))
        benign_class = ds.meta.get("benign_class") or 0
        sample = SimpleNamespace(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            num_classes=ds.num_classes, benign_class=benign_class,
            anomaly=d5._unsupervised_feature_anomaly(X_train), graph=graph,
        )

        noise_target, noise_target_name = _noise_target(dataset_name, y_train, ds.meta)
        tail = _tail_noise_labels(dataset_name, y_train, noise_target)
        noisy, flip = p2e._inject_tail_asymmetric(y_train, tail, noise_target, rate, seed=seed)
        evidence = compute_evidence(noisy, {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}}, anomaly=sample.anomaly)
        cdm = d5._cdm_from_observed_labels(noisy, evidence, graph, ds.num_classes)
        hard_weights = d5._weights_for_hard(cdm, evidence)
        soft_rescue = graph_cdm.soft_weights(cdm, evidence, p2e.SOFT_CFG)

        dup = _duplicate_intersection(X_train, X_test)

        for method in METHODS:
            weights, train_labels, _info = p2e._method_training_plan(
                method, sample, noisy, cdm, evidence, hard_weights, soft_rescue, seed
            )
            _y_train_final, y_test_pred = p2e._fit_predict_train_test(
                X_train, train_labels, X_test, weights, method, seed
            )
            seen_classes = np.unique(y_train)
            seen_mask = np.isin(y_test, seen_classes)
            macro_f1_seen = _macro_f1(y_test[seen_mask], y_test_pred[seen_mask]) if seen_mask.any() else float("nan")
            row = {
                "dataset": dataset_name,
                "condition": condition,
                "method": method,
                "seed": int(seed),
                "noise_rate": float(rate),
                "noise_target_class": int(noise_target),
                "noise_target_name": noise_target_name,
                "macro_f1": _macro_f1(y_test, y_test_pred),
                "macro_f1_seen_classes": macro_f1_seen,
                "tail_f1": _tail_f1(y_test, y_test_pred, tail),
                "unseen_test_class_fraction": float(1.0 - seen_mask.mean()),
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                **dup,
                **{f"split_{k}": str(v) for k, v in {**split_notes, **s3_info}.items()},
            }
            if condition == "C5_S0_transductive" and method == "Graph-CoLD-soft":
                row.update(_transductive_contrast(X_train, noisy, X_test, y_test, tail, seed))
            rows.append(row)

    frame = pd.DataFrame(rows)
    out_csv = out_dir / f"p3_proxy_audit_{dataset_name}_{condition}.csv"
    frame.to_csv(out_csv, index=False)

    summary = _condition_summary(frame, dataset_name, condition, leaked, split_notes, s3_info, feature_names)
    report_json = reports_dir / f"p3_proxy_audit_{dataset_name}_{condition}.json"
    report_json.write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(_jsonable(summary["headline"]), indent=2, allow_nan=False), flush=True)
    return summary


def _condition_summary(frame, dataset_name, condition, leaked, split_notes, s3_info, feature_names) -> dict[str, Any]:
    pivot = frame.groupby("method")[["macro_f1", "macro_f1_seen_classes", "tail_f1"]].agg(["mean", "std"])
    soft = frame[frame.method == "Graph-CoLD-soft"].set_index("seed")
    hard = frame[frame.method == "ablation_hard"].set_index("seed")
    cold = frame[frame.method == "CoLD"].set_index("seed")
    joined = soft[["tail_f1", "macro_f1"]].join(hard[["tail_f1", "macro_f1"]], lsuffix="_soft", rsuffix="_hard")
    joined = joined.join(cold[["macro_f1"]].rename(columns={"macro_f1": "macro_f1_cold"}))
    tail_delta = float(joined["tail_f1_soft"].mean() - joined["tail_f1_hard"].mean())
    dup_cols = frame[["exact_dup_rate", "near_dup_rate"]].mean().to_dict()
    headline = {
        "dataset": dataset_name,
        "condition": condition,
        "tail_f1_delta_soft_minus_hard": tail_delta,
        "macro_f1_soft": float(joined["macro_f1_soft"].mean()),
        "macro_f1_hard": float(joined["macro_f1_hard"].mean()),
        "macro_f1_cold": float(joined["macro_f1_cold"].mean()),
        "exact_dup_rate": float(dup_cols["exact_dup_rate"]),
        "near_dup_rate": float(dup_cols["near_dup_rate"]),
    }
    return {
        "stage": "P3-P0-2-screening",
        "dataset": dataset_name,
        "condition": condition,
        "seeds": sorted(int(s) for s in frame.seed.unique()),
        "noise_rate": float(frame.noise_rate.iloc[0]),
        "noise_type": "tail_asymmetric",
        "noise_target_class": int(frame.noise_target_class.iloc[0]),
        "noise_target_name": str(frame.noise_target_name.iloc[0]),
        "leaked_proxy_features_present": leaked,
        "n_features": len(feature_names),
        "split_notes": {**split_notes, **s3_info},
        "headline": headline,
        "per_method": {m: {"macro_f1_mean": float(pivot.loc[m, ("macro_f1", "mean")]),
                           "tail_f1_mean": float(pivot.loc[m, ("tail_f1", "mean")])} for m in pivot.index},
        "transductive": frame[["transductive_lp_macro_f1", "transductive_lp_tail_f1"]].dropna().mean().to_dict()
        if "transductive_lp_macro_f1" in frame else {},
    }


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
    parser = argparse.ArgumentParser(description="P3/P0-2 proxy leakage audit (screening)")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    args = parser.parse_args()
    run_condition(args.dataset, args.condition, args.configs, args.out, args.reports)


if __name__ == "__main__":
    main()
