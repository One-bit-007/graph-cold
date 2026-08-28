"""P0-4 Confident Learning + Graph-CDM combination experiment.

The G2 primary comparison is ``CL+Graph-soft`` versus ``Confident-Learning``
under proxy-masked, strongest-available split protocols.  Two issue-set fusion
rules are exploratory:

* issue-intersection: prune only rows flagged by both CL and Graph-CDM;
* issue-union: prune rows flagged by either detector.

Screening uses 3 seeds x one 0.6 rate.  Full mode uses the frozen 10 seeds x
three rates and writes resumable, scenario-level evidence.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
import statsmodels.formula.api as smf

from src.baselines.base import ensure_class_coverage
from src.baselines.confident_learning import ConfidentLearningBaseline
from src.data.loaders import load_dataset
from src.experiments import d5
from src.metrics import macro_f1
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence
from src.paper import p2d_clean_rerun as p2d
from src.paper import p2e_salvage as p2e
from src.paper import p2f_tighten as p2f
from src.paper import p3_proxy_audit as p3


DATASETS = p3.DATASETS
METHODS = (
    "Confident-Learning",
    "Graph-CoLD-soft",
    "CL+Graph-soft",
    "CL-issue-intersection",
    "CL-issue-union",
)
PRIMARY_METHOD = "CL+Graph-soft"
REFERENCE_METHOD = "Confident-Learning"
SCREEN_SEEDS = (0, 1, 2)
FULL_SEEDS = tuple(range(10))
SCREEN_RATES = (0.6,)
FULL_RATES = (0.4, 0.6, 0.8)
TRAIN_WINDOW = 10_000
TEST_WINDOW = 5_000
MACRO_NI_DELTA = 0.01
CLASS_F1_GUARD = 0.05
FAMILY_B_SIZE = len(DATASETS) * 2 * len(FULL_RATES)
RESULT_SCHEMA_VERSION = 2

PROTOCOLS: dict[str, dict[str, str]] = {
    "cicids2017": {
        "mask": "tier1",
        "split": "temporal",
        "label": "C1-tier1 + S1 temporal-by-source-file",
        "limitation": "FINAL frozen on the ML copy; S2 entity isolation attempted on the raw copy with union-find grouping and ruled infeasible by construction (one giant component spans 99.96% of valid flows; see reports/p6_cicids_entity_isolation.md)",
    },
    "cesnet_tls_year22": {
        "mask": "tier2",
        "split": "temporal_entity",
        "label": "C1b-tier2 + S1∩S2 temporal/entity-disjoint",
        "limitation": "audit frame spans about seven hours",
    },
    "unsw_nb15": {
        "mask": "tier2",
        "split": "stratified",
        "label": "C1b-tier2 + official fixed partition",
        "limitation": "S1/S2 unavailable because the fixed files lack timestamp/entity columns",
    },
}


def _dataset_cfg(dataset_name: str, configs_dir: Path) -> dict[str, Any]:
    base = p3._formal_cfg(dataset_name, configs_dir)
    mask_condition = "C1_S0_masked_t1" if PROTOCOLS[dataset_name]["mask"] == "tier1" else "C1b_S0_masked_t2"
    cfg = p3._condition_cfg(dataset_name, mask_condition, base)
    cfg[dataset_name]["split_strategy"] = PROTOCOLS[dataset_name]["split"]
    return cfg


def _load_primary_dataset(dataset_name: str, configs_dir: Path):
    cfg = _dataset_cfg(dataset_name, configs_dir)
    ds = load_dataset(dataset_name, cfg)
    if dataset_name == "cicids2017":
        ds = d5._deduplicate_cicids_train(ds)
    feature_names = [str(name) for name in ds.meta.get("feature_names", [])]
    blocked = list(p3.PROXY_TIERS[dataset_name]["tier1"])
    if PROTOCOLS[dataset_name]["mask"] == "tier2":
        blocked += p3.PROXY_TIERS[dataset_name]["tier2"]
    leaked = [name for name in blocked if any(f == name or f.startswith(f"{name}_") for f in feature_names)]
    if leaked:
        raise AssertionError(f"Proxy fields survived masking for {dataset_name}: {leaked}")
    return ds


def _sample(ds, dataset_name: str, seed: int):
    train_idx = p2d._stratified_sample(ds.y_train, min(TRAIN_WINDOW, len(ds.y_train)), seed)
    test_idx = p2d._stratified_sample(ds.y_test, min(TEST_WINDOW, len(ds.y_test)), seed + 1000)
    X_train = ds.X_train[train_idx]
    y_train = np.asarray(ds.y_train[train_idx], dtype=np.int64)
    X_test = ds.X_test[test_idx]
    y_test = np.asarray(ds.y_test[test_idx], dtype=np.int64)
    graph = p2d._feature_graph(X_train, active_views=list(ds.meta.get("active_views", ["ip"])))
    target, target_name = p3._noise_target(dataset_name, y_train, ds.meta)
    tail = p3._tail_noise_labels(dataset_name, y_train, target)
    return SimpleNamespace(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        graph=graph,
        anomaly=d5._unsupervised_feature_anomaly(X_train),
        num_classes=ds.num_classes,
        tail=tail,
        noise_target=target,
        noise_target_name=target_name,
        split_notes=dict(ds.meta.get("split_notes", {})),
    )


def _predict_weighted(sample, noisy: np.ndarray, weights: np.ndarray, seed: int, hard: bool) -> np.ndarray:
    mode = "ablation_hard" if hard else "Graph-CoLD-soft"
    _, pred = p2e._fit_predict_train_test(sample.X_train, noisy, sample.X_test, weights, mode, seed)
    return np.asarray(pred, dtype=np.int64)


def _safe_ratio(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def _scenario_rows(dataset_name: str, ds, seed: int, rate: float) -> list[dict[str, Any]]:
    sample = _sample(ds, dataset_name, seed)
    noisy, flip = p2e._inject_tail_asymmetric(
        sample.y_train, sample.tail, sample.noise_target, rate, seed=seed
    )
    evidence = compute_evidence(
        noisy,
        {"evidence_preserving": {"freq_protect": "log", "gamma_anomaly": 1.0}},
        anomaly=sample.anomaly,
    )
    cdm = d5._cdm_from_observed_labels(noisy, evidence, sample.graph, sample.num_classes)
    soft = graph_cdm.soft_weights(cdm, evidence, p2e.SOFT_CFG)
    graph_issue = cdm > float(p2e.SOFT_CFG["evidence_preserving"]["theta"])

    cl = ConfidentLearningBaseline(seed=seed, noise_rate=float(np.mean(flip)), n_estimators=24)
    cl_result = cl.fit_predict(sample.X_train, noisy, sample.X_test, sample.num_classes)
    cl_keep = np.asarray(cl_result.retained_mask, dtype=bool)
    cl_issue = ~cl_keep

    intersection_keep = ~(cl_issue & graph_issue)
    union_keep = ~(cl_issue | graph_issue)
    confidence = 1.0 - np.asarray(cdm, dtype=np.float64)
    intersection_keep = ensure_class_coverage(intersection_keep, confidence, noisy)
    union_keep = ensure_class_coverage(union_keep, confidence, noisy)

    predictions = {
        "Confident-Learning": np.asarray(cl_result.y_pred, dtype=np.int64),
        "Graph-CoLD-soft": _predict_weighted(sample, noisy, soft, seed, hard=False),
        "CL+Graph-soft": _predict_weighted(sample, noisy, cl_keep.astype(float) * soft, seed, hard=False),
        "CL-issue-intersection": _predict_weighted(sample, noisy, intersection_keep.astype(float), seed, hard=True),
        "CL-issue-union": _predict_weighted(sample, noisy, union_keep.astype(float), seed, hard=True),
    }
    keeps = {
        "Confident-Learning": cl_keep,
        "Graph-CoLD-soft": soft > 0.0,
        "CL+Graph-soft": cl_keep,
        "CL-issue-intersection": intersection_keep,
        "CL-issue-union": union_keep,
    }
    overlap = int(np.sum(cl_issue & graph_issue))
    union = int(np.sum(cl_issue | graph_issue))
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        pred = predictions[method]
        keep = keeps[method]
        issue = ~keep
        rows.append({
            "dataset": dataset_name,
            "protocol": PROTOCOLS[dataset_name]["label"],
            "seed": int(seed),
            "tail_flip_rate": float(rate),
            "effective_flip_rate": float(np.mean(flip)),
            "noise_target_class": int(sample.noise_target),
            "noise_target_name": sample.noise_target_name,
            "method": method,
            "macro_f1": macro_f1(sample.y_test, pred),
            "tail_macro_f1": p2e._tail_macro_f1(sample.y_test, pred, sample.tail),
            "tail_recall": p2e._tail_recall(sample.y_test, pred, sample.tail),
            "per_class_f1_json": json.dumps({
                str(int(label)): float(score)
                for label, score in zip(
                    np.unique(sample.y_test),
                    f1_score(sample.y_test, pred, labels=np.unique(sample.y_test), average=None, zero_division=0),
                )
            }, sort_keys=True),
            "retained_fraction": float(np.mean(keep)),
            "flag_precision_vs_injected": _safe_ratio(int(np.sum(issue & flip)), int(np.sum(issue))),
            "flag_recall_vs_injected": _safe_ratio(int(np.sum(issue & flip)), int(np.sum(flip))),
            "cl_issue_rate": float(np.mean(cl_issue)),
            "graph_issue_rate": float(np.mean(graph_issue)),
            "issue_overlap_jaccard": _safe_ratio(overlap, union),
            "n_train": int(sample.X_train.shape[0]),
            "n_test": int(sample.X_test.shape[0]),
            "split_notes": json.dumps(sample.split_notes, sort_keys=True, default=str),
            "cleanlab_engine": str(cl_result.details.get("issue_detector", "")),
            "cl_probability_protocol": str(cl_result.details.get("probability_protocol", "")),
        })
    return rows


def _noninferiority(frame: pd.DataFrame, method: str, reference: str, delta: float) -> dict[str, Any]:
    part = frame[frame.method.isin([method, reference])]
    pivot = part.pivot_table(index=["seed"], columns="method", values="macro_f1")
    pivot = pivot[[method, reference]].dropna()
    diff = (pivot[method] - pivot[reference]).to_numpy(dtype=float)
    if len(diff) < 2:
        return {"paired_rows": int(len(diff)), "mean_delta": float(np.mean(diff)) if len(diff) else None, "p_noninferior": 1.0, "passed": False}
    p_value = float(stats.ttest_1samp(diff, popmean=-delta, alternative="greater").pvalue)
    return {"paired_rows": int(len(diff)), "mean_delta": float(np.mean(diff)), "delta": float(delta), "p_noninferior": p_value, "passed": bool(np.mean(diff) > -delta and p_value < 0.05)}


def _class_guard(frame: pd.DataFrame, method: str, reference: str, limit: float) -> dict[str, Any]:
    part = frame[frame.method.isin([method, reference])]
    by_method: dict[str, dict[str, list[float]]] = {method: {}, reference: {}}
    for row in part.itertuples(index=False):
        values = json.loads(row.per_class_f1_json)
        for label, value in values.items():
            by_method[str(row.method)].setdefault(str(label), []).append(float(value))
    labels = sorted(set(by_method[method]) & set(by_method[reference]), key=int)
    regressions = {
        label: float(np.mean(by_method[method][label]) - np.mean(by_method[reference][label]))
        for label in labels
    }
    worst_label = min(regressions, key=regressions.get) if regressions else None
    worst_delta = regressions[worst_label] if worst_label is not None else None
    return {
        "limit": float(limit),
        "worst_class": worst_label,
        "worst_mean_delta": worst_delta,
        "passed": bool(worst_delta is not None and worst_delta >= -limit),
        "class_mean_deltas": regressions,
    }


def _mixed_effect_robustness(frame: pd.DataFrame, method: str, reference: str) -> dict[str, Any]:
    part = frame[frame.method.isin([method, reference])]
    pivot = part.pivot_table(index=["seed", "tail_flip_rate"], columns="method", values="tail_macro_f1")
    pivot = pivot[[method, reference]].dropna()
    data = (pivot[method] - pivot[reference]).rename("delta").reset_index()
    if data.seed.nunique() < 2 or data.tail_flip_rate.nunique() < 2:
        return {"available": False, "reason": "requires at least two seeds and two rates"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = smf.mixedlm("delta ~ C(tail_flip_rate)", data, groups=data["seed"]).fit(
            reml=True, method="powell", disp=False
        )
    contrast = np.zeros((1, len(fit.fe_params)), dtype=float)
    contrast[0, 0] = 1.0
    if contrast.shape[1] > 1:
        contrast[0, 1:] = 1.0 / float(data.tail_flip_rate.nunique())
    test = fit.t_test(contrast)
    estimate = float(np.asarray(test.effect).reshape(-1)[0])
    z_value = float(np.asarray(test.tvalue).reshape(-1)[0])
    p_two_sided = float(test.pvalue)
    p_greater = float(p_two_sided / 2.0 if z_value >= 0 else 1.0 - p_two_sided / 2.0)
    ci = np.asarray(test.conf_int(alpha=0.05)).reshape(-1)
    return {
        "available": True,
        "model": "tail_delta ~ C(tail_flip_rate) + (1|seed)",
        "paired_rows": int(len(data)),
        "seed_groups": int(data.seed.nunique()),
        "estimate_average_rate_delta": estimate,
        "z_value": z_value,
        "p_value_greater": p_greater,
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "random_intercept_variance": float(fit.cov_re.iloc[0, 0]),
        "residual_variance": float(fit.scale),
        "converged": bool(fit.converged),
        "warnings": [str(item.message) for item in caught],
    }


def _aggregate(frame: pd.DataFrame, mode: str) -> dict[str, Any]:
    means = frame.groupby(["dataset", "protocol", "method"])[["macro_f1", "tail_macro_f1", "tail_recall", "retained_fraction"]].mean().reset_index()
    family_b: list[dict[str, Any]] = []
    for dataset in sorted(frame.dataset.unique()):
        part = frame[frame.dataset == dataset]
        for rate in sorted(part.tail_flip_rate.unique()):
            rate_part = part[np.isclose(part.tail_flip_rate, rate)]
            for metric in ("macro_f1", "tail_macro_f1"):
                item = p2f._paired_stat(rate_part, PRIMARY_METHOD, REFERENCE_METHOD, metric)
                item.update({
                    "dataset": dataset,
                    "tail_flip_rate": float(rate),
                    "method": PRIMARY_METHOD,
                    "reference": REFERENCE_METHOD,
                    "metric": metric,
                })
                family_b.append(item)
    if len(family_b) > FAMILY_B_SIZE:
        raise AssertionError(f"Family B overflow: {len(family_b)} > {FAMILY_B_SIZE}")
    padded = [float(item["p_value_greater"]) for item in family_b] + [1.0] * (FAMILY_B_SIZE - len(family_b))
    adjusted = p2f._holm(padded)[:len(family_b)]
    for item, p_holm in zip(family_b, adjusted):
        item["p_holm_greater"] = float(p_holm)

    g2: dict[str, Any] = {}
    for dataset in sorted(frame.dataset.unique()):
        part = frame[frame.dataset == dataset]
        rate_results: list[dict[str, Any]] = []
        for rate in sorted(part.tail_flip_rate.unique()):
            rate_part = part[np.isclose(part.tail_flip_rate, rate)]
            primary_tail = next(
                item for item in family_b
                if item["dataset"] == dataset and item["tail_flip_rate"] == float(rate) and item["metric"] == "tail_macro_f1"
            )
            macro_test = next(
                item for item in family_b
                if item["dataset"] == dataset and item["tail_flip_rate"] == float(rate) and item["metric"] == "macro_f1"
            )
            ni = _noninferiority(rate_part, PRIMARY_METHOD, REFERENCE_METHOD, MACRO_NI_DELTA)
            guard = _class_guard(rate_part, PRIMARY_METHOD, REFERENCE_METHOD, CLASS_F1_GUARD)
            tail_pass = bool(
                primary_tail["mean_delta"] > 0
                and primary_tail["p_holm_greater"] < 0.05
                and primary_tail["cohens_dz"] >= 0.3
                and primary_tail["bootstrap_ci95_low"] > 0
            )
            rate_results.append({
                "tail_flip_rate": float(rate),
                "tail_primary": primary_tail,
                "macro_superiority_descriptive": macro_test,
                "macro_noninferiority": ni,
                "class_f1_guard": guard,
                "passed": bool(mode == "full" and tail_pass and ni["passed"] and guard["passed"]),
                "screening_direction_positive": bool(primary_tail["mean_delta"] > 0),
            })
        g2[dataset] = {
            "rates": rate_results,
            "passed": bool(any(item["passed"] for item in rate_results)),
            "screening_direction_positive": bool(any(item["screening_direction_positive"] for item in rate_results)),
            "mixed_effect_robustness": _mixed_effect_robustness(part, PRIMARY_METHOD, REFERENCE_METHOD),
        }
    pass_count = sum(bool(item["passed"]) for item in g2.values())
    verdict = "robust" if pass_count >= 2 else "narrow" if pass_count == 1 else "null"
    return {
        "means": means.to_dict(orient="records"),
        "family_b_tests": family_b,
        "family_b_size": FAMILY_B_SIZE,
        "g2_by_dataset": g2,
        "global_verdict": verdict,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# P0-4 CL + Graph-CDM {report['mode'].title()} Report",
        "",
        f"- Rows: {report['row_count']} / expected {report['expected_row_count']}",
        f"- Seeds: {report['seeds']}; rates: {report['rates']}",
        "- Primary G2 method: `CL+Graph-soft` vs `Confident-Learning`.",
        f"- Frozen family B size: {report['summary']['family_b_size']} rate-specific tests; unrun cells conservatively enter Holm as p=1.",
        "- Per-rate G2 pass requires positive Tail-F1, Holm p<0.05, dz>=0.3, CI lower bound >0, Macro-F1 non-inferiority at delta=0.01, and no class mean F1 loss beyond 0.05.",
        "",
        "## Mean metrics",
        "",
        "| dataset | protocol | method | macro F1 | tail F1 | tail recall | retained |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report["summary"]["means"]:
        lines.append(f"| {item['dataset']} | {item['protocol']} | {item['method']} | {item['macro_f1']:.4f} | {item['tail_macro_f1']:.4f} | {item['tail_recall']:.4f} | {item['retained_fraction']:.3f} |")
    lines += ["", "## G2 status", ""]
    lines.append(f"- **Global verdict: {report['summary']['global_verdict']}**")
    for dataset, item in report["summary"]["g2_by_dataset"].items():
        lines.append(f"- **{dataset}**: dataset pass={item['passed']}.")
        for rate_item in item["rates"]:
            tail = rate_item["tail_primary"]
            ni = rate_item["macro_noninferiority"]
            guard = rate_item["class_f1_guard"]
            lines.append(
                f"  - rate={rate_item['tail_flip_rate']:.1f}: tail delta {tail['mean_delta']:+.4f}, "
                f"Holm p={tail['p_holm_greater']:.3g}, dz={tail['cohens_dz']:.3f}, "
                f"CI=[{tail['bootstrap_ci95_low']:+.4f}, {tail['bootstrap_ci95_high']:+.4f}]; "
                f"macro delta {ni['mean_delta']:+.4f}, NI p={ni['p_noninferior']:.3g}; "
                f"worst class delta {guard['worst_mean_delta']:+.4f}; pass={rate_item['passed']}."
            )
        mixed = item["mixed_effect_robustness"]
        if mixed["available"]:
            lines.append(
                f"  - Mixed-effect robustness: average tail delta {mixed['estimate_average_rate_delta']:+.4f}, "
                f"one-sided p={mixed['p_value_greater']:.3g}, CI=[{mixed['ci95_low']:+.4f}, {mixed['ci95_high']:+.4f}], "
                f"converged={mixed['converged']}; warnings={len(mixed['warnings'])}."
            )
        else:
            lines.append(f"  - Mixed-effect robustness unavailable: {mixed['reason']}.")
    lines += ["", "## Protocol limits", ""]
    for dataset in report["datasets"]:
        protocol = PROTOCOLS[dataset]
        lines.append(f"- **{dataset}** — {protocol['label']}; limitation: {protocol['limitation']}.")
    lines.append("")
    return "\n".join(lines)


def run(
    mode: str,
    configs_dir: str | Path = "configs",
    out_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
    datasets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    seeds = SCREEN_SEEDS if mode == "screening" else FULL_SEEDS
    rates = SCREEN_RATES if mode == "screening" else FULL_RATES
    selected = tuple(DATASETS if datasets is None else datasets)
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    if not selected:
        raise ValueError("At least one dataset must be selected")
    configs = Path(configs_dir)
    out = Path(out_dir)
    reports = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    suffix = mode if selected == tuple(DATASETS) else f"{mode}_{'_'.join(selected)}"
    partial = out / f"p4_cl_graph_combo_{suffix}.partial.csv"
    rows: list[dict[str, Any]] = []
    done: set[tuple[str, int, float]] = set()
    if partial.exists():
        existing = pd.read_csv(partial)
        if "per_class_f1_json" in existing.columns:
            rows = existing.to_dict(orient="records")
            counts = existing.groupby(["dataset", "seed", "tail_flip_rate"])["method"].nunique()
            done = {(str(key[0]), int(key[1]), float(key[2])) for key, count in counts.items() if int(count) == len(METHODS)}
    for dataset in selected:
        ds = _load_primary_dataset(dataset, configs)
        for seed in seeds:
            for rate in rates:
                key = (dataset, int(seed), float(rate))
                if key in done:
                    continue
                print(f"[p4] {mode} {dataset} seed={seed} rate={rate}", flush=True)
                rows.extend(_scenario_rows(dataset, ds, seed, rate))
                pd.DataFrame(rows).to_csv(partial, index=False)
    frame = pd.DataFrame(rows)
    final_csv = out / f"p4_cl_graph_combo_{suffix}.csv"
    frame.to_csv(final_csv, index=False)
    expected = len(selected) * len(seeds) * len(rates) * len(METHODS)
    report = {
        "stage": "P0-4",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "mode": mode,
        "completed": int(len(frame)) == expected,
        "row_count": int(len(frame)),
        "expected_row_count": int(expected),
        "seeds": list(seeds),
        "rates": list(rates),
        "methods": list(METHODS),
        "datasets": list(selected),
        "cleanlab_version": importlib.metadata.version("cleanlab"),
        "cl_probability_protocol": "full_oof_stratified_cv",
        "protocols": PROTOCOLS,
        "summary": _aggregate(frame, mode),
        "outputs": {"results": str(final_csv).replace("\\", "/"), "partial": str(partial).replace("\\", "/")},
    }
    json_path = reports / f"p4_cl_graph_combo_{suffix}.json"
    md_path = reports / f"p4_cl_graph_combo_{suffix}.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-4 CL + Graph-CDM combination experiment")
    parser.add_argument("--mode", choices=("screening", "full"), default="screening")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--datasets", default=None, help="Comma-separated survivor datasets; default: all")
    args = parser.parse_args()
    selected = None if args.datasets is None else tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    report = run(args.mode, args.configs, args.out, args.reports, selected)
    print(json.dumps({"completed": report["completed"], "row_count": report["row_count"], "g2": report["summary"]["g2_by_dataset"]}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
