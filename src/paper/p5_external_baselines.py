"""P5 external-baseline supplement under the exact P2f clean protocol.

Runs two external noisy-label anchors — Confident Learning (cleanlab) and full
Co-Teaching (dual-MLP, small-loss co-exchange) — through the identical P2f
scenario pipeline: same dataset contracts, same deterministic audit windows
(train 10,000 / test 5,000), same tail-asymmetric noise injection at
{40, 60, 80}% with seeds 0-9, and the same metric functions.

Integrity rules:
- never modifies any pre-existing results file; all new output goes to
  ``results/p5_external_baselines.csv`` (plus tables/reports p5_* files);
- reports failures honestly instead of fabricating numbers.
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.confident_learning import ConfidentLearningBaseline
from src.baselines.coteaching_full import CoTeachingFullBaseline
from src.experiments import d5
from src.metrics import false_negative_rate, false_positive_rate, macro_f1
from src.paper import p2d_clean_rerun as p2d
from src.paper import p2e_salvage as p2e
from src.paper import p2f_tighten as p2f

DATASETS = p2f.DATASETS
RATES = p2f.RATES
SEEDS = p2f.SEEDS
DEFAULT_TRAIN_SIZE = p2f.DEFAULT_TRAIN_SIZE
DEFAULT_TEST_SIZE = p2f.DEFAULT_TEST_SIZE
METHODS = ("Confident-Learning", "Co-Teaching")
REFERENCE_METHOD = "Graph-CoLD-soft"
METRICS = ("macro_f1", "tail_macro_f1", "tail_recall")


def _baseline_result(method: str, sample: Any, noisy: np.ndarray, effective_rate: float, seed: int):
    if method == "Confident-Learning":
        baseline = ConfidentLearningBaseline(
            seed=seed,
            noise_rate=effective_rate,
            n_estimators=24,  # match the P2f shared downstream budget
        )
    elif method == "Co-Teaching":
        baseline = CoTeachingFullBaseline(seed=seed, noise_rate=effective_rate)
    else:  # pragma: no cover - guarded by METHODS
        raise ValueError(f"Unknown P5 external baseline: {method}")
    return baseline.fit_predict(
        sample.X_train,
        noisy,
        sample.X_test,
        sample.num_classes,
    )


def _row(
    sample: Any,
    method: str,
    rate: float,
    seed: int,
    noisy: np.ndarray,
    flip: np.ndarray,
    tail_labels: np.ndarray,
    result: Any,
    runtime_sec: float,
    memory_mb: float,
) -> dict[str, Any]:
    y_test_pred = np.asarray(result.y_pred, dtype=np.int64)
    return {
        "dataset": sample.dataset_key,
        "reported_as": sample.reported_as,
        "dataset_hash": sample.dataset_hash,
        "actual_data_path": sample.actual_data_path,
        "class_policy": sample.class_policy,
        "sample_policy": sample.sample_policy,
        "train_rows": int(sample.X_train.shape[0]),
        "test_rows": int(sample.X_test.shape[0]),
        "num_classes": int(sample.num_classes),
        "noise_type": "tail_asymmetric",
        "tail_flip_rate": float(rate),
        "effective_flip_rate_train": float(np.mean(flip)),
        "seed": int(seed),
        "method": method,
        "training_mode": str(result.details.get("classifier", "")),
        "macro_f1": macro_f1(sample.y_test, y_test_pred),
        "fpr": false_positive_rate(sample.y_test, y_test_pred, sample.benign_class),
        "fnr": false_negative_rate(sample.y_test, y_test_pred, sample.benign_class),
        "tail_macro_f1": p2e._tail_macro_f1(sample.y_test, y_test_pred, tail_labels),
        "tail_recall": p2e._tail_recall(sample.y_test, y_test_pred, tail_labels),
        "retained_fraction": float(result.details.get("retained_fraction", np.nan)),
        "runtime_sec": float(runtime_sec),
        "memory_mb": float(memory_mb),
        "implementation_status": str(result.implementation_status),
        "engine_detail": json.dumps(result.details, sort_keys=True, default=str),
        "active_views": sample.active_views,
        "source_verified": sample.source_verified,
        "replacement_for": sample.replacement_for,
    }


def _run_matrix(
    configs: Path,
    reports: Path,
    train_size: int,
    test_size: int,
    partial_path: Path,
    only_datasets: tuple[str, ...] = DATASETS,
    only_seeds: tuple[int, ...] = SEEDS,
    only_rates: tuple[float, ...] = RATES,
) -> pd.DataFrame:
    scale = d5.write_scale_policy_report(reports)
    rows: list[dict[str, Any]] = []
    done: set[tuple[str, float, int, str]] = set()
    if partial_path.exists():
        existing = pd.read_csv(partial_path, keep_default_na=False)
        rows = existing.to_dict(orient="records")
        done = {
            (str(r["dataset"]), float(r["tail_flip_rate"]), int(r["seed"]), str(r["method"]))
            for r in rows
        }
        print(f"[p5] resume: {len(done)} scenario-method rows already present", flush=True)
    failures: list[dict[str, Any]] = []
    failures_path = partial_path.parent / "p5_external_baselines.failures.json"
    if failures_path.exists():
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
    for dataset_name in only_datasets:
        for seed in only_seeds:
            sample = p2d._sample_bundle(dataset_name, seed, configs, reports, scale, train_size, test_size)
            tail_labels = p2e._tail_malicious_labels(sample.y_train, sample.benign_class)
            for rate in only_rates:
                noisy, flip = p2e._inject_tail_asymmetric(
                    sample.y_train, tail_labels, sample.benign_class, rate, seed=seed
                )
                effective_rate = float(np.mean(flip))
                for method in METHODS:
                    key = (dataset_name, float(rate), int(seed), method)
                    if key in done:
                        continue
                    print(f"[p5] {dataset_name} seed={seed} tail_asymmetric={rate:.0%} {method}", flush=True)
                    start = time.perf_counter()
                    tracemalloc.start()
                    try:
                        result = _baseline_result(method, sample, noisy, effective_rate, seed)
                        _, peak = tracemalloc.get_traced_memory()
                        row = _row(
                            sample,
                            method,
                            rate,
                            seed,
                            noisy,
                            flip,
                            tail_labels,
                            result,
                            time.perf_counter() - start,
                            peak / (1024 * 1024),
                        )
                        rows.append(row)
                        done.add(key)
                    except Exception as exc:  # honest failure logging, never fabricate
                        tracemalloc.stop()
                        failure = {
                            "dataset": dataset_name,
                            "tail_flip_rate": float(rate),
                            "seed": int(seed),
                            "method": method,
                            "error": repr(exc),
                        }
                        failures.append(failure)
                        print(f"[p5][FAILURE] {failure}", flush=True)
                    finally:
                        if tracemalloc.is_tracing():
                            tracemalloc.stop()
                    partial_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(partial_path, index=False)
                    (partial_path.parent / "p5_external_baselines.failures.json").write_text(
                        json.dumps(failures, indent=2), encoding="utf-8"
                    )
    return pd.DataFrame(rows)


def _paired_tests(frame: pd.DataFrame, reference: pd.DataFrame) -> list[dict[str, Any]]:
    """Scenario-paired tests of each external baseline vs Graph-CoLD-soft."""
    tests: list[dict[str, Any]] = []
    for dataset in sorted(frame["reported_as"].unique()):
        part_new = frame[frame["reported_as"] == dataset]
        part_ref = reference[reference["reported_as"] == dataset]
        combined = pd.concat([part_new, part_ref], ignore_index=True)
        dataset_tests: list[dict[str, Any]] = []
        for method in METHODS:
            for metric in METRICS:
                stat = p2f._paired_stat(combined, method, REFERENCE_METHOD, metric)
                stat.update({"dataset": dataset, "method": method, "metric": metric})
                dataset_tests.append(stat)
        adjusted = p2f._holm([item["p_value_greater"] for item in dataset_tests])
        for item, p_holm in zip(dataset_tests, adjusted):
            item["p_holm"] = p_holm
            item["external_beats_soft"] = bool(item["mean_delta"] > 0.0 and item["p_holm"] < 0.05)
            item["soft_beats_external"] = bool(item["mean_delta"] < 0.0 and item["p_holm"] < 0.05)
            tests.append(item)
    return tests


def _tail_by_rate(reference: pd.DataFrame) -> pd.DataFrame:
    """A5: presentation-only per-noise-rate tail-F1 view of the frozen P2f runs."""
    grouped = (
        reference.groupby(["reported_as", "method", "tail_flip_rate"], dropna=False)["tail_macro_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "tail_macro_f1_mean", "std": "tail_macro_f1_std", "count": "n_seeds"})
    )
    return grouped


def run_p5_external_baselines(
    configs_dir: str | Path = "configs",
    out_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
    tables_dir: str | Path = "tables",
    train_size: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    only_datasets: tuple[str, ...] = DATASETS,
    only_seeds: tuple[int, ...] = SEEDS,
    only_rates: tuple[float, ...] = RATES,
    finalize_only: bool = False,
) -> dict[str, Any]:
    configs = Path(configs_dir)
    out = Path(out_dir)
    reports = Path(reports_dir)
    tables = Path(tables_dir)
    for directory in (out, reports, tables):
        directory.mkdir(parents=True, exist_ok=True)

    reference_path = out / "p2f_tail_powered.csv"
    if not reference_path.exists():
        raise FileNotFoundError("results/p2f_tail_powered.csv (frozen P2f matrix) is required for pairing.")
    reference = pd.read_csv(reference_path, keep_default_na=False)
    reference = reference[reference["method"] == REFERENCE_METHOD].copy()

    partial_path = out / "p5_external_baselines.partial.csv"
    if not finalize_only:
        frame = _run_matrix(configs, reports, train_size, test_size, partial_path, only_datasets, only_seeds, only_rates)
        frame.to_csv(out / "p5_external_baselines.csv", index=False)
    else:
        if not partial_path.exists() and not (out / "p5_external_baselines.csv").exists():
            raise FileNotFoundError("No P5 partial or final results found to finalize.")
        source = partial_path if partial_path.exists() else out / "p5_external_baselines.csv"
        frame = pd.read_csv(source, keep_default_na=False)
        frame.to_csv(out / "p5_external_baselines.csv", index=False)

    expected = int(len(DATASETS) * len(SEEDS) * len(RATES) * len(METHODS))
    if len(frame) < expected:
        print(f"[p5] partial run: {len(frame)}/{expected} rows; aggregation deferred until complete.", flush=True)
        return {"stage": "P5-A", "completed": False, "row_count": int(len(frame)), "expected_row_count": expected}

    failures_path = out / "p5_external_baselines.failures.json"
    failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.exists() else []

    tests = _paired_tests(frame, reference)
    pd.DataFrame(tests).to_csv(tables / "table_p5_external_tests.csv", index=False)

    tail_by_rate = _tail_by_rate(pd.read_csv(reference_path, keep_default_na=False))
    tail_by_rate.to_csv(tables / "table_p5_tail_by_rate.csv", index=False)

    hash_check = {
        dataset: sorted(set(frame[frame["reported_as"] == dataset]["dataset_hash"]))
        == sorted(set(reference[reference["reported_as"] == dataset]["dataset_hash"]))
        for dataset in sorted(frame["reported_as"].unique())
    }

    summary = (
        frame.groupby(["reported_as", "method"], dropna=False)[list(METRICS)]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    report = {
        "stage": "P5-A",
        "completed": True,
        "scope": {
            "datasets": [p2d.DATASET_LABELS[name] for name in DATASETS],
            "methods": list(METHODS),
            "reference_method": REFERENCE_METHOD,
            "rates": [float(r) for r in RATES],
            "seeds": list(SEEDS),
            "train_size": int(train_size),
            "test_size": int(test_size),
            "protocol": "identical P2f clean de-oracled scenario pipeline (p2d._sample_bundle + p2e._inject_tail_asymmetric)",
        },
        "row_count": int(len(frame)),
        "expected_row_count": int(len(DATASETS) * len(SEEDS) * len(RATES) * len(METHODS)),
        "failures": failures,
        "dataset_hash_matches_frozen_p2f": hash_check,
        "summary_means": summary,
        "paired_tests_vs_graph_cold_soft": tests,
        "outputs": {
            "results": "results/p5_external_baselines.csv",
            "tests": "tables/table_p5_external_tests.csv",
            "tail_by_rate": "tables/table_p5_tail_by_rate.csv",
        },
        "reproduction_commands": [
            "python -m src.paper.p5_external_baselines --configs configs --out results --reports reports --tables tables"
        ],
    }
    (reports / "p5_external_baselines.json").write_text(
        json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    (reports / "p5_external_baselines.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P5 External Baselines (Confident Learning + full Co-Teaching) under the P2f Protocol",
        "",
        f"- Rows: {report['row_count']} / expected {report['expected_row_count']}",
        f"- Failures: {len(report['failures'])}",
        f"- Dataset hashes match frozen P2f: {report['dataset_hash_matches_frozen_p2f']}",
        "",
        "## Mean metrics (over seeds x rates)",
        "",
        "| dataset | method | macro_f1 | tail_macro_f1 | tail_recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["summary_means"]:
        lines.append(
            f"| {item['reported_as']} | {item['method']} | {item['macro_f1']:.4f} "
            f"| {item['tail_macro_f1']:.4f} | {item['tail_recall']:.4f} |"
        )
    lines += [
        "",
        "## Scenario-paired tests vs Graph-CoLD-soft (Holm-corrected per dataset)",
        "",
        "| dataset | method | metric | mean delta | p_holm | cohens_dz | CI95 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["paired_tests_vs_graph_cold_soft"]:
        lines.append(
            f"| {item['dataset']} | {item['method']} | {item['metric']} | {item['mean_delta']:+.4f} "
            f"| {item['p_holm']:.2e} | {item['cohens_dz']:+.2f} "
            f"| [{item['bootstrap_ci95_low']:+.4f}, {item['bootstrap_ci95_high']:+.4f}] |"
        )
    if report["failures"]:
        lines += ["", "## Failures (reported honestly, no substitution)", ""]
        for failure in report["failures"]:
            lines.append(f"- {failure}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 external baselines under the P2f protocol.")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--datasets", default=None, help="Comma-separated subset of dataset keys.")
    parser.add_argument("--seeds", default=None, help="Comma-separated subset of seeds.")
    parser.add_argument("--rates", default=None, help="Comma-separated subset of tail flip rates.")
    parser.add_argument("--finalize-only", action="store_true", help="Aggregate existing partial results only.")
    args = parser.parse_args()
    only_datasets = tuple(args.datasets.split(",")) if args.datasets else DATASETS
    only_seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else SEEDS
    only_rates = tuple(float(r) for r in args.rates.split(",")) if args.rates else RATES
    run_p5_external_baselines(
        configs_dir=args.configs,
        out_dir=args.out,
        reports_dir=args.reports,
        tables_dir=args.tables,
        train_size=args.train_size,
        test_size=args.test_size,
        only_datasets=only_datasets,
        only_seeds=only_seeds,
        only_rates=only_rates,
        finalize_only=args.finalize_only,
    )


if __name__ == "__main__":
    main()
