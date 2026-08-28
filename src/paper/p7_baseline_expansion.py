"""P7 (P1-2 + P1-3): FINE formal inclusion + Co-Teaching F-5 fix, frozen protocol.

Runs FINE and Co-Teaching through the identical P2f scenario pipeline used by
p5_external_baselines (same windows, noise injection, metrics), but with the
F-5 fix: both baselines receive ``noise_rate=None`` and estimate the noise rate
from the noisy training set only (src/baselines/noise_estimate.py). The true
injected rate never reaches either baseline.

Integrity rules:
- pre-existing result files (p2f/p5) are never modified; all output goes to
  results/p7_baseline_expansion(.partial).csv and reports/p7_baseline_expansion.json;
- pairing vs frozen Graph-CoLD-soft / Confident-Learning references is
  DESCRIPTIVE (delta + bootstrap CI); no new inferential family per F-6.
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

from src.baselines.coteaching_full import CoTeachingFullBaseline
from src.baselines.fine_style import FINEBaseline
from src.paper import p2d_clean_rerun as p2d
from src.paper import p2e_salvage as p2e
from src.paper import p2f_tighten as p2f
from src.paper import p5_external_baselines as p5

DATASETS = p2f.DATASETS
RATES = p2f.RATES
SEEDS = p2f.SEEDS
METHODS = ("FINE", "Co-Teaching")
PARTIAL = Path("results/p7_baseline_expansion.partial.csv")
OUT_CSV = Path("results/p7_baseline_expansion.csv")
OUT_JSON = Path("reports/p7_baseline_expansion.json")
FAILURES = Path("results/p7_baseline_expansion.failures.json")


def _baseline_result(method: str, sample: Any, noisy: np.ndarray, seed: int):
    """noise_rate=None everywhere: F-5 compliant estimation inside the baseline."""
    if method == "FINE":
        baseline = FINEBaseline(seed=seed, noise_rate=None)
    elif method == "Co-Teaching":
        baseline = CoTeachingFullBaseline(seed=seed, noise_rate=None)
    else:  # pragma: no cover - guarded by METHODS
        raise ValueError(f"Unknown P7 baseline: {method}")
    return baseline.fit_predict(sample.X_train, noisy, sample.X_test, sample.num_classes)


def _run_matrix(configs: Path, reports: Path, train_size: int, test_size: int,
                only_datasets, only_seeds, only_rates) -> pd.DataFrame:
    from src.experiments import d5

    rows: list[dict[str, Any]] = []
    scale = d5.write_scale_policy_report(reports)
    done: set[tuple[str, float, int, str]] = set()
    if PARTIAL.exists():
        existing = pd.read_csv(PARTIAL, keep_default_na=False)
        rows = existing.to_dict(orient="records")
        done = {(str(r["dataset"]), float(r["tail_flip_rate"]), int(r["seed"]), str(r["method"])) for r in rows}
        print(f"[p7] resume: {len(done)} scenario-method rows already present", flush=True)
    failures: list[dict[str, Any]] = []
    if FAILURES.exists():
        failures = json.loads(FAILURES.read_text(encoding="utf-8"))
    for dataset_name in only_datasets:
        for seed in only_seeds:
            # Efficiency only: skip the expensive sample load when every
            # (rate, method) cell for this seed is already in the partial CSV.
            if all((dataset_name, float(rate), int(seed), method) in done
                   for rate in only_rates for method in METHODS):
                continue
            sample = p2d._sample_bundle(dataset_name, seed, configs, reports, scale, train_size, test_size)
            tail_labels = p2e._tail_malicious_labels(sample.y_train, sample.benign_class)
            for rate in only_rates:
                noisy, flip = p2e._inject_tail_asymmetric(
                    sample.y_train, tail_labels, sample.benign_class, rate, seed=seed
                )
                for method in METHODS:
                    key = (dataset_name, float(rate), int(seed), method)
                    if key in done:
                        continue
                    print(f"[p7] {dataset_name} seed={seed} rate={rate:.0%} {method}", flush=True)
                    start = time.perf_counter()
                    tracemalloc.start()
                    try:
                        result = _baseline_result(method, sample, noisy, seed)
                        _, peak = tracemalloc.get_traced_memory()
                        row = p5._row(sample, method, rate, seed, noisy, flip, tail_labels, result,
                                      time.perf_counter() - start, peak / (1024 * 1024))
                        row["noise_rate_source"] = str(result.details.get("noise_rate_source", ""))
                        rows.append(row)
                        done.add(key)
                    except Exception as exc:  # honest failure logging, never fabricate
                        failures.append({"dataset": dataset_name, "tail_flip_rate": float(rate),
                                         "seed": int(seed), "method": method, "error": repr(exc)})
                        print(f"[p7][FAILURE] {failures[-1]}", flush=True)
                    finally:
                        if tracemalloc.is_tracing():
                            tracemalloc.stop()
                    PARTIAL.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(PARTIAL, index=False)
                    FAILURES.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return pd.DataFrame(rows)


def _paired_delta(new: pd.DataFrame, ref: pd.DataFrame, method: str, reference: str, metric: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    combined = pd.concat([new, ref], ignore_index=True)
    for dataset in sorted(combined["reported_as"].unique()):
        for rate in sorted(combined["tail_flip_rate"].unique()):
            part = combined[(combined["reported_as"] == dataset) & (combined["tail_flip_rate"] == rate)]
            pivot = part.pivot_table(index="seed", columns="method", values=metric, aggfunc="mean")
            if method not in pivot.columns or reference not in pivot.columns:
                continue
            joined = pivot[[method, reference]].dropna()
            if joined.empty:
                continue
            diff = (joined[method] - joined[reference]).to_numpy(dtype=float)
            lo, hi = p2f._bootstrap_ci(diff)
            out.append({"dataset": dataset, "rate": float(rate), "metric": metric,
                        "method": method, "reference": reference, "paired_seeds": int(len(diff)),
                        "mean_method": float(joined[method].mean()), "mean_reference": float(joined[reference].mean()),
                        "mean_delta": float(np.mean(diff)),
                        "bootstrap_ci95_low": float(lo), "bootstrap_ci95_high": float(hi)})
    return out


def run(only_datasets=DATASETS, only_seeds=SEEDS, only_rates=RATES,
        configs_dir="configs", reports_dir="reports", finalize_only: bool = False,
        shard: str = "") -> dict[str, Any]:
    configs, reports = Path(configs_dir), Path(reports_dir)
    if not finalize_only:
        frame = _run_matrix(configs, reports, p2f.DEFAULT_TRAIN_SIZE, p2f.DEFAULT_TEST_SIZE,
                            only_datasets, only_seeds, only_rates)
        if shard:
            print(f"[p7] shard={shard}: {len(frame)} rows; aggregation deferred.", flush=True)
            return {"rows": int(len(frame)), "shard": shard, "complete": False}
        frame.to_csv(OUT_CSV, index=False)
    else:
        source = PARTIAL if PARTIAL.exists() else OUT_CSV
        frame = pd.read_csv(source, keep_default_na=False)
        frame.to_csv(OUT_CSV, index=False)

    expected = int(len(DATASETS) * len(SEEDS) * len(RATES) * len(METHODS))
    if len(frame) < expected:
        print(f"[p7] partial run: {len(frame)}/{expected} rows; aggregation deferred.", flush=True)
        return {"rows": int(len(frame)), "expected": expected, "complete": False}

    # Descriptive pairing against frozen references (no new inferential family).
    ref_soft = pd.read_csv("results/p2f_tail_powered.csv", keep_default_na=False)
    ref_soft = ref_soft[ref_soft["method"] == "Graph-CoLD-soft"]
    ref_cl = pd.read_csv("results/p5_external_baselines.csv", keep_default_na=False)
    ref_cl = ref_cl[ref_cl["method"] == "Confident-Learning"]
    means = (frame.groupby(["reported_as", "tail_flip_rate", "method"])[["macro_f1", "tail_macro_f1"]]
             .mean().round(4).reset_index())
    summary = {
        "complete": True,
        "rows": int(len(frame)),
        "f5_compliance": "noise_rate=None for every run; estimated from noisy train only",
        "noise_rate_sources": sorted(frame["noise_rate_source"].unique().tolist()),
        "per_dataset_rate_means": means.to_dict("records"),
        "delta_vs_soft": _paired_delta(frame, ref_soft, "FINE", "Graph-CoLD-soft", "tail_macro_f1")
        + _paired_delta(frame, ref_soft, "Co-Teaching", "Graph-CoLD-soft", "tail_macro_f1"),
        "delta_vs_cl": _paired_delta(frame, ref_cl, "FINE", "Confident-Learning", "tail_macro_f1")
        + _paired_delta(frame, ref_cl, "Co-Teaching", "Confident-Learning", "tail_macro_f1"),
        "delta_vs_soft_macro": _paired_delta(frame, ref_soft, "FINE", "Graph-CoLD-soft", "macro_f1")
        + _paired_delta(frame, ref_soft, "Co-Teaching", "Graph-CoLD-soft", "macro_f1"),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rows": summary["rows"], "sources": summary["noise_rate_sources"]}, indent=2))
    return summary


def main() -> None:
    global PARTIAL, FAILURES
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--rates", default=",".join(str(r) for r in RATES))
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--shard", default="",
                        help="shard name: writes results/p7_baseline_expansion.<shard>.partial.csv "
                             "and skips aggregation so multiple shards can run in parallel")
    args = parser.parse_args()
    if args.shard:
        PARTIAL = Path(f"results/p7_baseline_expansion.{args.shard}.partial.csv")
        FAILURES = Path(f"results/p7_baseline_expansion.{args.shard}.failures.json")
    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = tuple(range(int(lo), int(hi) + 1))
    else:
        seeds = tuple(int(s) for s in args.seeds.split(","))
    run(only_datasets=tuple(args.datasets.split(",")), only_seeds=seeds,
        only_rates=tuple(float(r) for r in args.rates.split(",")), finalize_only=args.finalize_only,
        shard=args.shard)


if __name__ == "__main__":
    main()
