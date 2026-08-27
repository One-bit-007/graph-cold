"""P5b / P0-5 companion: per-class F1 guard for family A (#4 soft vs #3 hard).

The p5 full-tier CSV stores aggregate metrics only. F-6.3 requires the
per-class regression guard (any single-class F1 regression <= 0.05, paired
mean) before a non-inferiority statement may stand. This companion recomputes
the #3/#4 cell predictions on the frozen grid (10 seeds x 3 rates x 3
datasets), stores per-class F1 in a NEW companion CSV (existing p5 artifacts
untouched), and evaluates the guard per dataset x rate.

Note: predictions are deterministic modulo the documented n_jobs=-1 runtime
float noise (~1e-4 macro-F1), far below the 0.05 guard threshold.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.paper import p4_cl_graph_combo as p4
from src.paper import p5_factor_ablation as p5


CELLS = ("3_graph_cdm_hard", "4_graph_cdm_soft")
GUARD_LIMIT = 0.05


def run(
    configs_dir: str | Path = "configs",
    out_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
    tables_dir: str | Path = "tables",
    datasets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    configs, out, reports, tables = Path(configs_dir), Path(out_dir), Path(reports_dir), Path(tables_dir)
    for d in (out, reports, tables):
        d.mkdir(parents=True, exist_ok=True)
    selected = tuple(p5.DATASETS if datasets is None else datasets)

    partial = out / "p5_perclass_guard.partial.csv"
    rows: list[dict[str, Any]] = []
    done: set[tuple[str, int, float]] = set()
    if partial.exists():
        existing = pd.read_csv(partial)
        rows = existing.to_dict(orient="records")
        counts = existing.groupby(["dataset", "seed", "tail_flip_rate"])["cell"].nunique()
        done = {(str(k[0]), int(k[1]), float(k[2])) for k, c in counts.items() if int(c) == len(CELLS)}

    for dataset in selected:
        ds = p4._load_primary_dataset(dataset, configs)
        for seed in p5.FULL_SEEDS:
            for rate in p5.FULL_RATES:
                if (dataset, seed, rate) in done:
                    continue
                t0 = time.time()
                scenario = p5._prepare_scenario(ds, seed, rate)
                for cell in CELLS:
                    weights, train_labels, fit_mode = p5._cell_weights(cell, scenario)
                    y_pred = p5._fit_predict_cell(
                        scenario["X_train"], train_labels, scenario["X_test"], weights, fit_mode, seed + 7
                    )
                    y_test = scenario["y_test"]
                    rows.append({
                        "dataset": dataset, "cell": cell, "seed": seed, "tail_flip_rate": rate,
                        "per_class_f1_json": json.dumps({
                            str(int(label)): float(score)
                            for label, score in zip(
                                np.unique(y_test),
                                f1_score(y_test, y_pred, labels=np.unique(y_test), average=None, zero_division=0),
                            )
                        }, sort_keys=True),
                    })
                print(f"[p5b] {dataset} seed={seed} rate={rate} done in {time.time()-t0:.1f}s", flush=True)
                pd.DataFrame(rows).to_csv(partial, index=False)

    frame = pd.DataFrame(rows).rename(columns={"cell": "method"})
    final_csv = out / "p5_perclass_guard.csv"
    frame.to_csv(final_csv, index=False)

    guards: list[dict[str, Any]] = []
    for dataset in selected:
        part = frame[frame.dataset == dataset]
        for rate in p5.FULL_RATES:
            rate_part = part[np.isclose(part.tail_flip_rate, rate)]
            guard = p4._class_guard(rate_part, "4_graph_cdm_soft", "3_graph_cdm_hard", GUARD_LIMIT)
            guard.update({"dataset": dataset, "tail_flip_rate": float(rate)})
            guards.append(guard)
    guard_frame = pd.DataFrame(guards)
    guard_frame.to_csv(tables / "table_p5_perclass_guard.csv", index=False)

    report = {
        "stage": "P5b-P0-5-perclass-guard",
        "row_count": int(len(frame)),
        "expected_row_count": len(selected) * len(p5.FULL_SEEDS) * len(p5.FULL_RATES) * len(CELLS),
        "completed": int(len(frame)) == len(selected) * len(p5.FULL_SEEDS) * len(p5.FULL_RATES) * len(CELLS),
        "guard_limit": GUARD_LIMIT,
        "guards": guards,
        "all_guards_pass": bool(all(g.get("passed", False) for g in guards)),
        "outputs": {"results": str(final_csv).replace("\\", "/"),
                    "guards": str(tables / "table_p5_perclass_guard.csv").replace("\\", "/")},
    }
    (reports / "p5_perclass_guard.json").write_text(
        json.dumps(p5._jsonable(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="P5b per-class guard for family A")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--datasets", default=None, help="Comma-separated; default: all")
    args = parser.parse_args()
    selected = None if args.datasets is None else tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    report = run(args.configs, args.out, args.reports, args.tables, selected)
    print(json.dumps({"completed": report["completed"], "all_guards_pass": report["all_guards_pass"]},
                     indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
