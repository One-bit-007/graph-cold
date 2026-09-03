"""P10 staged UNSW-only rerun; immutable history, full-family stop gates."""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from src.paper import p3_proxy_audit as p3, p4_cl_graph_combo as p4
from src.paper import p5_factor_ablation as p5, p6_leakage_ablation as p6
from src.paper.p9_tail_metric_unification import array_hash
from src.paper.p10_support import ROOT, assert_preserved, sha256

OUT = ROOT / "results/p10"
REPORTS = ROOT / "reports/p10"
TABLES = ROOT / "tables/p10"
DATASET = "unsw_nb15"
KEY = ["dataset", "seed", "tail_flip_rate"]
AC_SOURCE = ROOT / "results/p5_factor_ablation_full.csv"
GUARD_SOURCE = ROOT / "results/p5_perclass_guard.csv"
B_SOURCE = ROOT / "results/p4_cl_graph_combo_full_family_b_complete.csv"


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p5._jsonable(data), indent=2, allow_nan=False), encoding="utf-8")


def contract():
    assert_preserved()
    for d in (OUT, REPORTS, TABLES):
        d.mkdir(parents=True, exist_ok=True)
    inputs = ["configs/datasets.yaml", "src/data/loaders.py", "src/metrics_canonical.py",
              "src/paper/p3_proxy_audit.py", "src/paper/p4_cl_graph_combo.py",
              "src/paper/p5_factor_ablation.py", "src/paper/p6_leakage_ablation.py",
              "src/paper/p10_unsw_rerun.py", "src/paper/p10_support.py",
              "reports/p10_feature_audit.json", "reports/p9_tail_metric_unification.json"]
    value = {"schema": 1, "submission_ready": False, "datasets_rerun": [DATASET],
             "seeds": list(p5.FULL_SEEDS), "rates": list(p5.FULL_RATES),
             "tail_metric": "canonical full-test per-class F1 averaged over frozen tail classes",
             "label_exclusion": "permanent, including PRX/ALL", "input_sha256": {x: sha256(ROOT/x) for x in inputs}}
    path = REPORTS / "run_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError("STOP: run contract changed; do not reuse cached rows")
    else:
        dump(path, value)


def validate_grid(frame, methods, *, datasets=(DATASET,), method_col="cell"):
    keys = KEY + [method_col]
    expected = set(itertools.product(datasets, p5.FULL_SEEDS, p5.FULL_RATES, methods))
    actual = set(frame[keys].itertuples(index=False, name=None))
    if frame.duplicated(keys).any() or actual != expected:
        raise AssertionError(f"Incomplete/duplicate grid: expected={len(expected)}, actual={len(actual)}")


def merge_unsw(source, replacement, target):
    """Copy each non-UNSW CSV record as original bytes; serialize new UNSW only."""
    raw = source.read_bytes().splitlines(keepends=True)
    header = next(csv.reader([raw[0].decode("utf-8-sig")]))
    kept = []
    for line in raw[1:]:
        values = next(csv.reader([line.decode("utf-8")]))
        if len(values) != len(header):
            raise AssertionError("Unexpected multiline/invalid CSV; refusing lossy merge")
        if dict(zip(header, values))["dataset"] != DATASET:
            kept.append(line)
    if set(replacement.dataset) != {DATASET}:
        raise AssertionError("Replacement must contain UNSW only")
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore", lineterminator="\n")
    writer.writerows(replacement.to_dict("records"))
    target.write_bytes(raw[0] + b"".join(kept) + buf.getvalue().encode("utf-8"))
    actual_kept = target.read_bytes().splitlines(keepends=True)[1:1+len(kept)]
    assert actual_kept == kept
    return {"source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
            "destination": str(target.relative_to(ROOT)), "retained_other_dataset_rows": len(kept),
            "other_dataset_record_bytes_unchanged": True, "destination_sha256": sha256(target)}


def cell_and_guard(scenario, cell):
    w, labels, mode = p5._cell_weights(cell, scenario)
    pred = p5._fit_predict_cell(scenario["X_train"], labels, scenario["X_test"], w, mode, scenario["seed"]+7)
    y = scenario["y_test"]
    seen = np.isin(y, np.unique(scenario["y_train"]))
    row = {"dataset": DATASET, "cell": cell, "seed": scenario["seed"], "tail_flip_rate": scenario["rate"],
           "noise_target": int(scenario["noise_target"]), "macro_f1": p3._macro_f1(y, pred),
           "macro_f1_seen_classes": p3._macro_f1(y[seen], pred[seen]) if seen.any() else float("nan"),
           "tail_macro_f1": p3._tail_f1(y, pred, scenario["tail"]),
           "unseen_test_class_fraction": float(1-seen.mean()), "weight_min": float(w.min()), "weight_mean": float(w.mean())}
    scores = dict(zip(map(lambda x: str(int(x)), np.unique(y)),
                      map(float, f1_score(y, pred, labels=np.unique(y), average=None, zero_division=0))))
    guard = {"dataset": DATASET, "method": cell, "seed": scenario["seed"], "tail_flip_rate": scenario["rate"],
             "per_class_f1_json": json.dumps(scores, sort_keys=True), "noise_target": int(scenario["noise_target"])}
    return row, guard


def run_ac():
    contract()
    partial = OUT / "p5_unsw.partial.csv"
    guard_partial = OUT / "p5_guard_unsw.partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    guards = pd.read_csv(guard_partial).to_dict("records") if guard_partial.exists() else []
    windows = {int(w["seed"]): w for w in json.loads((ROOT/"reports/p9_tail_metric_unification.json").read_text())["windows"] if w["dataset"] == DATASET}
    ds = p4._load_primary_dataset(DATASET, ROOT / "configs")
    assert "label" not in ds.meta["feature_names"] and ds.X_train.shape[1] == 184
    for seed in p5.FULL_SEEDS:
        for rate in p5.FULL_RATES:
            matches = lambda r: r["seed"] == seed and r["tail_flip_rate"] == rate
            old = [r for r in rows if matches(r)]
            oldg = [r for r in guards if matches(r)]
            if len(old) == 4 and {r["cell"] for r in old} == set(p5.CELL_FULL) and len(oldg) == 2:
                continue
            rows = [r for r in rows if not matches(r)]
            guards = [r for r in guards if not matches(r)]
            s = p5._prepare_scenario(ds, DATASET, seed, rate)
            win = windows[seed]
            assert array_hash(s["noisy"]) == win["noise_by_rate"][str(rate)]["noisy_y_sha256"]
            assert int(s["noise_target"]) == win["noise_target"]
            for cell in p5.CELL_FULL:
                row, guard = cell_and_guard(s, cell)
                rows.append(row)
                if cell in p5.CELL_FULL[:2]:
                    guards.append(guard)
            pd.DataFrame(rows).to_csv(partial, index=False)
            pd.DataFrame(guards).to_csv(guard_partial, index=False)
            print(f"[p10 AC] seed={seed} rate={rate}: {len(rows)}/120 + {len(guards)}/60 guard", flush=True)
    frame, gf = pd.DataFrame(rows), pd.DataFrame(guards)
    validate_grid(frame, p5.CELL_FULL)
    validate_grid(gf, p5.CELL_FULL[:2], method_col="method")
    frame.to_csv(OUT/"p5_unsw.csv", index=False)
    gf.to_csv(OUT/"p5_guard_unsw.csv", index=False)
    merges = [merge_unsw(AC_SOURCE, frame, OUT/"p5_full_merged.csv"),
              merge_unsw(GUARD_SOURCE, gf, OUT/"p5_guard_full_merged.csv")]
    full = pd.read_csv(OUT/"p5_full_merged.csv")
    fullg = pd.read_csv(OUT/"p5_guard_full_merged.csv")
    validate_grid(full, p5.CELL_FULL, datasets=p5.DATASETS)
    validate_grid(fullg, p5.CELL_FULL[:2], datasets=p5.DATASETS, method_col="method")
    oldf = pd.read_csv(AC_SOURCE)
    new_stats, old_stats = ac_stats(full), ac_stats(oldf)
    changes = decision_changes(ac_decisions(old_stats), ac_decisions(new_stats))
    new_guards, old_guards = class_guards(fullg), class_guards(pd.read_csv(GUARD_SOURCE))
    changes += decision_changes({f"guard/{x['dataset']}/{x['rate']}": x["passed"] for x in old_guards},
                                {f"guard/{x['dataset']}/{x['rate']}": x["passed"] for x in new_guards})
    result = {"submission_ready": False, "completed_ac": True, "merged_rows": len(full), "guard_rows": len(fullg),
              "merges": merges, "old": old_stats, "new": new_stats,
              "old_guards": old_guards, "new_guards": new_guards, "decision_changes": changes,
              "stop_required": bool(changes), "full_family_sizes": {"A": 9, "C": 18}}
    dump(REPORTS/"family_ac_complete.json", result)
    for family in ("family_a", "family_c"):
        pd.DataFrame(new_stats[family]["tests"]).to_csv(TABLES/f"{family}.csv", index=False)
    assert_preserved()
    print(json.dumps({"A": new_stats["family_a"]["verdict"], "C_freq": new_stats["family_c"]["vs_freqfloor"]["verdict"],
                      "C_cbce": new_stats["family_c"]["vs_cbce"]["verdict"], "decision_changes": changes}), flush=True)
    if changes:
        raise SystemExit("STOP: AC or guard decision changed; no B/leakage/P9 rerun authorized past this gate")


def ac_stats(frame):
    a = p5._family_tests(frame, [("4_graph_cdm_soft", "3_graph_cdm_hard")], size=9)
    c = p5._family_tests(frame, [("4_graph_cdm_soft", "5_graph_cdm_freqfloor"), ("4_graph_cdm_soft", "6_nograph_cbce")], size=18)
    assert len(a) == 9 and len(c) == 18
    assert all(t["paired_rows"] == 10 for t in a+c)
    return {"family_a": {"tests": a, **p5._family_verdict(frame, a, "4_graph_cdm_soft", "3_graph_cdm_hard")},
            "family_c": {"tests": c, "vs_freqfloor": p5._family_verdict(frame, c, "4_graph_cdm_soft", "5_graph_cdm_freqfloor"),
                         "vs_cbce": p5._family_verdict(frame, c, "4_graph_cdm_soft", "6_nograph_cbce")}}


def ac_decisions(stats):
    result = {}
    for name, item in [("A", stats["family_a"]), ("C_freq", stats["family_c"]["vs_freqfloor"]), ("C_cbce", stats["family_c"]["vs_cbce"])]:
        result[name] = item["verdict"]
        for ds, val in item["per_dataset"].items():
            result[f"{name}/{ds}"] = val["passed"]
            for rate in val["rates"]:
                prefix = f"{name}/{ds}/{rate['tail_flip_rate']}"
                result[prefix] = rate["passed"]
                result[prefix+"/macro_NI"] = rate["macro_noninferiority"]["passed"]
                # Descriptive reverse signal, not a new confirmatory hypothesis.
                if name.startswith("C"):
                    result[prefix+"/reverse_CI_excludes_zero"] = rate["test"]["bootstrap_ci95_high"] < 0
    return result


def class_guards(frame):
    return [{"dataset": ds, "rate": rate, **p4._class_guard(frame[(frame.dataset == ds) & np.isclose(frame.tail_flip_rate, rate)],
                "4_graph_cdm_soft", "3_graph_cdm_hard", 0.05)} for ds in p5.DATASETS for rate in p5.FULL_RATES]


def decision_changes(old, new):
    assert old.keys() == new.keys()
    return [{"decision": key, "before": old[key], "after": new[key]} for key in old if old[key] != new[key]]


def require_gate(path):
    if not path.exists() or json.loads(path.read_text())["stop_required"]:
        raise RuntimeError(f"STOP: prerequisite gate not clear: {path.name}")


def run_b():
    contract()
    require_gate(REPORTS/"family_ac_complete.json")
    partial = OUT/"p4_unsw.partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    ds = p4._load_primary_dataset(DATASET, ROOT/"configs")
    for seed in p4.FULL_SEEDS:
        for rate in p4.FULL_RATES:
            matches = lambda r: r["seed"] == seed and r["tail_flip_rate"] == rate
            old = [r for r in rows if matches(r)]
            if len(old) == len(p4.METHODS) and {r['method'] for r in old} == set(p4.METHODS):
                continue
            rows = [r for r in rows if not matches(r)]
            new = p4._scenario_rows(DATASET, ds, seed, rate)
            assert {r['cleanlab_engine'] for r in new} == {'cleanlab'}
            rows.extend(new)
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f"[p10 B] seed={seed} rate={rate}: {len(rows)}/150", flush=True)
    frame = pd.DataFrame(rows)
    validate_grid(frame, p4.METHODS, method_col="method")
    frame.to_csv(OUT/"p4_unsw_canonical.csv", index=False)
    # Derive the historical canonical view from P9 saved scores, never masked vs canonical.
    source = pd.read_csv(B_SOURCE, dtype=str, keep_default_na=False)
    canonical = pd.read_csv(ROOT/"results/p9_canonical_tail_f1.csv", dtype=str, keep_default_na=False)
    canonical = canonical[canonical.source_file == "results/p4_cl_graph_combo_full_family_b_complete.csv"]
    keys = KEY+["method"]
    derived = source.drop(columns="tail_macro_f1").merge(canonical[keys+["canonical_tail_f1"]], on=keys, validate="one_to_one")
    derived = derived.rename(columns={"canonical_tail_f1": "tail_macro_f1"})[list(source.columns)]
    assert len(derived) == 450
    for col in source.columns:
        if col != "tail_macro_f1":
            assert derived[col].tolist() == source[col].tolist()
    derived.to_csv(OUT/"p4_before_canonical_from_p9.csv", index=False)
    merge = merge_unsw(OUT/"p4_before_canonical_from_p9.csv", frame, OUT/"p4_full_merged_canonical.csv")
    full = pd.read_csv(OUT/"p4_full_merged_canonical.csv")
    validate_grid(full, p4.METHODS, datasets=p4.DATASETS, method_col="method")
    before = p4._aggregate(pd.read_csv(OUT/"p4_before_canonical_from_p9.csv"), "full")
    after = p4._aggregate(full, "full")
    assert len(after["family_b_tests"]) == 18
    changes = decision_changes(b_decisions(before), b_decisions(after))
    result = {"submission_ready": False, "completed_b": True, "merge": merge, "before_canonical": before,
              "after_canonical": after, "decision_changes": changes, "stop_required": bool(changes),
              "metric_note": "P9 canonical historical view; original masked family B artifacts unchanged."}
    dump(REPORTS/"family_b_complete.json", result)
    pd.DataFrame(after["family_b_tests"]).to_csv(TABLES/"family_b.csv", index=False)
    assert_preserved()
    print(json.dumps({"G2": after["global_verdict"], "changes": changes}), flush=True)
    if changes:
        raise SystemExit("STOP: B decision changed; do not run leakage/P9")


def b_decisions(stats):
    result = {"G2": stats["global_verdict"]}
    for ds, item in stats["g2_by_dataset"].items():
        result[ds] = item["passed"]
        for rate in item["rates"]:
            prefix = f"{ds}/{rate['tail_flip_rate']}"
            result[prefix] = rate["passed"]
            result[prefix+"/macro_NI"] = rate["macro_noninferiority"]["passed"]
            result[prefix+"/class_guard"] = rate["class_f1_guard"]["passed"]
    return result


def run_leakage():
    contract()
    require_gate(REPORTS/"family_ac_complete.json")
    require_gate(REPORTS/"family_b_complete.json")
    p6.run(configs_dir=ROOT/"configs", out_dir=OUT, reports_dir=REPORTS, tables_dir=TABLES, datasets=(DATASET,))
    new = pd.read_csv(OUT/"p6_leakage_ablation_fixedtail_unsw_nb15.csv")
    assert len(new) == 180 and not new.duplicated(["dataset", "condition", "seed", "method"]).any()
    merged = OUT/"p6_full_merged.csv"
    info = merge_unsw(ROOT/"results/p6_leakage_ablation_fixedtail_all.csv", new, merged)
    p6._delta_table(pd.read_csv(merged)).to_csv(TABLES/"p6_full_deltas.csv", index=False)
    dump(REPORTS/"leakage_complete.json", {"submission_ready": False, "merge": info, "completed": True})
    assert_preserved()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ac", "b", "leakage"])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {"ac": run_ac, "b": run_b, "leakage": run_leakage}[args.stage]()
