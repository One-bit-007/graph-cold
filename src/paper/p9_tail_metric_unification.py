"""P9 Task 0: read-only reaggregation; writes NEW P9 outputs only.

The canonical sensitivity analysis reuses the existing 18-member Family B
gate, NOT a new inferential family. It never replaces preregistered results.
Run: python -m src.paper.p9_tail_metric_unification
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.base import array_hash
from src.metrics_canonical import canonical_tail_f1_from_perclass
from src.paper import p2d_clean_rerun as p2d, p2e_salvage as p2e
from src.paper import p3_proxy_audit as p3, p4_cl_graph_combo as p4

P4_SOURCE = Path("results/p4_cl_graph_combo_full_family_b_complete.csv")
P5_SOURCE = Path("results/p5_factor_ablation_full.csv")
GUARD_SOURCE = Path("results/p5_perclass_guard.csv")
KEY = ["dataset", "seed", "tail_flip_rate", "method"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_snapshot():
    """Protect all tracked prior results, tables, reports, manuscript and code."""
    paths = subprocess.check_output(["git", "ls-files", "-z"], text=True).split("\0")
    return {p: sha256(p) for p in paths if p and Path(p).is_file()
            and p.startswith(("results/", "tables/", "reports/", "paper/", "src/", "configs/", "reproducibility/"))
            and "p9_" not in p and p != "src/baselines/registry.py"}


def assert_unchanged(snapshot):
    changed = [p for p, digest in snapshot.items() if not Path(p).exists() or sha256(p) != digest]
    if changed:
        raise AssertionError(f"Protected artifacts changed: {changed}")


def validate_grid(frame, methods):
    expected = {(d, s, r, m) for d in p4.DATASETS for s in p4.FULL_SEEDS
                for r in p4.FULL_RATES for m in methods}
    actual = set(frame[KEY].itertuples(index=False, name=None))
    if frame.duplicated(KEY).any() or actual != expected:
        raise AssertionError(f"Grid mismatch: missing={expected-actual}, extra={actual-expected}")


def restore_windows():
    """Restore inputs/labels only. No graph, detector or classifier fitting."""
    windows = {}
    for dataset in p4.DATASETS:
        ds = p4._load_primary_dataset(dataset, Path("configs"))
        for seed in p4.FULL_SEEDS:
            train_idx = p2d._stratified_sample(ds.y_train, min(p4.TRAIN_WINDOW, len(ds.y_train)), seed)
            test_idx = p2d._stratified_sample(ds.y_test, min(p4.TEST_WINDOW, len(ds.y_test)), seed + 1000)
            y_train, y_test = ds.y_train[train_idx], ds.y_test[test_idx]
            target, name = p3._noise_target(dataset, y_train, ds.meta)
            tail = p3._tail_labels(y_train)
            old_tail = p3._tail_noise_labels(dataset, y_train, target)
            noise_hashes = {}
            for rate in p4.FULL_RATES:
                noisy, flip = p2e._inject_tail_asymmetric(y_train, tail, target, rate, seed)
                other, other_flip = p2e._inject_tail_asymmetric(y_train, old_tail, target, rate, seed)
                np.testing.assert_array_equal(noisy, other)
                np.testing.assert_array_equal(flip, other_flip)
                noise_hashes[str(rate)] = {"noisy_y_sha256": array_hash(noisy),
                                          "effective_flip_rate": float(flip.mean())}
            windows[(dataset, seed)] = {
                "dataset": dataset, "seed": int(seed), "noise_target": int(target),
                "noise_target_name": name, "tail_labels": tail.tolist(),
                "p4_tail_labels": old_tail.tolist(), "test_labels": np.unique(y_test).tolist(),
                "n_train": len(train_idx), "n_test": len(test_idx),
                "train_idx_sha256": array_hash(train_idx), "test_idx_sha256": array_hash(test_idx),
                "X_train_sha256": array_hash(ds.X_train[train_idx]),
                "X_test_sha256": array_hash(ds.X_test[test_idx]),
                "y_train_sha256": array_hash(y_train), "y_test_sha256": array_hash(y_test),
                "noise_by_rate": noise_hashes, "protocol": p4.PROTOCOLS[dataset]["label"],
            }
        print(f"[p9 canonical] restored {dataset}: 10 windows, 30 noise pairs agree", flush=True)
    return windows


def reaggregate(frame, windows, source):
    rows = []
    for row in frame.to_dict("records"):
        win = windows[(row["dataset"], int(row["seed"]))]
        scores = json.loads(row["per_class_f1_json"])
        if set(map(int, scores)) != set(win["test_labels"]):
            raise AssertionError(f"Test-class coverage mismatch: {row['dataset']}/{row['seed']}")
        if source == P4_SOURCE:
            if int(row["noise_target_class"]) != win["noise_target"]:
                raise AssertionError("Frozen p4 noise target mismatch")
            if row["n_train"] != win["n_train"] or row["n_test"] != win["n_test"]:
                raise AssertionError("Frozen p4 window mismatch")
            expected_flip = win["noise_by_rate"][str(float(row["tail_flip_rate"]))]["effective_flip_rate"]
            if not np.isclose(row["effective_flip_rate"], expected_flip, rtol=0, atol=1e-12):
                raise AssertionError("Frozen p4 effective noise mismatch")
        elif pd.notna(row["noise_target"]) and int(row["noise_target"]) != win["noise_target"]:
            raise AssertionError("Frozen p5 guard noise target mismatch")
        elif source == GUARD_SOURCE and row["dataset"] == "cesnet_tls_year22" and pd.isna(row["noise_target"]):
            raise AssertionError("Corrected CESNET guard must log F-3 target")
        score = canonical_tail_f1_from_perclass(scores, win["tail_labels"])
        if not np.isfinite(score):
            raise AssertionError("No evaluable tail classes")
        rows.append({
            **{k: row[k] for k in KEY}, "source_file": source.as_posix(),
            "source_sha256": sha256(source), "noise_target": win["noise_target"],
            "noise_target_evidence": "historically_logged" if source == P4_SOURCE or pd.notna(row.get("noise_target")) else "reconstructed_protocol_only_not_logged_in_legacy_guard",
            "tail_labels_json": json.dumps(win["tail_labels"]),
            "test_labels_json": json.dumps(win["test_labels"]),
            "per_class_f1_json": row["per_class_f1_json"],
            "canonical_tail_f1": score, "stored_tail_f1": row.get("tail_macro_f1"),
            "macro_f1": row.get("macro_f1"), "metric_reaggregated_for_p9": True,
        })
    return pd.DataFrame(rows)


def run():
    snapshot = frozen_snapshot()
    source = pd.read_csv(P4_SOURCE)
    guard = pd.read_csv(GUARD_SOURCE)
    main = pd.read_csv(P5_SOURCE).rename(columns={"cell": "method"})
    validate_grid(source, p4.METHODS)
    validate_grid(guard, ["3_graph_cdm_hard", "4_graph_cdm_soft"])
    validate_grid(main, ["3_graph_cdm_hard", "4_graph_cdm_soft", "5_graph_cdm_freqfloor", "6_nograph_cbce"])
    if set(source.cleanlab_engine) != {"cleanlab"}:
        raise AssertionError("CL source includes a fallback engine")
    windows = restore_windows()
    Path("reports/p9_protocol_windows.json").write_text(
        json.dumps({"windows": list(windows.values()), "protected_sha256": snapshot}, indent=2), encoding="utf-8")
    canonical = reaggregate(source, windows, P4_SOURCE)
    guard_canonical = reaggregate(guard, windows, GUARD_SOURCE).drop(columns=["macro_f1", "stored_tail_f1"])
    guard_canonical = guard_canonical.merge(main[KEY + ["macro_f1", "tail_macro_f1"]],
                                           on=KEY, validate="one_to_one")
    guard_canonical = guard_canonical.rename(columns={"tail_macro_f1": "stored_tail_f1"})
    guard_canonical["macro_source_file"] = P5_SOURCE.as_posix()
    guard_canonical["macro_source_sha256"] = sha256(P5_SOURCE)
    guard_difference = float(np.max(np.abs(guard_canonical.canonical_tail_f1 - guard_canonical.stored_tail_f1)))
    if guard_difference > 0.001:
        raise AssertionError(f"p5 companion/main disagreement exceeds documented runtime scale: {guard_difference}")
    combined = pd.concat([canonical, guard_canonical], ignore_index=True).sort_values(KEY)
    adjusted = source.drop(columns="tail_macro_f1").merge(
        canonical[KEY + ["canonical_tail_f1"]], on=KEY, validate="one_to_one"
    ).rename(columns={"canonical_tail_f1": "tail_macro_f1"})
    sensitivity = p4._aggregate(adjusted, "full")
    original = json.loads(Path("reports/p4_cl_graph_combo_full_family_b_complete.json").read_text(encoding="utf-8"))["summary"]
    changes = []
    for (dataset, method), part in canonical[canonical.method.isin([p4.REFERENCE_METHOD, p4.PRIMARY_METHOD])].groupby(["dataset", "method"]):
        changes.append({"dataset": dataset, "method": method, "n": len(part),
                        "masked_mean": float(part.stored_tail_f1.mean()),
                        "masked_std": float(part.stored_tail_f1.std(ddof=1)),
                        "canonical_mean": float(part.canonical_tail_f1.mean()),
                        "canonical_std": float(part.canonical_tail_f1.std(ddof=1)),
                        "mean_change": float((part.canonical_tail_f1-part.stored_tail_f1).mean())})
    report = {"stage": "P9 Task 0", "submission_ready": False,
              "analysis_status": "post-hoc metric sensitivity; not replacement preregistration",
              "row_count": len(combined), "window_count": len(windows),
              "original_global_verdict": original["global_verdict"],
              "canonical_global_verdict": sensitivity["global_verdict"],
              "table2_changes": changes, "sensitivity_summary": sensitivity,
              "p5_guard_main_max_abs_tail_difference": guard_difference,
              "legacy_guard_missing_target_rows": int(guard.noise_target.isna().sum()),
              "windows": list(windows.values()), "protected_sha256": snapshot}
    assert_unchanged(snapshot)
    combined.to_csv("results/p9_canonical_tail_f1.csv", index=False)
    Path("reports/p9_tail_metric_unification.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# P9 Task 0：canonical Tail-F1 与主表影响", "",
             "状态：数据重汇总；不重训，不修改原 A/B/C、生成器或正文。", "",
             f"- 完整性：{len(combined)} 行（p4 450 + p5 护栏 180），30 个窗口、90 个注入配对均一致。",
             "- canonical = 完整测试集逐类 F1 后，对测试中出现的尾类取均值；尾类由冻结训练窗口恢复。",
             f"- p5 主表与独立护栏重跑的最大 Tail-F1 差：{guard_difference:.12g}；逐类来源与主表 Macro 来源单独标注。",
             f"- 历史护栏中 {int(guard.noise_target.isna().sum())} 行未落盘 noise_target（CICIDS/UNSW）；这些行仅按冻结协议恢复目标，未伪称历史记录齐全。CESNET 修正护栏要求显式记录 11。",
             "- 主表下列均值/样本标准差跨 10 seeds × 3 rates；这不是 30 个独立推断样本。", "",
             "| 数据集 | 方法 | masked 均值 ± std | canonical 均值 ± std | Δ |",
             "|---|---|---:|---:|---:|"]
    for r in changes:
        lines.append(f"| {r['dataset']} | {r['method']} | {r['masked_mean']:.6f} ± {r['masked_std']:.6f} | {r['canonical_mean']:.6f} ± {r['canonical_std']:.6f} | {r['mean_change']:+.6f} |")
    lines += ["", "## Family B：事后指标敏感性检查（不替换预注册结论）", "",
              f"原预注册 masked G2：**{original['global_verdict']}**；canonical 敏感性 G2：**{sensitivity['global_verdict']}**。",
              "复用原 18 项 Family B 的同一比较与门槛（含 Macro 项），不是新增假设族；FINE 未纳入此分析。", "",
              "| 数据集 | rate | canonical Δ Tail | 原 masked Δ Tail | Holm p（敏感性） | dz | bootstrap 95% CI | G2 单格通过 |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for dataset, result in sensitivity["g2_by_dataset"].items():
        for rate in result["rates"]:
            t = rate["tail_primary"]
            old = next(x for x in original["family_b_tests"] if x["dataset"] == dataset and x["tail_flip_rate"] == rate["tail_flip_rate"] and x["metric"] == "tail_macro_f1")
            lines.append(f"| {dataset} | {rate['tail_flip_rate']} | {t['mean_delta']:+.6f} | {old['mean_delta']:+.6f} | {t['p_holm_greater']:.6g} | {t['cohens_dz']:.4f} | [{t['bootstrap_ci95_low']:+.6f}, {t['bootstrap_ci95_high']:+.6f}] | {rate['passed']} |")
    lines += ["", "所有数值、混合效应输出、窗口/噪声哈希及输入文件哈希见同名 JSON；逐行重汇总见 `results/p9_canonical_tail_f1.csv`。",
              "Family A/C 本身已使用 canonical 口径，未重算；主表及原 Family B 指标名称需协调者清晰区分。",
              "同一族内部两臂定义一致，不足以证明更换定义后的 delta 或裁决必然不变；以上为实际敏感性结果。",
              "canonical 下 null 也不是等效性证据；不能把未检出优势解释为方法相同。", ""]
    if sensitivity["global_verdict"] != original["global_verdict"]:
        lines += ["**STOP：canonical 敏感性裁决改变，暂停 FINE 并交协调者裁定。**", ""]
    Path("reports/p9_tail_metric_unification.md").write_text("\n".join(lines), encoding="utf-8")
    assert_unchanged(snapshot)
    print(json.dumps({k: report[k] for k in ["row_count", "original_global_verdict", "canonical_global_verdict", "p5_guard_main_max_abs_tail_difference"]}), flush=True)
    return report


if __name__ == "__main__":
    result = run()
    if result["canonical_global_verdict"] != result["original_global_verdict"]:
        raise SystemExit("STOP: canonical G2 differs; report written, no FINE experiment started")
