"""P9: FINE(tabular) only; frozen p4/p5 scenarios and canonical metrics.

No FINE hypothesis tests or Holm family. Existing method predictions are NOT
retrained. Run ``python -m src.paper.p9_fine_final_protocol``; interrupted runs
resume only if the implementation/input fingerprints still match.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from src.baselines.base import array_hash
from src.baselines.fine_tabular import FINETabularBaseline, METHOD, FAITHFULNESS
from src.baselines.registry import get_baseline_entry
from src.metrics_canonical import canonical_tail_f1_from_perclass
from src.paper import p3_proxy_audit as p3, p4_cl_graph_combo as p4, p5_factor_ablation as p5
from src.paper.p9_tail_metric_unification import assert_unchanged, sha256, validate_grid, KEY

PARTIAL = Path("results/p9_fine_final_protocol.partial.csv")
FINAL = Path("results/p9_fine_final_protocol.csv")
CONTRACT = Path("reports/p9_fine_run_contract.json")
CANONICAL = Path("results/p9_canonical_tail_f1.csv")
SENSITIVITY = Path("reports/p9_tail_metric_unification.json")


def assert_non_oracle_schema(ds):
    banned = ("attempted", "label", "flip_mask", "is_noisy", "clean_y", "ground_truth", "y_true", "__source_file", "__row_in_file")
    leaked = [name for name in ds.meta.get("feature_names", []) if any(x in str(name).lower() for x in banned)]
    if leaked or any("attempted" in str(c).lower() for c in ds.meta.get("class_names", [])):
        raise AssertionError(f"Attempted/oracle fields or classes present: {leaked}")


def prepare_scenario(ds, dataset, seed, rate, window):
    scenario = p5._prepare_scenario(ds, dataset, seed, rate)
    target, _ = p3._noise_target(dataset, scenario["y_train"], ds.meta)
    if scenario["noise_target"] != target or target != window["noise_target"]:
        raise AssertionError("F-3 target mismatch")
    for key in ("train_idx", "test_idx", "X_train", "X_test", "y_train", "y_test"):
        if array_hash(scenario[key]) != window[f"{key}_sha256"]:
            raise AssertionError(f"Frozen input mismatch: {dataset}/{seed}/{key}")
    if array_hash(scenario["noisy"]) != window["noise_by_rate"][str(float(rate))]["noisy_y_sha256"]:
        raise AssertionError("Frozen noisy training labels differ")
    np.testing.assert_array_equal(scenario["tail"], window["tail_labels"])
    return scenario


def fit_baseline(scenario, seed):
    # Only these arrays cross the training API: no clean labels, rate, flip,
    # validation set, graph-oracle metadata or scenario object is forwarded.
    model = FINETabularBaseline(seed=seed)
    return model.fit_predict(scenario["X_train"], scenario["noisy"],
                             scenario["X_test"], scenario["num_classes"])


def evaluate(dataset, scenario, result):
    y_test, pred = scenario["y_test"], result.y_pred
    labels = np.unique(y_test)
    scores = {str(int(k)): float(v) for k, v in zip(labels, f1_score(y_test, pred, labels=labels, average=None, zero_division=0))}
    tail = canonical_tail_f1_from_perclass(scores, scenario["tail"])
    if not np.isfinite(tail) or result.details["training_label_hash"] != array_hash(scenario["noisy"]):
        raise AssertionError("Invalid evaluation or training label provenance")
    return {"dataset": dataset, "seed": scenario["seed"], "tail_flip_rate": scenario["rate"],
            "method": METHOD, "faithfulness_level": FAITHFULNESS,
            "protocol": p4.PROTOCOLS[dataset]["label"], "noise_target": scenario["noise_target"],
            "noise_rate_source": "not_used", "macro_f1": p3._macro_f1(y_test, pred),
            "tail_macro_f1": tail, "tail_metric_definition": "full_test_set_per_class_tail_mean",
            "per_class_f1_json": json.dumps(scores, sort_keys=True),
            "tail_labels_json": json.dumps(scenario["tail"].tolist()),
            "confusion_labels_json": json.dumps(list(range(scenario["num_classes"]))),
            "confusion_matrix_json": json.dumps(confusion_matrix(y_test, pred, labels=np.arange(scenario["num_classes"])).tolist()),
            "n_train": len(scenario["noisy"]), "n_test": len(y_test),
            "noisy_y_sha256": array_hash(scenario["noisy"]), "prediction_sha256": array_hash(pred),
            "retained_fraction": float(result.retained_mask.mean()),
            "fallback_class_count": result.details["fallback_class_count"],
            "fallback_sample_count": result.details["fallback_sample_count"],
            "metadata_json": json.dumps(result.details, sort_keys=True, allow_nan=False)}


def contract_for(audit):
    files = ["src/baselines/fine_tabular.py", "src/baselines/registry.py", "src/metrics_canonical.py",
             "src/paper/p9_fine_final_protocol.py", str(CANONICAL), str(SENSITIVITY)]
    return {"schema_version": 1, "method": METHOD, "config": FINETabularBaseline.CONFIG,
            "implementation_and_input_sha256": {p.replace("\\", "/"): sha256(p) for p in files},
            "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "threadpoolctl"]},
            "python": platform.python_version(), "blas_threads": 1,
            "seeds": list(p4.FULL_SEEDS), "rates": list(p4.FULL_RATES), "datasets": list(p4.DATASETS),
            "bootstrap": {"unit": "paired seed within dataset and rate", "n_resamples": 2000, "seed": 42,
                          "interval": "pointwise percentile 95%; descriptive, not multiplicity adjusted"},
            "protected_sha256": audit["protected_sha256"], "submission_ready": False}


def paired_deltas(frame, reference, label):
    joined = frame.merge(reference, on=["dataset", "seed", "tail_flip_rate"], suffixes=("_fine", "_ref"), validate="one_to_one")
    if len(joined) != 90:
        raise AssertionError("Incomplete frozen reference pairing")
    rows = []
    for (dataset, rate), part in joined.groupby(["dataset", "tail_flip_rate"]):
        if sorted(part.seed.tolist()) != list(p4.FULL_SEEDS):
            raise AssertionError("Missing paired seeds")
        for metric in ["macro_f1", "tail_macro_f1"]:
            diff = (part[f"{metric}_fine"] - part[f"{metric}_ref"]).to_numpy()
            rng = np.random.default_rng(42)
            means = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
            rows.append({"dataset": dataset, "tail_flip_rate": rate, "reference": label,
                         "metric": metric, "n_pairs": len(diff), "mean_delta_fine_minus_reference": float(diff.mean()),
                         "bootstrap_ci95_low": float(np.quantile(means, 0.025)),
                         "bootstrap_ci95_high": float(np.quantile(means, 0.975)),
                         "analysis": "descriptive_pointwise_only"})
    return rows


def finalize(frame, audit, contract):
    validate_grid(frame, [METHOD])
    if set(frame.noise_rate_source) != {"not_used"} or not np.isfinite(frame[["macro_f1", "tail_macro_f1"]].to_numpy()).all():
        raise AssertionError("Invalid FINE result grid")
    canonical = pd.read_csv(CANONICAL)
    references = {}
    for method, name in [(p4.REFERENCE_METHOD, p4.REFERENCE_METHOD), ("4_graph_cdm_soft", "Graph-CoLD-soft")]:
        ref = canonical[canonical.method == method].copy()
        validate_grid(ref, [method])
        ref["tail_macro_f1"] = ref.canonical_tail_f1
        ref["method"] = name
        references[name] = ref
    columns = KEY + ["macro_f1", "tail_macro_f1"]
    all_methods = pd.concat([frame[columns]] + [r[columns] for r in references.values()], ignore_index=True)
    table = all_methods.groupby(["dataset", "tail_flip_rate", "method"]).agg(
        n=("seed", "size"), macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        tail_f1_mean=("tail_macro_f1", "mean"), tail_f1_std=("tail_macro_f1", "std")).reset_index()
    deltas = pd.DataFrame([r for name, ref in references.items() for r in paired_deltas(frame, ref[columns], name)])
    pooled = all_methods.groupby(["dataset", "method"])[["macro_f1", "tail_macro_f1"]].mean().reset_index()
    fallback = frame.groupby("dataset").agg(scenarios=("seed", "size"),
                mean_retained_fraction=("retained_fraction", "mean"),
                class_fallback_events=("fallback_class_count", "sum"),
                samples_in_fallback_classes=("fallback_sample_count", "sum")).reset_index()
    assert_unchanged(contract["protected_sha256"])
    frame.sort_values(KEY).to_csv(FINAL, index=False)
    table.to_csv("tables/table_p9_fine_final.csv", index=False)
    deltas.to_csv("tables/table_p9_fine_paired_deltas.csv", index=False)
    report = {"completed": True, "row_count": len(frame), "submission_ready": False,
              "method": METHOD, "faithfulness_level": FAITHFULNESS,
              "canonical_g2_sensitivity": audit["canonical_global_verdict"],
              "table": table.to_dict("records"), "paired_descriptive_deltas": deltas.to_dict("records"),
              "pooled_means_descriptive_only": pooled.to_dict("records"), "fallback_summary": fallback.to_dict("records"),
              "output_sha256": {str(FINAL): sha256(FINAL), "tables/table_p9_fine_final.csv": sha256("tables/table_p9_fine_final.csv")}}
    Path("reports/p9_fine_final_protocol.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# P9：FINE detector (tabular adaptation) 最终协议补充基线", "",
             "完成 3 数据集 × 10 seeds × 3 rates = 90 场景；逐类 F1、混淆矩阵和回退信息已落盘。",
             "F-5：训练只接收特征及噪声标签；不输入真实/估计噪声率或干净验证集。",
             "FINE 核心为论文 Algorithm 1 的 Gram/平方对齐/GMM；静态 SVD 表示和 ExtraTrees 分类头是适配，非完整原始深度实现。", "",
             "## 同协议比较（每格 10 seeds；均值 ± 样本标准差）", "",
             "所有 Tail-F1 均由完整测试集逐类 F1 使用 canonical helper 汇总。CL 引用 p4，Graph 引用修正 p5 的逐类护栏；Macro 引用对应冻结主产物。",
             "两项既有方法均未重训；每个 FINE 场景均与任务 0 保存的输入/噪声哈希逐项核对。", "",
             "| 数据集 | rate | 方法 | Macro-F1 | canonical Tail-F1 |", "|---|---:|---|---:|---:|"]
    for r in table.to_dict("records"):
        lines.append(f"| {r['dataset']} | {r['tail_flip_rate']} | {r['method']} | {r['macro_f1_mean']:.6f} ± {r['macro_f1_std']:.6f} | {r['tail_f1_mean']:.6f} ± {r['tail_f1_std']:.6f} |")
    lines += ["", "## 配对差值（FINE − 引用方法；仅描述性）", "",
              "逐数据集/噪声率按 seed 配对，n=10；2000 次成对 bootstrap，seed=42，95% percentile CI。",
              "区间未作多重性校正，不用于新增显著性结论；未合并三个噪声率伪称 n=30 独立样本。", "",
              "| 数据集 | rate | 引用 | 指标 | Δ | 点态 95% CI |", "|---|---:|---|---|---:|---|"]
    for r in deltas.to_dict("records"):
        lines.append(f"| {r['dataset']} | {r['tail_flip_rate']} | {r['reference']} | {r['metric']} | {r['mean_delta_fine_minus_reference']:+.6f} | [{r['bootstrap_ci95_low']:+.6f}, {r['bootstrap_ci95_high']:+.6f}] |")
    lines += ["", "## 选择回退与局限", "", "| 数据集 | 平均保留比例 | 类回退事件数 | 回退类样本数（跨场景，非去重样本） |", "|---|---:|---:|---:|"]
    for r in fallback.to_dict("records"):
        lines.append(f"| {r['dataset']} | {r['mean_retained_fraction']:.6f} | {r['class_fallback_events']} | {r['samples_in_fallback_classes']} |")
    lines += ["", "回退原因逐场景记录于 metadata_json：小类、退化对齐、GMM 未收敛/数值失败等按预先定义策略保留该类全部样本，不能忽略其对检测有效性的影响。",
              "CICIDS/UNSW 不具备严格实体隔离；CESNET 仅是短窗口服务分类压力测试，不能解释为攻击检测或 SOC 效果。",
              "无性能调参、无额外基线、无新增推断族。", "",
              "## 诚实定位与协调者交接", "",
              "本工作在最终协议下补齐一个经过组件保真度核验的外部方法适配基线，降低但不清零 A7 风险。",
              "不是完整原始 FINE 复现，不能据此清除全部外部基线保真度疑问；不得声称普遍超越 CL。",
              "实际方向以以上全部单格数值为准，不把运行前‘可能不及 CL’的预期当作结果。",
              "Table 2 的 CL/CL+Graph 数值影响另见 p9_tail_metric_unification.md；生成器、正文、答复信和数字审计待协调者更新。",
              "原 A/B/C 与 F-3 修正产物保持原哈希；submission_ready=false；未推送或发布。", ""]
    Path("reports/p9_fine_final_protocol.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def run(finalize_only=False):
    audit = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    if audit["canonical_global_verdict"] != audit["original_global_verdict"]:
        raise RuntimeError("STOP: canonical sensitivity changes G2")
    entry = get_baseline_entry(METHOD)
    if not entry.verified or not entry.include_in_formal_results:
        raise RuntimeError("F-5 implementation tests must pass before registry inclusion and run")
    contract = contract_for(audit)
    assert_unchanged(contract["protected_sha256"])
    if CONTRACT.exists():
        if json.loads(CONTRACT.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Contract changed: refuse stale partial/final reuse")
    else:
        if PARTIAL.exists() or FINAL.exists():
            raise RuntimeError("Existing results lack run contract")
        CONTRACT.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    frame = pd.read_csv(PARTIAL) if PARTIAL.exists() else pd.DataFrame()
    if not frame.empty and frame.duplicated(KEY).any():
        raise AssertionError("Duplicate partial scenario")
    rows = frame.to_dict("records")
    done = {(r["dataset"], int(r["seed"]), float(r["tail_flip_rate"])) for r in rows}
    windows = {(r["dataset"], r["seed"]): r for r in audit["windows"]}
    if not finalize_only:
        with threadpool_limits(limits=1):
            for dataset in p4.DATASETS:
                if all((dataset, s, r) in done for s in p4.FULL_SEEDS for r in p4.FULL_RATES):
                    continue
                ds = p4._load_primary_dataset(dataset, Path("configs"))
                assert_non_oracle_schema(ds)
                for seed in p4.FULL_SEEDS:
                    for rate in p4.FULL_RATES:
                        if (dataset, seed, rate) in done:
                            continue
                        start = time.monotonic()
                        scenario = prepare_scenario(ds, dataset, seed, rate, windows[(dataset, seed)])
                        result = fit_baseline(scenario, seed)
                        rows.append(evaluate(dataset, scenario, result))
                        temporary = PARTIAL.with_suffix(".tmp.csv")
                        pd.DataFrame(rows).to_csv(temporary, index=False)
                        temporary.replace(PARTIAL)
                        print(f"[p9 FINE] {len(rows)}/90 {dataset} seed={seed} rate={rate} ({time.monotonic()-start:.1f}s)", flush=True)
    result = finalize(pd.DataFrame(rows), audit, contract)
    assert_unchanged(contract["protected_sha256"])
    print(json.dumps({"completed": result["completed"], "rows": result["row_count"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-only", action="store_true")
    run(parser.parse_args().finalize_only)
