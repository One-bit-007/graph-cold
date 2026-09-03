"""P10 A/B: enumerate every final feature and scan association, no training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.baselines.base import array_hash
from src.paper import p2d_clean_rerun as p2d, p3_proxy_audit as p3, p4_cl_graph_combo as p4
from src.paper.leak_scan import scan_features
from src.paper.p10_support import assert_preserved, sha256


def run():
    assert_preserved()
    previous = json.loads(Path("reports/p9_tail_metric_unification.json").read_text(encoding="utf-8"))
    old_windows = {(r["dataset"], r["seed"]): r for r in previous["windows"]}
    old_label_values = json.loads(Path("reports/p9_unsw_label_blocker.json").read_text(encoding="utf-8"))["partitions"]["train"]["scaled_feature_values"]
    frames, dataset_info = [], []
    for dataset in p4.DATASETS:
        ds = p4._load_primary_dataset(dataset, Path("configs"))
        names = list(ds.meta["feature_names"])
        bad = [name for name in names if any(token in str(name).casefold() for token in ["label", "attack_cat", "category", "attempted", "__source_file", "__row_in_file"])]
        if bad:
            raise AssertionError(f"Unexpected target/metadata columns after fix: {dataset}: {bad}")
        checks = []
        for seed in p4.FULL_SEEDS:
            old = old_windows[(dataset, seed)]
            ti = p2d._stratified_sample(ds.y_train, min(p4.TRAIN_WINDOW, len(ds.y_train)), seed)
            vi = p2d._stratified_sample(ds.y_test, min(p4.TEST_WINDOW, len(ds.y_test)), seed+1000)
            for label_key, values in [("y_train", ds.y_train[ti]), ("y_test", ds.y_test[vi]), ("train_idx", ti), ("test_idx", vi)]:
                if array_hash(values) != old[f"{label_key}_sha256"]:
                    raise AssertionError(f"Labels/sampling changed: {dataset}/{seed}/{label_key}")
            for part, idx in [("train", ti), ("test", vi)]:
                X = getattr(ds, f"X_{part}")[idx]
                if dataset == "unsw_nb15":
                    # Restore the previously audited binary column solely to
                    # verify that removing it was the ONLY input change.
                    labels = getattr(ds, f"y_{part}")[idx]
                    old_column = np.where(labels == 0, old_label_values[0], old_label_values[1]).astype(np.float32)
                    X = np.insert(X, 30, old_column, axis=1)
                if array_hash(X) != old[f"X_{part}_sha256"]:
                    raise AssertionError(f"Unexpected feature change: {dataset}/{seed}/{part}")
            checks.append(seed)
        for split in ["train", "test"]:
            frame = scan_features(getattr(ds, f"X_{split}"), getattr(ds, f"y_{split}"), names,
                                  benign_class=ds.meta.get("benign_class"), seed=0)
            frame.insert(0, "dataset", dataset)
            frame.insert(1, "split", split)
            frames.append(frame)
        dataset_info.append({"dataset": dataset, "feature_names": names, "n_features": len(names),
                             "n_train": len(ds.y_train), "n_test": len(ds.y_test),
                             "class_names": ds.meta["class_names"], "benign_class": ds.meta.get("benign_class"),
                             "protocol": p4.PROTOCOLS[dataset], "window_checks_passed": checks})
        if dataset == "unsw_nb15":
            # Positive control reconstructs the old audited label column. It
            # is scanned separately and never reintroduced to a model input.
            old_label = np.where(ds.y_train == 0, old_label_values[0], old_label_values[1])
            canary = scan_features(old_label[:, None], ds.y_train, ["legacy_label_positive_control"], benign_class=0)
        print(f"[p10 audit] {dataset}: {len(names)} features; all 10 frozen windows match (UNSW minus label only)", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    payload = {"completed": True, "scope": "feature enumeration and association scan, not proof of no leakage",
               "datasets": dataset_info, "scan_settings": {"max_rows_per_split": 20000, "quantile_bins": 32,
               "normalized_mi_threshold": 0.95, "max_abs_ovr_correlation_threshold": 0.995, "seed": 0},
               "flagged_features": frame[frame.flagged].replace({np.nan: None}).to_dict("records"),
               "legacy_label_positive_control": canary.to_dict("records"),
               "code_sha256": {p: sha256(p) for p in ["src/data/loaders.py", "configs/datasets.yaml", "src/paper/leak_scan.py"]}}
    if canary.iloc[0].binary_normalized_mi < 1-1e-12 or not canary.iloc[0].flagged:
        raise AssertionError("Legacy label positive control not detected")
    frame.to_csv("results/p10_feature_association_scan.csv", index=False)
    Path("reports/p10_feature_audit.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# P10 三数据集完整特征审计", "", "只读清点与统计扫描，不训练模型。UNSW label 已永久排除；CICIDS/CESNET 输入逐窗口字节一致。",
             "UNSW 恢复旧 label 列后与 P9 旧特征哈希逐位一致，证明本次仅移除这一列，采样及标签未变。", "",
             "MI 原始值以 nats 计；归一化分数为 I(X;Y)/H(Y)，不是 sklearn 对称 NMI。",
             "高基数字段用至多 32 分位箱，避免每个连续值唯一导致伪 MI=1；每个 split 最多均匀抽 20000 行，seed=0。",
             "红标条件：归一化 MI≥0.95 或单类 one-v-rest 绝对相关≥0.995；高相关不等于标签泄漏，低于阈值也不是安全证明。",
             "CESNET 无攻击/正常语义，二元指标为 N/A，仅扫描服务多类目标。", "",
             f"旧 UNSW label 正控：binary normalized MI={canary.iloc[0].binary_normalized_mi:.12f}，raw MI={canary.iloc[0].binary_mi_nats:.6f} nats；只用于工具核验，不进入训练。", ""]
    for info in dataset_info:
        data = frame[frame.dataset == info["dataset"]]
        lines += [f"## {info['dataset']}：{info['n_features']} 列", "", "| 列名 | 是否标签/代理 | 处置 | train/test 最大 normalized MI | train/test 最大 OVR 相关 |", "|---|---|---|---:|---:|"]
        for name in info["feature_names"]:
            part = data[data.feature == name]
            mi = float(part[["binary_normalized_mi", "multiclass_normalized_mi"]].max().max())
            corr = float(part[["binary_abs_correlation", "multiclass_max_abs_ovr_correlation"]].max().max())
            flag = bool(part.flagged.any())
            status = "待语义复核：高关联红标" if flag else "未识别标签字段；统计未红标"
            action = "暂停判定，不自动删除" if flag else "保留；不等于已证明无代理"
            lines.append(f"| {name.replace('|', '&#124;')} | {status} | {action} | {mi:.6f} | {corr:.6f} |")
        lines.append("")
    lines += ["## 边界", "", "该报告枚举了实际输入的每列，不以模板代替清点。任何红标须结合字段来源单独裁定；不据此自动删特征或改协议。",
              "已禁止的标签、IP、端口及 CATEGORY 等字段由配置/加载器门禁另行约束；关联扫描不能替代字段来源审计。", ""]
    Path("reports/p10_feature_audit.md").write_text("\n".join(lines), encoding="utf-8")
    assert_preserved()
    print(json.dumps({"features": {i['dataset']:i['n_features'] for i in dataset_info}, "flagged_rows": int(frame.flagged.sum())}), flush=True)
    return payload


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
