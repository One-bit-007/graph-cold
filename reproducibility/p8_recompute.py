"""P8 runtime aggregation only; does not train or change frozen estimators.

Run from the repository root: python -m reproducibility.p8_recompute family
then (after the CESNET-only guard run) python -m reproducibility.p8_recompute guard.
Exit 2 means a decision changed: stop downstream publication/manuscript work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.paper import p4_cl_graph_combo as p4
from src.paper import p5_factor_ablation as p5

ARCHIVE = Path("reproducibility/history/p8_before_f3_20260903")
CESNET = "cesnet_tls_year22"
KEY = ["dataset", "cell", "seed", "tail_flip_rate"]
SOFT, HARD, FLOOR, CBCE = p5.CELL_FULL[1], p5.CELL_FULL[0], p5.CELL_FULL[2], p5.CELL_FULL[3]


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(p5._jsonable(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def write_rows(path, fields, records):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def grid(frame, datasets, cells):
    assert len(frame) == len(datasets) * 10 * 3 * len(cells)
    assert not frame.duplicated(KEY).any()
    assert set(frame.dataset) == set(datasets)
    expected = {(d, c, s, r) for d in datasets for c in cells for s in p5.FULL_SEEDS for r in p5.FULL_RATES}
    assert set(frame[KEY].itertuples(index=False, name=None)) == expected


def aggregate(frame):
    a = p5._family_tests(frame, [(SOFT, HARD)], size=9)
    c = p5._family_tests(frame, [(SOFT, FLOOR), (SOFT, CBCE)], size=18)
    assert len(a) == 9 and len(c) == 18
    assert all(t["paired_rows"] == 10 for t in a + c)
    return {
        "family_a": {"tests": a, **p5._family_verdict(frame, a, SOFT, HARD)},
        "family_c": {"tests": c,
                     "vs_freqfloor": p5._family_verdict(frame, c, SOFT, FLOOR),
                     "vs_cbce": p5._family_verdict(frame, c, SOFT, CBCE)},
    }


def decisions(result):
    out = {}
    for name, family in [("A", result["family_a"]), ("C_floor", result["family_c"]["vs_freqfloor"]),
                         ("C_cbce", result["family_c"]["vs_cbce"])]:
        out[name] = family["verdict"]
        for ds, item in family["per_dataset"].items():
            out[f"{name}/{ds}"] = item["passed"]
            for rate in item["rates"]:
                stem = f"{name}/{ds}/{rate['tail_flip_rate']}"
                out[stem + "/superiority"] = rate["passed"]
                out[stem + "/macro_noninferiority"] = rate["macro_noninferiority"]["passed"]
    return out


def family():
    old_path = ARCHIVE / "results/p5_factor_ablation_full.csv"
    new_path = Path(f"results/p5_factor_ablation_full_{CESNET}.csv")
    old, new = read_csv(old_path), read_csv(new_path)
    grid(old, p5.DATASETS, p5.CELL_FULL)
    grid(new, [CESNET], p5.CELL_FULL)
    assert new.noise_target.eq(11).all()
    unchanged = {}
    shard_differences = []
    for ds in ("cicids2017", "unsw_nb15"):
        rel = f"results/p5_factor_ablation_full_{ds}.csv"
        assert sha(rel) == sha(ARCHIVE / rel), f"Changed protected shard: {ds}"
        unchanged[ds] = sha(rel)
        base = old[old.dataset == ds].set_index(KEY).sort_index()
        shard = read_csv(rel).set_index(KEY).sort_index()
        assert base.index.equals(shard.index)
        for key in base.index:
            for col in base.columns:
                if base.loc[key, col] != shard.loc[key, col]:
                    shard_differences.append({"key": list(key), "column": col,
                                              "committed_full": base.loc[key, col], "shard": shard.loc[key, col]})

    # Preserve exact historic numeric tokens, not a round-trip float serialization.
    fields, old_records = rows(old_path)
    _, new_records = rows(new_path)
    key_of = lambda row: tuple(row[k] for k in KEY)
    replacements = {key_of(row): row for row in new_records}
    merged_records = [replacements[key_of(row)] if row["dataset"] == CESNET else {**row, "noise_target": ""}
                      for row in old_records]
    fields = fields + ["noise_target"]
    merged_path = Path("results/p5_factor_ablation_full.csv")
    write_rows(merged_path, fields, merged_records)
    write_rows("results/p5_factor_ablation_full.partial.csv", fields, merged_records)
    merged = read_csv(merged_path)
    grid(merged, p5.DATASETS, p5.CELL_FULL)
    original_columns = list(old.columns)
    pd.testing.assert_frame_equal(old[old.dataset != CESNET].reset_index(drop=True),
                                  merged.loc[merged.dataset != CESNET, original_columns].reset_index(drop=True),
                                  check_exact=True)
    _, written = rows(merged_path)
    assert all({k: after[k] for k in original_columns} == before
               for before, after in zip(old_records, written) if before["dataset"] != CESNET)

    before = aggregate(old)
    after = aggregate(merged)
    # Independently compare the new decision booleans against the frozen JSON.
    frozen = json.loads((ARCHIVE / "reports/p5_factor_ablation_full.json").read_text(encoding="utf-8"))
    assert decisions(before) == decisions(frozen), "Historical decision reconstruction differs from frozen report"
    old_decisions, new_decisions = decisions(before), decisions(after)
    changes = [{"decision": k, "old": v, "new": new_decisions[k]} for k, v in old_decisions.items() if v != new_decisions[k]]
    report = {**frozen, **after, "implementation_amendment": "PA-3: CESNET target corrected to 11",
              "outputs": {"results": str(merged_path), "partial": "results/p5_factor_ablation_full.partial.csv"}}
    dump("reports/p5_factor_ablation_full.json", report)
    for name in ("a", "c"):
        pd.DataFrame(p5._flatten_tests(after[f"family_{name}"]["tests"])).to_csv(f"tables/table_p5_family_{name}.csv", index=False)
    mixed = {ds: p4._mixed_effect_robustness(p5._as_method_frame(merged[merged.dataset == ds]), SOFT, HARD)
             for ds in p5.DATASETS}
    dump("reports/p5_family_a_mixed_effects.json", mixed)

    comparison = []
    for cell in p5.CELL_FULL:
        for rate in p5.FULL_RATES:
            a = old[(old.dataset == CESNET) & (old.cell == cell) & (old.tail_flip_rate == rate)]
            b = merged[(merged.dataset == CESNET) & (merged.cell == cell) & (merged.tail_flip_rate == rate)]
            comparison.append({"cell": cell, "rate": rate,
                               **{f"{prefix}_{metric}": float(f[metric].mean())
                                  for prefix, f in [("old", a), ("new", b)] for metric in ["macro_f1", "tail_macro_f1"]}})
    audit = {"status": "STOP_DECISION_CHANGED" if changes else "FAMILIES_RECOMPUTED",
             "decision_changes": changes, "old_decisions": old_decisions, "new_decisions": new_decisions,
             "row_count": len(merged), "cesnet_new_rows": len(new), "cesnet_noise_target": 11,
             "unaffected_rows_exact": 240, "unaffected_numeric_tokens_exact": True,
             "unaffected_shard_sha256": unchanged, "historical_shard_discrepancies": shard_differences,
             "source_precedence": "Committed full baseline for CICIDS/UNSW; new CESNET shard only replaces CESNET",
             "comparison": comparison, "before": before, "after": after,
             "family_a_test_count": 9, "family_c_test_count": 18,
             "submission_ready": False}
    dump("reports/p8_cesnet_f3_recompute.json", audit)
    print(json.dumps({k: audit[k] for k in ["status", "decision_changes", "cesnet_new_rows", "unaffected_rows_exact"]}, indent=2))
    print("Family verdicts:", {k: new_decisions[k] for k in ["A", "C_floor", "C_cbce"]})
    return 2 if changes else 0


def guard():
    audit = json.loads(Path("reports/p8_cesnet_f3_recompute.json").read_text(encoding="utf-8"))
    assert not audit["decision_changes"], "Stop: family decision changed"
    old_path = ARCHIVE / "results/p5_perclass_guard.csv"
    new_path = Path("results/p8_guard/p5_perclass_guard.csv")
    old, new = read_csv(old_path), read_csv(new_path)
    assert new.noise_target.eq(11).all()
    grid(new.rename(columns={"method": "cell"}), [CESNET], [HARD, SOFT])
    fields, old_records = rows(old_path)
    _, new_records = rows(new_path)
    key = lambda x: (x["dataset"], x["method"], x["seed"], x["tail_flip_rate"])
    replacements = {key(x): x for x in new_records}
    records = [replacements[key(x)] if x["dataset"] == CESNET else {**x, "noise_target": ""} for x in old_records]
    write_rows("results/p5_perclass_guard.csv", fields + ["noise_target"], records)
    partial_records = [{("cell" if k == "method" else k): v for k, v in x.items()} for x in records]
    write_rows("results/p5_perclass_guard.partial.csv", ["cell" if x == "method" else x for x in fields] + ["noise_target"], partial_records)
    frame = read_csv("results/p5_perclass_guard.csv")
    pd.testing.assert_frame_equal(old[old.dataset != CESNET].reset_index(drop=True),
                                  frame.loc[frame.dataset != CESNET, old.columns].reset_index(drop=True), check_exact=True)
    grid(frame.rename(columns={"method": "cell"}), p5.DATASETS, [HARD, SOFT])
    guards = []
    for ds in p5.DATASETS:
        for rate in p5.FULL_RATES:
            g = p4._class_guard(frame[(frame.dataset == ds) & (frame.tail_flip_rate == rate)], SOFT, HARD, 0.05)
            guards.append({**g, "dataset": ds, "tail_flip_rate": rate})
    pd.DataFrame(guards).to_csv("tables/table_p5_perclass_guard.csv", index=False)
    previous = json.loads((ARCHIVE / "reports/p5_perclass_guard.json").read_text(encoding="utf-8"))
    previous_by_key = {(g["dataset"], g["tail_flip_rate"]): g for g in previous["guards"]}
    changes = [{"dataset": g["dataset"], "rate": g["tail_flip_rate"],
                "old": previous_by_key[(g["dataset"], g["tail_flip_rate"])]["passed"], "new": g["passed"]}
               for g in guards if g["passed"] != previous_by_key[(g["dataset"], g["tail_flip_rate"])]["passed"]]
    report = {**previous, "guards": guards, "row_count": 180, "completed": True,
              "all_guards_pass": all(g["passed"] for g in guards), "implementation_amendment": "PA-3"}
    dump("reports/p5_perclass_guard.json", report)
    audit.update({"guard_decision_changes": changes, "guard_new_cesnet_rows": 60,
                  "unaffected_guard_rows_exact": 120, "guard_all_pass": report["all_guards_pass"],
                  "status": "STOP_GUARD_CHANGED" if changes else "DATA_RECOMPUTATION_COMPLETE"})
    dump("reports/p8_cesnet_f3_recompute.json", audit)
    print(json.dumps({"status": audit["status"], "guard_decision_changes": changes, "all_guards_pass": report["all_guards_pass"]}, indent=2))
    return 2 if changes else 0


def handoff():
    """Render the explicitly requested Markdown handoff from frozen results."""
    audit = json.loads(Path("reports/p8_cesnet_f3_recompute.json").read_text(encoding="utf-8"))
    assert audit["status"] == "DATA_RECOMPUTATION_COMPLETE"
    assert not audit["decision_changes"] and not audit["guard_decision_changes"]
    root = ET.parse("reports/p8_pytest.xml").getroot()
    cases = list(root.iter("testcase"))
    failures = [{"name": c.attrib.get("name"), "class": c.attrib.get("classname"),
                 "message": c.find("failure").attrib.get("message", "")}
                for c in cases if c.find("failure") is not None]
    counts = {"tests": len(cases), "failures": len(failures),
              "errors": sum(c.find("error") is not None for c in cases),
              "skipped": sum(c.find("skipped") is not None for c in cases)}
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    claim_cases = [c for c in cases if "test_w5_claim_gates" in c.attrib.get("classname", "")]
    claim_failures = sum(c.find("failure") is not None or c.find("error") is not None for c in claim_cases)
    audit_run = json.loads(Path("reports/p8_number_audit.json").read_text(encoding="utf-8"))
    numeric_failures = [x.strip()[2:] for x in audit_run["stdout"].splitlines() if x.strip().startswith("- ")]
    qa = {"pytest": counts, "failures": failures, "claim_gates": {"tests": len(claim_cases), "failures": claim_failures},
          "number_audit_returncode": audit_run["returncode"], "number_audit_failures": numeric_failures,
          "manuscript_not_modified": True, "submission_ready": False}
    dump("reports/p8_validation_summary.json", qa)
    before, after = audit["before"], audit["after"]
    old_guard = json.loads((ARCHIVE / "reports/p5_perclass_guard.json").read_text(encoding="utf-8"))["guards"]
    new_guard = json.loads(Path("reports/p5_perclass_guard.json").read_text(encoding="utf-8"))["guards"]
    mixed = json.loads(Path("reports/p5_family_a_mixed_effects.json").read_text(encoding="utf-8"))
    old_mixed = json.loads((ARCHIVE / "reports/p5_family_a_mixed_effects.json").read_text(encoding="utf-8"))
    lines = [
        "# P8 CESNET F-3 重跑与完整统计族重算交付报告", "",
        "日期：2026-09-03。**数据重算完成；A/C、逐类护栏裁决均未变化；正文数字同步与最终投稿门禁待协调者复核。**", "",
        "本报告替代先前的前置阻塞报告。旧报告及旧实验产物已保存于 `reproducibility/history/p8_before_f3_20260903/`。", "",
        "## 1. 本轮执行范围", "",
        "- 协调者代码提交 `3e86d10` 已在本地；未做多余 pull，未修改训练器、模型超参或冻结统计函数。",
        "- 一致性测试文件 6 项，加 P5 测试文件 6 项，合计 **12/12 通过**（不是单文件 12 项）。",
        "- 17 份旧结果、缓存、统计表及报告先复制归档并逐文件核对 SHA-256。",
        "- 删除活动 CESNET P5 final/partial 和旧护栏 partial；归档副本可恢复。未删除其他数据集分片。",
        "- CESNET P5 **120/120 method-runs**（4 cells × 10 seeds × 3 rates），逐类护栏另跑 **60/60**（2 cells × 10 × 3）。没有重训 CICIDS/UNSW。",
        "- 冻结参数保持 seeds 0–9、rates 0.4/0.6/0.8、train/test=10,000/5,000；CESNET tier2 屏蔽 + temporal/entity-disjoint。",
        "- 新 CESNET 主实验与护栏记录均有 `noise_target=11`。旧 IDS 记录没有该运行时字段，合并时留空，不伪造历史元数据。",
        "- 新增 `reproducibility/p8_recompute.py` 仅用于合并、调用既有统计函数和出具报告；不包含训练或调参。", "",
        "## 2. 完整族裁决（非 CESNET 子族）", "",
        "| 项目 | 重算范围与结果 | 原裁决是否变化 |", "| --- | --- | --- |",
        "| Family A | 合并 360 行，完整 9 项 Holm；9/9 条件通过，3/3 数据集，robust | 否 |",
        "| Family C | 同一合并帧，完整 18 项 Holm；无正向显著，vs freq-floor=null，vs CBCE=null | 否 |",
        "| Family A Macro 非劣性 | 逐噪声率 n=10，δ=0.01；9/9 通过 | 否 |",
        "| 逐类护栏 | 合并 180 行，9 个数据集×噪声率条件全部通过，阈值 0.05 | 否 |",
        "| Family B / G2 | 不在本次重跑范围，原产物未修改 | 未重新裁决 |", "",
        "重算调用 `_family_tests(size=9/18)`、`_family_verdict`、`_flatten_tests`；不仅比较总体裁决，也检查了各数据集、各噪声率的优效与非劣效通过状态，均未变化。单数据集 runner 输出的 narrow 不用于全文裁决。", "",
        "统计边界：沿用既有单侧 Macro 非劣性检验，不将其表述为双侧等效性已成立；未因本次重跑更改检验方法。混合效应模型显式建模 seed 随机截距与 rate 固定效应，不是把 30 行当作独立种子做配对检验。", "",
        "## 3. CESNET 新旧均值（每行 10 seeds）", "",
        "旧值：buggy target=0；新值：F-3 target=11。", "",
        "| Cell | Rate | Macro 旧 | Macro 新 | Tail 旧 | Tail 新 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in audit["comparison"]:
        lines.append(f"| {row['cell']} | {row['rate']:.1f} | {row['old_macro_f1']:.6f} | {row['new_macro_f1']:.6f} | {row['old_tail_macro_f1']:.6f} | {row['new_tail_macro_f1']:.6f} |")
    lines += ["", "## 4. CESNET Family A：soft − hard", "",
              "| Rate | Δ旧 | Δ新 | Holm p旧 | Holm p新 | 新 d_z | 新 95% CI |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for t in after["family_a"]["tests"]:
        if t["dataset"] != CESNET:
            continue
        old = next(x for x in before["family_a"]["tests"] if x["dataset"] == CESNET and x["tail_flip_rate"] == t["tail_flip_rate"])
        lines.append(f"| {t['tail_flip_rate']:.1f} | {old['mean_delta']:+.6f} | {t['mean_delta']:+.6f} | {old['p_holm_greater']:.8f} | {t['p_holm_greater']:.8f} | {t['cohens_dz']:.4f} | [{t['bootstrap_ci95_low']:+.6f}, {t['bootstrap_ci95_high']:+.6f}] |")
    lines += ["", "## 5. CESNET Family C：无正向显著优势", "",
              "| Reference | Rate | Δ旧 | Δ新 | 新 Holm p | 新 95% CI |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for t in after["family_c"]["tests"]:
        if t["dataset"] != CESNET:
            continue
        old = next(x for x in before["family_c"]["tests"] if x["dataset"] == CESNET and x["tail_flip_rate"] == t["tail_flip_rate"] and x["reference"] == t["reference"])
        lines.append(f"| {t['reference']} | {t['tail_flip_rate']:.1f} | {old['mean_delta']:+.6f} | {t['mean_delta']:+.6f} | {t['p_holm_greater']:.3f} | [{t['bootstrap_ci95_low']:+.6f}, {t['bootstrap_ci95_high']:+.6f}] |")
    lines += ["", "## 6. 逐类护栏与混合效应", "",
              "| CESNET Rate | 旧最差类 Δ | 新最差类 Δ | 新最差类标签 | 通过 |", "| --- | ---: | ---: | ---: | --- |"]
    for g in new_guard:
        if g["dataset"] == CESNET:
            old = next(x for x in old_guard if x["dataset"] == CESNET and x["tail_flip_rate"] == g["tail_flip_rate"])
            lines.append(f"| {g['tail_flip_rate']:.1f} | {old['worst_mean_delta']:+.6f} | {g['worst_mean_delta']:+.6f} | {g['worst_class']} | {g['passed']} |")
    m, om = mixed[CESNET], old_mixed[CESNET]
    lines += ["", f"CESNET 混合效应平均 Δ 从 {om['estimate_average_rate_delta']:+.6f} 更新为 {m['estimate_average_rate_delta']:+.6f}，95% CI [{m['ci95_low']:+.6f}, {m['ci95_high']:+.6f}]，单侧 p={m['p_value_greater']:.4g}。模型收敛，但随机截距方差接近零，保留 MLE boundary 警告；不把该稳健性结果替代主检验。", "",
              "## 7. 不变性检查与历史源冲突", "",
              "- CICIDS/UNSW 分片 SHA-256 均与归档逐字节一致；两分片各 120 行，没有重写。",
              "- 合并主表中非 CESNET 的 **240 行、全部原有列的数值和原始 CSV 数字字符串完全一致**。CSV 新增 noise_target 列，旧行该列为空；因此不声称整个新主表字节不变。",
              "- 护栏中非 CESNET 的 **120 行完全保留**（原有列、逐类 JSON）；没有重训。",
              "- 合并前发现 UNSW 分片中 `5_graph_cdm_freqfloor / seed 8 / rate 0.8` 的 Macro-F1 与已入库主表相差约 0.0000423494（seen-class Macro 同样不同，Tail 相同）。这是重跑前已有差异，本轮没有推定其成因。",
              "- 为满足批复中非 CESNET 逐位不变的优先约束，以已入库全量表为这 240 行的权威来源，只替换 CESNET；没有静默引入该 UNSW 分片差异。精确两个字段值见 recompute JSON 的 historical_shard_discrepancies。",
              "- Holm 是完整族校正：CESNET p 值变化可以使未重训数据集的校正 p 改变，例如 UNSW rate 0.4 从约 0.00909 到 0.00727。这不代表原实验指标改变。", "",
              "| Protected shard | SHA-256 |", "| --- | --- |"]
    for ds, value in audit["unaffected_shard_sha256"].items():
        lines.append(f"| {ds} | `{value}` |")
    lines += ["", "## 8. 门禁结果与协调者待办", "",
              f"- 全仓测试目录：**{counts['passed']} passed / {counts['failures']} failed / {counts['skipped']} skipped / {counts['errors']} errors**；原始记录 `reports/p8_pytest.xml`。",
              f"- 声明门：{len(claim_cases) - claim_failures}/{len(claim_cases)} 通过；数字审计子测试随正文旧数字失败，未绕过。",
              f"- 数字审计：17/19 通过，两项失败，exit={audit_run['returncode']}；原始输出 `reports/p8_number_audit.json`。",
              "- `famA other range`：正文及审计期望仍是 +0.025 到 +0.052，新上界为 +0.056。",
              "- `famA dz range`：正文及审计期望仍是 0.77 到 3.14，新上界为 3.34。",
              "- 已重新生成 7 张 LaTeX 表，实际变动为 tab_main、tab_family_a、tab_family_c。",
              "- 协调者须据新数据同步正文、摘要、答复信与数字审计中的相应期望，并补充 PA-3，再完整复核。没有修改审计规则来制造全绿。",
              "- 本轮未更新论文 PDF、src_pkg、source.zip、EM 粘贴包或公开发布物；现有这些产物仍需协调者重建。历史 P5/P0-5 文字报告未重写，其 CESNET 数字已由本报告取代。",
              "- submission_ready 仍为 false；未 push、未发布 Zenodo、未执行投稿。", "",
              "## 9. PA-3 披露段草稿（待协调者写入稿件）", "",
              "> **PA-3 — Correction of the CESNET asymmetric-noise implementation.** A post-analysis code audit identified a discrepancy between the frozen F-3 mapping and the P5 factor-ablation implementation: CESNET corruption was directed to label 0 (`apple-itunes`) instead of the prescribed label 11 (`microsoft-diagnostic`). We corrected the implementation to use the shared target-selection function, added target metadata to newly generated result rows, and invalidated the affected cached runs. We reran only the CESNET full-tier ablations (120 method-runs) and the accompanying per-class checks (60 method-runs), without changing the frozen seeds, noise rates, train/test windows, masking, splits, model settings, or statistical families. The CICIDS and UNSW entries in the committed full-table baseline were preserved exactly. Family A (9 hypotheses) and Family C (18 hypotheses) were recomputed on the complete three-dataset frame, including non-inferiority and mixed-effects checks. CESNET soft-minus-hard Tail-F1 differences changed from +0.0340/+0.0516/+0.0470 to +0.0406/+0.0459/+0.0557 at rates 0.4/0.6/0.8. Family A remains robust, with all nine rate-level superiority tests passing; neither Family C comparison supports a superiority claim, and all per-class regression guards remain satisfied. P4 Family B was outside the affected implementation path and was not rerun. Superseded artifacts and the before/after comparison are retained. This is a correction of a protocol implementation error discovered after analysis, not a newly preregistered experiment.", "",
              "## 10. 变更文件清单与复核入口", "",
              "- 数据更新：`results/p5_factor_ablation_full.csv`、`results/p5_factor_ablation_full_cesnet_tls_year22.csv`、`results/p5_perclass_guard.csv`、`results/p8_guard/p5_perclass_guard.csv`。两个 IDS 分片仅补充纳入版本控制，字节未改。",
              "- 统计更新：三份 `tables/table_p5_{family_a,family_c,perclass_guard}.csv`，以及 `reports/p5_factor_ablation_full.json`、`reports/p5_family_a_mixed_effects.json`、`reports/p5_perclass_guard.json`。",
              "- 表格更新：`paper/springer/tables_w5/tab_main.tex`、`tab_family_a.tex`、`tab_family_c.tex`；其余生成表无内容变化。",
              "- 新增交接证据：本报告、`reports/p8_cesnet_f3_recompute.json`、`reports/p8_validation_summary.json`、`reports/p8_number_audit.json`、`reports/p8_pytest.xml`、`reports/p8_artifact_manifest.json`。",
              "- 复核工具：`reproducibility/p8_recompute.py`、`tests/test_p8_recompute.py`。新增 `.gitattributes` 仅对 P8 历史快照与相关结果 CSV 禁用换行转换，保护跨 checkout 的字节不变检查；不改模型或协议。",
              "- 历史归档：`reproducibility/history/p8_before_f3_20260903/` 中 17 个原始文件，逐文件列表及 SHA-256 见 manifest。活动 partial 缓存未作为新结果纳入提交。", "",
              "- 机器可读新旧对照和所有判据：`reports/p8_cesnet_f3_recompute.json`。",
              "- 重算/护栏原始结果：`results/p5_factor_ablation_full.csv`、`results/p5_perclass_guard.csv`。",
              "- 完整统计族与护栏：`tables/table_p5_family_a.csv`、`tables/table_p5_family_c.csv`、`tables/table_p5_perclass_guard.csv`。",
              "- 独立 SciPy / statsmodels 复核、原值不变测试：`tests/test_p8_recompute.py`。",
              "- 输入/输出哈希、环境与提交基线：`reports/p8_artifact_manifest.json`。",
              "- 代码提交哈希在会话最终交付中给出；无线上发布。", "",
              "```powershell",
              "# 清理/隔离缓存且保留归档后执行；不要删除已有归档。",
              "python -m src.paper.p5_factor_ablation --tier full --datasets cesnet_tls_year22 --reports reports/p8_staging --tables tables/p8_staging",
              "python -m reproducibility.p8_recompute family",
              "# 上一步 exit=2 必须停下，不得继续下面步骤。",
              "python -m src.paper.p5_perclass_guard --datasets cesnet_tls_year22 --out results/p8_guard --reports reports/p8_staging --tables tables/p8_staging",
              "python -m reproducibility.p8_recompute guard",
              "# 若护栏裁决变化，必须停下。",
              "python -m src.paper.w5_result_tables",
              "python -m src.paper.w5_number_audit",
              "python -m pytest -q tests --junitxml=reports/p8_pytest.xml",
              "```", ""]
    Path("reports/p8_cesnet_f3_rerun.md").write_text("\n".join(lines), encoding="utf-8")
    artifact_paths = ["results/p5_factor_ablation_full.csv", "results/p5_perclass_guard.csv",
                      f"results/p5_factor_ablation_full_{CESNET}.csv", "results/p8_guard/p5_perclass_guard.csv",
                      "tables/table_p5_family_a.csv", "tables/table_p5_family_c.csv", "tables/table_p5_perclass_guard.csv",
                      "reports/p5_factor_ablation_full.json", "reports/p5_family_a_mixed_effects.json", "reports/p5_perclass_guard.json",
                      "reports/p8_cesnet_f3_rerun.md", "reports/p8_cesnet_f3_recompute.json", "reports/p8_validation_summary.json",
                      "reports/p8_number_audit.json", "reports/p8_pytest.xml",
                      "src/paper/p5_factor_ablation.py", "src/paper/p5_perclass_guard.py", "src/paper/p3_proxy_audit.py",
                      "src/paper/p4_cl_graph_combo.py", "src/paper/p2f_tighten.py", "reproducibility/noise_map.json",
                      "reproducibility/p8_recompute.py", "tests/test_p8_recompute.py", ".gitattributes"]
    artifact_paths += [str(p).replace("\\", "/") for p in Path("paper/springer/tables_w5").glob("*.tex")]
    import scipy, sklearn, statsmodels
    manifest = {"recorded_at": datetime.now(timezone.utc).isoformat(),
                "hash_scope": "Raw bytes in the execution worktree; archive and listed result CSVs are protected by -text attributes. Other text files may be newline-normalized by Git on other platforms.",
                "coordinator_baseline": subprocess.check_output(["git", "rev-parse", "3e86d10"], text=True).strip(),
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                "scipy": scipy.__version__, "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__},
                "archive": {str(p.relative_to(ARCHIVE)).replace("\\", "/"): sha(p) for p in sorted(ARCHIVE.rglob("*")) if p.is_file()},
                "artifacts": {p: sha(p) for p in artifact_paths}, "submission_ready": False}
    dump("reports/p8_artifact_manifest.json", manifest)
    print(json.dumps(qa, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["family", "guard", "handoff"])
    raise SystemExit({"family": family, "guard": guard, "handoff": handoff}[parser.parse_args().stage]())
