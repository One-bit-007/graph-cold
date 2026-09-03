# P8 CESNET F-3 重跑与完整统计族重算交付报告

日期：2026-09-03。**数据重算完成；A/C、逐类护栏裁决均未变化；正文数字同步与最终投稿门禁待协调者复核。**

本报告替代先前的前置阻塞报告。旧报告及旧实验产物已保存于 `reproducibility/history/p8_before_f3_20260903/`。

## 1. 本轮执行范围

- 协调者代码提交 `3e86d10` 已在本地；未做多余 pull，未修改训练器、模型超参或冻结统计函数。
- 一致性测试文件 6 项，加 P5 测试文件 6 项，合计 **12/12 通过**（不是单文件 12 项）。
- 17 份旧结果、缓存、统计表及报告先复制归档并逐文件核对 SHA-256。
- 删除活动 CESNET P5 final/partial 和旧护栏 partial；归档副本可恢复。未删除其他数据集分片。
- CESNET P5 **120/120 method-runs**（4 cells × 10 seeds × 3 rates），逐类护栏另跑 **60/60**（2 cells × 10 × 3）。没有重训 CICIDS/UNSW。
- 冻结参数保持 seeds 0–9、rates 0.4/0.6/0.8、train/test=10,000/5,000；CESNET tier2 屏蔽 + temporal/entity-disjoint。
- 新 CESNET 主实验与护栏记录均有 `noise_target=11`。旧 IDS 记录没有该运行时字段，合并时留空，不伪造历史元数据。
- 新增 `reproducibility/p8_recompute.py` 仅用于合并、调用既有统计函数和出具报告；不包含训练或调参。

## 2. 完整族裁决（非 CESNET 子族）

| 项目 | 重算范围与结果 | 原裁决是否变化 |
| --- | --- | --- |
| Family A | 合并 360 行，完整 9 项 Holm；9/9 条件通过，3/3 数据集，robust | 否 |
| Family C | 同一合并帧，完整 18 项 Holm；无正向显著，vs freq-floor=null，vs CBCE=null | 否 |
| Family A Macro 非劣性 | 逐噪声率 n=10，δ=0.01；9/9 通过 | 否 |
| 逐类护栏 | 合并 180 行，9 个数据集×噪声率条件全部通过，阈值 0.05 | 否 |
| Family B / G2 | 不在本次重跑范围，原产物未修改 | 未重新裁决 |

重算调用 `_family_tests(size=9/18)`、`_family_verdict`、`_flatten_tests`；不仅比较总体裁决，也检查了各数据集、各噪声率的优效与非劣效通过状态，均未变化。单数据集 runner 输出的 narrow 不用于全文裁决。

统计边界：沿用既有单侧 Macro 非劣性检验，不将其表述为双侧等效性已成立；未因本次重跑更改检验方法。混合效应模型显式建模 seed 随机截距与 rate 固定效应，不是把 30 行当作独立种子做配对检验。

## 3. CESNET 新旧均值（每行 10 seeds）

旧值：buggy target=0；新值：F-3 target=11。

| Cell | Rate | Macro 旧 | Macro 新 | Tail 旧 | Tail 新 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3_graph_cdm_hard | 0.4 | 0.476411 | 0.492407 | 0.342244 | 0.353825 |
| 3_graph_cdm_hard | 0.6 | 0.409790 | 0.420872 | 0.208628 | 0.217612 |
| 3_graph_cdm_hard | 0.8 | 0.335595 | 0.343399 | 0.064604 | 0.063524 |
| 4_graph_cdm_soft | 0.4 | 0.500483 | 0.521785 | 0.376271 | 0.394421 |
| 4_graph_cdm_soft | 0.6 | 0.441018 | 0.452144 | 0.260278 | 0.263537 |
| 4_graph_cdm_soft | 0.8 | 0.366947 | 0.377593 | 0.111629 | 0.119201 |
| 5_graph_cdm_freqfloor | 0.4 | 0.502331 | 0.523886 | 0.380284 | 0.398985 |
| 5_graph_cdm_freqfloor | 0.6 | 0.437322 | 0.450112 | 0.253822 | 0.263560 |
| 5_graph_cdm_freqfloor | 0.8 | 0.366559 | 0.376074 | 0.113291 | 0.115942 |
| 6_nograph_cbce | 0.4 | 0.503647 | 0.519579 | 0.379674 | 0.391137 |
| 6_nograph_cbce | 0.6 | 0.434426 | 0.454017 | 0.249827 | 0.263800 |
| 6_nograph_cbce | 0.8 | 0.366285 | 0.377028 | 0.107596 | 0.117507 |

## 4. CESNET Family A：soft − hard

| Rate | Δ旧 | Δ新 | Holm p旧 | Holm p新 | 新 d_z | 新 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.4 | +0.034026 | +0.040596 | 0.01506097 | 0.00099449 | 1.7282 | [+0.027136, +0.055574] |
| 0.6 | +0.051650 | +0.045926 | 0.00003690 | 0.00009211 | 2.4755 | [+0.036282, +0.057676] |
| 0.8 | +0.047025 | +0.055677 | 0.00001698 | 0.00001027 | 3.3372 | [+0.045859, +0.065290] |

## 5. CESNET Family C：无正向显著优势

| Reference | Rate | Δ旧 | Δ新 | 新 Holm p | 新 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| 5_graph_cdm_freqfloor | 0.4 | -0.004013 | -0.004563 | 1.000 | [-0.012063, +0.004941] |
| 6_nograph_cbce | 0.4 | -0.003403 | +0.003284 | 1.000 | [-0.003160, +0.009925] |
| 5_graph_cdm_freqfloor | 0.6 | +0.006456 | -0.000023 | 1.000 | [-0.008703, +0.008602] |
| 6_nograph_cbce | 0.6 | +0.010452 | -0.000263 | 1.000 | [-0.010142, +0.009454] |
| 5_graph_cdm_freqfloor | 0.8 | -0.001662 | +0.003259 | 1.000 | [-0.003314, +0.010219] |
| 6_nograph_cbce | 0.8 | +0.004033 | +0.001694 | 1.000 | [-0.004310, +0.007246] |

## 6. 逐类护栏与混合效应

| CESNET Rate | 旧最差类 Δ | 新最差类 Δ | 新最差类标签 | 通过 |
| --- | ---: | ---: | ---: | --- |
| 0.4 | -0.028942 | -0.015363 | 1 | True |
| 0.6 | -0.036460 | -0.027788 | 5 | True |
| 0.8 | -0.007781 | -0.027261 | 14 | True |

CESNET 混合效应平均 Δ 从 +0.044234 更新为 +0.047400，95% CI [+0.040320, +0.054479]，单侧 p=1.231e-39。模型收敛，但随机截距方差接近零，保留 MLE boundary 警告；不把该稳健性结果替代主检验。

## 7. 不变性检查与历史源冲突

- CICIDS/UNSW 分片 SHA-256 均与归档逐字节一致；两分片各 120 行，没有重写。
- 合并主表中非 CESNET 的 **240 行、全部原有列的数值和原始 CSV 数字字符串完全一致**。CSV 新增 noise_target 列，旧行该列为空；因此不声称整个新主表字节不变。
- 护栏中非 CESNET 的 **120 行完全保留**（原有列、逐类 JSON）；没有重训。
- 合并前发现 UNSW 分片中 `5_graph_cdm_freqfloor / seed 8 / rate 0.8` 的 Macro-F1 与已入库主表相差约 0.0000423494（seen-class Macro 同样不同，Tail 相同）。这是重跑前已有差异，本轮没有推定其成因。
- 为满足批复中非 CESNET 逐位不变的优先约束，以已入库全量表为这 240 行的权威来源，只替换 CESNET；没有静默引入该 UNSW 分片差异。精确两个字段值见 recompute JSON 的 historical_shard_discrepancies。
- Holm 是完整族校正：CESNET p 值变化可以使未重训数据集的校正 p 改变，例如 UNSW rate 0.4 从约 0.00909 到 0.00727。这不代表原实验指标改变。

| Protected shard | SHA-256 |
| --- | --- |
| cicids2017 | `1cb6e8ef5b603b3d8c99ad1a997ea486f8302e46370e583fd11b9afc6135d30d` |
| unsw_nb15 | `4360a44e66b6a85853d160128e85edd139ad6a857ca51af04d5651af834005d0` |

## 8. 门禁结果与协调者待办

- 全仓测试目录：**289 passed / 1 failed / 3 skipped / 0 errors**；原始记录 `reports/p8_pytest.xml`。
- 声明门：24/25 通过；数字审计子测试随正文旧数字失败，未绕过。
- 数字审计：17/19 通过，两项失败，exit=1；原始输出 `reports/p8_number_audit.json`。
- `famA other range`：正文及审计期望仍是 +0.025 到 +0.052，新上界为 +0.056。
- `famA dz range`：正文及审计期望仍是 0.77 到 3.14，新上界为 3.34。
- 已重新生成 7 张 LaTeX 表，实际变动为 tab_main、tab_family_a、tab_family_c。
- 协调者须据新数据同步正文、摘要、答复信与数字审计中的相应期望，并补充 PA-3，再完整复核。没有修改审计规则来制造全绿。
- 本轮未更新论文 PDF、src_pkg、source.zip、EM 粘贴包或公开发布物；现有这些产物仍需协调者重建。历史 P5/P0-5 文字报告未重写，其 CESNET 数字已由本报告取代。
- submission_ready 仍为 false；未 push、未发布 Zenodo、未执行投稿。

## 9. PA-3 披露段草稿（待协调者写入稿件）

> **PA-3 — Correction of the CESNET asymmetric-noise implementation.** A post-analysis code audit identified a discrepancy between the frozen F-3 mapping and the P5 factor-ablation implementation: CESNET corruption was directed to label 0 (`apple-itunes`) instead of the prescribed label 11 (`microsoft-diagnostic`). We corrected the implementation to use the shared target-selection function, added target metadata to newly generated result rows, and invalidated the affected cached runs. We reran only the CESNET full-tier ablations (120 method-runs) and the accompanying per-class checks (60 method-runs), without changing the frozen seeds, noise rates, train/test windows, masking, splits, model settings, or statistical families. The CICIDS and UNSW entries in the committed full-table baseline were preserved exactly. Family A (9 hypotheses) and Family C (18 hypotheses) were recomputed on the complete three-dataset frame, including non-inferiority and mixed-effects checks. CESNET soft-minus-hard Tail-F1 differences changed from +0.0340/+0.0516/+0.0470 to +0.0406/+0.0459/+0.0557 at rates 0.4/0.6/0.8. Family A remains robust, with all nine rate-level superiority tests passing; neither Family C comparison supports a superiority claim, and all per-class regression guards remain satisfied. P4 Family B was outside the affected implementation path and was not rerun. Superseded artifacts and the before/after comparison are retained. This is a correction of a protocol implementation error discovered after analysis, not a newly preregistered experiment.

## 10. 变更文件清单与复核入口

- 数据更新：`results/p5_factor_ablation_full.csv`、`results/p5_factor_ablation_full_cesnet_tls_year22.csv`、`results/p5_perclass_guard.csv`、`results/p8_guard/p5_perclass_guard.csv`。两个 IDS 分片仅补充纳入版本控制，字节未改。
- 统计更新：三份 `tables/table_p5_{family_a,family_c,perclass_guard}.csv`，以及 `reports/p5_factor_ablation_full.json`、`reports/p5_family_a_mixed_effects.json`、`reports/p5_perclass_guard.json`。
- 表格更新：`paper/springer/tables_w5/tab_main.tex`、`tab_family_a.tex`、`tab_family_c.tex`；其余生成表无内容变化。
- 新增交接证据：本报告、`reports/p8_cesnet_f3_recompute.json`、`reports/p8_validation_summary.json`、`reports/p8_number_audit.json`、`reports/p8_pytest.xml`、`reports/p8_artifact_manifest.json`。
- 复核工具：`reproducibility/p8_recompute.py`、`tests/test_p8_recompute.py`。新增 `.gitattributes` 仅对 P8 历史快照与相关结果 CSV 禁用换行转换，保护跨 checkout 的字节不变检查；不改模型或协议。
- 历史归档：`reproducibility/history/p8_before_f3_20260903/` 中 17 个原始文件，逐文件列表及 SHA-256 见 manifest。活动 partial 缓存未作为新结果纳入提交。

- 机器可读新旧对照和所有判据：`reports/p8_cesnet_f3_recompute.json`。
- 重算/护栏原始结果：`results/p5_factor_ablation_full.csv`、`results/p5_perclass_guard.csv`。
- 完整统计族与护栏：`tables/table_p5_family_a.csv`、`tables/table_p5_family_c.csv`、`tables/table_p5_perclass_guard.csv`。
- 独立 SciPy / statsmodels 复核、原值不变测试：`tests/test_p8_recompute.py`。
- 输入/输出哈希、环境与提交基线：`reports/p8_artifact_manifest.json`。
- 代码提交哈希在会话最终交付中给出；无线上发布。

```powershell
# 清理/隔离缓存且保留归档后执行；不要删除已有归档。
python -m src.paper.p5_factor_ablation --tier full --datasets cesnet_tls_year22 --reports reports/p8_staging --tables tables/p8_staging
python -m reproducibility.p8_recompute family
# 上一步 exit=2 必须停下，不得继续下面步骤。
python -m src.paper.p5_perclass_guard --datasets cesnet_tls_year22 --out results/p8_guard --reports reports/p8_staging --tables tables/p8_staging
python -m reproducibility.p8_recompute guard
# 若护栏裁决变化，必须停下。
python -m src.paper.w5_result_tables
python -m src.paper.w5_number_audit
python -m pytest -q tests --junitxml=reports/p8_pytest.xml
```
