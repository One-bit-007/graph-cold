# P10 UNSW 标签泄漏修复：阶段性交付与暂停报告

日期：2026-09-03。工作包：`docs/prompts/prompt_codex_p10_unsw_label.md`。

**状态：A/B 审计任务完成；C 的 Family A/C 与逐类护栏完成。局部统计判定变化触发暂停；Family B、泄漏分解、P9 未继续。P10 尚未全部完成，`submission_ready=false`。**

这里“B 审计任务”指特征/一致性排查，不指推断 Family B。不得用旧 G2=null 代替尚未执行的新 Family B。

## 1. 根因、影响与修复

UNSW 同时提供多类 `attack_cat` 与二元 `label`。旧配置只丢弃 `id`，特征构建只丢弃选定的多类目标，导致 `label` 进入所有 UNSW 条件（包括 tier-2 FINAL）。训练目标接受合成噪声翻转时，该二元特征仍保留干净 attack/normal 信息。这是直接标签投影泄漏，影响所有使用该加载器的旧 UNSW 方法结果，非单一算法的问题。

旧管线中该特征经标准化后仍与二元目标逐行一致。修复前主协议为 185 维；修复后为 184 维。配置显式 `drop_cols: [id, label]`，加载器 `_feature_inputs` 又在所有 UNSW 入口永久排除 `label/attack_cat` 及其大小写、空格、下划线别名。二元任务反过来以 `label` 作目标时，仍不可把 `attack_cat` 作为输入。PRX/ALL 不可恢复这些标签字段。

P3 Tail-F1 已委托 P9 canonical helper；P4 后续执行同样用 canonical，且不再给 CL 构造器传真实注入率。旧 P4 masked CSV 未修改；新的完整 B 必须把 P9 canonical 派生的另两数据集与新 UNSW 合并，不能混用两种定义。

历史留存：`reproducibility/history/p10_before_unsw_label_20260903/` 保存 19 份关键修复前代码/配置/结果/统计副本；`reports/p10_preservation_manifest.json` 锁定 494 份历史材料哈希。本轮没有删除或覆盖旧结果。

## 2. 三数据集审计与一致性

- 完整输入列数：CICIDS 77、CESNET 29、UNSW 184；逐列判定见 `p10_feature_audit.md`，逐列统计见 `results/p10_feature_association_scan.csv`。
- 30 个冻结窗口的训练/测试标签与采样索引哈希一致；CICIDS/CESNET 特征字节一致。UNSW 仅在诊断时重新插入旧二元列即可恢复旧 X 哈希，确认唯一特征变化是删掉该列；诊断重建列从未进入训练。
- 自动扫描：每 split 最多 20,000 行，seed=0；高基数列分 32 分位箱。红标阈值 normalized MI≥0.95 或 max absolute one-v-rest correlation≥0.995。旧 `label` 正控二元 normalized MI≈1；raw MI=0.542995 nats，二者不可混称。现输入没有达到阈值的列。
- CESNET 只作服务类别目标，不构造攻击/正常指标。扫描未红标不代表彻底无代理；环境/场景、重复与划分风险仍须保留。
- 噪声目标、tail 定义、canonical 指标、屏蔽、划分、窗口与历史例外清单见 `p10_consistency_audit.md`。历史 masked 指标路径被显式列出，而非虚称全仓历史口径已经统一。

**新增待裁定风险：预处理尚非完全 train-only。** 两条加载路径均在训练/测试分离前调用 `_build_feature_frame`，其中中位数缺失填补、全缺失列判定、数值/类别类型判断和 one-hot 词表由合并输入生成；只有 StandardScaler 明确仅拟合训练集。这不是新的直接标签字段，数值影响尚未量化，但使“全流程严格归纳”声明缺乏依据。本轮未改动该步骤；修复可能影响三数据集，需协调者另行授权并冻结范围。当前新结果应准确称为 **label-only correction，保留原预处理**。

## 3. 实际完成的重跑与完整族重算

本轮只训练 UNSW：10 seeds × 3 rates × 4 cells = **120 行**（#3/#4/#5/#6）；从同一批 #3/#4 预测记录 **60 行逐类护栏**，没有再拟合一套模型冒充完全相同预测。每场景核对 noisy-y 哈希与冻结目标 0。

合并后 `p5_full_merged.csv` **360 行**、护栏 **180 行**；CICIDS/CESNET 分别 240/120 行逐条原始 CSV 字节保留。A/C 在三数据集合并帧上调用 `_family_tests(size=9/18)`，不是 UNSW 子族补 p=1。

| 对象 | 修复前 | 修复后 | 说明 |
|---|---|---|---|
| Family A | robust；3/3 数据集 PASS | robust；3/3 数据集 PASS | 9/9 正向 Holm 检验通过 |
| Family C：soft vs freq-floor | null | null | 未检出 soft 优势，不是等效性结论 |
| Family C：soft vs CBCE | null | null | 未检出 soft 优势，不是等效性结论 |
| A 逐类护栏 | 9/9 通过 | 9/9 通过 | UNSW 最差类均值回归仍远小于 0.05 |
| Family B / G2 | 历史 null（受污染 UNSW） | **未重跑 / 不作裁决** | 因前置局部判定变化暂停 |
| UNSW 泄漏分解 | 历史数字受污染 | **未重跑** | 不能继续用于修复后的定量主张 |

注意：CICIDS/CESNET 实验行不变，不要求其 Holm 校正 p 值不变；完整族内 UNSW 的 p 值改变会影响校正排序，这正是必须重算完整族的原因。

### UNSW 新旧均值（仅描述性，10 seeds × 3 rates）

下表 30 场景均值不是 30 个独立推断样本，正式推断仍按每个 rate 的 10 个配对 seeds。

| cell | 旧 Macro-F1 | 新 Macro-F1 | 旧 canonical Tail-F1 | 新 canonical Tail-F1 |
|---|---:|---:|---:|---:|
| #3 hard | 0.459849 | 0.402271 | 0.161055 | 0.154821 |
| #4 soft | 0.482239 | 0.425452 | 0.199508 | 0.195551 |
| #5 freq-floor | 0.491839 | 0.433285 | 0.216584 | 0.211166 |
| #6 CBCE | 0.488209 | 0.428416 | 0.209633 | 0.201481 |

删除泄漏后的 Macro-F1 明显下降，不能因为总体机制裁决仍存活就淡化旧绝对性能数字被污染的事实。

| UNSW rate | 新 ΔTail（soft−hard） | 完整 A Holm p | bootstrap 95% CI |
|---|---:|---:|---|
| 0.4 | +0.030654 | 0.00125246 | [+0.018916, +0.042159] |
| 0.6 | +0.042368 | 0.000193897 | [+0.031493, +0.053728] |
| 0.8 | +0.049170 | 0.0000630893 | [+0.038067, +0.060125] |

## 4. 触发暂停的四项局部变化

**总体 A/C 标签没有翻转；变化发生于附属非劣判定和反向效应区间。** 按“任何裁决变化即停下上报”从严执行，等待协调者接受这些影响后再运行下游。

| 比较 / rate | 修复前 | 修复后 | 解释 |
|---|---|---|---|
| soft−freq-floor Tail，0.8 | CI [−0.020156, +0.000629] | CI [−0.021236, −0.001494] | 从跨零变为全负；反向证据加强 |
| soft vs CBCE Macro NI，0.4 | p=0.144594，不通过 | p=0.017114，通过 | δ=0.01 的单侧非劣检验 |
| soft vs CBCE Macro NI，0.6 | p=0.246144，不通过 | p=0.020567，通过 | δ=0.01 的单侧非劣检验 |
| soft−CBCE Tail，0.6 | CI [−0.025178, −0.003416] | CI [−0.017705, +0.005390] | 从全负变为跨零；反向证据减弱 |

这些 bootstrap 区间是描述性逐格区间，不是新增且完成 Holm 校正的“反向显著性族”。Macro NI 也不等于双侧 TOST 等效性。本轮没有把这些变化改写成性能优越主张。

## 5. 数值复核与门禁

`src/paper/p10_verify_delivery.py` 不重训，独立用 scipy 单样本差值检验与 statsmodels Holm 重算 27 项检验，逐项匹配完整族输出；重建 bootstrap seed=42、2,000 次的区间。60 行护栏回算 canonical Tail-F1 与主结果最大差 `8.326672684688674e-17`；修复前 A/C 重算匹配旧冻结报告。历史哈希与另两数据集原始记录字节验证通过。

验证输出：`reports/p10/validation.json`。当前资料可交协调者复核，但 **不是完整 P10 验收，更不是投稿就绪**。新运行契约为 `reports/p10/run_contract.json`，代码/输入变化时禁止复用旧缓存；恢复需显式审阅，不可删除契约绕过。

全仓回归：`python -m pytest -q` → **340 passed, 3 skipped, 10 warnings**（125.29 秒）。警告为测试中的优化器收敛与近常量样本统计精度提示；不把 skipped 计为通过。定向新增测试 17 项通过；完整复核脚本另行成功执行。Family B/泄漏阶段的编排代码已提供，但因停止条件尚未得到真实运行验收。

## 6. PA-4 披露草稿（截至当前，非完整修复宣告）

> During the cross-runner audit, we identified a binary label projection in our own UNSW-NB15 feature pipeline. The `label` field, which indicates attack versus normal traffic, had remained in the feature matrix while the multiclass training target was subjected to synthetic label corruption. This contaminated the earlier UNSW results across methods. We permanently excluded both target fields from model inputs and preserved the previous artifacts for traceability. We reran the 120 UNSW factor-ablation cells and recorded per-class scores for the 60 hard/soft cells on the unchanged sampling and noise grid. After recomputing complete Families A and C with the unchanged CICIDS and CESNET records, Family A remained robust and neither Family C superiority claim was supported. However, some UNSW non-inferiority decisions and descriptive reverse-effect intervals changed; these changes are reported explicitly. The correction is not yet complete: Family B and the UNSW leakage decomposition await rerunning after review of this gate. Accordingly, we make no claim at this stage that the corrected G2 verdict is unchanged. The detection illustrates the value of applying the audit to our own pipeline, while also revealing a serious omission in the earlier validation; it does not establish that the remaining pipeline is leakage-free.

配套限制草稿：

> This rerun changes the label exclusion only. The retained preprocessing still derives imputation statistics and categorical feature vocabulary before separating the training and test rows, although scaling is fitted on training rows alone. We have not yet quantified the impact of replacing that preprocessing with train-only transformations. Accordingly, these results should not be characterized as a fully inductive end-to-end pipeline.

最终 PA-4 必须在 B 与泄漏分解跑完后据实更新，不能预先填入“G2 remains null”。发现错误可以体现审计的作用，但不能掩盖初始质控失败或作为整体协议有效性的充分证明。

## 7. 精确交付与待批复事项

已交付：

- `reports/p10_feature_audit.md/.json`：三表完整列清单与运行时窗口核验；`results/p10_feature_association_scan.csv`：580 行 train/test 特征扫描。
- `reports/p10_consistency_audit.md`：共享概念、例外与恢复注意事项。
- `src/paper/leak_scan.py`、两组新增守卫测试、P10 分阶段运行器及独立复核脚本。
- `results/p10/p5_unsw.csv`、`p5_guard_unsw.csv`：修复后 UNSW 120/60 行；对应 partial 完整保留。
- `results/p10/p5_full_merged.csv`、`p5_guard_full_merged.csv`：三数据集合并 360/180 行。
- `reports/p10/family_ac_complete.json`：新旧完整 A/C、9 格护栏、四项变化与 stop 标识；`tables/p10/family_a.csv`、`family_c.csv`。
- `reports/p10/validation.json`、`run_contract.json`；历史归档与保存清单。

待协调者批复：**接受四项局部变化后，先继续仅修 `label` 的 Family B/泄漏分解，还是扩大到 train-only 预处理修复及可能的三数据集重跑？** 后者超出当前“另两数据集不动”的授权，必须先裁定范围。批准前者则应保留预处理限制，先保存门禁审阅记录，再恢复下游；如 B 再有变化仍须停止。P9 最后恢复，并显式迁移其代码/输入契约以保留已有 60 行。

协调者后续稿件影响面：主表 UNSW 的 soft/hard/CL/CL+Graph；Family A/C/B 相关表与叙事；UNSW 泄漏分解；PA-4、答复信与数字门禁；旧 P7 iid UNSW 基线也不应被当作无污染支持。本文稿、表生成器、远端与发布资料本轮均未修改。
