# Codex 工作包 P10：UNSW `label` 泄漏修复 + 系统性泄漏/一致性排查

> **背景（Codex 在 P9 中发现，协调者已独立核验并确认严重）**：
> UNSW-NB15 分片文件同时含 `attack_cat`（多类目标）与 `label`（二元攻击指示，
> =1 iff 攻击）。`configs/datasets.yaml` 只 drop `id`；`src/paper/p3_proxy_audit.py`
> 的 `PROXY_TIERS["unsw_nb15"]` tier1=`is_sm_ips_ports`、tier2=`ct_*`——**`label`
> 不在任何屏蔽层**。因此二元 `label` 在所有条件（含冻结 tier-2 主协议）下都留在
> UNSW 特征里；且噪声注入翻转的是 `attack_cat` 训练标签，`label` 特征仍持**干净
> 二元真值** → 等于 clean-label 二元代理进了特征。**污染所有 UNSW 冻结结果。**
>
> 这是本文自身审计的那类泄漏出现在自己管线里。必须修复 + 重跑 UNSW 全线 + 诚实披露，
> **并顺带做一次系统性排查，杜绝同类遗漏**（这是连续第三个"停下才发现"的问题）。

## 不可违反的规则

1. **CICIDS/CESNET 冻结产物不动**（它们不含此泄漏）；只重跑 UNSW 受影响实验。
2. 禁 oracle 泄漏；F-5 合规；`p3._noise_target` + canonical Tail-F1（P9 helper）沿用。
3. 已有产物只增不改；旧 UNSW 产物归档披露，不覆盖。
4. `submission_ready` 保持 false。发现预期外问题即停下报告。
5. **不改正文/生成器**——只出数据+报告，协调者据此更新稿件。

## 【协调者裁定，2026-09-03 更新】执行顺序改为“先审计穷尽 → 一次综合修复 → 一次最终重跑”

> Codex 已在 label 修复中又发现**第四个**问题（预处理泄漏：`_build_feature_frame`
> 在 load 时对**全集**算 `numeric.median()` 与 `get_dummies` 词表，划分在其后→
> 测试统计量泄漏进训练预处理）。协调者已独立核实属实。
> 连续四个“停下才发现”说明“修一个跑一遍”不可取。**不要**先把 label-only 的
> Family B / 泄漏分解 / P9 跑完（预处理修复会让它们再变）。按以下顺序：

### 阶段 1：审计穷尽（不跑昂贵重训，先把“还漏什么”一次列完）
1. 完成任务 B 的全部子项（特征列互信息扫描 + 跨 runner 一致性 + `leak_scan.py`）；
2. **新增预处理边界审计**：确认所有 fit-依赖步骤（中位数填补、one-hot 词表、
   任何 scaler/归一化、图构建的邻居统计）**是否在划分前对全集 fit**；逐一列出；
3. 产出 `reports/p10_full_audit.md`：一张“待修清单”（label 泄漏、预处理泄漏、
   任何互信息扫描新命中项、任何跨 runner 不一致项），每项标注修复方案。
   **此阶段目标：把所有同类隐患一次挖净，不再出第五个。**完成后停下，交协调者
   确认清单完整再进阶段 2。

### 阶段 2：一次综合修复（待阶段 1 清单定稿后）
- **预处理改为 train-only fit**：把 `_build_feature_frame` 的 fit-依赖步骤（中位数、
  one-hot 词表、scaler）从 load 时全集下**移到场景构建器内、仅对训练 fold fit、
  变换测试**（结构性修正，无论幅度）；label 永久剔除（任务 A）已做；
- **先量化**：对一个代表性切片（如 CICIDS seed0 rate0.6）测 train-only vs 全集预处理的
  指标 Δ，写入报告（判断幅度）；无论大小都采用该结构性修正。

### 阶段 3：一次最终重跑（两项修正合并，三数据集）
- 在 label 剔除 + train-only 预处理两项修正下，**一次性重跑三数据集的**
  Family A / B / C、逐类护栏、泄漏分解，并重算全部裁决（A/B/C 完整族）；
- **CICIDS/CESNET 不再可声称逐位不变**（预处理变了）——改为报告新旧对比 Δ；
- 任何裁决变化（尤其 G2、Family A robust）**立即停下上报**，不自行改写；
- 随后恢复 P9（FINE 剔 30 场景 + 主表 canonical）在修正后数据上完成。

---

## 任务 A：修复 UNSW `label` 泄漏（根因）【已完成，保留记录】

1. 把 `label` 加入 UNSW 的**标签构造字段**处理：正确做法是它属于 tier1 直接标签字段
   （二元真值），加入 `PROXY_TIERS["unsw_nb15"]["tier1"]`（或等价的"永久排除"列表，
   因为它在**任何**条件下都不该作特征——它不是"可解除屏蔽做审计"的代理，而是标签本身的
   二元投影）。**裁定：`label` 永久从 UNSW 特征剔除**（写入 config drop_cols 与/或加载器
   硬排除），并加守卫测试断言 UNSW 特征集不含 `label`。
2. 自检：修复后 UNSW 特征列清单打印出来，人工确认无 `label`、无其它二元/多类目标投影。

## 任务 B：系统性泄漏/一致性排查（防止第四个同类问题）

**这是本工作包的核心增值——不要只修 UNSW 一列。**

1. **逐数据集特征列全清点**：对 CICIDS/CESNET/UNSW，在冻结主协议下打印**最终进入特征
   矩阵的完整列名清单**，逐列判定是否为标签/标签构造代理，产出
   `reports/p10_feature_audit.md`（三张表，每列一行：列名 / 是否代理 / 处置）。
   重点查：任何与 label 完全或高度相关的列（互信息/相关系数≈1 的自动扫描）。
2. **跨运行器共享概念一致性**：列出所有"同名概念多处实现"的点，逐一断言各 runner
   调用**同一冻结函数**。已知需覆盖：噪声目标（p3._noise_target，已修）、Tail-F1
   （canonical helper，P9 修）、tail 类定义（`_tail_labels`）、屏蔽层应用、划分策略。
   产出 `reports/p10_consistency_audit.md` + 新守卫测试
   `tests/test_cross_runner_consistency.py`（把这些断言固化）。
3. **自动代理扫描**（可复用）：写 `src/paper/leak_scan.py`——对任一数据集的特征矩阵，
   计算每列与二元 attack/normal 及与多类标签的互信息，超阈值即标红。UNSW `label`
   应被它以互信息=1 命中（作为该工具的验证用例）。

## 任务 C：重跑 UNSW 受影响的冻结实验（去泄漏后）

- 用修复后的 UNSW 特征，重跑 UNSW 的：Family A（#3/#4）、Family C（#4 vs #5/#6）、
  Family B（CL / CL+Graph / Graph-soft）、逐类护栏、泄漏分解 UNSW 列；
- **在合并帧上重算完整 Family A/B/C**（沿 P8 机制：CICIDS/CESNET 行逐位不变，仅换 UNSW）；
- 裁决影响评估：Family A 预期仍 robust（CICIDS+CESNET 不受影响，≥2/3），但以实际为准；
  G2 预期仍 null；Family C 的 UNSW 反向信号可能变化——**如实报告，任何裁决变化即停下上报**。

## 任务 D：披露草稿（PA-4）
`reports/p10_unsw_label_fix.md`：泄漏机制、影响的表/裁决、新旧 UNSW 对照、
PA-4 披露段草稿（供修改稿 Protocol Amendments 附录）。措辞方向：
"我们的审计协议在自身 UNSW 特征中发现并移除了一处二元标签代理；修复后重跑，
Family A 仍 robust / G2 仍 null" —— 把它作为审计协议有效性的正面证据，不淡化。

## 交付（分阶段）
- 阶段 1：`reports/p10_full_audit.md`（待修清单）+ leak_scan 工具 + 跨 runner 一致性测试
  + 预处理边界审计。**停下交协调者确认清单。**
- 阶段 2/3（确认后）：train-only 预处理代码 + 量化报告 + 三数据集一次重跑
  + 完整族重算 + 新旧对比 + PA-4/PA-5 披露草稿 + 守卫测试 + commit hash。

## 给协调者的后续（P10 完成后）
更新稿件：UNSW 全线数字、PA-4 附录、泄漏分解 UNSW、Family C UNSW 叙事、数字审计、
答复信（把这次自查作为审计协议有效性的证据补进 A2/M5）。
