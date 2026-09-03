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

## 任务 A：修复 UNSW `label` 泄漏（根因）

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

## 交付
任务 A–D 的报告 + 新守卫测试 + leak_scan 工具 + 重跑后的 UNSW 产物与完整族重算 JSON
+ 各数据集特征列清单 + 裁决影响表 + git commit hash。
恢复 P9（FINE 剩余 30 场景 + 主表 canonical 收尾）在 P10 之后。

## 给协调者的后续（P10 完成后）
更新稿件：UNSW 全线数字、PA-4 附录、泄漏分解 UNSW、Family C UNSW 叙事、数字审计、
答复信（把这次自查作为审计协议有效性的证据补进 A2/M5）。
