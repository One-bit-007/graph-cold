# Codex 工作包 P9：FINE 保真度核验 + 最终协议补跑（回应评审 A7）

> **背景（外部专家二审）**：现有 `src/baselines/fine_style.py` 不是完整原始 FINE。
> 差异（专家 + 协调者已核实）：
> - **表示**：现用 TruncatedSVD + 类内中心化对齐分数；原论文用类内表示 Gram 矩阵的
>   **主特征向量**与**平方对齐分数**（Kim et al., NeurIPS 2021, Algorithm 1）；
> - **筛选**：现按**估计噪声率**保留固定比例；原论文对每类对齐分数拟合 **2-成分 GMM**，
>   按后验概率分 clean/noisy，**不需要噪声率**；
> - **协议**：现有 P7 runner（`p7_baseline_expansion.py`）走 p2f 遗留 iid 采样，故稿件标为
>   "legacy window, non-comparable"。
>
> 专家结论：直接跑满 90 场景**不能消除 A7**，还可能重演 CoLD 假身份问题。
> **必须先核验保真度、明确命名，再接入最终协议补跑。**

## 不可违反的规则

1. **不改任何已冻结裁决/结果**：Family A（robust）、G2（null）、Family C（null）、
   泄漏分解、CESNET F-3 修正后的产物——全部不动。P9 是**附加基线**，纯描述性，
   **不新增 Holm 推断族**（FINE 是对照锚点，不是假设检验）。
2. F-5 合规：不用干净验证集、不用真实注入噪声率。GMM 路径天然无需噪声率——优先采用。
3. 最终协议 = 各数据集冻结主口径（tier 屏蔽 + 最强可用划分 + `p3._noise_target`），
   与 p4/p5 完全一致；**不要复用 p7 的 p2f 遗留采样**。
4. 禁 oracle 泄漏；相关回归测试保持全绿。已有产物只增不改，新产物新文件名。
5. `submission_ready` 保持 false。发现预期外问题即停下报告。

## 任务 0：统一 Tail-F1 口径（协调者已批准，先于 FINE）

**已核实的问题**：跨运行器 Tail-F1 定义不一致——
- p5 `p3._tail_f1`：在**完整测试集**上算逐类 F1，再取尾类均值（假阳性扣分）；
- p4 `_tail_macro_f1`：只在**真实尾类行**上算 macro-F1（更宽松）。
- 后果：主表 Table 2 把 p5 的 soft/hard 与 p4 的 CL/CL+Graph 并列，Tail-F1 列混用两种定义。

**已核实的可行性**：p4 的 `per_class_f1_json` 存的是
`f1_score(y_test, pred, average=None)`（全集逐类 F1），因此 CL/CL+Graph 的 canonical
Tail-F1 **可从已存逐类结果直接重汇总，无需重训**。

**任务**：
1. **canonical 定义 = p5 全集口径**（更保守、更标准）。写可复用 helper
   `canonical_tail_f1_from_perclass(per_class_f1_json, tail_labels)`（放合适模块，
   p9 与协调者共用）；
2. **不重训、不改 A/B/C 的推断产物**：Family A/C 本就是全集口径（=canonical，无需动）；
   Family B 的预注册 delta 保持 masked 口径（两臂一致，G2 null 不动）；
3. **出主表影响报告** `reports/p9_tail_metric_unification.md`：从已存 p4 逐类结果重算
   CL、CL+Graph 的 canonical Tail-F1，列出主表这两行 Tail-F1 masked→canonical 的变化，
   并**附带**在 canonical 口径下重算 Family B 配对 delta，确认 G2 **仍 null**（稳健性佐证）；
   产物写 `results/p9_canonical_tail_f1.csv`，**不改** w5_result_tables 生成器与正文
   （协调者据此更新 tab_main、RQ1 叙事、数字审计、答复信）。

## 任务 1：FINE 保真度核验（出报告）
- 写 `reports/p9_fine_fidelity.md`：逐组件对照原论文 Algorithm 1 vs 本实现
  （表示、对齐分数、筛选准则、分类器头），如实列出每一处适配及理由。

### 2. 提升实现向原始 FINE 靠拢（在忠实与可行间取平衡）
- **核心机制对齐**：实现类内表示的 Gram 矩阵主特征向量 + **平方对齐分数**
  `s_i = ⟨z_i, v_c⟩²`（c 为样本 i 的观测噪声类），并对每类分数拟合 **2-成分 GMM**，
  按"高对齐=clean"的后验分离 clean/noisy——**移除对估计噪声率的依赖**（同时更忠实、更 F-5 干净）；
- 表示层若仍需适配（tabular 无深度表示），**保留但显式标注**为适配项；
- **命名**：`FINE detector (tabular adaptation)`；`faithfulness_level` 字段如实写明
  "Gram-eigenvector + per-class GMM selection as in Kim et al.; representation is a
  tabular adaptation, not the original deep features"；
- 小类回退（GMM 需最小样本）如实处理并记录，不静默保留全部。

### 3. 最终协议补跑（描述性）
- 新建 `src/paper/p9_fine_final_protocol.py`，**复用 p4/p5 的冻结协议场景机制**
  （`p4._load_primary_dataset`、tier 屏蔽、最强划分、`p3._noise_target`），
  跑 FINE(tabular) 于 3 数据集 × 10 seeds × 3 rates = 90 场景，逐类结果落盘；
- **Tail-F1 统一用任务 0 的 canonical helper**（全集口径）；FINE / CL / Graph-CoLD-soft
  三者同表并列的 Tail-F1 均为 canonical 口径（CL/Graph 数值从已冻结 p4/p5
  逐类结果重汇总，不重跑；FINE 为新增），配对 Δ + bootstrap CI（描述性，无 Holm）；
- 产物：`results/p9_fine_final_protocol.csv`、`reports/p9_fine_final_protocol.md`、
  新表 `tables/table_p9_fine_final.csv`。

### 4. registry + 守卫测试
- `registry.py`：登记 FINE(tabular) 的 `faithfulness_level`、`uses_noisy_y_train=True`、
  `uses_clean_y_test_for_eval_only=True`；`include_in_formal_results` 置真前确认 F-5 合规；
- 新守卫测试：GMM 筛选不依赖噪声率（断言无 true-rate 输入路径）、FINE 走冻结协议
  `p3._noise_target`（复用一致性测试模式）、Attempted/oracle 不泄漏。

### 5. 诚实定位（写入报告，供协调者更新正文）
- 明确：FINE(tabular) 在最终协议下**大概率仍不及 CL**（预期，如实报告）；
- A7 的处理表述为"在最终协议下补齐了一个保真度核验过的额外强基线，**降低但不清零** A7 风险"；
- **不得**把 FINE 包装成完整原始实现或声称超越 CL。

## 交付
`reports/p9_fine_fidelity.md`（逐组件对照）+ `reports/p9_fine_final_protocol.md`
（含 FINE/CL/Graph 同协议对照表、诚实定位段）+ `results/p9_fine_final_protocol.csv`
+ `tables/table_p9_fine_final.csv` + registry 变更 + 新守卫测试 + git commit hash。
协调者随后：把 FINE 从稿件的 "legacy iid non-comparable" 表移入最终协议对照，
更新数字审计、答复信 A7 与 M2 条目。

## 不在本工作包范围
- 不改 Family A/B/C、G2、泄漏分解、CESNET F-3 产物；
- 不新增推断族；不追加 FINE 之外的基线（止损：一个保真度核验过的强基线即可）；
- 正文/答复信改写由协调者统一处理。
