# Codex 工作包 P8：CESNET F-3 噪声映射修复 — 重跑 Family A/C

> **背景（外部专家二审发现，协调者已独立核验）**：`src/paper/p5_factor_ablation.py`
> 旧代码用 `benign_class or 0` 作 CESNET 噪声翻转目标。CESNET 无 benign 类 →
> 回退到标签 **0（apple-itunes）**，而冻结 F-3（`reproducibility/noise_map.json` line 17）
> 规定 CESNET 目标是标签 **11（microsoft-diagnostic）**。P4/Family B 用的是正确的
> `p3._noise_target()`，P5/Family A/C 却走了错误分支。**只影响 CESNET；CICIDS/UNSW
> 两路径一致（协调者已用代码+数据核验：CICIDS benign_class=0=frozen target）。**
>
> **根因代码已由协调者修复**（commit 已入库）：`_prepare_scenario` 现调用
> `p3._noise_target(dataset_name, ...)`，并新增 `tests/test_noise_target_consistency.py`
> 锁死"所有 runner 共用冻结目标函数"。本工作包只做**重跑 + 重算 + 诚实披露**。

## 不可违反的规则

1. 冻结协议不变：seeds 0–9、rates {0.4,0.6,0.8}、train 10k/test 5k、tier-2 屏蔽 +
   S1∩S2 划分、F-6 统计族（A=9, C=18）。**只改噪声目标（现已由代码修复），其余一切不动。**
2. 禁 oracle 泄漏；相关回归测试保持全绿。
3. **保留旧产物、不覆盖历史**：旧 `results/p5_factor_ablation_full.csv` 的 CESNET 行是
   buggy 基线（已入库）；新结果写入后，git diff 必须清楚显示 CESNET buggy→correct，
   CICIDS/UNSW 行应逐位不变（它们不受影响——可作为正确性自检）。
4. `submission_ready` 保持 false。

## 任务

### 1. 前置自检（不跑训练）
- 确认 `tests/test_noise_target_consistency.py` 全绿（4 项）；
- 确认 `p3._noise_target("cesnet_tls_year22", y_train, meta)` 在真实 CESNET 训练窗口返回 11
  （若标签 11 不在窗口内会抛错——那是另一个需上报的问题，不得静默回退）。

### 2. 重跑 CESNET 的 P5 全量（只重跑 CESNET，不碰 CICIDS/UNSW）
- `python -m src.paper.p5_factor_ablation --tier full`，但**仅 CESNET 分片**
  （用 `--shard`/`--datasets` 机制，或临时限定 DATASETS）；
- 网格：#3/#4/#5/#6 × 10 seeds × 3 rates = **120 method-runs**，保存逐类结果；
- 续跑 partial CSV 机制照旧；产物合并回 `results/p5_factor_ablation_full.csv`
  （CICIDS/UNSW 行原样保留，仅替换 CESNET 行）。
- **自检**：重跑后 grep 确认新 CESNET 的 noise_target 记录为 11；CICIDS/UNSW 行与旧文件逐位一致。

### 3. 重算完整 Family A / Family C（不要只换三行）
- 用新 CESNET 数据 + 不变的 CICIDS/UNSW 数据，**重算完整 9 项 Family A 与 18 项 Family C**
  的 Holm 校正、TOST 非劣效、逐类护栏、bootstrap CI、混合效应稳健性；
- 逐类护栏运行器 `p5_perclass_guard.py` 复用 p5._prepare_scenario，随之自动纠正——一并重跑；
- 更新 `tables/table_p5_family_a.csv`、`table_p5_family_c.csv`、`table_p5_perclass_guard.csv`
  与相关 JSON。

### 4. 裁决与影响评估（如实，按 F-6 用语）
- **Family A**：CICIDS/UNSW 不受影响且效应强（dz 0.77~3.14）——即便 CESNET 重算后方向或
  显著性变化，Family A 在 CICIDS+UNSW 至少 2/3，**大概率仍 robust**；但必须以重算结果为准，
  若 CESNET 翻转导致任何裁决变化，立即停下上报，不得自行改写；
- **Family C**：null 裁决由 UNSW 反向 + 整体不显著驱动；重算 CESNET 不太可能把 null 翻正，
  但同样以实际结果为准；
- 产出 `reports/p8_cesnet_f3_rerun.md`：新旧 CESNET 数字对照表、受影响的表/裁决、
  **协议偏离披露段**（供修改稿 Protocol Amendments 附录新增 PA-3）。

### 5. 稿件与发布物更新（数据齐后交协调者，Codex 只出数据+报告）
- 重新生成表格：`python -m src.paper.w5_result_tables`；
- 重跑数字审计：`python -m src.paper.w5_number_audit`——正文 CESNET 相关数字若变，
  **审计会红**，这是预期；协调者据新数字更新正文 + 摘要 + 答复信，再令审计转绿；
- 全仓测试 + 声明门全绿。

## 交付
变更文件清单、新旧 CESNET 对照表、Family A/C 重算裁决、CICIDS/UNSW 逐位不变自检结果、
PA-3 披露段草稿、与预期不符处（如有则停下）、git commit hash。
