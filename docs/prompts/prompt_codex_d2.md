# Codex Prompt — D2 (five-view graph construction + CoLD-aligned representation)

> Authored by GPT-pro, coordinator-approved to run. Scope: Stage-1 representation
> only (NO classifier). Two D1 math items (D_pred target, graph-noise β=0) are
> being corrected by GPT-pro before D3/D4 and do NOT affect this task.

## 【角色】
你是 Graph-CoLD 系统工程实现负责人，实现：CoLD 表示学习模块 + 五视图异构图编码 +
SOC 级对比学习框架。必须与 CoLD baseline 完全对齐、可复现(seed=42)、支持
CICIDS-2017 / MALTLS-22。

## 【背景要点】
Graph-CoLD 在 CoLD 上的扩展：IID 样本 → 异构多视图图；hard deletion → 证据软加权；
classification → representation + SOC ranking。本次仅实现 Stage-1 表示学习 backbone
(无分类器)。权威规格：docs/spec_method_impl.md §1–§2。

## 【本次任务】
### 1. 五视图图构建（必须）
host / ip / process / temporal / threat-intel，输出 G^(m)=(V, E^(m))。
要求：sparse adjacency；batch 支持；view mask 支持。

### 2. 表示学习（CoLD 三件套必须实现）
- 特征混淆(CoLD Eq.2)：x'_v = x_v ⊙ Bernoulli(1-δ)
- 对比学习(InfoNCE)：正样本=同节点跨视图，负样本=batch 内其他节点，温度 τ
  L_con = -log[ exp(sim(z_v^a, z_v^b)/τ) / Σ_u exp(sim(z_v, z_u)/τ) ]
- 时序对齐：L_temporal = || z_v^t - z_v^{t+1} ||_2^2
- 重构损失：L_recon = || x_v - x̂_v ||_2^2
- 总损失：L_rep = L_con + α·L_temporal + β·L_recon

### 3. 编码器结构
优先 HGT；fallback RGCN（异构关系）或 GAT（按 view 拆图）。
每视图独立 message passing；view fusion = MEAN（CoLD optimal）。

### 4. 输出接口
返回 z_v ∈ R^128、z_v^(m)、重构 x̂_v。

## 【严格契约】
- 遵守 docs/spec_method_impl.md §1–§2。
- 图结构必须与后续 Graph-CDM 输入一致。
- 不得提前引入 classifier。embedding dim = 128 固定。

## 【Git 要求】
- branch: feature/rep-learning-v2
- commit: "CoLD representation learning + multi-view encoder"
- tag: v0.2-rep-d2（合并 main 后再打日 tag d2）

## 【验收标准（供协调者核对）】
- 五视图均非空；node embedding shape = [N, 128]。
- L_rep 单调收敛（保存曲线）。
- linear probe / kNN 在 clean labels 上 Macro-F1 > random baseline + 30%。
- seed=42 完全可复现。
- CoLD baseline 表示层已替换为同一 representation encoder（公平对齐，替换 PCA）。
