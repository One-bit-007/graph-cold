# Codex Prompt — D5 (full experiment matrix + ablations + baseline comparisons)

## 【角色】

你是 Graph-CoLD D5 实验工程负责人。

你的任务是：

> 构建完整实验矩阵 + 消融实验 + baseline 对比 + 统计显著性分析

---

## 【背景】

当前系统已完成：

- representation learning (D2)
- Graph-CDM + evidence + ranking (D3)
- graph-consistency noise + OpTC case (D4)

当前进入：

> Stage-3: Full experimental validation for journal submission

---

## 【本次任务】

### 1. 全实验矩阵构建（核心任务）

必须执行：

数据集：

- CICIDS-2017
- MALTLS-22
- synthetic OpTC (D4)

噪声设置：

- symmetric: `r in {10%, 20%, 40%, 60%}`
- asymmetric: `r in {10%, 20%, 40%, 60%}`
- graph-consistency: `beta in {0.0, 0.3, 0.6, 1.0}`

基线方法：

必须实现对比：

- CoLD (self-implemented)
- MCRe
- MORSE
- FINE
- Co-Teaching(+/+)
- Decoupling
- Flash
- Argus
- cleanlab (confidence learning baseline)

---

### 2. 消融实验（必须）

必须包含：

- w/o Graph-CDM
- w/o D_neigh
- w/o D_view
- w/o evidence (`w(v)=1`)
- ablation_hard (`rho=0`)
- w/o ranking
- w/o temporal

---

### 3. 核心指标

必须输出：

- Macro-F1
- FPR / FNR
- ERR
- Tail-ERR
- compression ratio
- runtime
- memory usage

---

### 4. 统计显著性

必须：

- t-test (Graph-CoLD vs CoLD)
- p-value reporting
- mean ± std over seeds `{0, 1, 2}`

---

### 5. 输出文件（必须）

必须生成：

```text
results/table_main.csv
results/table_ablation.csv
results/table_optc.csv
results/stat_tests.json
results/runtime.json
```

---

### 6. 图表生成

必须生成：

- Fig2: Macro-F1 vs noise rate
- Fig3: ERR vs compression ratio
- Fig4: ablation study bar chart
- Fig5: OpTC case ranking performance

---

## 【严格约束】

- 不得修改 Graph-CDM
- 不得修改 encoder
- 不得改变 noise model
- 必须 reuse D2/D3/D4 模块

---

## 【Git要求】

branch:

```text
feature/d5-experiments
```

tag:

```text
v0.5-d5-experiments
```

---

## 【验收标准】

CK-6 becomes primary:

- Graph-CoLD vs CoLD: `p < 0.05`
- ERR improves under high noise (`>40%`)
- ranking stability > CoLD baseline
- ablation shows monotonic degradation

---

## 【最终目标】

D5 完成后系统必须具备：

> 可投稿 Computers & Security 的完整 experimental section
