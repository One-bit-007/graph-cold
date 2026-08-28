# P7 基线扩展报告：FINE 正式纳入 + Co-Teaching F-5 修复（P1-2 / P1-3，附 P1-6 讨论）

日期：2026-08-28。执行范围：提示词 `docs/prompts/prompt_codex_p7_phase4.md` 优先级第 2、3 项（P1-3、P1-2），附节覆盖 P1-6（仅讨论，不新跑）。

## 0. TL;DR

- **F-5 修复落地**：FINE 与 Co-Teaching 的噪声率参数从「调用方传入真实注入率」改为「仅由带噪训练集估计」（`src/baselines/noise_estimate.py`，3 折分层 crossfit ExtraTrees 折外自信度）。真实注入率不再到达任何基线。
- **全量冻结网格完成**：3 数据集 × 10 seeds × 3 噪声率 × 2 方法 = **180/180 行**，零失败，全部行 `noise_rate_source=estimated_from_noisy_train`。
- **结论一（P1-3）**：FINE-est 与 Graph-CoLD-soft 互有胜负（9 个条件下 3 正 6 负，幅度均 ≤0.08），但**全面低于 Confident-Learning**（9/9 条件为负，Δ −0.06 ~ −0.32）。CL 仍是最强外部基线。
- **结论二（P1-2）**：Co-Teaching-est 尾类在 rate 0.6/0.8 下基本**崩溃**（Tail-F1 0.00–0.03；UNSW 三档全 0.00）。p5 历史 Co-Teaching 数字依赖 oracle forget rate，稿件中所有引用必须改用本报告的 F-5 合规数字，措辞按 §5 软化。
- 统计口径：仅描述性配对 Δ + bootstrap 95% CI，未新增推断族（F-6 合规）。

## 1. F-5 修复机制

| 项 | 内容 |
|---|---|
| 估计器 | `estimate_noise_rate(X_train, y_noisy, seed)`：3 折分层 crossfit，`ExtraTreesClassifier(n_estimators=24, class_weight="balanced")`，估计 = mean(1 − P_oof(y_noisy))，clip [0, 0.95] |
| 输入边界 | 只吃带噪训练集；函数签名不含任何干净标签 / 真实注入率参数（守卫测试锁定签名） |
| Co-Teaching | `noise_rate=None` → fit 时估计，`forget_rate` 调度与末尾 retained mask 使用估计值；`details` 记录 `noise_rate_source`、`forget_rate_target` |
| FINE | `noise_rate=None` → fit 时估计，保留比例目标使用估计值；`details` 记录 `noise_rate_source`、`estimated_noise_rate` |
| 旧行为 | 显式传值仍可用，但标记 `caller_provided_legacy`；正式网格一律 None |

**冒烟期发现并修复的缺陷**：初版 `n_splits = min(3, counts[counts>0].min(initial=0))` 中 `.min(initial=0)` 恒返回 0，退化为全量拟合 + 样本内自信度，估计恒为 0.0。修复为对非零类计数取 min；所有基于错误估计的冒烟行（2 行）已删除并全量重跑，未混入正式产物。

## 2. 估计噪声率 vs 实际注入（训练集）

| 数据集 | 名义 rate | 估计均值 (min–max) | 训练集实际翻转率 |
|---|---:|---:|---:|
| CESNET | 0.4 | 0.175 (0.171–0.181) | 0.125 |
| CESNET | 0.6 | 0.173 (0.169–0.175) | 0.187 |
| CESNET | 0.8 | 0.121 (0.120–0.122) | 0.250 |
| CICIDS | 0.4 | 0.025 (0.023–0.027) | 0.010 |
| CICIDS | 0.6 | 0.023 (0.021–0.025) | 0.016 |
| CICIDS | 0.8 | 0.019 (0.017–0.020) | 0.021 |
| UNSW | 0.4 | 0.258 (0.253–0.265) | 0.040 |
| UNSW | 0.6 | 0.257 (0.254–0.261) | 0.060 |
| UNSW | 0.8 | 0.249 (0.246–0.252) | 0.079 |

如实记录：尾类非对称注入下估计的是**全局**自疑率，不逼近名义 rate 属预期（CICIDS 接近实际全局翻转率；CESNET rate 0.8 低估；UNSW 因类间重叠系统性高估）。两个基线在同一场景下得到完全相同的估计（同 seed 同估计器），一致性通过。

## 3. 主结果（Tail-F1 / Macro-F1 均值，10 seeds）

| 数据集 | rate | FINE-est Tail | CT-est Tail | FINE-est Macro | CT-est Macro |
|---|---:|---:|---:|---:|---:|
| CESNET | 0.4 | 0.7119 | 0.4692 | 0.8053 | 0.6630 |
| CESNET | 0.6 | 0.5108 | 0.0069 | 0.6980 | 0.4161 |
| CESNET | 0.8 | 0.2805 | 0.0000 | 0.5815 | 0.4053 |
| CICIDS | 0.4 | 0.6170 | 0.1702 | 0.7931 | 0.5870 |
| CICIDS | 0.6 | 0.4136 | 0.0311 | 0.7062 | 0.5244 |
| CICIDS | 0.8 | 0.1791 | 0.0000 | 0.5997 | 0.5074 |
| UNSW | 0.4 | 0.2361 | 0.0000 | 0.4936 | 0.3609 |
| UNSW | 0.6 | 0.1633 | 0.0000 | 0.4622 | 0.3560 |
| UNSW | 0.8 | 0.0812 | 0.0000 | 0.4281 | 0.3593 |

## 4. 配对增量（描述性 Δ + bootstrap 95% CI，逐 seed 配对，无 Holm）

### 4.1 对 Graph-CoLD-soft（冻结参照，p2f）

**Tail-F1**

| 数据集 | rate | FINE-est Δ [CI] | CT-est Δ [CI] |
|---|---:|---|---|
| CESNET | 0.4 | −0.034 [−0.040, −0.029] | −0.277 [−0.292, −0.260] |
| CESNET | 0.6 | **+0.034 [+0.026, +0.042]** | −0.470 [−0.483, −0.457] |
| CESNET | 0.8 | **+0.080 [+0.075, +0.084]** | −0.201 [−0.211, −0.192] |
| CICIDS | 0.4 | −0.037 [−0.053, −0.016] | −0.484 [−0.514, −0.453] |
| CICIDS | 0.6 | −0.024 [−0.055, −0.001] | −0.407 [−0.438, −0.376] |
| CICIDS | 0.8 | −0.032 [−0.051, −0.008] | −0.211 [−0.250, −0.174] |
| UNSW | 0.4 | −0.012 [−0.038, +0.013] | −0.248 [−0.280, −0.220] |
| UNSW | 0.6 | −0.026 [−0.045, −0.006] | −0.189 [−0.206, −0.170] |
| UNSW | 0.8 | −0.001 [−0.013, +0.011] | −0.082 [−0.093, −0.071] |

**Macro-F1**：FINE-est 同构（CESNET 0.6/0.8 为正 +0.006/+0.035，其余 −0.017 ~ −0.031）；CT-est 全部显著为负（−0.086 ~ −0.276）。

### 4.2 对 Confident-Learning（冻结参照，p5）

Tail-F1：**FINE-est 9/9 条件为负**（−0.059 ~ −0.323）；**CT-est 9/9 条件为负**（−0.185 ~ −0.728）。CL 在 F-5 合规口径下仍是三数据集上最强基线。

## 5. 对稿件的影响与措辞软化（P1-2 收尾）

1. **FINE 定位**：FINE-est 是「不使用 oracle 噪声率时与 soft 保留同档」的过滤式基线（9 条件 3 正 6 负，幅度 ≤0.08），但不挑战 CL；可作为鲁棒性证据保留在正式结果中，不承载性能声明。
2. **Co-Teaching 声明替换**：p5 中 Co-Teaching 的数字是在 oracle forget rate 下取得的，F-5 合规后其尾类性能在 rate ≥0.6 时崩溃（≤0.03，UNSW 全 0）。稿件任何「Co-Teaching 可比/更优」的表述必须删除或改写为：在无法获知真实噪声率的部署设定下，依赖 forget-rate 先验的协同训练方法对尾类非对称噪声敏感，其性能高度依赖该先验的准确性。
3. **不回流**：p5 历史行保留为历史记录（registry notes 已标注 superseded），正式引用一律用 `results/p7_baseline_expansion.csv` 与本报告数字。
4. **估计器披露**：方法/附录需说明两个基线的噪声率为自估计（3 折 crossfit ExtraTrees 自疑率），并给出 §2 的估计—实际对照表，避免「不公平地调弱基线」的审稿质疑。

## 6. 附节（P1-6）：freq-floor 讨论段（复用既有结果，未新跑）

族 C #5（freq-floor 重加权）全量结果已定（`reports/p5_factor_ablation.md` §3）：#4 evidence-rescue 对 #5 的 9 项比较裁决为 null，且 UNSW rate 0.4/0.6 上 #4 **显著劣于** #5（CI 上界 −0.011/−0.013）。结合本轮观察（纯频率驱动的 FINE-est 在 CESNET 高噪声档反超 soft），讨论段建议措辞：

> 在尾类非对称噪声下，仅由观测类频率驱动的简单重加权（freq-floor）在部分设定上已足够强；图诊断信号（CDM）的增量价值不体现在训练加权性能上，其定位应收窄为审计/诊断工具——这与 G2=null 及族 C null 的裁决一致。稿件应避免任何「CDM 加权带来性能收益」的暗示。

此段不新增任何实验行，仅为讨论统一口径。

## 7. 验证与复现

- 网格完整性：180/180 行，键 (dataset, rate, seed, method) 无重复无缺失；`noise_rate_source` 单一值 `estimated_from_noisy_train`；指标列零缺失；`failures.json` 为空。
- 守卫测试：`tests/test_p7_baseline_expansion.py` **7/7 通过**（估计器签名 F-5 锁定、合成数据可回收噪声且确定、退化类尺寸安全、两基线 None→估计且 metadata 正确、runner 永不传真实率、网格完整性）。
- 相关回归：P7 基线扩展、P7 自然噪声、baseline registry、no-oracle 与 proxy-field 五组测试合计 **33/33 通过**。
- 全仓回归（2026-08-28 复核）：**256 passed, 3 skipped, 20 failed**。20 项失败均位于 P7 之外：19 项属于缺失的 D6--D11 Elsevier 历史稿件/装配资产或旧两数据集装配约束，1 项属于 2026-07-07 已存在的 `src/experiments/d5.py` 中 `fallback` 字符串门禁。故本轮 P7 变更通过相关回归，但不得将全仓状态表述为“全绿”。
- 复现：`./.venv/Scripts/python.exe -m src.paper.p7_baseline_expansion`（项目根；partial CSV 逐场景落盘，支持超时续跑；`--shard <name>` 支持分数据集并行，`--finalize-only` 仅聚合）。
- 产物：`results/p7_baseline_expansion.csv`、`reports/p7_baseline_expansion.json`、本报告。既有 p2f/p5 结果只读未改。
