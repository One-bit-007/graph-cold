# Codex Prompt — P0 SUBMISSION-BLOCKER FIXES (must pass before any C&S submission)

> A simulated Computers & Security review rated the current draft ~98% reject due
> to three FATAL issues. These P0 fixes are the critical path. Do ONLY P0 here;
> P1/P2 come after. Work on branch `fix/p0-submission-blockers`, push, tag `p0`.

## 背景（为什么必须做）
审稿模拟给出的三条一票否决级问题：
- FATAL-1：实验被写成 synthetic fallback / deterministic emulation（作者自曝非真实数据）。
- FATAL-2：结果好到不可信（100% F1、0% FPR/FNR、p≈1e-93）→ 泄漏/合成信号。
- FATAL-3：ERR 消融方向与核心论点相反（Table 2：full ERR=69.4% < ablation_hard ERR=100%），
  且 ERR 实现有 bug——用连续软权重 w·e，使软加权“机械性”低于硬删除。

要读的权威文件：docs/spec_method_impl.md §5–§6、docs/spec_graph_noise.md、
docs/PLAN.md（Pending corrections）、src/metrics.py、src/models/cold_baseline.py、
src/experiments/d5.py、src/data/loaders.py、configs/*.yaml。

---

## FIX-1（FATAL-1 + FATAL-2）：用真实数据，移除一切合成回退
目标：所有结果表来自真实 CICIDS-2017 + MALTLS-22（主实验），OpTC 用真实 provenance 特征
（企业案例）。禁止任何占位/合成/仿真数字进入 experiments/results。

实施：
1. src/data/loaders.py：实现对真实 CICIDS-2017(精炼9类，删<1000类,降采样) 与 MALTLS-22(23类)
   的加载；数据放 data/（.gitignore 已忽略）。若本地缺数据，必须 **fail loud**
   （raise FileNotFoundError 并打印下载/放置说明），**不得**回退到合成数据。
2. src/experiments/d5.py：删除 run_d5_experiments 内的 synthetic fallback 分支与
   “deterministic SOC emulation”代码路径；OpTC 走真实 Flash/Argus provenance 特征 + XGBoost 插件。
3. 全局搜索并移除字符串/逻辑：synthetic、fallback、emulation、deterministic SOC、draft placeholder。
4. 结果 CSV 顶部写明数据来源与版本，禁止任何“to be refreshed later”式占位。

验收：
- 缺数据时报错而非造数；有数据时 D5 全矩阵真实产出。
- grep 全仓库无 synthetic/emulation/fallback 结果路径。

## FIX-2（FATAL-3）：重定义并修正 ERR，使消融方向正确
问题根因：src/metrics.py 的 `_weighted_retention` 用连续 w·e。软权重恒<1 → 软方法 ERR
被系统性压低，硬删除(0/1) 反而更高。这与“证据保持保留更多证据”的论点相反。

实施（改 src/metrics.py）：
1. 引入保留阈值 τ_ret（configs/model.yaml: evidence_preserving.retention_threshold，默认 0.1）。
2. 定义“被保留”为二值：retained(v) = 1[ w(v) ≥ τ_ret ]。硬删除天然 w∈{0,1}。
3. ERR 仅在 clean∩informative 子集上度量证据保留：
   - clean = 未被翻转样本（用 flip_mask 的补集；调用方必须传真实 clean_mask，
     禁止把全 1 当 clean）。
   - informative = 低频/尾部类 ∪ 边界/异常高分样本（沿用现有 tail 逻辑 + anomaly 分位）。
   - ERR = Σ_{v∈clean∩inform} retained(v)·e(v) / Σ_{v∈clean∩inform} e(v)
   - 保留 Tail-ERR 与 ERR_final=½(ERR+Tail-ERR)。
4. 更新 evidence_retention_rate / evidence_retention_components 使用 retained 二值化。

验收（新增单测 tests/test_err.py）：
- 断言：在同一数据与噪声下 ERR(Graph-CoLD 软加权) **>** ERR(ablation_hard 硬删除)。
- 断言：当所有软权重 ≥ τ_ret 时软方法 ERR=1.0；硬删除删除任一 clean∩inform 样本则 ERR<1.0。
- 断言：clean_mask 必须由 flip_mask 推出，传全 1 触发告警/异常。

## FIX-3（MAJOR-1，支撑 FATAL-2 可信度）：公平的 CoLD 基线
问题：自研 CoLD 若停留在 PCA/弱实现，无噪 F1 远低于原文(≈92 MALTLS/≈99 CICIDS)，
对比不公平，且放大 Graph-CoLD 假优势，加重 FATAL-2 的不可信印象。

实施：
1. src/models/cold_baseline.py：fit_representation 用 D2 的真实对比编码器（非 PCA），
   与 Graph-CoLD 共享同一 encoder 家族，保证公平。
2. 校准到量级：无噪 MALTLS Macro-F1 ≈ 90+，CICIDS ≈ 99（对照 CoLD 原文趋势）。
3. 确认 ablation_hard(ρ=0 + 硬阈值) 数值 ≈ 自研 CoLD（差值=证据保持增益，且方向为正）。

验收：
- 无噪 CoLD 基线达标量级；random seed=42 可复现。
- 报告 ablation_hard vs 自研 CoLD 的差值表，方向与 FIX-2 的 ERR 结论一致。

---

## Git
- branch: fix/p0-submission-blockers
- commit(s): "fix(p0): real-data only + ERR redefinition + fair CoLD baseline"
- 合并 main 后打 tag p0，push origin main/dev/tags。

## 总验收（Definition of Done，P0 全绿才可谈投稿）
1. 无数据即报错，绝不合成；全矩阵在真实数据上产出，数字有波动/标准差、无 100%/0% 异常。
2. tests/test_err.py 通过：软加权 ERR > 硬删除 ERR（FATAL-3 反转修复）。
3. CoLD 基线达原文量级，ablation_hard≈CoLD 且证据保持增益方向为正。
4. 输出一页“P0 复核报告”：三处修复前后的关键数字对照 + 复现命令。
