# Codex 工作包 P7（Phase 4）：P1 提升说服力（按 G2=null 裁剪）

> 前置：Phase 0–3 全部完成，G2=null（Graph-CDM 降格为审计/诊断工具），族 A robust
> （软保留>硬删除），族 C null。论文定位 = 审计驱动边界研究。本工作包按计划 v3 的
> G2 裁剪规则执行 P1：**上调**审计叙事项（P1-4/P1-7），保留 P1-1/P1-3，**下调**性能叙事项
> （P1-2/P1-6，只做合规修复与措辞软化，不扩规模）。
>
> 优先级顺序：P1-4+P1-7（最高）→ P1-3 → P1-1 → P1-2 → P1-6。

## 不可违反的规则（沿用全部既有约束）

1. 冻结协议不可改：seeds 0–9 / rates {0.4,0.6,0.8}、train 10k/test 5k、F-2 tail 定义、
   F-3 噪声映射、**F-5 调参协议（禁干净验证集、禁真实注入噪声率）**、F-6 统计族与用语。
2. 禁 oracle 泄漏；proxy/oracle/leakage 回归测试保持全绿。
3. 已有结果只增不改；新产物新文件名。`submission_ready` 保持 false。
4. 新增基线纳入正式结果前，必须在 `src/baselines/registry.py` 显式登记
   （`verified`、`faithfulness_level`、`include_in_formal_results`、`uses_noisy_y_train` 等字段如实填写）。
5. 每项完成后：报告/表/测试入库，git commit（不 push）。任何协议偏离先停下报告。

---

## P1-4 + P1-7【最高优先，审计叙事上调】自然噪声实验 + 文献重定位

**目标**：用**已发表的 CICIDS 修正标签**构造真实标注错误实验，验证软保留与审计结论在
真实（非合成）噪声下是否成立；同时完成 CICIDS 缺陷文献的重定位落地。

**数据依赖（人工前置，类似 WP-3）**：需要 Engelen et al. 2021 与 Lanvin et al. 2023 发布的
CICIDS-2017 修正标签文件。二者均为公开 GitHub 资源：
- Engelen et al. 2021, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study"
  （改进标签 + 重标注流）；
- Lanvin et al. 2023, "Errors in the CICIDS2017 dataset and the significant differences in
  detection performances it makes"（修正标签 + 工具）。
- **若你（Codex）有网络**：尝试克隆对应公开仓库，把修正标签放到
  `E:/graphcold-data/cicids2017_corrected/`，记录来源 URL 与 commit/文件哈希；
- **若无网络**：停下并产出一份「所需文件清单 + 官方仓库指引」，由人工获取后再触发本项
  （其余 P1 项不受阻，先做 P1-3/P1-1）。

**实验设计（就位后）**：
1. 对齐：把修正标签按流键（Flow ID 或 (src,dst,port,timestamp) 组合）对齐到原始 CICIDS 记录；
   原始标签 vs 修正标签的**差集 = 真实标注错误集**，报告其规模、类分布、是否集中于尾类；
2. **真实噪声版主对比**：以原始（含真实错误）标签为训练噪声、修正标签为"干净"评估参照，
   在冻结协议口径下跑 {CoLD, ablation_hard, Graph-CoLD-soft, Confident-Learning}，
   报告软保留 vs 硬删除的尾类 Δ（族 A 语义的真实噪声版）、以及 CL 对比；
   — 这是**描述性**验证（真实错误集不是我们注入的、无 seed 重复的对称设计），
   按 F-6 不新增 Holm 推断族，用 Δ + bootstrap CI 呈现；
3. **审计结论的真实噪声复核**：检查合成噪声下的两个核心发现在真实噪声下方向是否一致
   —— ① 软保留>硬删除方向；② 泄漏对删除式方法抬升更大（可选，若样本足够）；
4. 文献重定位（P1-7 落地）：在相关工作/威胁有效性节引入 Engelen/Lanvin，把本文审计增量
   相对其定位清楚（它们指出标签错误，本文量化标签构造代理泄漏 + 提供隔离协议与回归测试）。

**交付**：`reports/p7_natural_noise.md`（对齐、真实错误集刻画、真实噪声 Δ 表、方向一致性、
文献重定位段）；修正标签来源与哈希记录；相关守卫测试（对齐正确性、修正标签不泄漏进特征）。

---

## P1-3【保留】正式基线补齐：FINE / DivideMix

- 仓库已有 `src/baselines/fine_style.py`。任务：将其（或 DivideMix，至少一个）以**忠实实现**
  纳入最终协议正式结果——在 `registry.py` 把 `include_in_formal_results` 置真前，确认：
  F-5 合规（不用干净验证集/真实噪声率调参）、共享同一编码器与训练预算、`faithfulness_level`
  如实描述（若为 tabular 近似而非原论文 CNN/半监督全貌，必须写明）;
- 在冻结协议（3 数据集 × 10 seeds × 3 rates）下跑该基线，产出与 CoLD/CL/Graph-CoLD 同表的
  Macro-F1 / Tail-F1；如实报告其相对位置（预期 CL 仍强，诚实呈现）;
- **止损**：只补到"至少一个额外强基线纳入正式协议"即可，不无限扩基线。

**交付**：`reports/p7_baseline_expansion.md` + 更新正式结果表；`registry.py` 变更；基线守卫测试。

---

## P1-1【保留】CoLD 保真度

- 目标：解释原稿 CoLD 复现值 0.534 与其报告 0.688 的差距来源。两条路径任选其一并公开：
  (a) 在**原文设置**下复现其报告数字（若能对齐数据/协议）；或
  (b) 公开对齐版调参协议，说明差距来自协议差异（窗口/划分/调参/类策略），逐项归因；
- 不得为抬高 CoLD 而使用干净验证集或真实噪声率（F-5）；差距归因必须可核。

**交付**：`reports/p7_cold_fidelity.md`（差距逐项归因表 + 复现命令）。

---

## P1-2【下调，仅合规修复 + 措辞软化】Co-Teaching

- **F-5 违规修复（必做）**：`src/baselines/coteaching_full.py` 当前 `_forget_rate` 用
  `noise_rate`（真实注入率）——违反冻结 F-5（禁用真实噪声率）。改为**估计噪声率**
  （cleanlab `get_noise_indices` / GMM 估计），估计过程只依赖带噪训练集；
- 措辞：把「遗忘机制牺牲稀有证据」的强表述软化为受实现/超参影响的观察，不作机制定论；
- **不扩规模**：只修复 + 重跑受影响的 Co-Teaching 对照，不新增 Co-Teaching 变体。

**交付**：`coteaching_full.py` 变更 + 受影响结果重跑 + 简报段落；守卫测试断言未使用真实注入率。

---

## P1-6【下调，仅讨论】freq_protect 变体

- 在方法节/讨论提前说明「保护观测噪声标签 ỹ」的设计局限（4.4 节），并给一个基于预测分布
  或先验类频率的对照说明（可复用 P0-3 族 C 的 #5 freq-floor 结果作证据，不新跑）；
- 与族 C 结论一致：CDM 加权形式无增量，价值在审计/诊断。**不新增实验**。

**交付**：讨论段落草稿（并入 `reports/p7_baseline_expansion.md` 附节或单独简报）。

---

## 不在本工作包范围（防扩散）

- 不改任何 G2/族 A/B/C 裁决；不重跑已冻结的 P0 结果；
- 不把 P1-4 真实噪声实验升级为新的推断族（描述性即可，F-6）;
- 论文正文修改由协调者统一处理（P1 产出为其提供素材与答复信条目）。

## 汇报格式

每项输出：变更文件、结果表关键数字、新增/既有测试状态、F-5 合规确认、
与预期不符之处（如有则停下等裁定）、git commit hash。P1-4 若因数据未就位而阻塞，
明确标注「等待人工获取修正标签」并继续其余项。
