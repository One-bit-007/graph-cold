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
**数据依赖（人工前置，协调者已核实来源，2026-08-28）**：
- **源 1（主）Engelen/Liu CNS2022**——已核实并**已克隆到位**：
  `E:/graphcold-data/cicids2017_corrected/CNS2022_Code/`（`github.com/GintsEngelen/CNS2022_Code`）。
  权威规则源 = `Labelling/CICIDS2017_original_version_labelling.ipynb`（吃 CIC 原始 CSV，行级对齐）。
  - 改进版数据集 CSV（可选，仅聚合交叉验证）：下载页
    `https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html`
    → 目录 `.../CNS2022/Datasets/`，放 `E:/graphcold-data/cicids2017_corrected/engelen_improved_csv/`。
- **源 2（可选交叉验证）Lanvin CRiSIS2022**：`https://gitlab.inria.fr/mlanvin/crisis2022`
  （原始流上只改标签；有反爬，需人工下载）。下不到不阻塞。
- **就位判定**：`CNS2022_Code` 已克隆 → **本项可立即启动**。CSV/Lanvin 缺失只降级交叉验证。

**关键设计（协调者裁定，避免流对齐陷阱）**：
Engelen **用修复版 CICFlowMeter 重新生成了流**，改进版 CSV 的流边界与原始不同，**不能**与
原始 CICIDS 做行级 1:1 匹配。**但仓库已提供直接可用的解法**：
`CNS2022_Code/Labelling/CICIDS2017_original_version_labelling.ipynb`（已克隆到
`E:/graphcold-data/cicids2017_corrected/CNS2022_Code/`）——README 明确它「用于仅研究标注错误
对分类器的影响」，**直接接受 CIC 原始发布 CSV 作输入**（正是我们的 WP-3 raw 副本 schema），
把 `Label` 清空为 `NeedManualLabel` 后用一组硬编码 `label_flows(...)` 规则重标，输出修正标签
+ `Attempted Category` 列，**行级对齐原始流天然成立**。
- **主路径 = 移植该 notebook**：把其 `label_flows` 规则调用、`format_csv_for_labelling`、
  尤其是**时间戳重建**逻辑原样移植为一个 Python 模块，套用到 WP-3 raw 流上得到修正标签；
  与原始发布 `Label` 逐行 diff = 真实标注错误集。**不要**改规则、不要自行反推。
- **时间戳重建必须精确复刻**（notebook 内逻辑，易错）：解析 `%d/%m/%Y %H:%M` →
  `hour < 7` 则 +12h（作者用 12 小时制但删了 AM/PM，流量集中 9:00–17:00）→ 再 +3h
  从 New Brunswick 夏令时(UTC-3)转 UTC。攻击时间窗在 notebook 中以 unix 纳秒硬编码，须原样保留。
  注意与 WP-3 loader 既有时间戳处理的一致性/差异，如实记录。
- **「Attempted」处理纪律（源站强制）**：凡 `Attempted Category != -1`（意图攻击但无恶意行为）
  的流一律**归为 Benign**；**绝不**把 Attempted 作独立 ML 类别。实现里硬编码并加断言。
- 改进版 CSV / Lanvin 仅作**聚合层面**交叉验证（差集规模、类分布量级），不做行级对齐。

**实验设计（就位后）**：
1. **规则派生 + 差集**：在 raw 副本上派生修正标签，原始 vs 修正差集 = 真实标注错误集；
   报告其规模、类分布、是否集中于尾类；用 Engelen 改进版 CSV（若下到）在聚合计数层面
   交叉验证差集规模合理（不做行级对齐，只比总量/类分布）；Lanvin（若下到）做二次印证；
2. **真实噪声版主对比**：以原始（含真实错误）标签为训练噪声、修正标签为"干净"评估参照，
   在冻结协议口径下跑 {CoLD, ablation_hard, Graph-CoLD-soft, Confident-Learning}，
   报告软保留 vs 硬删除的尾类 Δ（族 A 语义的真实噪声版）、以及 CL 对比；
   — 这是**描述性**验证（真实错误集非注入、无对称 seed 设计），
   按 F-6 不新增 Holm 推断族，用 Δ + bootstrap CI 呈现；
3. **审计结论的真实噪声复核**：检查合成噪声下的核心发现在真实噪声下方向是否一致
   —— ① 软保留>硬删除方向；② 泄漏对删除式方法抬升更大（可选，若样本足够）；
4. 文献重定位（P1-7 落地）：在相关工作/威胁有效性节引入 Engelen(CNS2022)/Lanvin，把本文审计
   增量相对其定位清楚（它们指出/修正标签错误，本文量化标签构造代理泄漏 + 提供隔离协议与回归测试）。

**交付**：`reports/p7_natural_noise.md`（规则派生说明、真实错误集刻画、真实噪声 Δ 表、
方向一致性、文献重定位段）；修正规则来源 URL 与 commit 哈希记录；守卫测试
（Attempted→Benign 断言、修正标签/派生用的实体列不泄漏进特征、差集规模对账）。

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
