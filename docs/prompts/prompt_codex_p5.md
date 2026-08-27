# Codex 工作包 P5：族 B 补跑（PA-1）+ P0-3 因子消融（含族 A/族 C 推断）+ CICIDS 实体列适配

> 角色：你是本项目的实现工程师。协调者（Claude）已把导师 2026-08-27 的裁决固化为
> 《论文提升计划 v3》与协议冻结清单的 Protocol Amendments（PA-1 / PA-2）。
> 本工作包含三个子包，**WP-1 优先级最高**，完成并出报告后再启动 WP-2；WP-3 为条件触发。

## 不可违反的完整性规则（沿用全部既有约束）

1. 冻结协议不可改：seeds 0–9（全量）/ 0–2（筛选）、rates {0.4,0.6,0.8}（全量）/ {0.6}（筛选）、
   train 10,000 / test 5,000、F-2 tail 定义、F-3 噪声映射（CESNET 目标 `microsoft-diagnostic`）、
   F-5 调参协议（禁干净验证集、禁真实噪声率）、F-6 统计族与用语（robust/narrow/null，禁 salvaged）。
2. 禁 oracle 泄漏：flip_mask / clean label 不得进入特征、图、诊断、权重的任何计算路径；
   `tests/test_no_oracle_leakage.py`、`tests/test_proxy_field_regression.py` 必须全绿。
3. 已有结果文件（results/、tables/、reports/ 中既有产物）只增不改；新产物用新文件名。
4. `submission_ready` 保持 false；任何协议偏离先停下来报告，不得自行"修正"冻结项。
5. 每个 WP 完成后：报告 + 表格入库，git commit（不 push 由人类决定）。

---

## WP-1【最高优先级】族 B 补跑：CICIDS/UNSW 全量（PA-1 救济）

**背景**：冻结族 B = 3 数据集 × (Macro-F1, Tail-F1) × 3 rates = 18 项，但 P0-4 全量仅跑了
CESNET；CICIDS/UNSW 因粗筛未存活被跳过——该 futility 规则非预注册，构成选择性检验。
导师裁定补跑而非事后修正案：G2 是全文最重要的裁决，不留口子。

**任务**：
1. 复用 `src/paper/p4_cl_graph_combo.py`，执行
   `--mode full --datasets cicids2017,unsw_nb15`（键名以 `p3.DATASETS` 实际为准）；
   各数据集使用 `PROTOCOLS` 中已冻结的最强可用口径（CICIDS = C1-tier1 + S1 temporal；
   UNSW = tier2 + temporal_entity），不得降级。
2. 跑完后合并 CESNET 既有全量结果，对**完整 18 项族 B** 重做 Holm 校正
   （分噪声率配对检验 n=10 + TOST δ=0.01 + 逐类回退护栏 + 混合效应稳健性，全按 F-6）。
3. 产出 `reports/p4_cl_graph_combo_full_family_b_complete.md`：
   - 18 项完整表（Δ、Holm p、95% CI、dz）；
   - G2 终裁决行（预期维持 null，但必须以完整族为据；若任何一项 Holm 显著为正，
     立即停止并上报，不得自行改写 G2 结论）；
   - PA-1 披露段草稿（供修改稿 Protocol Amendments 附录直接引用）。
4. 结果行数自检：每数据集 150 行（10 seeds × 3 rates × 5 methods）、30 配对完整。

**验收**：18/18 项有 Holm 后 p 与 CI；相关测试全绿；报告含 PA-1 披露段。

## WP-2 P0-3 因子消融（导师裁决 Q2 范围 + 族 A/族 C 推断，PA-2 搭车）

**执行范围（以存活声明载重为界，不得扩展）**：

| 档 | 配置 | 协议 |
|----|------|------|
| 全量 | #3 图+CDM+hard、#4 图+CDM+soft、#5 图+CDM+类频率最小权重、#6 无图 class-balanced CE 与 focal（二选一先 CB-CE，focal 作敏感性） | 10 seeds × 3 rates × 3 数据集，屏蔽 + group-disjoint 主口径（与 WP-1 相同 PROTOCOLS） |
| 筛选 | #1 无图 CE、#2 图无 CDM、#7 仅 Dpred / 仅 Dneigh / 仅 Dview | 3 seeds × 0.6，描述性，不做推断 |

**统计（严格按 F-6 冻结族）**：
- **族 C**（18 项）：{3 数据集 × Tail-F1 × (#4 vs #5, #4 vs #6) × 3 rates}，Holm——
  「evidence-rescue 优于简单重加权」声明只依据族 C；若 #4 输给 #5 或 #6，如实报告，
  该声明按计划删除，不做任何补救实验；
- **族 A**（9 项）：{3 数据集 × Tail-F1 × (#4 vs #3) × 3 rates}，Holm——这是 PA-2 要求的
  F-E 正式确认：`robust`（≥2 数据集 PASS）方可保留「软保留机制发现」为存活声明。

**实现要求**：
- 新建 `src/paper/p5_factor_ablation.py`，复用 p3/p4 的数据装载、屏蔽与划分代码路径
  （同一编码器、同一训练预算，F-5 冻结超参）；
- #5 类频率最小权重基线：w(v) = max(τ_ret, freq-based weight)，实现须与 4.4 节将来的
  披露一致，写入 docstring；#6 class-balanced CE 用 effective number (Cui et al. 2019)
  或 1/√freq，选择理由写入报告；
- 新增守卫测试：消融各格共享同一划分索引与噪声注入（同 seed 同 rate 下逐位一致），
  防止"格间差异来自数据而非因子"。

**交付**：`reports/p5_factor_ablation.md` + `tables/table_p5_family_a.csv` +
`tables/table_p5_family_c.csv` + 筛选档描述性附录表；结论行明确给出族 A 与族 C 的
robust/narrow/null 判定。

## WP-3【条件触发，一周时间盒】CICIDS 实体列适配（导师裁决 Q3）

**背景**：本地 MachineLearningCVE 已剥离 IP/时间戳，S2 实体隔离不可执行——
`PROTOCOLS['cicids2017'].limitation` 已如实记录。导师裁定：值得获取原始
GeneratedLabelledFlows CSV（含 Flow ID / Source IP / Destination IP / Timestamp 列），
时间盒一周。

**人类任务（等待用户完成）**：从 UNB 官方渠道下载 GeneratedLabelledFlows 并放置于
`E:\graphcold-data\cicids2017_raw\`（或用户指定路径），提供文件清单。

**Codex 任务（数据就位后触发）**：
1. `src/data/loaders.py` 适配新 schema：实体列（Source IP, Destination IP）保留为
   split-only 元数据，**绝不进入特征或图构建**（代理字段回归测试同步扩展覆盖）;
2. 与既有 MachineLearningCVE 版本做行级对账（特征列分布一致性抽查），差异如实记录；
3. 实现 CICIDS S2（源/目的实体对分组隔离）与 S1∩S2，更新 `PROTOCOLS['cicids2017']`
   的 limitation 字段；重跑 P0-2 粗筛口径的 S2 重复交集测量，补进审计报告。
4. **时间盒规则**：若数据在启动 WP-2 完成前仍未就位，写一段「限制如实声明 + 以
   CESNET/UNSW 的 S2 结果作为该因素证据」的正文草稿放入报告，WP-3 关闭不再等待。

## 不在本工作包范围（防扩散）

- P1-4 自然噪声实验（Engelen/Lanvin 修正标签路径）——Phase 4 另开工作包；
- 任何论文正文修改——由协调者在 P0-5/P0-6 数据齐后统一执行；
- 族 C 之外的额外消融格——一律拒绝，理由见计划 v3「深度止损线」。

## 汇报格式

每个 WP 结束输出：变更文件清单、测试结果（含新增守卫测试）、关键数字表、
与预期不符之处（如有，停下等待协调者裁定）、git commit hash。
