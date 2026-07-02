# GPT-pro Prompt — D2 (reconcile D1 math to specs + author D2 Codex prompt)

> Paste to GPT-pro. Coordinator (Copilot) has reviewed your D1 output against the
> authoritative specs and Codex's D1 code. Below are binding reconciliation
> directives, then your D2 orchestration task.

# 状态同步（Codex D1 已完成并推送）
- 远端 One-bit-007/graph-cold：main/dev/tag d1 = 7387deb。
- Codex 已实现并通过冒烟：src/data/loaders.py、src/data/noise.py(sym/asym，
  graph_consistency 留 TODO D4)、src/models/cold_baseline.py(相关→MST→DFS、
  逐子集 GMM-CDM、硬删除ε=0)。
- 协调者标注一处待修：CoLD 基线 fit_representation 目前用 PCA，而非 CoLD 真正的
  自监督对比编码(L_la+L_gr)。D2 需将 CoLD 基线表示层升级为真实编码器，确保与
  Graph-CoLD 公平对比(否则低估 CoLD)。

# 一、纠偏指令（你的 D1 定稿必须按此修订，或更新 docs 规格并给出理由待协调者签核）
权威规格：docs/spec_method_impl.md、docs/spec_graph_noise.md。你的 D1 写法有 6 处偏离：

1. D_pred：回退为逐视图 GMM 簇标签不一致 (1/M)·Σ_k 1[ỹ_v^(k) ≠ y_v]，
   继承 CoLD CDM。熵/margin 仅作“可选软变体”附注，不得替换主定义。
2. D_neigh：回退到“标签空间” KL(ŷ_v ‖ Agg_{u∈N(v)} ŷ_u)，归一化到[0,1]。
   禁止用嵌入 L2 作为主定义。
3. D_view：回退到“标签空间”——逐视图预测标签的众数不一致 (1 - mode-agreement)。
   禁止用嵌入 L2 作为主定义。
   【关键理由，务必在论文中显式说明】Graph-CDM 必须停留在标签/预测空间，
   以保住“因果散度/噪声转移概率”的解释力，并干净退化到 CoLD；若改用嵌入距离，
   等于重新引入 CoLD 明确批判的“距离度量受局部一致性干扰”缺陷，会被审稿直接质疑。
4. e(v)：回退为显式可解释式 e(v)=freq_protect(n_{y_v})·(1+γ·anom(v)) 并 minmax 归一。
   freq_protect∈{log,inverse}。可学习证据编码器仅列为 future work，不作主方法(7天不可控)。
5. ERR：主定义 ERR=Σ_v w(v)e(v) / Σ_v e(v)（保留）。删除你写的“结构版”
   (E[w·e/e]·E[w]=E[w]² 是错误式)。如需结构版，改为“按低频/尾部类分层的 ERR”，
   即先在低频类子集内计算 ERR 再宏平均，并给出正式式。
6. 图噪声：保留 docs/spec_graph_noise.md 的“多类 + β(consistency_bias) 一致性边偏置”
   生成机制，且保持“β=0 ⇒ 退化为 symmetric”性质(CK-2 依赖)。你的 SOC 噪声来源叙述
   (SIEM 误触发/攻击链传播/多源视图冲突)很好，采纳为“动机与来源”段落，但生成式仍用
   β-加权一致性边翻转(多类)，不得改成二分类泛化转移。

Figure 1 概念图规格保留你的版本（左CoLD/右Graph-CoLD，冷→暖，Evidence Extension 箭头），
但把右侧 Graph-CDM 四项标注改为“标签空间一致性(prediction/neighborhood/view/chain)”。

# 二、你的 D2 编排任务：产出 prompt_codex_d2.md
依据 docs/spec_method_impl.md §1(五视图)+§2(表示学习)，产出给 Codex 的实现提示词，
命名 docs/prompts/prompt_codex_d2.md，固定结构【角色/背景要点/本次任务(限当日)/严格契约/
Git 与 tag/验收标准】。必须修正你上一版 D2 样例的 3 个问题：

A. 表示层必须包含 CoLD 对齐的三件套(不得省略)：
   - 特征混淆(Bernoulli(δ) mask, CoLD Eq.2) 作为图节点特征扰动；
   - 局部对齐对比损失 L_con(InfoNCE，正对=同节点跨视图，负对=batch 内他节点，温度τ)；
   - 全局重构损失 L_recon(单视图重构全局节点特征) + 时序对齐 L_temporal(相邻快照同节点对齐)。
   目标 L_rep = L_con + α·L_temporal + β·L_recon。视图聚合默认 MEAN(CoLD 最优)。
B. 编码器：优先异构可用的 HGT/RGCN；若用 GAT/GraphSAGE 需按视图分别建边并支持异构关系，
   在提示词中明确 fallback 与理由。嵌入维 d=128。
C. 验收标准修正：stage-1 无分类器，禁止用“与 CoLD Macro-F1 差<0.02”。改为：
   - 五视图均非空、node embedding shape=[N,128]；
   - L_rep 收敛(记录曲线)；
   - 线性探针(linear probe) 或 kNN 在干净标签上 Macro-F1 优于随机基线(证明表示有效)；
   - seed=42 可复现；
   - 同时把 CoLD 基线表示层从 PCA 升级为该对比编码器(公平对比要求)。

# 三、协调者关键检查点(你的产出需可被核对)
CK-4 Graph-CDM 数学与代码一致(标签空间四项); CK-5 消融退化正确(ρ=0+硬阈值≈CoLD);
CK-6 指标定义对齐(ERR 主式 + 分层版) + t 检验对标 CoLD; 其余 CK-1/2/3/7 同前。

# 四、输出格式
1) 修订后的 D1 四份定稿(标注每处相对上一版的改动)。
2) docs/prompts/prompt_codex_d2.md 全文。
3) 全程强调 SOC/企业网络工程与安全运营意义(C&S 风格)，避免纯 ML 抽象表达。
