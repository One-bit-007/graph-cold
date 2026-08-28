# P1-4 + P1-7：CICIDS-2017 真实标注错误实验（Engelen 修正标签）与文献重定位

日期：2026-08-28 ｜ 工作包：P7（Phase 4）最高优先项 ｜ 统计性质：**描述性**（真实错误集非注入、无对称 seed 设计；F-6 下不新增推断族）

## 0. 结论速览

1. **真实错误集**：原始发布标签 vs Engelen 修正标签逐行 diff = **128,247 行（有效行的 4.53%）**，其中 34,415 行为 Attempted（意图攻击但无恶意行为），按源站纪律一律归 BENIGN。
2. **族 A 方向在真实噪声下成立**：冻结 FINAL 口径（temporal 臂）Graph-CoLD-soft − ablation_hard 的 Tail-F1 Δ = **+0.284 [+0.089, +0.487]**，Macro-F1 Δ = +0.073 [+0.010, +0.133]；分层敏感性臂同向（Tail +0.010 [-0.001, +0.020]，弱正）。「软保留优于硬删除」不依赖合成噪声假设。
3. **CL 强度与 G2=null 语境一致**：temporal 臂 CL 尾类 0.910 ≥ soft 0.864（Δ -0.046 [-0.132, +0.001]）；CoLD 尾类 0.000 复现 FINAL 行为。图诊断相对 CL 无增量——真实噪声下同样成立。
4. **结构性发现**：真实标注错误几乎全在训练期（train 池翻转率 4.70%，test 池 0.31% 且全部是 Bot→BENIGN）——冻结时间窗下「评估参照」几乎干净，实验灵敏度主要来自训练期噪声。分层臂（补充、非冻结口径）让噪声进入评估窗，结论方向不变。

## 1. 来源与保真度

- 规则源：`Labelling/CICIDS2017_original_version_labelling.ipynb`，仓库 `github.com/GintsEngelen/CNS2022_Code`，本地克隆 commit **f0ce502**（`E:/graphcold-data/cicids2017_corrected/CNS2022_Code/`）。该 notebook 直接接受 CIC 原始发布 CSV（即 WP-3 raw 副本 schema），行级对齐天然成立。
- 移植：`src/paper/p7_engelen_labels.py` 逐字移植 `label_flows` / `format_csv_for_labelling` / 全部 8 文件的规则（攻击时间窗 unix 纳秒、端口列表、additional_filters、flip 语义、规则顺序均原样保留；Monday 无规则=全 BENIGN）。
- **与 notebook 的差异（行为保持，逐项记录）**：
  1. 只读规则所需的 13 列（usecols，控内存）；
  2. pandas≥3 移除 `floor('T')` 别名 → `floor('min')`（语义相同）；
  3. pandas≥3 Copy-on-Write 下 `df["Label"].mask(..., inplace=True)` 为无操作 → 改为赋值形式（否则全部标签退化为 NeedManualLabel→BENIGN，移植即失效；这是本移植最关键的适配）；
  4. raw 镜像的 Thursday-Afternoon-Infilteration 等文件含少量带秒时间戳（`%d/%m/%Y %H:%M:%S`）→ 精确格式失败后回退 `format="mixed", dayfirst=True`（语义相同，警告记录）；
  5. 标签输出改为内存 DataFrame + gzip CSV（`results/p7_corrected_labels.csv.gz`，3,119,345 行 × (source_file,row_in_file) 行级键）。
- 时间戳重建（易错点，已精确复刻）：`%d/%m/%Y %H:%M` 解析 → hour<7 加 12h（作者 12 小时制删了 AM/PM）→ 再加 3h（New Brunswick 夏令时 UTC-3 → UTC）。守卫测试锁定两个时刻样例。
  **与 WP-3 loader 的差异（如实记录）**：WP-3 的 temporal 划分用 `_timestamps_to_seconds` 直接解析原始字符串、未做 +12h/+3h 重建；WP-3 的退化裁定不依赖时刻正确性（连通分量与划分尺寸不变），但任何未来基于绝对时刻的对齐必须用本模块的重建结果。
- **Attempted 纪律**：`apply_attempted_discipline` 硬编码 + 断言：凡 `Attempted Category != -1` → BENIGN；输出中不允许任何 "Attempted" 字样残留（测试锁定）。
- 可选交叉验证源未就位、不阻塞：Engelen 改进版 CSV（未下载）与 Lanvin CRiSIS2022（反爬需人工）仅作聚合计数级交叉验证用；本版报告只含内部一致性对账。

## 2. 真实错误集刻画（reports/p7_engelen_diff_summary.json）

总量：3,119,345 行（含 288,602 空行瑕疵，见 WP-3 报告）；有效标签行 2,830,743；差异行 **128,247（4.53%）**。

| 原始 → 修正 | 行数 | 解读 |
|---|---:|---|
| BENIGN → Infiltration - Portscan | 65,634 | 原始发布漏标的 NMAP/渗透扫描（修正版新类） |
| BENIGN → DDoS | 31,339 | 原始发布漏标 DDoS（修正后 DDoS 达 159,366） |
| DoS Hulk → BENIGN | 10,290 | 原始把 attempted/伪影流标成攻击 |
| FTP-Patator → BENIGN | 3,966 | 同上 |
| BENIGN → DoS Hulk | 3,667 | 原始漏标 |
| DoS Slowhttptest → BENIGN | 3,099 | attempted 伪影 |
| SSH-Patator → BENIGN | 2,937 | 同上 |
| DoS GoldenEye → BENIGN | 2,579 | 同上 |
| DoS Slowloris → BENIGN | 1,726 | 同上 |
| Web Attack - Brute Force → BENIGN | 1,434 | **该类 1,507 例中 95% 是标注伪影**（修正后仅剩 74 正例） |
| 其余 16 对 | ≤636/对 | 含 Botnet→BENIGN 494、各类零星漏标 |

尾类集中度：Web Attack 族（BF/XSS/Sql）在修正后合计 <100 正例——这些「稀有类」很大程度上是原始标注管线的伪影，这直接呼应本文「标签构造代理泄漏」的核心论点。

## 3. 实验设计（冻结协议口径 + 双标签）

- 数据：`cicids2017_raw`，FINAL 对齐 = tier1 屏蔽（加 drop Destination Port）+ **按源文件星期序时间划分**（ML 副本语义；strict 时间戳切分会让测试窗只剩 DDoS/PortScan 导致 Tail-F1 无定义——loader 新增 `temporal_order: source_file` 旗标）+ 训练精确去重（d5 语义，索引级携带双标签）。
- 训练标签 = **原始发布标签**（含真实错误）；评估参照 = **Engelen 修正标签**。修正标签落在冻结 11 类空间之外的测试行（如 Infiltration - Portscan）从指标中剔除并计数（构造上不可预测）；训练行保留原样（这正是真实噪声）。
- 窗口 train 10k / test 5k，按原始标签分层，seeds 0–9，四方法共享窗口（逐 seed 配对）。
- 方法：CoLD / ablation_hard / Graph-CoLD-soft（CDM 来自观测原始标签，evidence freq_protect log）+ Confident-Learning（`noise_rate=0.0` → cleanlab `get_noise_indices` 用折外概率；**F-5 合规：全流程无真实注入率、无干净验证集**）。
- 尾类：F-2 窗口感知、按原始训练标签（训练者视角），方法内逐 seed 一致。
- 产物：`results/p7_natural_noise.csv`（temporal 臂 40 行）、`results/p7_natural_noise_stratified.csv`（分层敏感性臂 40 行，**非冻结口径，仅描述性补充**）、`reports/p7_natural_noise{,_stratified}.json`。

## 4. 结果

### 4.1 主臂（FINAL 对齐 temporal 窗，修正标签参照，10 seeds 配对均值）

| 方法 | Macro-F1 | Tail-F1 |
|---|---:|---:|
| Graph-CoLD-soft | 0.257 | 0.864 |
| Confident-Learning | 0.270 | 0.910 |
| ablation_hard | 0.184 | 0.581 |
| CoLD | 0.050 | 0.000 |

配对 Δ（bootstrap 95% CI，描述性）：

| 对比 | Tail-F1 Δ | Macro-F1 Δ |
|---|---|---|
| soft − hard | **+0.284 [+0.089, +0.487]** | **+0.073 [+0.010, +0.133]** |
| soft − CoLD | +0.864 [+0.734, +0.989] | — |
| soft − CL | -0.046 [-0.132, +0.001] | — |
| CL − hard | +0.329 [+0.131, +0.529] | — |

原始标签参照下数字几乎相同（测试窗修正标签与原标签重合度 99.7%），两组 Δ 完全一致。

### 4.2 分层敏感性臂（非冻结口径，补充）

| 对比 | Tail-F1 Δ (clean ref) |
|---|---|
| soft − hard | +0.010 [-0.001, +0.020]（弱正） |
| soft − CoLD | **+0.274 [+0.251, +0.301]** |
| soft − CL | **+0.037 [+0.025, +0.053]** |
| CL − hard | -0.027 [-0.047, -0.011] |

分层臂中真实错误进入评估窗（orig-ref 与 clean-ref 差距显著，如 hard Macro-F1 0.882 vs 0.709），族 A 方向仍保持但幅度大幅收窄；CL 在此臂不再最强。两臂合并解读：**「软保留 ≥ 硬删除」方向稳健；CL 与 soft 的相对位置随协议而变，不支持任何一方的一致优势声称**——与 G2=null / 族 C null 的既有裁决自洽。

## 5. 审计结论的真实噪声复核

| 合成噪声核心发现 | 真实噪声下 | 判定 |
|---|---|---|
| ① 软保留 > 硬删除（族 A） | 两臂同向（temporal 强、stratified 弱正） | **方向一致** |
| ② CoLD 删除式在干净协议下尾类崩溃 | CoLD Tail-F1 = 0.000（temporal 臂） | **复现** |
| ③ 图诊断相对 CL 无增量（G2=null） | temporal 臂 CL ≥ soft；stratified 臂 soft 略优但 CL−hard 为负 | **方向一致**（无一方一致占优） |
| ④ 泄漏对删除式方法抬升更大 | 需泄漏条件 × 真实噪声交叉网格，样本/语义不支持本工作包内完成 | 未测（如实标注） |

## 6. 限制

1. 冻结时间窗的结构性限制：真实错误集中在训练期（4.70% vs 0.31%），测试窗几乎干净，主臂 Tail-F1 实质由 PortScan 单类构成（FINAL 已知「尾类为单类」问题的延伸）；分层臂仅为补充证据，不进任何正式声明。
2. 修正标签本身也是规则派生（非人工逐行复核）；Engelen 改进版 CSV / Lanvin 的聚合级交叉验证留作后续（不阻塞）。
3. Web Attack 族在修正后正例 <100，相关逐类结论不可做。
4. 训练池中 6,533 行的修正标签落在冻结类空间外（Infiltration - Portscan 等），以原始标签留在训练集中（真实噪声的一部分），不参与评估。
5. 本实验全部描述性；不新增 Holm 族、不改任何 P0 冻结裁决。

## 7. P1-7 文献重定位段（供协调者并入相关工作/威胁有效性节）

> 草稿（英文骨架，协调者统稿）：
> Engelen et al. (CNS 2022) systematically documented and repaired CICIDS-2017 labeling errors, releasing corrected labels derived from rule-based relabeling of the original flows; Lanvin et al. (CRiSIS 2022) independently catalogued dataset errors and showed that correction materially changes measured detection performance. Our work is complementary and goes beyond label correction in three ways: (i) we show that the *label-construction pipeline itself* leaks into standard benchmark features (label-proxies such as destination port), inflating measured performance by 25–36 points under a four-factor isolation audit; (ii) we provide a frozen isolation protocol with regression tests so that entity/temporal/duplicate leakage cannot silently re-enter; (iii) using Engelen et al.'s corrected labels as a *real* noise reference, we show that the central methodological finding of this paper — soft evidence-preserving retention outperforms hard deletion under label noise — holds for the actual documented annotation errors of CICIDS-2017, not only under synthetic noise. We additionally report that entity-disjoint evaluation is infeasible by construction on CICIDS-2017 (a single connected component covers 99.96% of valid flows), a dataset-topology limitation that prior audits did not quantify.

## 8. 守卫测试与合规

- `tests/test_p7_natural_noise.py` 9 项：Attempted→BENIGN 断言 ×2、命名空间映射、label_flows 窗口/翻转语义、时间戳重建样例、差集对账（总量/有效行/attempted 数/两大 published 方向）、P7 配置屏蔽列 + FINAL 对齐旗标、`__row_in_file` 不入特征且与划分对齐、双臂网格完整性（40+40 行）。
- 回归：proxy/oracle/p6/p4/entity-isolation 合计 41/41 绿。
- F-5：CL `noise_rate=0.0`（cleanlab 折外概率自估计）；无干净验证集；无真实注入率。
- 冻结：未改任何 FINAL/P0 结果；结果文件均为新文件名。
