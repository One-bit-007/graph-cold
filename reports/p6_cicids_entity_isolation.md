# WP-3 / P0-6 补格：CICIDS-2017 raw 副本实体隔离（ENT）测量 — 裁定报告

日期：2026-08-28 ｜ 执行：WP-3（prompt_codex_wp3.md）｜ 状态：**完成，裁定为「构造性不可行（degenerate-by-construction）」**

## 0. 裁定（先说结论）

**CICIDS-2017 的 ENT 因素在 union-find 实体定义下不可测量：流图连通分量退化。**
raw 副本 19,130 个 IP 只形成 **4 个连通分量**，其中一个巨型分量覆盖 **19,058 个 IP（99.6%）与 99.96% 的有效流**。按 F-4「任何 IP（无论 src 或 dst）都不跨划分」的忠实语义做 `temporal_entity` 划分时，测试窗触及巨型分量，训练侧清除后仅剩 **68 行单一类（BENIGN）**，划分退化。加载器按既定纪律**如实回退 stratified 并记录 note，未伪装通过**（守卫测试 `test_giant_component_temporal_entity_falls_back_honestly` 锁定此行为）。

因此：

- **没有产出任何 ENT 性能 Δ 数字**——在退化划分上跑 10 seeds × 3 方法得到的将是 stratified 伪装的假测量，本报告明确拒绝此举；
- p6 泄漏消融主表中 CICIDS 的 ENT 单元格保持 **N/A**，但脚注由「本地副本无实体列（WP-3 待触发）」更新为「**已在 raw 副本上以 union-find 实体定义尝试，判定构造性不可行**」，并指向本报告；
- 这本身是一个应进正文限制节的**数据集层面发现**：CICIDS-2017 单网络捕获拓扑（共享受害服务器/基础设施 IP）使实体级组不相交划分在其上无意义——与 UNSW「官方固定划分本身含 38.4% 精确碰撞」并列，同属「基准数据集自带泄漏结构」的证据。

## 1. 设计与配置

- 新增独立配置键 `cicids2017_raw`（`configs/datasets.yaml`），指向 `E:/graphcold-data/cicids2017_raw/TrafficLabelling/`（8 个 CSV，约 1.2 GB，协调者已 MD5 校验）。**冻结的 `cicids2017`（ML 副本）条目原样未动；FINAL 未重跑、未替换。**
- `drop_cols = [Flow ID, Source IP, Destination IP, Source Port, Protocol, Timestamp]`。
  **对提示词的显式偏离（已核验为提示词笔误）**：提示词列出的 drop 清单含 `Destination Port`、不含 `Protocol`；但 raw 85 列 = ML 79 列 + 6 个多出列（Flow ID / Source IP / Source Port / Destination IP / **Protocol** / Timestamp），Destination Port 在 ML 副本中本来就是特征。故正确 drop 是上述 6 列，Destination Port 保留为特征。守卫测试 `test_entity_columns_never_enter_features` 同时断言「六列不入特征」与「Destination Port 在特征中」。
- 实体定义：union-find——每条流的 (Source IP, Destination IP) 视为无向边，scipy `connected_components` 求连通分量，**逐行**实体 id = 该行的分量 id；`meta["entity_component_ips"]` 保存分量→成员 IP 映射供审计。union-find 在**下采样/过滤之前的全量行**上构建（最保守）。
- 实体列只作 split 元数据：抽取发生在 `drop_cols` 之前、特征矩阵构建完全在其后，永不进特征/图。
- 编码回退：raw 的 Thursday-Morning-WebAttacks 文件含 cp1252 字节 0x96（en-dash），UTF-8 读取整文件失败；`_read_table` 增加 cp1252 回退并 `warnings.warn`。0x96 → U+2013，标签还原为「Web Attack – Brute Force」。

## 2. raw vs ML 副本对账（reports/p6_cicids_raw_reconcile.json）

| 项目 | raw 副本 | ML 副本 | 判定 |
|---|---|---|---|
| 文件 | 8 CSV，同名同序 | 8 CSV | 一致 |
| 列头 | 85 列 | 79 列 | raw 恰多 6 列（见 §1），逐文件列头一致 |
| 特征集（drop 后） | 78 个 | 78 个 | **集合完全相等**（diff 双向为空） |
| postfilter11 类计数 | 11 类，总计 787,999 | 11 类，总计 787,999 | **11 类计数逐类完全相等**（BENIGN 231,073 / DoS Hulk 231,073 / PortScan 158,930 / DDoS 128,027 / … / Web Attack–BF 1,507；对照 `cicids_class_policy_report.json`） |
| temporal 划分 | 严格时间边界通过（train_max 1499398980 < test_min 1499399040），train 630,399 / test 157,600 | ML 副本无 Timestamp 列，用按文件星期序回退 | raw 支持真实时间戳，这是 WP-3 的附加价值 |
| 标签字符串 | 「Web Attack – Brute Force」（U+2013） | 「Web Attack \ufffd Brute Force」（替换符） | 装饰性差异；类身份按计数/顺序一致，raw 内部配对不受影响 |

对账中发现的 raw 文件瑕疵（如实记录）：Thursday-Morning-WebAttacks 文件自第 170,367 数据行起有 **288,602 行纯逗号空行**（全 85 列 NaN）。这些行 Label 为 NaN，被加载器 `valid_label_mask` 丢弃，不进入分类池（ML 副本的清洗等价地排除了它们；两侧 postfilter11 计数完全相等可互证）。两臂走同一加载路径，配对测量不受影响。

装载耗时：temporal 臂 43–53 s，temporal_entity 臂 52 s（含 union-find）。

## 3. 连通分量结构与退化证据

全量 3,119,345 行（含空行）上的 union-find 结果：

| 分量 | IP 数 | 流数 | 说明 |
|---|---:|---:|---|
| 0（巨型） | 19,058 | 2,829,491 | 单网络捕获拓扑：所有攻击源经共享受害/基础设施 IP 连成一体；含全部 11 个有效类 |
| 1 | 69 | 68 | 0.40.x/0.44.x/8.0.69.0 杂散外联地址，全部 BENIGN |
| 2 | 2 | 1,184 | 8.0.6.4 ↔ 8.6.0.1 孤立对，全部 BENIGN |
| 3 | 1（`<missing>`） | 288,602 | 即 §2 的纯逗号空行（IP/Label 全 NaN），Label 过滤后消失 |

`temporal_entity` 清除模拟（80/20 时间切，与加载器逻辑一致）：

- 测试期（最迟 20%，2017-07-07 ~11:04 UTC 起）触及分量 {0, 2}；
- 训练侧清除这些分量后剩 288,670 行 = 288,602 空行 + **68 行 BENIGN（分量 1）**；
- Label 过滤后训练池为**单一类 68 行** → 退化（加载器内部判据 `np.unique(y[train]).size >= 2` 失败）→ 如实回退 stratified，note 记录 `fallback`。

加载器实测（冒烟运行，`reports/p6_cicids_raw_reconcile.json` 的 `temporal_entity_arm`）：`applied_strategy = stratified`，带 `fallback` note——与模拟完全一致。非实体的 `entity` 策略同样退化（首个被抽中的分组即巨型分量，测试集会吞掉 ~99.96% 的行），无替代测量路径。

**scope 说明**：union-find 在过滤前全量行上构建；即使只在有效行上构建，结论不变（巨型分量在有效流中占比 99.96%）。

## 4. 对 p6 主表的回填口径

- `reports/p6_leakage_ablation.md` §2 主表 CICIDS·ENT：`N/A` → `N/A（构造性退化）`，新增脚注指向本报告；
- §1 条件设计表 ENT 行 CICIDS 列同步更新；
- §6 限制节第 1 条由「WP-3 待触发（时间盒 2026-09-03）」改为「已在 raw 副本上以 union-find 实体定义尝试并裁定构造性不可行（巨型连通分量覆盖 99.96% 有效流），见 p6_cicids_entity_isolation.md」；
- `p4.PROTOCOLS['cicids2017']` 的 limitation 字段同步（代码内字符串）。

**禁止事项（沿提示词纪律）**：raw 绝对数不与 ML FINAL 绝对数相减；未产出 Δ 即不存在该风险；未新增推断族。

## 5. 守卫测试（tests/test_cicids_entity_isolation.py，6/6 通过）

1. `test_union_find_components_hand_computed` — 合成边集连通分量与手算一致（含三角形闭合）；
2. `test_union_find_src_dst_share_component_and_missing_handled` — 逐行实体 id 正确、`<missing>` 节点处理；
3. `test_entity_columns_never_enter_features` — 六列实体/代理列不入特征，Destination Port 保留；
4. `test_temporal_entity_zero_cross_split_ip_overlap` — 合成双分量数据上跨划分 IP 数 == 0；
5. `test_giant_component_temporal_entity_falls_back_honestly` — 巨型分量拓扑必须回退并记 note，不许伪装通过；
6. `test_raw_feature_set_matches_ml_copy` — 真实列头级特征集对齐（数据目录缺席时 skip）。

回归：`tests/test_proxy_field_regression.py` + `test_no_oracle_leakage.py` + `test_p6_leakage_ablation.py` 22/22 通过。全量套件 240 passed / 20 failed，其中 20 个失败为 d6/d7/d8/d11 旧稿件管线门与 smoke 门，在干净 HEAD（691c36c）上 stash 复验同样失败，与 WP-3 无关。

## 6. 限制

1. ENT 的「不可行」裁定仅在 union-find（src,dst）连通分量定义下成立——这正是 F-4 忠实语义；任何更弱的实体定义（如仅 src IP、或剔除枢纽 IP）都会放松「任何 IP 不跨划分」的保证，未采用。
2. raw 副本与 ML 副本为同源不同清洗；本报告所有数字均出自 raw 副本内部，不与 ML FINAL 混算。
3. Thursday-Morning-WebAttacks 的 288,602 空行是 raw 文件固有瑕疵，已由 Label-NaN 过滤中性化；若未来换用官方修复版文件，分量结构结论预计不变（空行不参与连边）。
4. CICIDS ENT 缺位对论文结论的影响有界：p6 的四因素分解中 CICIDS 的 DUP/PRX/ORA/ALL 均在 ML 副本上完成且有效；ENT 在 CESNET（+0.187）与 UNSW（+0.002）上有测量，CICIDS 的 ENT 贡献以「数据集拓扑使其实体隔离不可定义」替代陈述。
