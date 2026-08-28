# Codex 工作包 WP-3：CICIDS 原始副本实体隔离（P0-6 ENT 因素补格）

> 触发条件已满足：`E:\graphcold-data\cicids2017_raw\TrafficLabelling\` 已就位（8 个 CSV，
> MD5 校验通过，85 列含 Flow ID / Source IP / Destination IP / Source Port / Destination Port /
> Timestamp / Label）。本工作包补齐 P0-6 泄漏消融表中 CICIDS 的 ENT 因素（当前标 N/A）。
>
> 协调者（Claude）已核验数据完整性，并核过 `src/data/loaders.py` 的 S2 机器
> （`_make_split` 的 `entity` / `temporal_entity` 策略、`entity_cols`、`entity_overlap` 追踪）
> 在 P0-2 已建好。本工作包是接通，不是重建。

## 不可违反的规则（沿用全部既有约束）

1. **FINAL 冻结不可改**：CICIDS 的冻结 FINAL = tier1 屏蔽 + temporal-by-source-file 划分 +
   训练去重，建立在**本地 MachineLearningCVE 副本**（`data/cicids2017/`）上。WP-1/P0-4/P0-5/P0-6
   的全部 CICIDS 数字都以它为准。**严禁**用 raw 副本替换 FINAL 或重跑任何已冻结的 CICIDS 结果。
2. **实体列只作 split 元数据，永不进特征/图**：Source IP / Destination IP / Flow ID / Source Port /
   Timestamp 只用于划分分组与审计度量；必须从特征矩阵与图构建中完全剔除。加回归测试锁死。
3. 禁 oracle 泄漏：flip_mask / clean label 不得进入特征、图、诊断、权重；既有 leakage/proxy
   回归测试必须保持全绿。
4. 已有结果文件只增不改；新产物用新文件名（如 `*_entity_raw` 后缀）。
5. `submission_ready` 保持 false；任何协议偏离先停下报告，不得自行"修正"冻结项。
6. 完成后：报告 + 表 + 测试入库，git commit（不 push，由人类决定）。

## 核心设计约束：raw ≠ ML 副本，ENT 必须内部自洽

raw GeneratedLabelledFlows（85 列）与冻结 FINAL 所用的 MachineLearningCVE 副本（~79 列）是
**不同文件**，特征列不完全一致。因此：

- ENT 因素的 Δ **必须在同一 raw 副本内部**测量：`temporal_entity`（时间边界 + 实体零重叠）
  相对 `temporal`（仅时间边界，实体可重叠）的配对 Δ——两臂都在 raw 副本上，Δ 归因于
  "是否强制实体隔离"，不受 raw-vs-ML 特征差异干扰；
- **不得**把 raw 副本上的绝对数字与 ML 副本上的 FINAL 绝对数字直接相减；
- 报告必须显式声明该测量在 raw 副本上进行，并给出 raw-vs-ML 的行级对账（下）。

## 任务

### 1. 特征集对账（先做，决定可比性）
- 新增 raw 配置项（不覆盖冻结的 `cicids2017.path`）：在 `configs/datasets.yaml` 加
  `cicids2017_raw`，`path: E:/graphcold-data/cicids2017_raw/TrafficLabelling/`，
  `class_policy: postfilter11`，`min_class_count: 1000`，`label_col: Label`，
  `drop_cols: [Flow ID, Source IP, Destination IP, Source Port, Destination Port, Timestamp]`
  （把 Source Port 也纳入 drop：raw 有而 ML 特征集无它，剔除以对齐特征集），
  `entity_cols: [Source IP]`（见任务 2 的实体定义）。
- 装载 raw 后，断言其 `feature_names` 与冻结 ML 副本的 `feature_names` **集合一致**
  （canonical CICIDS 特征集，通常含 Destination Port 作为特征；不含任何 IP/Flow ID/Timestamp/
  Source Port）。若有差集，如实列出并停下报告，不自行增删。
- 行级对账：raw 与 ML 副本按类计数、总行数、特征分布（各特征均值/方差抽查）对比，
  差异写入报告（预期：同源数据，postfilter11 后类计数应高度一致）。

### 2. 实体定义（S2 忠实于冻结 F-4："源/目的实体不跨划分"）
- F-4 要求"实体不跨划分"。单列 Source IP 分组不足以保证 Destination IP 也不跨划分。
  **实现 union-find（并查集）**：把每条流的 (Source IP, Destination IP) 视为一条边，
  连通分量为一个"实体组"；按连通分量做 group-disjoint 划分，保证**任何 IP（无论 src 或 dst）
  都不同时出现在 train 与 test**。
- 复用/扩展 `_make_split` 的 `entity` 与 `temporal_entity` 分支：把单列分组升级为
  "基于 (src,dst) 连通分量的分组"；`entity_overlap` 度量改为"跨划分出现的 IP 数"，
  成功判据 = 0。若无法达到 0（连通分量过大导致划分退化），如实回退并记 note，不伪装通过。
- 保留旧单列语义为可选，不破坏 CESNET/UNSW 既有调用。

### 3. ENT 因素补格（10 seeds × rate 0.6，与 p6 同口径）
- 复用 `src/paper/p6_leakage_ablation.py` 的场景机制，对 raw CICIDS 跑两臂：
  - `temporal`（仅时间边界，实体可重叠）
  - `temporal_entity`（时间边界 + 连通分量实体零重叠）
  - 方法 {CoLD, ablation_hard, Graph-CoLD-soft}，rate 0.6，seeds 0–9；
- 逐 seed 配对 Δ（temporal − temporal_entity）+ bootstrap 95% CI，**描述性**，不新增推断族（F-6）；
- 同时报告两臂的 `entity_overlap` 与 train↔test 精确碰撞率（seeds 0–2 足够）。
- 尾类锁定沿用 p6 的 fixedtail 机制（按 seed 锁到该臂的窗口），避免尾类边界漂移。

### 4. 守卫测试（新增，入复现包）
- `tests/test_cicids_entity_isolation.py`：
  - **实体列不入特征**：装载 raw CICIDS，断言 `feature_names` 不含
    {Flow ID, Source IP, Destination IP, Source Port, Timestamp} 任一；
  - **特征集对齐**：raw 的 feature 集合 == ML 副本 feature 集合（或如实标注差集常量）；
  - **实体零重叠**：`temporal_entity` 划分后跨划分 IP 数 == 0（union-find 正确性，用小型合成图）；
  - **union-find 正确性**：合成 (src,dst) 边集，断言连通分量与手算一致。
- 既有 `tests/test_proxy_field_regression.py` / `test_no_oracle_leakage.py` 保持全绿。

### 5. 报告与回填
- 新报告 `reports/p6_cicids_entity_isolation.md`：设计、raw-vs-ML 对账、ENT 两臂 Δ 表、
  entity_overlap=0 达成证据、碰撞率、限制（raw≠ML 副本，绝对数不与 ML FINAL 相减）；
- 更新 `reports/p6_leakage_ablation.md` 的 §2 主表：把 CICIDS 的 ENT 由 N/A 替换为
  raw 副本内部测得的 Δ（带脚注指向新报告与 raw-copy 限制）；同步更新 §6 限制节第 1 条
  （删除"WP-3 待触发"，改为"已在 raw 副本上补齐，见 p6_cicids_entity_isolation.md"）；
- 更新 `p4.PROTOCOLS['cicids2017']` 的 `limitation` 字段：从"本地副本无实体列"改为
  "FINAL 在 ML 副本上冻结；实体隔离已在 raw 副本上单独测量（见 WP-3 报告）"。

## 不在本工作包范围（防扩散）

- 不重跑任何 ML 副本上的冻结 CICIDS 结果（FINAL/WP-1/P0-4/P0-5/P0-6 其它因素）；
- 不改 G2/族 A/族 B/族 C 任何裁决；
- 不把 raw 副本提升为新 FINAL；
- 论文正文修改由协调者统一处理。

## 汇报格式

输出：变更文件清单、raw-vs-ML 对账结论、entity_overlap 达成情况、ENT Δ 表、
新增守卫测试结果、既有回归测试状态、与预期不符之处（如有则停下等裁定）、git commit hash。
