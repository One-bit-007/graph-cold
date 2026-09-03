# P10 跨运行器一致性审计

日期：2026-09-03。范围：P3–P10 当前主协议执行路径；历史协议作为显式例外保留，不修改冻结结果。`submission_ready=false`。

## 冻结概念与调用链

| 概念 | 唯一基础实现 / 主协议入口 | 当前调用者 | 验证与边界 |
|---|---|---|---|
| 噪声目标 F-3 | `p3._noise_target` | p4 `_sample`、p5 `_prepare_scenario`、p6 `_scenario_rows`、p9 `prepare_scenario`；p10 复用 p4/p5 | CESNET 固定 11；IDS 用元数据 benign_class；不重新用当前窗口众数决定 CESNET 目标。新 UNSW 行显式记录 0。 |
| Tail 类 F-2 | `p3._tail_labels`：训练窗口类计数 ≤ 中位数 | p5/p9 直接调用；p4/p6 经 `_tail_noise_labels` | IDS 噪声候选额外排除 benign。这不是第二种频率阈值定义；P9 已验证 30 窗口两种入口得到相同 tail。P6 对照沿用 FINAL 的逐 seed tail lock。 |
| Tail-F1 | `metrics_canonical.canonical_tail_f1_from_perclass` | p3 `_tail_f1` → p4/p5/p6/p7/p10；p9 直接调用 helper | 完整评估集逐类 F1，再对其中出现的 tail 类求均值；head→tail 假阳性计入。自然噪声 cleanref 先按可评估真值范围筛选，因此评估总体不同，不能和 synthetic 数值无条件合并。 |
| 标签排除 | `loaders._feature_inputs` | 普通加载路径与 UNSW 官方分片加载路径 | UNSW `label` 与 `attack_cat`（含大小写、空格、下划线变体）永久排除；选中的目标仍可供编码，二元/多类模式均有测试。PRX/ALL 不能恢复标签字段。 |
| 代理屏蔽 | `p3.PROXY_TIERS` + `_condition_cfg` | p4 `_dataset_cfg` → p5/p9/p10；p6 从同一入口构造因子开关 | 当前最终特征逐列清点 77/29/184；`_load_primary_dataset` 检查屏蔽字段及其 one-hot 派生列。实验性 PRX 是显式协议对照，非主协议。 |
| 划分 | `p4.PROTOCOLS` + `_dataset_cfg` → loaders | p4/p5/p9/p10 复用 `_load_primary_dataset`；p6 FINAL 同源 | CICIDS source-file 时间留出；CESNET temporal_entity；UNSW 官方固定分片。UNSW 的配置 `stratified` 不意味着官方文件被再随机划分，也不意味着实体隔离。 |
| 采样窗口 | `p2d._stratified_sample` | p4/p5/p6/p9/p10 | train=10,000/test=5,000，seed 与 seed+1000；主协议 seeds 0–9。P10 对全部 30 窗口验证 y、索引、X 哈希；UNSW 只删除一列，其余输入完全一致。 |
| 图构建 | `p2d._feature_graph` | p4/p5/p6；p9 按其基线契约不使用图 | 输入仅 X_train；使用各数据集 active_views；没有为本次修复换图策略。 |
| 标签噪声注入 | `p2e._inject_tail_asymmetric` | p4/p5/p6/p9 | P10 UNSW 每个 seed/rate 的 noisy y 哈希核对 P9 原协议，避免删特征时暗改标签。 |
| F-5 | 观测 noisy y 的 CDM、CL OOF | p4/p5/p9 主实验 | p4 不再向 CL 构造器传真实 effective flip rate（旧 cleanlab 路径未使用该参数，但其 fallback 通路不应收到它）。P10 正式 CL 必须 `cleanlab_engine=cleanlab`，否则停止。 |
| 完整推断族 | p5 `_family_tests(size=9/18)`、p4 `_aggregate(full)` | p10 合并帧 | A=9/C=18/B=18，单格配对 n=10；拒绝缺格/重复格。不得只对 UNSW 子族校正后冒充完整族。 |

## 历史路径与不能自动统一的口径

搜索 `src/paper/*.py` 所有 tail/noise helper 定义及调用点：`p2e._tail_macro_f1` 仍由历史 p2e/p2f、p5_external_baselines 和 p6_scale_validation 调用。这些是 **masked-tail 历史路径**，不是本轮 canonical 主表数据来源；不批量改写旧指标，也不声称全仓历史口径已经一致。旧 iid 外部基线和尺度结果不能无注释地并入 canonical 数值表。

原 p4 Family B CSV 的 Tail-F1 同样是 masked。P10 新 p4 执行改用 canonical；旧 CSV、旧推断 JSON 与 P9 的事后敏感性产物全部原样保留。完整 B 新计算必须使用 **P9 canonical 派生视图的 CICIDS/CESNET + P10 UNSW canonical**。两臂历史上同口径，并不能逻辑上保证改定义后的裁决不变，必须实际重算。

P6 ORA/ALL 有意把 injected flip mask 交给 oracle CDM，仅用于已明确标记的泄漏上界对照；主协议 FINAL 不允许这样做。UNSW ENT 是 pooled random permutation，不是严格分层、更不是实体隔离；保留历史实现，报告必须使用准确名称。

## 守卫覆盖

`tests/test_cross_runner_consistency.py` 固化标签永久排除、二元/多类及普通/分片加载器、PRX/ALL 禁恢复、共享 helper 路由、窗口/seeds、CL 不传注入率、MI 正控/连续高基数负控、CESNET 无二元攻击语义与 head→tail 假阳性计分。

`tests/test_p10_rerun.py` 核对 P10 cell 行计算与原 p5 实现完全一致，合并时非 UNSW 行原始字节保持，缺格/重复格拒绝，裁决变化后下游不可运行。真实输入的 30 窗口哈希验证另见 `p10_feature_audit.json`；源码路由断言不能替代运行时数值核验。

## 后续恢复注意

### 新发现：预处理拟合范围并非完全 train-only（待裁定，未修复）

`loaders.py` 普通路径在取训练/测试索引前调用 `_build_feature_frame`；UNSW 官方分片路径同样先对合并表构建特征，再取 `__graphcold_split` 对应索引。`_build_feature_frame` 在该表上判断数值列、剔除全缺失列、计算数值中位数填补并生成 `get_dummies` 词表。`StandardScaler` 才是仅 `fit_transform(X[train_idx])`，测试行只 transform。

因此“训练图不含测试节点”不等于“所有预处理严格归纳”。这是 **测试分布参与预处理的结构性风险**，不是已证实的第二个直接标签字段；当前未统计哪些列有缺失/测试独有类别，也未计算 train-only 转换后会改变多少输入或裁决。列级 MI 扫描无法排除这种风险。

本轮不擅自修复此路径，因为修复可能改变 CICIDS/CESNET 输入并要求三数据集重跑，与工作包“另两数据集冻结产物不动”的边界冲突。该问题须与四项局部判定变化一起交协调者裁定。若仅接受现有 label-only 修正，必须保留此限制；若要求完全 train-only 预处理，应另行冻结修正协议、授权更广重跑，不能把它悄悄并入本轮 UNSW 修复。

P9 已有 60 行 CICIDS/CESNET FINE 结果和旧契约保持不动。P10 修改了 loader/config；原 P9 契约应拒绝直接混用新旧代码哈希。待 P10 门禁签署后，需显式迁移契约、验证已有 60 行输入不变，再跑 UNSW 剩余 30 场景，不能删除契约以绕过守卫。
