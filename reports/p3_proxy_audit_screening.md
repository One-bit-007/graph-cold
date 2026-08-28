# P3 / P0-2 代理泄漏审计粗筛报告（Phase 1）

> 筛选协议：3 seeds × 1 噪声率（tail-asymmetric 0.6）× {CoLD, ablation_hard, Graph-CoLD-soft}，训练/测试窗口 10,000/5,000（与冻结协议 F-1 一致）。
> 运行器：`src/paper/p3_proxy_audit.py`；明细：`results/p3_proxy_audit_<dataset>_<condition>.csv`、`reports/p3_proxy_audit_<dataset>_<condition>.json`；回归测试：`tests/test_proxy_field_regression.py`。当前共有 18 个可执行条件结果、5 个明确不支持条件；CICIDS 的 C1b 因 tier2 为空而不重复运行。
>
> **协议修正（2026-08-27）**：早期 CESNET 粗筛错误沿用类别 0 作为翻转目标，违反 F-3 的服务→高频服务语义；相关结果已全部作废并以冻结目标 `microsoft-diagnostic`（编码 11）重跑。C4 在 CESNET/UNSW 上的类别留出仅为探索性未见类压力测试，不属于冻结的 S3；CESNET 已补跑严格的 C6=S1∩S2，UNSW 因缺失时间和实体列无法执行。

## 1. 条件总表（macro F1 为 3 seeds 均值）

| 数据集 | 条件 | 划分 | CoLD | hard | soft | tail Δ(soft−hard) | 精确特征碰撞 | 近似特征碰撞 |
|---|---|---|---|---|---|---|---|---|
| CICIDS | C0 S0 基线 | stratified | 0.460 | 0.653 | 0.684 | +0.061 | 0.9% | **82.7%** |
| CICIDS | C1 屏蔽端口 | stratified | 0.430 | 0.589 | 0.652 | **+0.104** | 0.6% | 80.0% |
| CICIDS | C2 时间隔离 | temporal_by_source_file（Mon–Thu→Fri，部分周五渗入 train） | 0.272 | 0.254 | 0.454 | n/a（tail 无交集） | 0.2% | 49.6% |
| CICIDS | C3 实体隔离 | **不支持**（本地文件已剥离 IP，见 §4） | — | — | — | — | — | — |
| CICIDS | C4 周五留出 | block_holdout_by_source_file | 0.045 | 0.049 | 0.043 | n/a | 0.0% | 30.3% |
| CICIDS | C5 传导式对照 | stratified | — | — | — | — | — | — |
| CESNET | C0 S0 基线 | stratified | 0.661 | 0.689 | 0.699 | +0.016 | 0.01% | **87.3%** |
| CESNET | C1 屏蔽 CATEGORY+DST_PORT | stratified | 0.589 | 0.611 | 0.641 | +0.039 | 0.01% | 85.9% |
| CESNET | C1b +屏蔽 DST_ASN | stratified | 0.544 | 0.559 | 0.602 | **+0.057** | 0.01% | 89.3% |
| CESNET | C2 时间隔离 | temporal（严格时间边界；窗口仅 ~7 小时） | 0.678 | 0.704 | 0.714 | +0.018 | 0.01% | 88.2% |
| CESNET | C3 实体隔离 | entity（SRC_IP 组） | 0.632 | 0.648 | 0.678 | +0.040 | 0.0% | **46.3%** |
| CESNET | C4 类别留出（探索性） | class_holdout；非冻结 S3 | 0.636 | 0.658 | 0.672 | +0.025 | 0.01% | 85.6% |
| CESNET | C5 传导式对照 | stratified | — | — | — | — | — | — |
| CESNET | C6 时间∩实体隔离 | temporal_entity；时间严格先后且 SRC_IP 零交集 | 0.583 | 0.638 | 0.657 | +0.039 | 0.0% | 69.8% |
| UNSW | C0 S0 基线 | fixed partition | 0.426 | 0.441 | 0.452 | +0.019 | 3.8% | 23.7% |
| UNSW | C1 屏蔽 is_sm_ips_ports | fixed partition | 0.432 | 0.442 | 0.452 | +0.018 | 3.8% | 23.7% |
| UNSW | C1b +屏蔽 ct_* | fixed partition | 0.443 | 0.454 | 0.481 | +0.047 | **38.4%** | **58.0%** |
| UNSW | C2/C3 | **不支持**（固定划分无时间戳/实体列） | — | — | — | — | — | — |
| UNSW | C4 攻击类留出（探索性） | class_holdout；非冻结 S3 | 0.427 | 0.439 | 0.452 | +0.026 | 3.9% | 23.8% |
| UNSW | C5 传导式对照 | fixed partition | — | — | — | — | — | — |

## 2. 四项关键发现

### F-A CESNET 存在直接标签代理泄漏（新发现，评审 A 问题 2 之外增量）
未屏蔽配置下，CESNET 特征包含 **`CATEGORY_*` 12 个 one-hot 列**（CATEGORY 是标签 APP 的父类别，来自同一标注管线）与 `DST_PORT`。按修正后的 F-3 噪声映射，屏蔽后 soft macro F1 0.699→0.641（−0.058），说明此前 CESNET 绝对数字受到标签代理影响。回归测试已锁定（`test_cesnet_category_and_dst_port_blocked_under_masked_policy`）。

### F-B 预处理后特征碰撞普遍且严重（评审 A 要求项）
iid 基线下，测试行在标准化特征空间中与训练行余弦相似度 ≥0.999 的比例为：CESNET 87.3%、CICIDS 82.7%、UNSW 23.7%。这衡量的是**模型输入表示碰撞**，不能直接等同原始流重复。UNSW 屏蔽 ct_* 后，近似碰撞升至 58.0%、精确特征向量碰撞升至 38.4%，主要原因是删列后表示维度下降；它提示可记忆性风险，但不能据此断言源数据重复率上升。CESNET 实体隔离将近似碰撞降至 46.3%。

### F-C CICIDS 场景隔离下的崩塌主要是「未见类效应」，不是图机制崩塌
- 严格周五留出（C4）：87.3% 测试行属于训练中未出现的类（DDoS/PortScan/Bot），整体 macro F1 ≈ 0.04（三种方法一致）；
- **已见类 macro F1 = 0.666**；它与 iid 的 0.684 类集合和权重口径不同，只能作为“崩塌由未见类主导”的描述性证据，不能作正式非劣效比较；
- 时间序 80/20（C2，部分周五渗入训练）：soft 0.454 vs hard 0.254，soft 显著更抗场景偏移。
结论：CICIDS 的图方法增益在场景隔离下**没有证据表明完全消失**，但 iid 绝对数字被近似重复与场景重叠抬高，必须在修改稿中以 group-disjoint 数字为主口径。

### F-D 当前主管线是归纳式；单一传导式对照未显示优势（评审 A 问题 2 末项）
现有主管线由训练集拟合分类器并逐样本预测测试集，代码路径不使用测试批图。作为压力测试，train+test 联合 kNN 的 LabelPropagation 在三个数据集上均低于归纳式 ExtraTrees（CICIDS 0.603 vs 0.684、CESNET 约 0.65 vs 0.699、UNSW 0.381 vs 0.452）。这只能说明**该 LabelPropagation 实现**没有造成抬高，不能推广为“所有传导式构图都安全”。

### F-E 软保留机制发现在全部可测条件下存活
tail Δ(soft−hard) 在全部可测条件下为正（约 +0.016 ~ +0.104），且屏蔽代理字段后增大（CICIDS +0.061→+0.104；CESNET +0.016→+0.057；UNSW +0.019→+0.047）。这是 3-seed 粗筛方向性证据，支持把这些配置送入全量实验，但尚不能写成显著性结论。

## 3. G1 判据初判（粗筛证据，待全量确认）

**初判：增益「存活但需降格陈述」——倾向「图诊断的增量价值」与「审计驱动边界研究」的混合定位。**

- 支持存活：F-C（seen-class 描述性稳定）、F-E（tail delta 全条件为正且屏蔽后增大）、F-D（主管线代码路径为归纳式）；
- 支持收缩：F-A（CESNET 绝对数字含标签代理水分，−0.055）、F-B（iid 数字受近似重复抬高，须改以 group-disjoint 口径为主）、CICIDS C2 的绝对水平大幅下降（0.68→0.45）。

**进入 Phase 2 的建议**：P0-3/P0-4 全量实验应以 C1（屏蔽后）+ C2/C3（group-disjoint）为主协议配置，C0 仅作历史对照。

## 4. 数据限制与后续动作

| 限制 | 影响 | 后续动作 |
|---|---|---|
| 本地 CICIDS CSV 已剥离 IP/时间戳（79 列） | C3 实体隔离不可做；S1/S3 仅能按文件星期 | 全量阶段获取原始 MachineLearningCVE 文件补 S2；或如实声明限制 |
| UNSW 固定划分无时间戳/实体列 | C2/C3 不可做 | 声明限制，或用完整 UNSW-NB15 原始文件替换 |
| CESNET 采样窗仅 ~7 小时（2022-01-01 单日） | C6 虽实现严格时间∩实体隔离，但外部效度有限 | F-1 规模验证时按周扩展窗口 |
| CICIDS C2/C4 的 tail Δ 不可测（tail 类与测试类无交集） | G1 尾部判据缺失 | 全量阶段报告 unseen-class 分解与 seen-class 指标（已实现 `macro_f1_seen_classes`） |

## 5. 复现

```bash
# 单条件（示例）
python -m src.paper.p3_proxy_audit --dataset cicids2017 --condition C2_S1_temporal
# 回归测试
python -m pytest tests/test_proxy_field_regression.py -q
```
