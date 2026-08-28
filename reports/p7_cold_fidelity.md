# P7 / P1-1：CoLD 保真度审计与协议差距归因

日期：2026-08-28

裁决路径：**(b) 公开协议差异与方法身份，不声称完成官方复现**。

## 0. 结论先行

原工作包提出的“CoLD 复现值 0.534 与其报告 0.688 的差距”并不是一组可比较的 CoLD 结果：

- `0.534492` 是本项目 P2f 表中历史标签为 `CoLD` 的行，实际实现是 **Graph-CoLD 邻域 CDM + 手工硬阈值 + ExtraTrees**；本文后续必须改称 **CoLD-aligned proxy**。
- `0.688497` 是同一张 P2f 表中的内部方法 `ablation_hard`，不是 CoLD 原论文数字。
- NDSS 2026 官方论文 Table IV 在其九类 refined CICIDS-2017 上报告的 CoLD Macro-F1 为 0.9971（无噪声）、0.9898（40% symmetric）、0.9832（40% asymmetric）等，数值约在 0.94--1.00；论文中不存在 0.688 这个 CoLD 对齐目标。

因此，差距的第一来源是**方法身份混淆**，不是“复现低了 15.4 个点”。现有 P2f `CoLD` 行不可继续作为官方 CoLD 复现或公平实现来表述。

## 1. 权威来源与版本固定

- 官方论文：Yang et al., *CoLD: Collaborative Label Denoising Framework for Network Intrusion Detection*, NDSS 2026, DOI `10.14722/ndss.2026.231950`：<https://www.ndss-symposium.org/wp-content/uploads/2026-s1950-paper.pdf>
- 官方代码：<https://github.com/kalendsyang/CoLD>
- 本次核验 commit：`f0b940affe521662667562888d29468279d420a8`（2026-06-04，仓库首个 commit）。

本项目 `src/models/cold_baseline.py` 早于官方代码发布，旧注释“no official code exists / faithful”现已撤回。更关键的是，P2f 正式表并未调用该模块，而是调用 `src/experiments/d5.py::_weights_for_cold` 的轻量代理路径。

## 2. 0.534 与 0.688 的可核对来源

口径：CICIDS-2017，10 seeds，tail-asymmetric 0.4/0.6/0.8，共 30 行；来源 `results/p2f_tail_powered.csv` 和 `tables/table_p2f_summary.csv`。

| P2f 方法身份 | rate 0.4 | rate 0.6 | rate 0.8 | 三档聚合 Macro-F1 |
|---|---:|---:|---:|---:|
| 历史 `CoLD`，应改称 `CoLD-aligned proxy` | 0.585134 | 0.519301 | 0.499043 | **0.534492** |
| `ablation_hard` | 0.785541 | 0.696014 | 0.583937 | **0.688497** |

`0.688497 - 0.534492 = 0.154005` 只说明两个**内部实现**在冻结尾类噪声协议下不同，不能用来评估官方 CoLD 的复现误差。

## 3. 官方 CoLD 与 P2f 代理的逐项差异

| 维度 | NDSS 论文 / 官方代码 | 当前 P2f 历史 `CoLD` 行 | 归因结论 |
|---|---|---|---|
| 数据 | refined CICIDS-2017，9 类；主导类下采样、少于 1000 的类删除 | postfilter11，11 类；10k/5k 审计窗 | 标签空间、类比例和样本支持均不同 |
| 划分 | 随机 8:2；官方结果重复 3 次 | 从项目既有 train/test 池分层抽样，10 seeds | 测试分布与方差口径不同 |
| 噪声 | symmetric：所有类参与；asymmetric：恶意类转 benign；10%--80% | 仅冻结 tail 类转 benign；40%/60%/80% | P2f 是更集中、更伤尾类的压力测试 |
| 表示学习 | 4 个重叠特征子集；mask 0.3；自监督局部对齐 + 重构；官方代码默认预训练 15 epochs | 不执行官方多子集编码器 | 核心表示机制缺失，不能称 faithful reproduction |
| 诊断 | 各子集和聚合表示经分类/聚类后做一致性净化；论文描述 GMM | 基于 kNN 图的 observed-label 邻域 CDM，混入频率/异常 evidence | 诊断变量与公式不是同一对象 |
| 删除规则 | 多表示一致性 / CDM，论文阈值 epsilon=0 | `cdm <= 0.5`，再删除保留集 evidence 最高 1% | 代理阈值为项目自定义 |
| 下游分类器 | 官方代码默认 XGBoost | ExtraTrees，24 trees，class-balanced | 模型容量与决策边界不同 |
| 汇总 | 每个噪声率单独报告 Macro-F1 | 0.534/0.688 是三档噪声率再聚合 | 即使方法相同也不能直接对数值 |

## 4. 为什么本轮不宣称“原文设置复现”

官方代码已经发布，但精确复现论文 Table IV 仍缺少必要数据契约：

1. 官方仓库只规定 `X_train.npy/y_train.npy/X_test.npy/y_test.npy` 接口，没有发布论文使用的 refined CICIDS 数组、split manifest 或哈希；
2. 论文说明删除少于 1000 的类并下采样主导类，但没有给出精确数组和抽样随机状态；
3. 发布代码默认 `feature_reordering=false`、`cluster_method=kmeans`，而论文方法与 Algorithm 1 描述 Feature Reordering 和 GMM，存在 paper--code 默认值歧义；
4. P2f 的目的本来就是 11 类、尾类定向噪声、10 seeds 的审计压力测试，不应为了追近论文数字而改回九类随机 8:2 协议。

在这些条件下报告“官方复现失败”或拿 0.688 当目标都不严谨。P1-1 因而采用路径 (b)：公开差异、冻结未来对齐协议，并把现有行降格为 proxy。

## 5. 未来 official-source aligned 协议（预注册，不在本轮伪造结果）

若导师决定补跑官方源码适配，必须新增结果文件，不覆盖 P2f：

| 项 | 冻结值 |
|---|---|
| 上游源码 | `kalendsyang/CoLD@f0b940affe521662667562888d29468279d420a8` |
| 数据/划分 | 复用最终泄漏屏蔽后的 P2f 10k/5k train/test manifest；逐数据集导出 `.npy`，不重新随机 8:2 |
| 噪声 | P2f tail-asymmetric 0.4/0.6/0.8；同一噪声 mask 逐方法配对 |
| seeds | 0--9；编码器 seed 与评估 seed 均由场景 seed 派生并记录 |
| 方法参数 | 4 subsets，overlap 0.75，mask 0.3，15 pretrain epochs，batch 32，Adam lr 0.001，mean aggregation |
| 歧义裁定 | 按论文方法启用 Feature Reordering 和 GMM；另以官方 YAML 的 no-reorder + k-means 作为敏感性臂，不据结果反向选择 |
| 下游 | XGBoost，官方 commit 默认参数；不使用干净验证集调参 |
| F-5 | 干净训练标签、真实 flip mask、真实噪声率不得进入表示、净化、阈值或模型选择；官方代码中的 `y_clean` 仅用于噪声检测报告，正式适配器应禁用该报告路径 |
| 输出身份 | `Official-CoLD-aligned`，与历史 `CoLD-aligned proxy` 并列；旧结果只读 |

该协议回答“官方算法结构在我们的最终数据协议下表现如何”，但仍不等于复现论文 Table IV，因为数据和噪声协议不同。

## 6. 对修改稿与答复信的强制影响

1. 把表格和正文中的现有 `CoLD` 改为 `CoLD-aligned proxy`，直到 official-source aligned 网格完成；
2. 删除“aligned CoLD baseline uses the same encoder/training budget”一类表述——现有 P2f 代理没有使用官方 CoLD 编码器；
3. 不再把 Graph-CoLD 相对该代理的增量写成“优于 CoLD 原方法”，只能写成“相对内部硬删除代理的协议内差异”；
4. 0.688 明确标为 `ablation_hard`，不得进入 CoLD 保真度叙事；
5. 若不补跑官方源码，答复信应主动承认：原稿的 CoLD 命名过强，修改稿已降格，并将最强外部锚点转为 F-5 合规的 Confident Learning / FINE。

## 7. 复现命令

核对本项目两个数字：

```powershell
@'
import pandas as pd
df = pd.read_csv("results/p2f_tail_powered.csv")
q = df[(df.reported_as == "CICIDS-2017") & df.method.isin(["CoLD", "ablation_hard"])]
print(q.groupby("method").macro_f1.mean())
print(q.groupby(["method", "tail_flip_rate"]).macro_f1.mean())
'@ | .\.venv\Scripts\python.exe -
```

固定官方源码版本并核查默认配置：

```powershell
git clone https://github.com/kalendsyang/CoLD.git external-cold-audit
git -C external-cold-audit checkout f0b940affe521662667562888d29468279d420a8
Get-Content external-cold-audit\config\default.yaml
```

守卫测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_p7_cold_fidelity.py -q
```

## 8. P1-1 验收状态

**完成的是保真度审计与路径 (b) 协议差异归因；未完成、也未声称完成官方 Table IV 复现。** 这一边界是本交付的核心结论。
