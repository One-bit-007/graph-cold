# P9：FINE 保真度核验（实现前审计）

日期：2026-09-03。审计基线：`a61a12889a0743330dcdb3461b066d03b40952a4`。

**状态：原论文与当前实现的组件核验已完成；新实现、registry 登记和 90 场景补跑尚未开始。**
最终协议预检发现 p4/p5 Tail-F1 定义不一致，已按 P9 的“发现预期外问题即停下报告”规则暂停。
本文件不是已完成实现的认证，也不能作为 A7 已闭合的证明。
详见 [指标口径阻塞报告](p9_fine_metric_blocker.md)。

## 1. 控制来源与核验边界

- 原论文：Taehyeon Kim, Jongwoo Ko, Sangwook Cho, Jinhwan Choi, Se-Young Yun,
  *FINE Samples for Learning with Noisy Labels*, NeurIPS 2021。
  [会议官方论文 PDF](https://proceedings.neurips.cc/paper/2021/file/ca91c5464e73d3066825362c3093a45f-Paper.pdf)。
  对照位置：Algorithm 1 / Definition 2；§4.1 的后验阈值；§4.2.1 的训练循环。
- 当前实现：`src/baselines/fine_style.py`；登记：`src/baselines/registry.py`；
  旧实验入口：`src/paper/p7_baseline_expansion.py`。
- [作者官方仓库](https://github.com/Kthyeon/FINE_official)已确认，远程 HEAD 为
  `d5ab341dec5356d5f9a2a8ebc3ba8e846188af61`，默认分支 `master`。
  本轮没有完成官方代码逐行对照或数值一致性测试：GitHub API 读取遇到限流。
  以下原始算法判断以会议论文为准，不声称与官方代码数值等价。

## 2. 逐组件对照

| 组件 | 原论文依据 | 当前本地实现 | P9 应如何处理（尚未实现） |
|---|---|---|---|
| 表示 | 特征提取器输出，原实验使用深层表示 | 训练集 StandardScaler + TruncatedSVD，随后行归一化；可接受外部表示 | 可保留表格表示，但明确属于适配，不等同于原深度特征 |
| 类分组 | 按观测标签分组 | 按 noisy training label 分组 | 保留；不得改用干净标签构造组 |
| 特征方向 | 类内未中心化 Gram 矩阵的最大特征值对应向量 | 对类内表示再中心化、归一化，然后求主方向 | 移除类内中心化；在所声明的表示上使用未中心化 Gram |
| 对齐分数 | 平方内积 | 绝对值内积 | 使用平方内积；通过手工矩阵样例核对 |
| 筛选 | 逐类两成分 GMM，高均值成分视为 clean，按后验筛选 | 按噪声率指定配额，保留最高分的固定比例 | 改为逐类 GMM 后验；论文实验阈值为 0.5 |
| 噪声率 | 检测器不需要噪声率 | 无输入时估计；也保留显式输入接口 | 新类移除真实与估计噪声率接口及依赖 |
| 小类 | Algorithm 1 未提供本地表格数据稀疏类回退规范 | 小于 20 个样本全部保留，并有类别覆盖修复 | 明定最小样本、退化分数、拟合失败处理；每次回退记录原因与数量；不静默保留全部 |
| 分类器及循环 | 原样本选择应用在 warmup 后逐轮筛选并训练网络 | 一次筛选后训练 ExtraTrees | 允许固定表格分类器，但须披露单次检测、无原深度训练循环；不称完整原始 FINE |
| 评估入口 | 本文须遵守冻结协议，非原论文网络流量协议 | P7 复用遗留 iid 窗口 | 新入口复用 p4/p5 最强可用划分、屏蔽和 p3 噪声映射；先解决指标不兼容 |

表示的全局标准化，与算法中禁止的“按观测类再中心化”是不同操作；
不得因保留前者而误称后者已经符合 Algorithm 1。

## 3. 命名及保真度声明

新类必须另行命名，旧类和 P7 产物保留供历史复现：

- `method`: `FINE detector (tabular adaptation)`。
- 目标 `faithfulness_level`（只有核心机制实现并经测试后才可登记）：
  `Gram-eigenvector + per-class GMM selection as in Kim et al.; representation is a tabular adaptation, not the original deep features`。
- 不声称完整原始实现，不把表格表示适配的结果归因于原论文全部训练体系。
- `include_in_formal_results=True` 的前置是 F-5 守卫通过；本轮未修改登记。

## 4. 待解除阻塞后完成的验证

1. 未中心化 Gram / 平方分数的数值样例、特征向量符号不变性与逐类 GMM 选择测试。
2. 检测器只接收训练特征及观测标签；拒绝真实噪声率、干净验证标签和 oracle 元数据。
3. 表示只在训练集拟合；测试特征和标签不参与筛选。
4. 小类和 GMM 退化路径有显式记录；实现配置在基准评估前固定，不按 FINE 排名调参。
5. 三数据集场景、F-3 目标、输入哈希、逐类指标及冻结引用的配对检查。
6. 描述性配对差值与 bootstrap CI，不新增推断族，不重算 A/B/C。

## 5. 当前可说与不可说

可说：旧 FINE-style 不是本工作包要求的核心机制对齐版本，差异已定位。

不可说：FINE 补跑已完成、保真实现测试已通过、A7 已降低、FINE 优于或不及 CL。
这些均需新实现与可比较的实际数据。当前 `submission_ready=false`，稿件和答复信未改。

## 6. 2026-09-03 批准恢复后的补充（保留上文历史暂停记录）

协调者在 `b52ca17` 批准 canonical 指标派生，并扩展为主表影响和 G2 敏感性核验。
任务 0 完成且未触发裁决停止条件后，才启动 FINE 训练。

本轮已读取官方固定提交 `d5ab341dec5356d5f9a2a8ebc3ba8e846188af61` 的
[svd_classifier.py](https://github.com/Kthyeon/FINE_official/blob/d5ab341dec5356d5f9a2a8ebc3ba8e846188af61/dynamic_selection/selection/svd_classifier.py)
与 [gmm.py](https://github.com/Kthyeon/FINE_official/blob/d5ab341dec5356d5f9a2a8ebc3ba8e846188af61/dynamic_selection/selection/gmm.py)。
官方 SVD 路径不做类内中心化；`get_score` 使用绝对内积，可对特征归一化；
`fit_mixture` 为逐类两成分 GMM，取高均值成分，后验严格大于阈值时保留。
它同时提供 KMeans 等可选路径。因此“符合论文 Algorithm 1”不能写成“所有官方代码路径逐项等价”。
P9 依照工作包采用论文规定的平方分数与 GMM；平方虽保持分数排序，却不保证 GMM 后验与绝对值版本相同。

### 6.1 基准训练前固定的适配配置

- 表示：仅训练特征拟合 StandardScaler；TruncatedSVD 上限 32 维（受矩阵维数限制）；行 L2 归一化。
  不按观测类中心化；Gram 在这个明确披露的静态表格表示上计算。
- 类内 Gram：`Z_c.T @ Z_c`，最大特征值对应向量，分数为平方内积。
- GMM：2 成分、full covariance、`tol=1e-6`、`max_iter=100`、`reg_covar=1e-6`、
  `n_init=1`、kmeans 初始化、随机种子为场景 seed；clean 为高均值成分，后验 `>0.5`。
- 小类：少于 4 个观测样本；退化分数：极差不超过 `1e-12`；
  或数值异常、未收敛、成分均值无法区分、后验无法给出任何保留样本时，
  **显式回退为保留该类全部样本**，记录原因、样本数、保留数和警告。
  这是保守的工程适配，不来自原论文定理；不得隐藏回退率。
- 分类头：保留样本的原始表格特征训练 ExtraTrees（24 棵，class_weight=balanced，
  seed+7，单线程以固定运行顺序），与 p5 硬删除分支的分类器家族及容量一致。
  表示仅用于筛选；没有深度 warmup、逐 epoch 重估或原论文的端到端表示学习。
- 不接受真实/估计噪声率、干净训练标签、验证标签或测试标签；不调用噪声率估计器。
- 这些参数在 FINE 基准评估前写定；不因 FINE 相对 CL 的结果而改动。

此节是实施配置，不是已通过实现测试的声明。运行完成状态以本报告后续验收节为准。

## 7. 本轮最终验收状态（优先于前述历史状态）

- 新实现位于 `src/baselines/fine_tabular.py`，旧 FINE/P7 未改。
- 核心机制、no-oracle/no-rate、canonical 计算及相关旧守卫 63 项通过；
  registry 登记启用后再次核验，最终全仓 323 passed、3 skipped。
- 已实际运行 CICIDS/CESNET 各 30 场景；UNSW 被标签字段守卫拦截，P9 仍未完成。
- 实现级 F-5 合规不等于基准数据无泄漏；不得把 registry 的实现登记状态当作完整投稿验收。
- 官方平方/绝对值路径差异、静态表格表示、单次检测与 GMM 回退仍是明确的适配边界。
- 不给出完整网格 FINE 排名、优于/不及 CL 的总体声明或 A7 关闭声明。

最新阻塞：[UNSW label 特征泄漏](p9_unsw_label_blocker.md)。
