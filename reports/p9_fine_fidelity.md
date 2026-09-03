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
