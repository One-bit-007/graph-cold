# P9 最终协议预检：Tail-F1 跨产物口径不一致（暂停待裁定）

日期：2026-09-03。代码基线：`a61a12889a0743330dcdb3461b066d03b40952a4`。

**结论：暂不能把 p4 CL 与 p5 Graph-CoLD-soft 的既有 `tail_macro_f1` 原列直接并列并计算配对差值。**
这不是 FINE 性能结果，而是开始 90 场景前发现的评估指标冲突。
P9 第 1/4 条要求冻结旧结果，第 5 条要求遇到预期外问题暂停，因此本轮没有自行更换口径、
没有实现或训练新 FINE，也没有重算任何 Family A/B/C 裁决。

## 1. 两条调用链不等价

| 冻结来源 | 调用链 | Tail-F1 实际定义 |
|---|---|---|
| p4 CL | `p4_cl_graph_combo._scenario_rows` → `p2e_salvage._tail_macro_f1` | 先筛出真实标签属于尾类的测试行，再计算这些行上的尾类 macro-F1 |
| p5 Graph-CoLD-soft | `p5_factor_ablation._cell_row` → `p3_proxy_audit._tail_f1` | 在完整测试集计算每个类的 F1，再对尾类平均 |

代码定位：`src/paper/p4_cl_graph_combo.py:199`、`src/paper/p2e_salvage.py:772`、
`src/paper/p5_factor_ablation.py:215`、`src/paper/p3_proxy_audit.py:194`。

p4 的实现排除了真实头类样本，因而不计入“真实头类被预测为尾类”造成的尾类假阳性。
在尾类集合一致时，其尾类 F1 可高于完整测试集上的逐类 F1；这不是小数舍入误差。

还需在恢复执行时逐场景确认尾类集合：p4 对 IDS 排除 benign target，p5 `_tail_labels` 不显式排除。
噪声注入函数本身排除了 target，因此集合实现差别不必然意味着注入样本不同；
但不能在未核验时声称所有场景指标和注入完全一致。

## 2. 最小反例（仅说明定义，不是实验数据）

```python
import numpy as np
from src.paper import p2e_salvage as p2e, p3_proxy_audit as p3
y_true = np.array([0, 0, 0, 0, 1, 1])
y_pred = np.array([1, 1, 0, 0, 1, 1])
tail = np.array([1])
print(p2e._tail_macro_f1(y_true, y_pred, tail))  # 1.0
print(p3._tail_f1(y_true, y_pred, tail))        # 0.6666666666666666
```

相同真实标签、预测和尾类定义，两个实现产生不同结果。本例已实际执行。

## 3. 冻结 CESNET 行的只读核验

范围：CESNET，seed 0，三个率；**没有拟合任何分类器**。
使用 p4 正式 loader 和冻结分层采样恢复训练尾类清单，
然后从 p4 已保存的 `per_class_f1_json` 汇总完整测试集口径的尾类 F1。

尾类编码：`[1,3,4,6,8,9,10,12,13,16,19,23,24]`。

| 噪声率 | p4 原存储 Tail-F1 | 已保存逐类 F1 的尾类平均（p5 口径） | 原存储 − 同口径重汇总 |
|---|---:|---:|---:|
| 0.4 | 0.3356717935284638 | 0.3126332204154286 | +0.0230385731130352 |
| 0.6 | 0.3322556554794755 | 0.3261273581488500 | +0.0061282973306255 |
| 0.8 | 0.2068805877667495 | 0.1989461371574701 | +0.0079344506092794 |

这是局部预检，不代表其余 87 场景差异的大小，也不是重算后的 Family B。

复现方法（在仓库根执行；只读）：

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.paper import p4_cl_graph_combo as p4, p3_proxy_audit as p3
from src.paper import p2d_clean_rerun as p2d
ds = p4._load_primary_dataset('cesnet_tls_year22', Path('configs'))
idx = p2d._stratified_sample(ds.y_train, 10000, 0)
tail = p3._tail_labels(ds.y_train[idx])
df = pd.read_csv('results/p4_cl_graph_combo_full_family_b_complete.csv')
part = df[(df.dataset == 'cesnet_tls_year22') & (df.seed == 0)
          & (df.method == 'Confident-Learning')]
for row in part.itertuples():
    per_class = json.loads(row.per_class_f1_json)
    aligned = np.mean([v for k, v in per_class.items() if int(k) in tail])
    print(row.tail_flip_rate, row.tail_macro_f1, aligned)
```

## 4. 建议协调者批准的最小处理（尚未执行）

仅对 **P9 新增描述性比较**，统一采用“完整测试集逐类 F1 的尾类平均”：

1. p4 CL：从冻结 `per_class_f1_json` 生成新的 P9 派生列，不重训、不覆写原列；
   明示 `metric_reaggregated_for_p9=true`，并保留原存储值与来源哈希供追溯。
2. p5 Graph-CoLD-soft：继续使用 F-3 修正后的冻结主 CSV；
   与护栏逐类结果核对，但不把历史独立重跑的浮点差异自动覆写进主 CSV。
3. 新 FINE：使用同一完整测试集定义，保存逐类结果、尾类集合与训练/测试标识哈希。
4. 完整核对 90 个场景键、尾类集合、F-3 映射、屏蔽、划分、采样及逐类覆盖；
   在通过前不生成 FINE−CL / FINE−Graph 的“可比”差值。
5. bootstrap CI 仅为描述性；原 A/B/C 表与结论保持不动。

以上会把提示词要求的“CL 数值直接复用”细化为“复用逐类冻结结果后在 P9 派生重汇总”，
属于需要协调者明确批准的评估口径裁定，不应由执行者静默决定。

如果协调者要求对整个论文统一指标、重新审计 G2 或更新既有推断族，应另行授权工作包。
本次证据**既不证明 G2 翻转，也不证明统一口径后 G2 一定不变**。

## 5. 交付与保护状态

- 已完成：论文级保真度组件核验；调用链审计；最小反例；CESNET 三行只读核验。
- 已写入：本报告、`reports/p9_fine_fidelity.md`。
- 未开始：新 FINE 实现、registry 变更、新守卫测试、90 场景训练、正式 P9 CSV/表、bootstrap。
- 未修改：既有代码、冻结结果、A/B/C 裁决、正文、答复信、投稿状态；未 push。
- `reports/d11/d11_final_audit.json` 中 `submission_ready` 仍为 `false`。
- 不宣称本轮全仓测试通过：本轮仅运行上述只读预检，尚未执行新实现测试或全仓 pytest。

关键输入 SHA-256（本轮读后记录，提交前再次校验）：

| 文件 | SHA-256 |
|---|---|
| `results/p4_cl_graph_combo_full_family_b_complete.csv` | `7db2ba0112475987274d5358c0e067c0f9f7cd9a7189b7c7974a3d4a7cbe5180` |
| `results/p5_factor_ablation_full.csv` | `68862c7d112ee98a73db13b622c9e6a97eb4ada6cfce20f894764f8559ce79f6` |
| `results/p5_perclass_guard.csv` | `9cf62b99f4c08517a03d6ea84d784a1da1c40e23dcefb31180c6a995f9783723` |

**待裁定：是否批准上述“仅在 P9 新表派生统一 Tail-F1，原 A/B/C 完全不动”的方案后继续执行？**
