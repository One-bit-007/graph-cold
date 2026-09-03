# P9 停止报告：UNSW 最终协议仍把二元 label 留作输入特征

日期：2026-09-03。严重性：高，影响“标签代理已屏蔽”的科学解释。

**P9 未完成：任务 0 已完成；新 FINE 完成 CICIDS 30 + CESNET 30 场景；
UNSW 在训练前被新守卫拦截，0/30。不能将本轮标为完整基线验收。**

## 1. 实证核验（不训练模型）

- 通过冻结 `p4._load_primary_dataset('unsw_nb15', configs)` 取得真实最终协议数据。
- 学习目标列为 `attack_cat`，9 个类别，`Normal` 编码 0。
- 185 个输入特征中，零基索引 30 的名称为 `label`。
- 该列经过标准化后只取两个值：Normal 行为 `-1.8342316150665283`；
  所有非 Normal 行为 `0.5451874136924744`。

| 划分 | 行数 | Normal 行数 | 攻击行数 | `(label_feature > 0) == (target != Normal)` |
|---|---:|---:|---:|---:|
| train | 154644 | 35433 | 119211 | 100% |
| test | 68726 | 23438 | 45288 | 100% |

这不是普通数值特征偶然名为 label；其二值分组完全对应目标的正常/攻击父标签。
本次最初直接拿标准化值与 0/1 比较不匹配，随后按实际两值分组核验，才得到上述正确的 100% 一致率。
该字段并未给出每种攻击的完整类别，但确实暴露干净目标的正常/攻击属性。
在尾类攻击标签被合成翻转为 Normal 时，输入中的这项属性不会随注入翻转，
因此也可能提供辨识被翻转样本的捷径。

## 2. 调用链原因

- `configs/datasets.yaml`：UNSW `label_col: attack_cat`，原始 `drop_cols: [id]`。
- tier1 只包含 `is_sm_ips_ports`；tier2 为八个 ct_* 字段，没有 `label`。
- `src/data/loaders.py` 的固定划分路径用
  `_build_feature_frame(work.drop(columns=[label_col]))` 构造特征，移除了 `attack_cat`，
  但没有移除另一标签列 `label`。
- 旧 p4 只检查 tier 列是否漏过；P9 新 `assert_non_oracle_schema` 另查标签/oracle 字段，
  报错：`Attempted/oracle fields or classes present: ['label']`。

这不是 FINE 新实现加入的字段，而是冻结输入构造中既有的遗漏。
既有测试全绿不等于真实数据端到端无代理字段：本轮全仓测试 323 passed、3 skipped，
实际运行守卫仍拦截了这项真实数据问题。

## 3. 影响与不能推断的事项

1. **本次 60 个 FINE 场景**：只涉及 CICIDS/CESNET，其输入、输出和合同原样保留；
   此发现本身不否定这些场景，但本轮没有对不完整网格生成最终并列表或完整基线结论。
2. **UNSW 既有 p4/p5**：共用上述数据配置与 loader；冻结窗口已核对。
   它们的实验协议存在该标签属性泄漏风险。不得以所有方法都能见到该列，推断相对差值完全不受影响。
3. **Family A/B/C**：未修改任何历史数字或裁决；但“冻结不动”不意味着它们的 UNSW 部分已通过无泄漏科学验收。
   删除该列后裁决是否改变，需要新的、另行授权的实验，不在本 P9 中擅自决定。
4. **canonical G2=null**：任务 0 的重汇总与计算仍然正确，回答的是“同一冻结输入更换指标定义后”的敏感性；
   不是“排除 label 泄漏后 G2 仍 null”的证据。
5. **主表**：已有的跨指标定义问题需要协调者修正；新增的 UNSW 数据问题不能仅靠换 Tail-F1 或改措辞解决。
6. **其他 UNSW 报告**：凡使用同一原始文件/多分类 loader 且未主动丢弃 label 的路径均需清点；
   本轮不宣称已逐一证明每个历史产物的影响大小。

## 4. 已执行的停止措施

- 不放宽或绕过标签字段守卫；未启动 UNSW FINE。
- 不修改原 loader、dataset config、proxy tiers；不私自改变冻结基准输入。
- 不覆盖/删除部分结果，不重跑既有 CL 或 Graph，不改 A/B/C 推断、w5 生成器、正文或答复信。
- `results/p9_fine_final_protocol.partial.csv` 保留 60 行；
  `reports/p9_fine_run_contract.json` 保留原实现/输入指纹，防止后续混入不同协议结果。
- 不创建完整的 `results/p9_fine_final_protocol.csv` 或 `tables/table_p9_fine_final.csv`。
- `submission_ready=false`；未推送或发布。

## 5. 建议协调者另行裁定

建议开独立的 **UNSW label 泄漏修复与重跑工作包**：

1. 从特征及图输入中去掉替代标签/父标签字段；大小写别名一并测试。
2. 定位所有受影响的 UNSW 冻结场景及其主表/推断依赖，保留历史快照和协议偏离披露。
3. 在一致新协议下重跑必要的 UNSW CL、Graph 及消融，重算完整受影响假设族，交协调者裁决。
4. 再补 UNSW FINE。不得仅给 FINE 去掉 label，却继续引用含 label 的旧 CL/Graph 做“同协议比较”。
5. 是否复用本次 CICIDS/CESNET 的 60 行，应由新工作包明确授权，并逐项验证配置/数据未改变；
   不能简单删除运行契约绕过 stale-cache 拒绝。

复现证据：`python -m src.paper.p9_unsw_label_audit`，
机器可读结果 `reports/p9_unsw_label_blocker.json`（含逐类交叉计数、实际 drop 列和文件哈希）。
