# P9 FINE 最终协议执行状态：未完成，暂停交协调者

日期：2026-09-03。**completed=false；60/90；submission_ready=false。**

## 已完成

- Task 0：canonical helper、630 行重汇总、Table 2 六个数据集/方法行的指标变化、
  原 Family B 的 canonical 敏感性检查；G2 在原输入下仍为 null，18 项均无 Holm 显著正增量。
- 新增 `FINE detector (tabular adaptation)`：未中心化 Gram、平方分数、逐类 GMM；
  静态表格表示与分类头如实披露；不接受真实/估计噪声率或干净验证标签。
- CICIDS 30 场景、CESNET 30 场景已训练并保存逐类 F1、混淆矩阵、输入/输出哈希和回退信息。
- 全仓测试：323 passed、3 skipped、10 warnings；警告来自既有 MORSE 收敛和统计精度测试。
  这不是 90 场景真实数据门禁通过的证明。
- 对 60 行保存的混淆矩阵独立重建 Macro-F1 和 canonical Tail-F1，最大误差
  `1.1102230246251565e-16`；场景键无重复，训练标签哈希与 metadata 一致。
- 共 45 次类级回退：25 次小类、4 次 GMM 均值不可区分、2 次后验选择为空、14 次未收敛；
  均在每行 metadata 中公开记录，未通过更换参数重跑来隐藏这些情况。

## 阻塞

UNSW 进入模型训练前，新守卫发现 `label` 父标签列仍在正式特征中。
该列在全部 train/test 数据上 100% 对应正常/攻击属性。
详见 [UNSW label 阻塞报告](p9_unsw_label_blocker.md)。

因此没有生成完整 FINE/CL/Graph 对照表、跨三个数据集的配对 bootstrap 或总体排名结论。
本轮不能说“已补齐最终协议强基线”或“A7 已关闭”；仅完成实现与部分运行。

## 数据与执行入口

- canonical 逐行数据：`results/p9_canonical_tail_f1.csv`。
- 部分 FINE 数据：`results/p9_fine_final_protocol.partial.csv`（60 行，不是完整正式产物）。
- 部分数据每行含 `per_class_f1_json`、`confusion_matrix_json`、`metadata_json`；
  小类/退化/GMM 失败均有回退原因，不静默剔除失败场景。
- 运行契约：`reports/p9_fine_run_contract.json`。
- 运行器：`python -m src.paper.p9_fine_final_protocol`；当前重启仍会在 UNSW 守卫处停止。
- canonical 报告：`reports/p9_tail_metric_unification.md`；保真度：`reports/p9_fine_fidelity.md`。

## 交接边界

已核验 573 个受保护文件保持原哈希。原 A/B/C、泄漏分解、F-3 修正结果、生成器、正文、答复信均未动。
但冻结 UNSW 输入存在新发现的标签属性泄漏；canonical G2=null 不能替代删除泄漏字段后的复验。

须先由协调者批准独立 UNSW 修复/重跑范围，再补完剩余 30 场景；不得仅修改 FINE 输入而引用旧 CL/Graph。
