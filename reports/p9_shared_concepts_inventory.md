# P9：跨运行器同名概念的局部清点

日期：2026-09-03；范围为本次最终协议比较涉及的 P3–P9 及其共享函数。
这是有限调用链清点，不是全仓“再无协议差异”的保证；未改写历史运行器或冻结产物。

| 概念 | 现有位置/用法 | 本轮处理与边界 |
|---|---|---|
| Tail-F1（真实尾类行子集） | `p2e_salvage._tail_macro_f1`；p4、p5_external_baselines、p6_scale_validation 使用 | 保留历史语义；p4 在 P9 中由已存全集逐类数据派生 canonical。其他历史实验不能直接宣称已统一 |
| Tail-F1（完整测试集逐类尾均值） | `p3_proxy_audit._tail_f1`；p5_factor_ablation、p6_leakage_ablation、p7_natural_noise 使用 | P9 以此为 canonical，新增公共 `src.metrics_canonical.canonical_tail_f1_from_perclass` |
| Tail 类集合 | `p3._tail_labels` 按计数中位数；`p3._tail_noise_labels` 对 IDS 排除 benign target | 本轮恢复的 30 个主协议窗口中两集合逐项一致；不保证未来窗口或历史非主协议窗口也一致 |
| 自然噪声指标参考 | p7 同时输出 corrected-reference 与 original-reference；前者排除无有效修正标签的测试行 | 两臂内部用相同定义，但总体与合成噪声主表不同；不可忽略参考标签与有效测试行的区别 |
| 噪声目标 | p3/p4/p5/p6 主注入路径调用 `p3._noise_target` | 主窗口 90 组 p4/p5 注入结果逐位一致；CESNET 目标 11。自然噪声 p7 不做合成注入 |
| `benign_class or 0` 字样 | p3/p6 仍用于 sample metadata，实际注入另调用 `_noise_target`；p7 限 CICIDS | 不能仅凭字符串命中判定 F-3 回潮；已核查相邻注入调用。不改历史早期 p2 路径 |
| 窗口和 seed | p4/p5 分别声明 10k/5k、0–9、0.4/0.6/0.8 | P9 直接调用 p5 场景构造，同时与任务 0 的输入/噪声哈希核对 |
| 分类头 | p5 soft 是加权 ExtraTrees；hard 是筛选后 balanced ExtraTrees；CL 有自己的 OOF 检测器 | 新 FINE 保留表格适配声明，采用同家族/容量分类头，不宣称与深度原版端到端对等 |
| FINE 名称 | 旧 `FINE`/`FINE-style` 是中心化+绝对对齐+配额；新算法依照论文采用 Gram+平方+GMM | 独立类及 registry 名称，不原地替换旧实现或把旧 P7 结果冒充新算法 |

## 后续门禁建议（未擅自扩展执行）

- 未来新增比较必须携带 metric definition、tail label set、noise target、输入/标签哈希和引用来源。
- 新运行器复用共享 helper；历史协议若改变定义，应新增派生版本及披露，不覆盖原预注册产物。
- P6 scale 与 legacy external baseline 的 Tail-F1 仍是子集口径，引用绝对值时须保留口径标签；
  本轮不以“方法两臂内部一致”推断 canonical 下差值一定不变。
- 本轮确认的 canonical G2 敏感性为 null，独立于原预注册 masked G2；两者应分别标注。

复核入口：`src/paper/p9_tail_metric_unification.py`、`tests/test_p9_canonical_tail.py`、
`tests/test_p9_fine_final_protocol.py`；逐窗口证据见 `reports/p9_tail_metric_unification.json`。
