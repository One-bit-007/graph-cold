# P8 CESNET F-3：前置核验与阻塞报告

日期：2026-09-03。状态：`BLOCKED_BEFORE_EXPERIMENTS`。

本报告不是重跑完成报告。遵照工作包“与预期不符处（如有则停下）”，在发现下列接口及续跑问题后，未启动 CESNET 正式模型实验，未删除、覆盖或合并任何历史实验产物，未改动论文、算法、配置或冻结协议。

## 1. 授权范围及基线

- 工作包：`docs/prompts/prompt_codex_p8_cesnet_f3.md`。
- 检查基线：`508a4600c357f59e38a3745877f097ea65576056`。
- 根因修复提交：`9cdb642`；P5 场景准备现调用 `p3._noise_target`。
- 本轮目标原定为 CESNET 120 method-runs，以及合并后三数据集的完整 Family A=9 / C=18 检验、逐类护栏、混合效应分析和 PA-3 披露草稿。
- 不运行 FINE，不 push，不发布 Zenodo，不授权投稿。
- 已核对 `reports/d11/d11_final_audit.json`、`reports/d9/d9_submission_lock_audit.json` 等文件中 `submission_ready` 为 false；这些文件未修改。

## 2. 已完成的前置核验

| 检查 | 实际结果 |
| --- | --- |
| `tests/test_noise_target_consistency.py` 与 `tests/test_p5_factor_ablation.py` | 合计 10 passed；前者包含工作包要求的 4 项 |
| `tests/test_no_oracle_leakage.py` 与 `tests/test_proxy_field_regression.py` | 合计 15 passed |
| 真实 CESNET 主协议数据加载 | 成功；tier2 屏蔽、`temporal_entity` 划分 |
| seeds 0–9 的各个 10,000 行训练窗口 | 全部存在目标类；目标均为 `(11, microsoft-diagnostic)` |
| 旧 P5 分片存在性 | CESNET、CICIDS、UNSW 的 final/partial 文件均存在 |
| 历史产物状态 | 全部原样保留；本轮未产生新的实验指标 |

上述 25 项为定向回归测试，不是全仓测试。测试可能使用小型合成样本；没有运行论文的真实数据训练网格。

真实窗口检查通过 `p4._load_primary_dataset` 加载主协议数据，使用 `p2d._stratified_sample` 与冻结的 seeds 0–9 / TRAIN_WINDOW=10,000 取样，再逐窗调用 `p3._noise_target`；没有调用训练函数。

## 3. 与工作包预期不符的三项问题

### B1 — 逐类护栏未适配已修改的接口

`src/paper/p5_factor_ablation.py:82` 的签名为：

```python
def _prepare_scenario(ds, dataset_name: str, seed: int, rate: float):
```

但 `src/paper/p5_perclass_guard.py:61` 仍调用：

```python
scenario = p5._prepare_scenario(ds, seed, rate)
```

无需训练，用参数绑定即可复现：

```python
inspect.signature(p5._prepare_scenario).bind(None, 0, 0.6)
# TypeError: missing a required argument: 'rate'
```

因此“逐类护栏已随根因修复”并不成立。清除旧缓存后会直接失败。建议最小修复为显式传入 `dataset`，并增加覆盖该调用路径的测试；本轮未擅自修改。

### B2 — 噪声目标只在内存中，没有写入结果行

`_prepare_scenario` 返回 `noise_target`，但 `_cell_row` 只写出：

```text
dataset, cell, seed, tail_flip_rate, macro_f1,
macro_f1_seen_classes, tail_macro_f1, unseen_test_class_fraction,
weight_min, weight_mean
```

AST 检查确认返回结果字典没有 `noise_target`，也没有 `per_class_f1_json`。原样执行工作包命令，不能满足“新 CESNET 行的 noise_target 记录为 11”；P5 主 runner 本身也不输出逐类结果，需要另行完成护栏路径。

建议在训练前明确记录方案：每个 CESNET 场景至少持久化目标 id/name，并用回归测试检查；可用独立 provenance 文件保持历史主表 schema 和未受影响数据行不变。不得为旧数据补造未经核验的运行元数据。

### B3 — 护栏历史缓存会静默复用错误 CESNET 结果

`results/p5_perclass_guard.partial.csv` 当前有 180 行，每数据集 60 行。护栏 runner 仅按 `(dataset, seed, tail_flip_rate)` 下两种 cell 是否齐全判断完成。

现有缓存使 CESNET 的 30 个场景全部进入 `done`。仅删除工作包列出的 P5 CESNET final/partial，**不会清除这个独立的护栏缓存**；护栏步骤会跳过全部 CESNET 训练，继续使用旧结果。

因此须同时解决 B1 与缓存隔离。建议先归档旧护栏产物，在独立输出目录重跑 CESNET 护栏，再与旧 CICIDS/UNSW 护栏结果合并并校验；不要删除整个三数据集护栏历史或重跑不受影响的数据集。

## 4. 历史产物指纹与保护范围

SHA-256（本轮读取；未重写这些文件）：

| 文件 | SHA-256 |
| --- | --- |
| `results/p5_factor_ablation_full.csv` | `7359f9006a35485af1d6a7b6daaa05d6a8b223dd72a834fd0398d527ece463bf` |
| `results/p5_factor_ablation_full_cesnet_tls_year22.csv` | `f74edf8fedcb7268961c29cd803d636ef11863db63abac01fb64094838b486ce` |
| `results/p5_factor_ablation_full_cesnet_tls_year22.partial.csv` | `f74edf8fedcb7268961c29cd803d636ef11863db63abac01fb64094838b486ce` |
| `results/p5_factor_ablation_full_cicids2017.csv` | `1cb6e8ef5b603b3d8c99ad1a997ea486f8302e46370e583fd11b9afc6135d30d` |
| `results/p5_factor_ablation_full_unsw_nb15.csv` | `4360a44e66b6a85853d160128e85edd139ad6a857ca51af04d5651af834005d0` |
| `results/p5_perclass_guard.csv` | `6bdd06f7a679df09c1c2d94557a931080ef8b4e4fd1cedf5660e63d743a51007` |
| `results/p5_perclass_guard.partial.csv` | `c7900429a899a56f6053b01411b40b7a1446086e3335930405ae06bb790498b8` |

注意：`git ls-files results/p5*` 显示全量主表已跟踪，但逐数据集分片和护栏原始 CSV 未被跟踪；这些文件的 `git diff` 为空不能证明其字节未变，后续必须核对上述哈希。不能仅依赖 Git 保存这些分片的历史。

## 5. 尚未完成的交付物

| 工作项 | 本轮状态 |
| --- | --- |
| CESNET 新旧指标对照 | 新指标未生成，不能填数 |
| 完整 Family A=9 / C=18、非劣性重算 | 未执行；不能确认或更新裁决 |
| CESNET 逐类护栏重跑及三数据集合并 | B1/B3 阻塞 |
| 混合效应分析重算 | 未执行 |
| 自动生成 LaTeX 表格、数字审计 | 数据尚未重跑，未执行 |
| 全仓测试与声明门 | 未执行；仅上述定向测试完成 |
| 稿件、摘要、答复信、发布包 | 未改动，按工作包留给协调者 |

历史 A/C 结果中的 CESNET 部分仍来自待纠正产物。本次检查不构成新的显著性、robust 或 null 裁决。Family B 未被本轮修改或重跑。

## 6. PA-3 披露草稿（当前阶段，不能表述为已完成重跑）

> **PA-3 — Correction of the CESNET asymmetric-noise implementation.** A post-analysis code audit identified an inconsistency between the frozen F-3 mapping and the P5 factor-ablation implementation. For CESNET, the earlier P5 implementation fell back to label 0 (`apple-itunes`), whereas F-3 specifies label 11 (`microsoft-diagnostic`). The target-selection code has been corrected to use the shared mapping function. The affected CESNET Family A/C estimates and per-class guard results remain pending recomputation and must not be treated as corrected evidence. Before the rerun, we additionally identified an outdated per-class-runner call, missing persisted target metadata, and a cache that would reuse the earlier CESNET results. These execution issues are being resolved without changing the frozen seeds, noise rates, sampling windows, masking, splits, model settings, or statistical families. After the corrected rerun, the complete three-dataset Family A and Family C families, non-inferiority results, per-class guards, and mixed-effects checks will be recomputed; the superseded artifacts and a before/after comparison will be retained. The P4 Family B runner used the prescribed target-selection path and is outside this correction.

最终 PA-3 须等重跑、完整族重算与验收后按事实更新时态，并写明裁决是否变化；此段尚未写入稿件。

## 7. 继续所需的最小裁定

请协调者确认允许先补齐 B1 调用适配、B2 结果元数据与 B3 缓存隔离，再执行原 P8 重跑范围。不得通过绕过护栏、复用旧 CESNET 缓存或只计算 CESNET 子族来完成验收。

本轮项目变更仅此报告；未改动进入任务时已有的无关未跟踪文件。报告提交哈希在会话交付中给出。
