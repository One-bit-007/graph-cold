# P0-4 族 B 完整报告（WP-1 / PA-1 救济）

- 裁决日期：2026-08-27；模式：full（10 seeds × 3 rates × 5 methods）
- 总行数：450（三数据集各 150，30/30 配对完整）
- cleanlab 版本：2.9.0；概率协议：full_oof_stratified_cv
- 主裁决：CL+Graph-soft vs Confident-Learning
- 统计：分噪声率配对检验（n=10）+ 完整 18 项族 B Holm；TOST δ=0.01 非劣效；逐类回退护栏 0.05；seed 随机截距混合效应稳健性

## 18 项族 B 完整检验表（无 p=1 占位项，全部为实跑）

| dataset | rate | metric | Δ mean | Holm p | 95% CI | dz |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| cesnet_tls_year22 | 0.4 | macro_f1 | +0.0023 | 1 | [-0.0040, +0.0090] | +0.20 |
| cesnet_tls_year22 | 0.6 | macro_f1 | +0.0072 | 0.608 | [+0.0007, +0.0130] | +0.66 |
| cesnet_tls_year22 | 0.8 | macro_f1 | -0.0027 | 1 | [-0.0103, +0.0040] | -0.22 |
| cesnet_tls_year22 | 0.4 | tail_macro_f1 | +0.0038 | 1 | [-0.0102, +0.0200] | +0.15 |
| cesnet_tls_year22 | 0.6 | tail_macro_f1 | +0.0118 | 1 | [-0.0034, +0.0256] | +0.45 |
| cesnet_tls_year22 | 0.8 | tail_macro_f1 | -0.0024 | 1 | [-0.0149, +0.0101] | -0.11 |
| cicids2017 | 0.4 | macro_f1 | +0.0115 | 1 | [-0.0126, +0.0377] | +0.26 |
| cicids2017 | 0.6 | macro_f1 | -0.0230 | 1 | [-0.0653, +0.0026] | -0.36 |
| cicids2017 | 0.8 | macro_f1 | +0.0016 | 1 | [-0.0048, +0.0111] | +0.11 |
| cicids2017 | 0.4 | tail_macro_f1 | +0.0040 | 1 | [+0.0000, +0.0116] | +0.34 |
| cicids2017 | 0.6 | tail_macro_f1 | -0.0422 | 1 | [-0.1265, +0.0000] | -0.32 |
| cicids2017 | 0.8 | tail_macro_f1 | -0.0032 | 1 | [-0.0095, +0.0000] | -0.32 |
| unsw_nb15 | 0.4 | macro_f1 | -0.0035 | 1 | [-0.0078, +0.0004] | -0.52 |
| unsw_nb15 | 0.6 | macro_f1 | -0.0031 | 1 | [-0.0073, +0.0010] | -0.42 |
| unsw_nb15 | 0.8 | macro_f1 | +0.0006 | 1 | [-0.0024, +0.0042] | +0.11 |
| unsw_nb15 | 0.4 | tail_macro_f1 | -0.0102 | 1 | [-0.0208, -0.0001] | -0.59 |
| unsw_nb15 | 0.6 | tail_macro_f1 | -0.0109 | 1 | [-0.0222, -0.0003] | -0.57 |
| unsw_nb15 | 0.8 | tail_macro_f1 | +0.0010 | 1 | [-0.0062, +0.0095] | +0.07 |

## G2 终裁决

**Global verdict: `null`（以完整 18 项族 B 为据）**

### cesnet_tls_year22 — dataset pass = False

| rate | tail Δ | Holm p | CI | NI p（δ=0.01） | 最差逐类 Δ | pass |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0.4 | +0.0038 | 1 | [-0.0102, +0.0200] | 0.00393 | -0.0320 | False |
| 0.6 | +0.0118 | 1 | [-0.0034, +0.0256] | 0.000391 | -0.0192 | False |
| 0.8 | -0.0024 | 1 | [-0.0149, +0.0101] | 0.0486 | -0.0354 | False |

混合效应稳健性：平均 tail Δ +0.0044，单侧 p=0.192，CI [-0.0055, +0.0144]，converged=True。

### cicids2017 — dataset pass = False

| rate | tail Δ | Holm p | CI | NI p（δ=0.01） | 最差逐类 Δ | pass |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0.4 | +0.0040 | 1 | [+0.0000, +0.0116] | 0.0782 | +0.0000 | False |
| 0.6 | -0.0422 | 1 | [-0.1265, +0.0000] | 0.732 | -0.0574 | False |
| 0.8 | -0.0032 | 1 | [-0.0095, +0.0000] | 0.0136 | -0.0032 | False |

混合效应稳健性：平均 tail Δ -0.0138，单侧 p=0.835，CI [-0.0415, +0.0139]，converged=True。

### unsw_nb15 — dataset pass = False

| rate | tail Δ | Holm p | CI | NI p（δ=0.01） | 最差逐类 Δ | pass |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0.4 | -0.0102 | 1 | [-0.0208, -0.0001] | 0.00732 | -0.0227 | False |
| 0.6 | -0.0109 | 1 | [-0.0222, -0.0003] | 0.00736 | -0.0252 | False |
| 0.8 | +0.0010 | 1 | [-0.0062, +0.0095] | 0.000114 | -0.0320 | False |

混合效应稳健性：平均 tail Δ -0.0067，单侧 p=0.978，CI [-0.0132, -0.0002]，converged=True。

结论：完整族 B 下，CL+Graph-soft 相对 Confident Learning 的 Tail-F1 增量在任一数据集、任一噪声率上均不显著为正
（tail 指标最大 95% CI 上界见上表；全部 Holm p = 1.000）。G2 维持 **null**——Graph-CDM 降格为审计/诊断工具，
不再作为性能增量声明。以效应上界语言陈述：组合相对 CL 的 Tail-F1 增益 95% CI 上界最大为 +0.0256（CESNET, rate=0.6）。

## Protocol Amendment PA-1 披露段（供修改稿附录引用）

> **PA-1（2026-08-27）** The frozen analysis family B comprised 3 datasets × 2 metrics × 3 noise rates = 18 tests. 
> In the initial full-power run, only CESNET-TLS-Year22 was executed at full power; CICIDS-2017 and UNSW-NB15 were 
> stopped after a screening pass (3 seeds × rate 0.6) showed non-positive increments. This futility-stopping rule was 
> **not pre-registered** and constituted selective testing. As remediation, we executed the full-power protocol 
> (10 seeds × 3 rates, each dataset's frozen primary protocol) on the two remaining datasets and recomputed the Holm 
> correction over the complete 18-test family with no p=1 placeholders. The screening-stop decision is disclosed here 
> as a post-hoc decision. The G2 verdict (null) is unchanged under the complete family.

## 协议与限制

- **cesnet_tls_year22** — C1b-tier2 + S1∩S2 temporal/entity-disjoint；limitation: audit frame spans about seven hours
- **cicids2017** — C1-tier1 + S1 temporal-by-source-file；limitation: S2 unavailable because the local copy lacks entity columns
- **unsw_nb15** — C1b-tier2 + official fixed partition；limitation: S1/S2 unavailable because the fixed files lack timestamp/entity columns

## 产物

- 合并结果：`results/p4_cl_graph_combo_full_family_b_complete.csv`（450 行）
- 机器可读：`reports/p4_cl_graph_combo_full_family_b_complete.json`
- 单数据集：`p4_cl_graph_combo_full_{cicids2017,unsw_nb15,cesnet_tls_year22}.{csv,json,md}`