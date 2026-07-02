# Codex Prompt — D1 (data pipeline + sym/asym noise + CoLD baseline + git push)

> Paste to Codex. Scope is limited to D1; do not do D2 work.

# 角色
你是 Graph-CoLD 项目的工程实现者(Codex)。目标期刊 Computers & Security。

# 背景
- 本地仓库已由协调者搭好骨架：C:\Users\g14370\graph-cold（Windows，已 git init，main 分支，已有 commit + tag d1）。
- 所有源码是"带契约的 stub"：每个函数/类的 docstring 写明了输入输出、公式与 TODO(Codex)。你必须严格按这些签名实现，不要改契约。
- 权威设计规格在 docs/：
  - docs/spec_method_impl.md（五视图/表示学习/Graph-CDM/证据保持软加权/排序/指标）
  - docs/spec_graph_noise.md（图结构一致性噪声形式化定义）
  - docs/PLAN.md（7天分工与每日目标）
- 基线：CoLD 无官方代码，需自研复现（src/models/cold_baseline.py），既作 baseline 又作改进对象。
- 数据集：主实验 CICIDS-2017(精炼9类) + MALTLS-22(23类)，在其上"构造多视图图"；OpTC 仅后续做企业案例。
- 环境提醒：本机 python 是 Store stub，不可用；请在你侧真实 Python≥3.10 环境安装 requirements.txt 后运行。

# 本次任务(D1)——只做以下 4 项，不要提前做 D2 内容
1. 数据管线：实现 src/data/loaders.load_dataset(name, cfg)，返回 Dataset 契约(X_train/y_train/X_test/y_test/num_classes/meta)。
   - CICIDS-2017：降采样主类、删除<1000样本的类、8:2切分、丢弃 configs/datasets.yaml drop_cols；时间戳仅存 meta 供 temporal 视图，不作分类特征。
   - MALTLS-22：保留原始不平衡分布、8:2切分。
   - 特征标准化(如 StandardScaler)，返回 meta.feature_names / class_names。
2. 噪声注入：实现 src/data/noise 的 inject_symmetric、inject_asymmetric，返回 (y_noisy, flip_mask)。仅注入训练集。graph_consistency 本次留 TODO(D4)。
3. CoLD 基线骨架：实现 src/models/cold_baseline.py 的 feature_reordering(相关矩阵→networkx 最大生成树→DFS 序)与 CoLD 类的 fit_representation / purify(GMM-CDM 硬删除,ε=0) / fit_classifier / predict，使其能端到端跑通(先在 MALTLS 上出一个 F1 数即可)。
4. 依赖与自检：补齐/校正 requirements.txt 的 CUDA 相关注释；添加 tests/：
   - test_noise.py：同 seed 复现；symmetric 翻转比例≈ratio；test 集无翻转。
   - test_loaders.py：CICIDS 无<1000类；切分比例正确。
   - test_cold_smoke.py：CoLD 在小采样 MALTLS 上端到端跑通不报错。

# Git（把推送也交给你）
- 若本机已配置 gh/凭据可推送则执行；否则输出准确的推送命令供人工执行。
- 设置仓库级身份：git config user.name "One-bit-007"；邮箱用占位或人工补。
- 关联远端并推送：
  git remote add origin https://github.com/One-bit-007/graph-cold.git
  git add -A && git commit -m "feat(d1): data pipeline + sym/asym noise + CoLD baseline skeleton + tests"
  git push -u origin main --tags
- 分支策略：日常提交在 dev，稳定合并到 main；每日打 tag(d1..d7)。

# 约束
- 严格遵守 stub 契约与 docs 规格；改动契约需在 PR 描述里说明理由。
- 全流程可复现：随机数用 configs/model.yaml train.seeds；报告 mean/std over seeds {0,1,2}。
- 不要提交原始数据集(已在 .gitignore)。

# 验收标准(Acceptance，供协调者核对 CK-1/CK-2/CK-3)
- pytest 全绿；CICIDS 类分布满足 min_class_count。
- symmetric 噪声在 β=0/给定 ratio 下分布正确、可复现。
- 自研 CoLD 在 MALTLS 无噪 F1 落在 ~90+ 量级、随噪声上升掉幅温和(与 CoLD slides 趋势一致)。
- 输出一段"运行说明"：如何用命令复现上述三项。
