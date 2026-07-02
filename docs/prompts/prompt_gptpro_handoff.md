# GPT-pro FULL HANDOFF — own the project end-to-end (D2→D7→submission)

> From here on GPT-pro fully drives the project and Codex executes. There is no
> external coordinator review anymore: GPT-pro OWNS all checkpoints (CK-1..CK-7),
> authors every Codex prompt, reads/points Codex to local files, verifies
> acceptance, and prepares the Computers & Security submission. This file is
> self-contained.

## 0. Mission & target
Extend CoLD (NDSS'26, sample-level label-noise purification for IDS) into
**Graph-CoLD**: an evidence-preserving collaborative graph denoising + SOC alert
prioritization framework. Target venue: **Computers & Security** (Elsevier).
Writing style: strong SOC/enterprise security-engineering framing, real
deployment value, avoid abstract ML-only phrasing. Timeline: finish within the
7-day sprint (currently entering D2).

## 1. What Graph-CoLD is (structure-aligned with CoLD, one-to-one upgrade)
- Feature-subset views → **five-view heterogeneous temporal graph**
  (host / ip / process / temporal / threat_intel).
- GMM-CDM → **Graph-CDM** (label/prediction-space consistency; four terms).
- Hard deletion (ε=0) → **evidence-preserving soft reweighting** w(v)>0 always;
  ρ=0 + hard threshold recovers CoLD (central ablation).
- No ranking → **SOC priority Top-K** + alert compression ratio + new metric
  **Evidence Retention Rate (ERR)**.

Scope locked: main experiments on **CICIDS-2017 + MALTLS-22** (constructed graphs);
**OpTC** = single enterprise case study (Flash/Argus + Graph-CoLD plug-in, XGBoost).

## 2. Repository state (github.com/One-bit-007/graph-cold; local C:\Users\g14370\graph-cold)
- Windows host; local python is a Store stub — Codex must use a real Python≥3.10
  env and `pip install -r requirements.txt` (CUDA build to match the GPU).
- main/dev/tag d1 pushed. Scaffold complete: contract stubs with docstrings +
  formulas + TODO markers.
- **Codex has completed D1**: src/data/loaders.py, src/data/noise.py
  (symmetric/asymmetric done; graph_consistency = TODO D4),
  src/models/cold_baseline.py (corr→MST→DFS, per-subset GMM-CDM, hard filter).
  Note: CoLD baseline `fit_representation` currently uses PCA and MUST be upgraded
  to the real contrastive encoder in D2 for fair comparison.

## 3. Authoritative local files Codex must read before coding
- `docs/spec_method_impl.md` — five views, representation, Graph-CDM, evidence
  soft weighting, ranking, metrics (SOURCE OF TRUTH for method).
- `docs/spec_graph_noise.md` — graph-consistency noise formal definition.
- `docs/PLAN.md` — 7-day plan + **Pending math corrections** section.
- `configs/*.yaml` — datasets / noise / model hyperparameters.
- `src/**` stubs — function/class contracts Codex must fill without changing.
- `docs/prompts/prompt_codex_d2.md` — the ready D2 task (already approved).
Every Codex prompt you author must list the exact files Codex should open.

## 4. TWO pending math corrections — fold into D1 finalization NOW (before D3/D4)
(Authoritative definitions already live in the specs; your earlier D1 draft
deviated. Fix the draft to match, do not change the specs.)
1. **D_pred (pre-D3, CK-4)**: compare each view's predicted label to the
   *observed* label y_v: `D_pred(v) = (1/M) Σ_k 1[ỹ_v^(k) ≠ y_v]`. Do NOT compare
   to the fused vote ŷ_v (that overlaps D_view and destroys CoLD-CDM
   noise-transfer semantics). See spec_method_impl.md §3.
2. **Graph-noise β=0 (pre-D4, CK-2)**: keep the target ratio r. Total flips = r·N,
   split into β·rN consistency-driven + (1-β)·rN random, so `β=0 ⇒ symmetric(r)`.
   Reject the `(1-β)δ_ij + β·π_ij` form (zero noise at β=0, drops r). See
   spec_graph_noise.md §4.

## 5. Graph-CDM (final, label/prediction space) — the definitions to implement
- `GraphCDM(v) = λ1 D_pred + λ2 D_neigh + λ3 D_view + λ4 D_chain`
  defaults (λ1,λ2,λ3,λ4)=(0.4,0.3,0.3,0.0); λ4>0 only on OpTC.
- `D_pred(v) = (1/M) Σ_k 1[ỹ_v^(k) ≠ y_v]`  (per-view GMM cluster label vs observed)
- `D_neigh(v) = KL(ŷ_v ‖ mean_{u∈N(v)} ŷ_u)` normalized to [0,1]  (label space)
- `D_view(v) = 1 − (1/M) max_c Σ_k 1[ỹ_v^(k)=c]`  (view mode disagreement)
- `D_chain(v) = 1 − sim(ŷ_{v_{t-1}}, ŷ_v, ŷ_{v_{t+1}})`  (OpTC only)
- Evidence: `e(v) = freq_protect(n_{y_v})·(1+γ·anom(v))`,
  `freq_protect ∈ {log(1+n_c), 1/n_c}`, minmax-normalized `ẽ(v)`.
- Soft weight: `w(v) = σ(−κ(GraphCDM(v)−θ))·(1−ρ) + ρ·ẽ(v)`  (always > 0).
- Loss: `L = L_cls + α L_con + β L_reg`, `L_cls = Σ_v w(v) CE(y_v, ŷ_v)`.
- Priority: `P(v) = α1·ŷ_mal(v) + α2·GraphCDM(v) + α3·e(v)`; output Top-K.
- ERR: `ERR = Σ_v w(v)e(v) / Σ_v e(v)`; `Tail-ERR` over `V_tail={v: n_{y_v}<τ}`;
  `ERR_final = ½(ERR + Tail-ERR)`.

## 6. Your dual role from now on
(a) Author & science lead: finalize math (fold §4 fixes), write the paper.
(b) Orchestrator: for each day, produce `docs/prompts/prompt_codex_dX.md` with the
    fixed structure 【角色/背景要点/本次任务(限当日,禁越界)/严格契约与要读文件/
    Git 与 tag/验收标准(可量化可运行)】, then verify Codex output against those
    acceptance criteria yourself (you now own CK-1..CK-7). Only advance a day after
    its acceptance passes.

## 7. Day-by-day plan you must drive (D2 already prompt-ready)
- **D2 — five-view graph + CoLD-aligned representation** (prompt_codex_d2.md ready).
  Deliver: five-view graph builder; feature-obfuscation mask; InfoNCE L_con;
  temporal alignment L_temporal; global reconstruction L_recon; HGT (fallback
  RGCN/GAT); MEAN fusion; embeddings [N,128]; upgrade CoLD baseline PCA→encoder.
  Accept: views non-empty; shape [N,128]; L_rep converges; linear-probe/kNN
  Macro-F1 > random+30% on clean labels; seed=42 reproducible.
- **D3 — Graph-CDM + evidence soft weighting + ranking + robust classifier**
  (apply the D_pred fix). Deliver: implement §5 exactly in
  src/models/graph_cdm.py + src/ranking/prioritize.py; weighted classifier in
  src/train.py; `ablation_hard` (ρ=0 + hard threshold) recovering CoLD.
  Accept: soft-weight tensor min > 0; `ablation_hard` numerics ≈ self-CoLD
  (report the gap = core evidence-preservation gain); single-dataset end-to-end.
- **D4 — OpTC case study + graph-consistency noise** (apply the β=0 fix).
  Deliver: inject_graph_consistency per spec_graph_noise §4 (unit test: β=0 matches
  symmetric distribution); OpTC ingest reusing Flash/Argus provenance features with
  XGBoost plug-in (target CoLD Table VIII style gains). λ4>0 for D_chain on OpTC.
  Accept: β=0≈symmetric test passes; OpTC Flash+/Argus+ show improvement.
- **D5 — full experiment matrix + ablations**. Datasets ×
  {symmetric, asymmetric, graph_consistency} × ratios {10,20,40,60(,80)} ×
  baselines {self-CoLD, MCRe, MORSE, FINE, Co-Teaching(+/+), Decoupling, Flash,
  Argus} × seeds {0,1,2}. Ablations: remove multi-view / D_neigh / D_view /
  evidence-preservation (→ ablation_hard) / ranking. Emit CSVs.
  Accept: run_all reproduces; mean/std over seeds; matrix complete.
- **D6 — figures + significance**. ERR / alert-compression / Top-K hit-rate /
  time+storage overhead figures; paired t-test vs self-CoLD (mirror CoLD p-values).
  Accept: every claim in the paper traces to a CSV; figures render from scripts.
- **D7 — manuscript + submission**. Write Intro / Related Work / Motivation
  (local-consistency → graph space; evidence-preservation motive with empirical
  figure) / Method §4 / Experiments §5 / Enterprise §6 / Discussion §7 /
  Conclusion §8; Figure 1 (CoLD→Graph-CoLD, label-space Graph-CDM annotations);
  README + reproduction scripts + release tags d2..d7. Prepare C&S package.

## 8. Baselines & metrics
Baselines: self-CoLD, MCRe, MORSE, FINE, Co-Teaching(+/+), Decoupling, Flash,
Argus (cleanlab may serve as a confident-learning reference).
Metrics: Macro-F1, FPR, FNR, alert compression ratio, ERR (+ Tail-ERR),
noise-detection P/R/F1, time/storage overhead.

## 9. Git & reproducibility (instruct Codex every day)
- Work on `dev`, merge stable to `main`; per-day tag d2..d7; push after each module.
- All randomness seeded (configs/model.yaml train.seeds); report mean/std over
  {0,1,2}; never commit raw datasets (.gitignore already covers them).
- Each Codex prompt names the exact files to read and ends with quantitative,
  runnable acceptance criteria you will personally verify.

## 10. Reviewer-risk guardrails to bake into the paper
- Keep Graph-CDM in label/prediction space (else "you re-introduced the
  distance-based weakness CoLD criticized").
- Show `ablation_hard ≈ CoLD` to prove the gain comes from evidence preservation,
  not architecture swaps.
- Define graph-consistency noise formally with the β=0⇒symmetric property.
- Provide complexity analysis + streaming/incremental note + overhead figures.
- ERR quantifies "evidence preserved, not discarded" (low-freq/APT/boundary).

Deliver now: (1) corrected final D1 math (fold §4); (2) confirm/patch
prompt_codex_d2.md; (3) begin driving D2 with Codex and proceed through D7.
