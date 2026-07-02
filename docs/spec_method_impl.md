# Spec D1-B — Method Implementation (five views, Graph-CDM, evidence soft weighting)

Status: implementation contract for Codex. GPT-pro finalizes the matching math in
`paper/method.tex`.

## 0. Two-stage pipeline (decoupled, mirrors CoLD's three stages)
```
raw alerts/flows
  -> (1) five-view heterogeneous temporal graph        [src/graph/build.py]
  -> (2) self-supervised multi-view representation      [src/models/encoders.py]
  -> (3) Graph-CDM three-term divergence                [src/models/graph_cdm.py]
  -> (4) evidence-preserving soft weights (NO deletion) [src/models/graph_cdm.py]
  -> (5) weighted robust classifier                     [src/train.py]
  -> (6) SOC priority Top-K                             [src/ranking/prioritize.py]
```

## 1. Five views (on flow datasets)
Lift CoLD's feature-subset views to a graph over samples. For each view $k$,
select the semantically-related feature block, build a k-NN (or thresholded
similarity) graph among samples on that block:
- **host**: host-identifier / endpoint features.
- **ip**: IP-communication statistics (bytes/pkts rates, flow duration).
- **process**: process/behavioral columns if present; else derived behavioral
  aggregates (fallback documented in code).
- **temporal**: connect samples within the same `temporal_window`; supports
  snapshot list for temporal alignment.
- **threat_intel**: shared IOC / signature attributes; sparse view.

Overlap between view feature blocks is allowed (CoLD permits subset overlap).
On OpTC, views come directly from the provenance graph; `lambda_chain>0`.

## 2. Stage-1 representation (self-supervised, label-free)
Per-view encoder $z^{(k)}_v=\text{Enc}_k(v)$ (HGT/RGCN). Objective:
$$
\mathcal{L}_{rep}=\mathcal{L}_{con}+\alpha\,\mathcal{L}_{temporal}+\beta\,\mathcal{L}_{recon}
$$
- $\mathcal{L}_{con}$: InfoNCE, positives = same node across two views, negatives =
  other nodes in the batch (temperature $\tau$). Includes feature-obfuscation:
  Bernoulli($\delta$) mask on node features before encoding (CoLD Eq. 2).
- $\mathcal{L}_{temporal}$: pull embeddings of the same node across adjacent
  snapshots.
- $\mathcal{L}_{recon}$: decode global node features from a single view (CoLD
  $\mathcal{L}_{gr}$).
Aggregate views by MEAN → $z_v$ (CoLD ablation shows MEAN best).

## 3. Graph-CDM (stage-2)
Per node $v$:
$$
\text{GraphCDM}(v)=\lambda_1 D_{pred}(v)+\lambda_2 D_{neigh}(v)+\lambda_3 D_{view}(v)+\lambda_4 D_{chain}(v)
$$
- $D_{pred}(v)=\frac1M\sum_k \mathbb{1}[\tilde y^{(k)}_v\neq y_v]$ — per-view GMM
  cluster label vs observed label (this is exactly CoLD CDM; fit GMM with
  `sklearn.mixture.GaussianMixture`, K components).
- $D_{neigh}(v)=\text{KL}\!\big(\hat y_v \,\|\, \text{Agg}_{u\in\mathcal N(v)}\hat y_u\big)$
  normalized to $[0,1]$ — disagreement with graph neighborhood soft-label mean.
- $D_{view}(v)=1-\text{mode-agreement}_k(\tilde y^{(k)}_v)$ — dispersion of
  per-view predicted labels.
- $D_{chain}(v)$: OpTC only; inconsistency along the provenance attack chain.
Default weights $(\lambda_1,\lambda_2,\lambda_3,\lambda_4)=(0.4,0.3,0.3,0.0)$.

## 4. Evidence-preserving soft weighting (KEY)
Evidence score protects informative samples:
$$
e(v)=\underbrace{\tfrac{1}{\log(1+n_{y_v})}}_{\text{low-freq protect}}\big(1+\gamma\cdot\text{anom}(v)\big),\quad \tilde e(v)=\text{minmax}(e(v))\in[0,1]
$$
where $\text{anom}(v)$ is a boundary/early-signal anomaly score (e.g. distance to
class centroid or isolation-forest score). Soft weight (never zero):
$$
w(v)=\sigma\!\big(-\kappa(\text{GraphCDM}(v)-\theta)\big)(1-\rho)+\rho\,\tilde e(v).
$$
Weighted robust training loss:
$$
\mathcal{L}=\sum_v w(v)\,\text{CE}(f(z_v),y_v).
$$
**Ablation `ablation_hard`**: set $\rho=0$ and hard-threshold
$w(v)=\mathbb{1}[\text{GraphCDM}(v)\le\theta]$ → recovers CoLD-style deletion.
Reporting the gap = the paper's central quantitative argument.

## 5. Priority ranking (stage-6)
$$
P(v)=\text{risk}(v)\,(1-\text{FP}_{prob}(v))+\eta\,\text{FN}_{risk}(v)
$$
- $\text{FP}_{prob}(v)$: increasing in GraphCDM for benign-labeled nodes.
- $\text{FN}_{risk}(v)$: high evidence + malicious neighborhood but benign label.
- $\text{risk}(v)$: node attributes (asset criticality proxy / class severity).
Output Top-K and alert-compression-ratio.

## 6. Metrics (src/metrics.py)
Macro-F1, FPR, FNR, alert compression ratio, **Evidence Retention Rate (ERR)**,
noise-detection P/R/F1, time/storage overhead.

## 7. Determinism & reproducibility
All randomness seeded (`configs/model.yaml: train.seeds`). Report mean/std over
seeds {0,1,2}; paired t-test vs CoLD (mirror CoLD's p-values).

## 8. Codex build order (matches D2–D6)
D2: §1 views + §2 representation. D3: §3 Graph-CDM + §4 soft weights + §5 ranking.
D4: OpTC views + graph-consistency noise. D5–D6: matrix + ablations + figures.
