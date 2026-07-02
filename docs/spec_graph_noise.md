# Spec D1-A — Graph-Consistency Label Noise (formal definition)

Status: draft for GPT-pro to finalize wording; Codex implements
`src/data/noise.inject_graph_consistency`.

## 1. Motivation
CoLD shows performance degradation is driven by *local consistency*: features of
different classes share similar distributions, letting noisy labels form spurious
associations. Existing benchmarks only test **symmetric** and **asymmetric**
noise, which are label-space constructs and ignore this structural mechanism.

We introduce **graph-consistency noise**, a noise model that flips labels
*preferentially between locally-consistent nodes*. It is a stronger, more
realistic stress test because it directly targets the failure mode CoLD
identified — and it is exactly the regime where hard-deletion purifiers lose
clean boundary samples. This motivates evidence-preserving soft reweighting.

## 2. Setup
Training set $\{(x_i,y_i)\}_{i=1}^N$, labels in $\{1,\dots,K\}$, benign class $b$.
Multi-view graph $G=\{G^{(k)}\}_{k\in\mathcal{V}}$,
$\mathcal{V}=\{$host, ip, process, temporal, threat_intel$\}$, each
$G^{(k)}=(V,E^{(k)})$ over the same node set $V$ (one node per training sample).

## 3. Local-consistency edge weight
For a view $k$ and edge $(i,j)\in E^{(k)}$ define a consistency weight
$$
c^{(k)}_{ij} = \mathbb{1}[y_i \neq y_j]\cdot s^{(k)}_{ij},
$$
where $s^{(k)}_{ij}\in[0,1]$ measures distributional overlap of the two nodes'
local neighborhoods in view $k$ (reuse the KS-based overlap from CoLD's
`local_consistency`; higher = more overlap). We only consider **cross-class**
edges because only those can carry a *misleading* flip.

Aggregate across views:
$$
c_{ij} = \frac{1}{|\mathcal{V}|}\sum_{k\in\mathcal{V}} c^{(k)}_{ij}.
$$

Per-node consistency exposure:
$$
C_i = \frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)} c_{ij}.
$$

## 4. Noise mechanism
Given target ratio $r$ and bias $\beta=$ `consistency_bias` (default 0.8):
1. Number of flips $m = \lfloor rN \rfloor$. Split into
   $m_c=\lfloor\beta m\rfloor$ consistency-driven and $m_r=m-m_c$ random.
2. **Consistency-driven flips** ($m_c$): sample source nodes with probability
   $\propto C_i$. For a chosen node $i$, flip $y_i \to y_j$ where $j$ is drawn
   from its cross-class neighbors with probability $\propto c_{ij}$ (i.e. relabel
   it to the class of the *most locally-consistent* opposing neighbor).
3. **Random flips** ($m_r$): uniform over remaining nodes and target classes
   (keeps comparability with symmetric noise).
4. Record `flip_mask` (bool $[N]$) for metrics.

Deterministic given `seed`.

## 5. Properties to state in the paper
- Reduces to symmetric noise when $\beta=0$.
- Produces a **structured** transition matrix conditioned on graph locality,
  unlike class-conditional asymmetric noise.
- Directly worsens the local-consistency confounding → the setting where
  Graph-CDM's neighborhood term $D_{neigh}$ and evidence preservation should show
  the largest gains over CoLD (planned ablation).

## 6. Deliverables
- Codex: `inject_graph_consistency(y, ratio, graph, cfg, rng)` returning
  `(y_noisy, flip_mask)`; unit test that $\beta=0$ matches symmetric distribution.
- GPT-pro: formal paragraph + a transition-matrix figure (cf. CoLD Fig. 6).
