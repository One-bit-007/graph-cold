# Response to Reviewers — CYSE-D-26-01464 (Major Revision)

**Manuscript:** When Does Graph Structure Help Noisy-Label Intrusion Detection? A Leakage-Audited Study of Rare-Class Training-Signal Retention

We thank both reviewers for reviews that fundamentally improved this work. Before the point-by-point responses, three framing notes.

**Every major change is driven by the reviewers' identifiability critiques.** Both reviews converged on one core objection: our experiments could not distinguish "graph structure helps denoising" from "the graph reconstructs the rules by which benchmark labels were generated." We accepted this objection in full. The revised manuscript is built around a frozen, leakage-audited protocol (proxy-masked features, strongest-available group-disjoint splits, pre-registered hypothesis families) designed specifically to answer it. The title, central claim, and headline results changed **because the reviewers' critique was correct**, not despite it; the traceability matrix below maps every change to the review comment that drove it.

**We proactively disclose three things the reviews did not know.**
1. **The requested combination experiment produced a null result.** Reviewer B's "biggest gap" (M1) — CL + graph diagnostic — was run at full power on all three datasets: the complete pre-registered 18-test family finds no detectable increment over Confident Learning alone (largest Tail-F1 CI upper bound +0.026). We report this as a first-class boundary result (§6.3) rather than burying it.
2. **Our earlier "CoLD" comparison was a false comparison.** The fidelity audit requested by Reviewer B (M2) revealed that the number we had previously discussed as "CoLD, 0.534" is an internal proxy implementation, and "0.688" is our own hard-deletion ablation — neither is official CoLD, whose published numbers (0.94–1.00, nine-class protocol) are not comparable in either direction. All rows are renamed *CoLD-aligned proxy*, and we claim neither reproduction of nor superiority over official CoLD (§7.1).
3. **Two protocol deviations occurred and are disclosed as amendments** (Appendix A): an unregistered futility stop in the combination experiment (remediated by complete full-power reruns) and a claim initially stated ahead of its evidence tier (remediated by the full pre-registered grid).

**Traceability matrix (comment → change → location).**

| Review comment | Change in revision | Location |
|---|---|---|
| A1 CESNET semantics | Service-class stress-test reframing; attack claims limited to CICIDS/UNSW; 178→25 filter table | §5.1, §9, App. B |
| A2 label-construction proxy leakage | Masking tiers + group-disjoint splits as *primary* protocol; PRX factor quantified; regression tests | §5, §6.4 |
| A3 undifferentiated ablation | Full factor grid; family A robust / family C null (claim withdrawn) | §6.2 |
| A4 statistical interpretation | F-6 protocol: per-rate n=10, Holm families, TOST δ=0.01 + per-class guard, mixed effects | §5, Tables 3–5 |
| A5 protocol inconsistency | Protocol frozen before final grid; deviations as amendments | §5, App. A |
| A6 method underspecification | Frozen-configuration subsection (all hyperparameters, tuning discipline); D_chain removed | §4 |
| A7 baselines / novelty | CL inside frozen protocol; FINE added; Co-Teaching de-oracled; proxy renaming | §5, §7.1 |
| A8 evidence over-extrapolation | "Training-signal retention" wording; triage-proxy subsection deleted; bounded claims | §1, throughout |
| A min. req. (splits / online inference) | S0–S3 on all datasets; inductive primary; transductive control; CICIDS entity finding | §5, §6.3 |
| B M1 CL combination missing | Complete 18-test family; G2 = null with effect ceiling | §6.3, Table 4 |
| B M2 CoLD fidelity | Identity audit; proxy renaming; non-comparability statement | §7.1 |
| B M3 scale / synthetic noise | 100k-window validation; real labelling errors via published corrected labels | §6.2, §6.4 |
| B M4 pseudo-replication | Per-rate inference unit; no pooled n=30 | §5 |
| B M5 audit novelty / literature | Benchmark-error literature positioning; proxy-leakage taxonomy increment | §2 |
| B minors 1–8 | See individual responses | various |

---

## Reviewer A

**A1. CESNET-TLS-Year22 carries no attack semantics; using it to support "rare-attack evidence" claims is a category error.**

Accepted in full; this was a factual error in the original manuscript, and fixing it was our first action. The revision (i) reframes CESNET everywhere as a long-tail *service-classification stress test* whose SNI-derived labels have no attack/benign semantics (§5.1); (ii) restricts all attack-level conclusions to CICIDS-2017 and UNSW-NB15 (§9, Conclusion); (iii) publishes the frozen 178→25 class-filter policy with per-class counts (Appendix B); and (iv) replaces the noise-flip target for CESNET with a high-frequency service class (`microsoft-diagnostic`) because "benign" is undefined there (§5.1). We also evaluated candidate replacement datasets against our own audit criteria before deciding to retain CESNET under the corrected semantics; CTU-13 failed hard criteria (labels derived from host-IP filters — the very proxy problem under study) and CIC-IDS-2018 inherits CICIDS-2017's generation pipeline.

**A2. The experiments cannot distinguish graph benefit from reconstruction of label-generation rules (timestamps, IPs, ports, attack schedule).**

This is the review's central critique and it reshaped the study. Actions:

1. *Proxy-field audit on all three datasets* (not only CICIDS): we identified and masked label-construction fields and their proxies per dataset (CICIDS tier-1; CESNET/UNSW tier-2). This surfaced a finding beyond the review's examples: CESNET's released features include `CATEGORY` one-hot columns that are *parents of the label* (§6.4).
2. *Group-disjoint splits as the primary protocol*: temporal-by-day for CICIDS, strict temporal∩entity for CESNET, official partition for UNSW, with train–test exact/near-duplicate intersection measured and reported for every split (§5).
3. *Quantification rather than assertion*: the four-factor decomposition (§6.4, Table 6) isolates the proxy-field factor — +0.184 Tail-F1 on CESNET; net *negative* on UNSW, where unmasking dilutes the 38.4% feature-collision memorization in the official partition. The masked, group-disjoint numbers are the paper's primary results; all inflated configurations appear only inside the decomposition.
4. *Regression tests* now enforce that no label-construction field, clean label, or noise mask can enter features, graphs, or diagnostics; the tests ship with the reproduction package.
5. On the reviewer's specific mechanism — the graph reconstructing the attack schedule — we additionally attempted entity-disjoint splitting on the raw CICIDS release (which retains IP columns). The flow graph collapses into one giant connected component covering 99.96% of valid flows: entity isolation is *infeasible by construction* on this benchmark. We report this honestly as a dataset-level finding (§9) rather than substituting a weaker entity definition that would appear to pass.

**A3. The ablation does not separate graph, diagnostic, and weighting factors.**

We ran the requested factor grid under the frozen protocol: shared-encoder cells for hard deletion, soft weighting, a frequency-floor reweighting, and a no-graph class-balanced CE, at full power (10 seeds × 3 rates × 3 datasets), plus screening-tier single-term diagnostic variants (descriptive). The outcome cuts both ways and we report both directions: soft-vs-hard is robust (family A, 9/9 corrected tests; §6.2), but the *specific* evidence-rescue weighting adds nothing over the simple frequency floor (family C, null) and is significantly worse on UNSW at two rates. We therefore withdrew the claim that our weighting form is superior to simple reweighting; the surviving claim is that *retention beats deletion*, with the form empirically interchangeable (§6.2).

**A4. Statistical interpretation errors (pooled observations, "no cost" from non-significance).**

The revision replaces the entire statistical apparatus with a pre-registered protocol (§5): the inference unit is the per-noise-rate paired seed comparison (n=10) — the 30 seed–rate observations are never pooled as independent; Holm correction is applied within three pre-registered families whose composition is fixed in the reproduction package; "no aggregate cost" claims are replaced by TOST non-inferiority at δ=0.01 *plus* a per-class guard (no class F1 regression >0.05), because a 0.01 macro margin could hide a concentrated single-class regression; and seed random-intercept mixed-effects models provide robustness confirmation. All headline claims state effect sizes and bootstrap CIs alongside corrected p-values.

**A5. Protocols differ across experiments.**

Correct, and this had already invalidated some of our own early results. The revision freezes one protocol (windows, tail definition, noise mapping, splits, tuning discipline, statistical families) *before* the final experiment grid; every result table in the paper comes from inside that protocol, and the two deviations that occurred are disclosed as amendments with their remediations (Appendix A). Legacy-protocol artifacts appear only where explicitly labelled non-comparable (Table 2) or inside the leakage decomposition.

**A6. The method is underspecified (edge rules, hyperparameters, thresholds).**

Added §4 "Frozen configuration and tuning discipline": all diagnostic weights, weighting parameters (κ, θ, λ_r, γ, τ_ret), the hard-deletion threshold, per-view edge rules (k-NN parameters, temporal window), encoder architecture, and training budget, together with the tuning discipline (no clean validation set, no true injected noise rate, for any method at any point). The provenance-chain term D_chain is removed outright (frozen λ₄=0) rather than left as an undefined component. Per-dataset active views are specified in the data contract (Table 1).

**A7. Baselines are thin and novelty is overstated.**

Three actions. (i) Confident Learning now runs *inside* the frozen protocol with full stratified out-of-fold probabilities, and it defines the performance bar — a result we highlight rather than hide (§6.1, §6.3). (ii) FINE was added; it ran under the legacy window and is reported as explicitly non-comparable (Table 2), which we state plainly. Co-Teaching's forget schedule was rebuilt to use an *estimated* noise rate after we found the previous version consumed the true injected rate — an unfairness we caught and disclose. (iii) The novelty claim is rescoped: the contribution is the audit protocol, the mechanism-level retention finding, and the boundary results — not an accuracy advance (§1).

**A8. "Evidence" language over-extrapolates benchmark rows into operational/forensic meaning.**

Accepted. The revision uses "training-signal retention" throughout; frequency protection is explicitly interpreted as *deletion cost*, not trustworthiness (§4.4); the triage-oriented priority score and its subsection are deleted because no supporting result remains in the formal matrix; and the claims section states explicitly that no SOC or forensic effectiveness is established.

**A. Minimum revision requirement: group-disjoint splits and online (inductive) inference on all datasets.**

Done as the primary protocol (see A2). Inductive per-sample inference is the primary mode; a transductive variant is reported only as a control, and it performs *worse* on all three datasets — evidence against test-batch information driving our numbers (§6.3).

**A. Minor issues** (aggregation inconsistencies; the 25-class filter; Fig. 3 status; §2 structure; the Cost section; stale citations): all addressed — a single frozen aggregation now generates every number via a table-generation script with a traceability manifest; the 25-class policy is Appendix B; the informativeness figure is explicitly descriptive/hypothesis-generating (§6.5); Related Work was merged with its redundant positioning subsection; the Cost section (legacy-protocol measurements) is deleted; the CoLD citation is updated to the NDSS 2026 official version.

---

## Reviewer B

**M1. The single biggest gap: no "CL + graph diagnostic" combination experiment despite CL winning everywhere.**

Run at full power, exactly as requested. CL+Graph-CDM (soft reweighting of CL-retained rows; intersection/union fusions as exploratory variants) versus CL alone, 10 seeds × 3 rates × 3 datasets under each dataset's frozen primary protocol, pre-registered as an 18-test Holm family. Result: **null** — no test significantly positive (all Holm p ≥ 0.608), largest Tail-F1 CI upper bound +0.026, mixed-effects confirmation in agreement, and CL near-saturated in the 100k-window check (§6.3, Table 4). We disclose that the first pass ran full power on only one dataset after a screening stop that had not been pre-registered; the complete family was rerun and the stop is disclosed as amendment PA-1 (Appendix A). The manuscript repositions the graph diagnostic as an audit instrument, and the abstract states the null result with its effect ceiling.

**M2. CoLD fidelity: the reproduced number is far below the published one; baseline may be unfaithful.**

The audit you requested found something worse than infidelity: a false comparison (see proactive disclosure 2). The comparator predates the official code release and implements neither multi-subset encoding, clustering-consistency purification, nor the XGBoost head; the "0.688" figure we had contrasted it against is our own ablation. The revision renames every such row *CoLD-aligned proxy*, adds §7.1 explaining precisely what the proxy is and is not, declares official CoLD's published numbers non-comparable in either direction (different class space, split design, and noise model), and registers an official-code alignment grid as future work. We chose honest renaming over a rushed "official reproduction" whose data contract (exact refined arrays, split manifests) the official release does not provide.

**M3. External validity: small windows and synthetic noise only.**

Two additions. (i) A one-shot 100k-window scale validation on CICIDS: the family-A direction holds at every rate; the G2 null is reinforced (CL saturates; the rate-0.8 point estimate is negative). It also reveals honestly that window size dominates tail learnability. (ii) A real-noise experiment: applying the published corrected-label rules of Liu et al. to the original CICIDS release yields 128,247 corrected rows (4.53%); training on original labels and evaluating against corrected labels preserves the family-A direction (+0.284 Tail-F1, descriptive). The same analysis shows ~95% of the rare "Web Attack – Brute Force" class are labelling artifacts — dataset-native corroboration of the audit's core concern (§6.2, §6.4).

**M4. Pseudo-replication in the statistics.**

Fixed as part of the frozen statistical protocol (see A4): per-rate paired tests (n=10 each), never a pooled n=30; families pre-registered; mixed-effects robustness models reported alongside.

**M5. The audit's novelty relative to prior CICIDS-error literature is unclear.**

Related Work now contains a dedicated positioning paragraph (§2): Engelen et al. diagnosed the errors, Liu et al. corrected the labelling logic, Lanvin et al. measured the performance impact. Our increments over that line: (i) extending the error taxonomy from *label mistakes* to *label-construction proxy leakage* (features and split structures that let a model reconstruct labelling rules); (ii) an isolation protocol (masking tiers × split taxonomy × four-factor decomposition) with regression tests other studies can reuse; (iii) using the published corrected labels as a validation instrument rather than only a critique.

**B. Questions for response 1–5.**
*Q1 (combination):* see M1 — run, null, disclosed. *Q2 (CoLD gap source):* see M2 — the gap was a method-identity confusion, now corrected and audited item-by-item (data policy, split, noise model, representation, diagnostic, deletion rule, classifier head). *Q3 (leakage magnitude under the final protocol):* the four-factor decomposition quantifies 25–36 points of Tail-F1 inflation on CESNET (ENT + PRX approximately additive) and shows leakage flatters deletion-style baselines most, *compressing* the soft-vs-hard gap (0.256 → 0.125 on CICIDS under DUP) — the honest direction of the earlier "+28-point" story (§6.4). *Q4 (statistics):* see M4/A4. *Q5 (freq_protect design):* §4.4 now discusses the structural limitation explicitly — protection is keyed on the observed noisy label, so asymmetric noise mis-protects flipped samples; family C confirms empirically that the specific form carries no advantage, which is why the claim was withdrawn.

**B. Minor comments 1–8.**
1. *Sensitivity curves for λ_r/κ/θ:* rather than sweeping parameters of a weighting form whose superiority claim we have withdrawn, we submit the stronger family-C result: the form itself (soft rescue vs. frequency floor vs. class-balanced CE) is empirically interchangeable for the retention benefit. A sweep supporting a withdrawn claim would be misleading precision; the frozen values are fully disclosed in §4.
2. *freq_protect protecting the observed noisy label:* addressed in §4.4 and empirically in family C (see Q5).
3. *Abstract length:* rewritten at ~240 words.
4. *Merge §2 and §2.1:* merged; unique citations retained.
5. *Cost section:* deleted (its measurements predate the frozen protocol).
6. *Fig. 3 (informativeness):* explicitly descriptive and hypothesis-generating (§6.5); it is retained as an audit instrument, not evidence of benefit.
7. *Stale citations:* CoLD updated to NDSS 2026; benchmark-error literature added.
8. *Internal file paths:* remaining paths reference reproduction-package artifacts only (data contracts, noise maps, table sources) and are labelled as such.

---

We believe the revised manuscript answers the identifiability critique with quantified evidence, reports the requested experiments regardless of direction, and converts the audit trail itself into reusable methodology. We thank both reviewers again for reviews that made the paper honest.
