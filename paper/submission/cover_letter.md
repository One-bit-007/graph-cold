# Cover Letter — Computers & Security

[DATE]

To the Editor-in-Chief,
*Computers & Security*

Dear Editor-in-Chief,

We submit our manuscript entitled **"Evidence-Preserving Graph Denoising for
Noisy-Label Intrusion Detection: A Leakage-Audited Study of When Graph
Structure Helps"** for consideration as a research article in *Computers &
Security*.

**Scope.** Intrusion-detection classifiers are routinely trained on delayed,
rule-generated, or inconsistent labels. A common remedy—deleting suspicious
samples—is risky in security operations because the deleted instances are often
the rare attack samples analysts most need. We revisit the CoLD collaborative
label-denoising framework and ask two questions: does a multi-view,
label-space graph consistency diagnostic (Graph-CDM) improve robustness, and
does evidence-preserving soft weighting protect rare-attack evidence better
than hard deletion?

**Contributions.** The study is, deliberately, an audit before it is a method:

1. A label-space multi-view graph diagnostic that modestly but significantly
   improves aggregate Macro-F1 over its aligned CoLD baseline on three real
   datasets (CICIDS-2017, CESNET-TLS-Year22, UNSW-NB15) under a
   leakage-audited protocol.
2. A mechanism-level finding: evidence-preserving soft weighting is
   statistically indistinguishable from hard deletion on aggregate accuracy,
   yet significantly improves rare/tail attack-class Macro-F1 under high
   asymmetric noise at no aggregate cost—within the shared pipeline.
3. An honest external baseline ordering under the identical protocol:
   Confident Learning outperforms our method on all datasets, while full
   Co-Teaching collapses on rare classes—empirically confirming that deletion-
   and forgetting-based mechanisms can sacrifice rare-attack evidence unless
   the noise diagnostic is precise.
4. A leakage audit showing that graph denoising results in security are easy
   to inflate (24% exact duplicates in CICIDS-2017; clean-label signal paths),
   collapsing an apparent +28-point advantage to a defensible +6.7 points,
   together with a graph-informativeness diagnostic bounding when graph
   structure helps.

We believe the audit discipline and the audited baseline ordering are of
direct practical relevance to the journal's readership, independent of the
specific method.

**Statements.** This manuscript is original, has not been published before,
and is not under consideration by any other journal. All authors have approved
the submission. Raw datasets are public; code, frozen result hashes, and the
reproduction command sequence are available as described in the Data
Availability section.

Authors: [AUTHOR NAMES]
Corresponding author: [CORRESPONDING AUTHOR, AFFILIATION, EMAIL]

Sincerely,
[AUTHOR NAMES]
