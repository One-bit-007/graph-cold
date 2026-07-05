# D11 Rebuttal Prewrite

## Reviewer concern
Why Graph-CDM is not just generic reweighting.

Author response:
Graph-CDM first computes a label-space structured inconsistency diagnostic across prediction, neighborhood, view, and chain terms, then maps that diagnostic into evidence-preserving weights. The weighting is downstream of graph label consistency rather than a generic confidence threshold.

Manuscript location already addressing it:
Method: Graph-CDM label-space diagnostic; Evidence-preserving training weights.

If revision requested, proposed manuscript change:
If requested, add a short ablation pointer to the existing hard-deletion and component tables.

Whether new experiment is needed: no

Risk if not addressed: Medium: novelty could be understated.

## Reviewer concern
Why label-space diagnostic is used instead of embedding distance.

Author response:
The noisy-label decision is made in label space, so label-space disagreement keeps the diagnostic aligned with the supervised corruption process and avoids interpreting representation geometry as evidence of label error.

Manuscript location already addressing it:
Problem Formulation and Method.

If revision requested, proposed manuscript change:
Add a sentence contrasting label-space CDM with embedding-distance filters.

Whether new experiment is needed: no

Risk if not addressed: Medium.

## Reviewer concern
Why FINE-style is excluded from formal results.

Author response:
FINE-style was implemented as a representation/eigenvector filtering baseline but failed the pre-registered real-data smoke gate on CICIDS-2017 symmetric noise, so the artifact excludes it instead of reporting an unstable value.

Manuscript location already addressing it:
D11 Limitations and D9.5 smoke report.

If revision requested, proposed manuscript change:
Already patched in D11.

Whether new experiment is needed: no

Risk if not addressed: Low after patch.

## Reviewer concern
Why Decoupling underperforms.

Author response:
Decoupling updates on prediction disagreement; structured SOC noise can make related alerts agree on the same wrong label, reducing the value of disagreement-only filtering.

Manuscript location already addressing it:
D11 Limitations.

If revision requested, proposed manuscript change:
Already patched in D11.

Whether new experiment is needed: no

Risk if not addressed: Medium.

## Reviewer concern
Why Co-Teaching-lite is named lite.

Author response:
The implementation is a lightweight tabular approximation and is explicitly not presented as full Co-Teaching, avoiding fidelity overclaiming.

Manuscript location already addressing it:
Baselines, ablations, and metrics.

If revision requested, proposed manuscript change:
Keep lite wording in all tables and captions.

Whether new experiment is needed: no

Risk if not addressed: Low.

## Reviewer concern
Why MALTLS-22 is omitted.

Author response:
The project did not have a verified source and license path, so it is excluded rather than used as an unverified dataset.

Manuscript location already addressing it:
Limitations and dataset docs.

If revision requested, proposed manuscript change:
None beyond current limitation statement.

Whether new experiment is needed: no

Risk if not addressed: Low.

## Reviewer concern
Why OpTC is omitted.

Author response:
Verified provenance events needed for the enterprise case were unavailable; reporting OpTC as a formal experiment would overstate evidence.

Manuscript location already addressing it:
Limitations.

If revision requested, proposed manuscript change:
None beyond current limitation statement.

Whether new experiment is needed: no

Risk if not addressed: Low.

## Reviewer concern
Why CESNET subset is acceptable.

Author response:
CESNET postfilter25 is declared as a deterministic audit-window subset and is interpreted as cross-domain stability and retention evidence, not full-archive coverage.

Manuscript location already addressing it:
Experimental Design, Discussion, D11 patch.

If revision requested, proposed manuscript change:
Already patched.

Whether new experiment is needed: no

Risk if not addressed: Medium.

## Reviewer concern
Why ERR=1.0 is not suspicious.

Author response:
ERR is not classification accuracy; it measures retention of clean informative samples under a retained-mask threshold. A value of 1.0 means all samples in that subset remained retained.

Manuscript location already addressing it:
D11 Discussion/Threats.

If revision requested, proposed manuscript change:
Already patched.

Whether new experiment is needed: no

Risk if not addressed: Low after patch.

## Reviewer concern
Why compression ratio is an operational proxy.

Author response:
Compression approximates review-load reduction and is paired with ERR so the paper does not claim analyst-time savings without an analyst study.

Manuscript location already addressing it:
Operational priority proxy; Discussion.

If revision requested, proposed manuscript change:
If requested, add examples of queue-review interpretation.

Whether new experiment is needed: no

Risk if not addressed: Medium.

## Reviewer concern
Why Graph-CoLD still matters when CESNET Macro-F1 is near ceiling.

Author response:
CESNET has strongly separable TLS/flow features in the postfilter25 subset, so Macro-F1 margin is not the primary signal; evidence retention and cross-domain stability remain informative.

Manuscript location already addressing it:
D11 Discussion.

If revision requested, proposed manuscript change:
Already patched.

Whether new experiment is needed: no

Risk if not addressed: Medium.

## Reviewer concern
Why excluded baselines are not reported.

Author response:
The artifact reports only implemented and real-data smoke-passed baselines to avoid fake, unstable, or unverified comparisons.

Manuscript location already addressing it:
Baselines section and feasibility audit.

If revision requested, proposed manuscript change:
Keep the bounded-comparison wording.

Whether new experiment is needed: no

Risk if not addressed: Medium.

## Reviewer concern
Why statistical tests are paired and scenario-level.

Author response:
Methods are compared under the same dataset, noise type, noise rate, graph beta, and seed, so paired scenario-level tests match the repeated evaluation design better than pooled tests.

Manuscript location already addressing it:
Experimental Design and statistics table.

If revision requested, proposed manuscript change:
Add a caveat that scenario settings are not independent operational deployments.

Whether new experiment is needed: no

Risk if not addressed: Medium.

## Reviewer concern
Why active view masks avoid artificial process/TI views.

Author response:
Views are enabled only when fields exist in the verified dataset contract; missing process or threat-intelligence fields are disabled rather than invented.

Manuscript location already addressing it:
Multi-view graph construction and dataset view policy report.

If revision requested, proposed manuscript change:
None.

Whether new experiment is needed: no

Risk if not addressed: Low.

## Reviewer concern
What future work covers.

Author response:
Future work covers faithful broader baselines, verified enterprise provenance data, analyst-in-the-loop validation, and sensitivity studies for evidence-score choices.

Manuscript location already addressing it:
Conclusion and Limitations.

If revision requested, proposed manuscript change:
Expand if reviewers request a roadmap.

Whether new experiment is needed: no

Risk if not addressed: Low.
