# Reviewer 1 - Methodology / ML

R1 summary
The paper is technically coherent and its label-space Graph-CDM is more specific than generic sample reweighting, but the reviewer will press for sharper novelty boundaries and careful baseline language.

Likely recommendation: minor-to-major revision
Score 0-10: 7
Fatal risk: no

## Major Concerns
- Graph-CDM novelty could be seen as graph-structured reweighting unless label-space terms are emphasized.
- Evidence score is heuristic and needs transparent interpretation.
- FINE-style failure must be described as an exclusion gate rather than hidden negative evidence.
- Decoupling should be described as a faithful disagreement-update baseline but not a strong neural comparator.

## Minor Concerns
- Co-Teaching-lite naming must remain explicit.
- Extreme p-values should be interpreted through paired scenario design, not as broad certainty.

## Questions to Authors
- Why diagnose in label space instead of embedding space?
- How sensitive is the method to evidence-score choices?
- Does Graph-CDM reduce to CoLD when graph/evidence terms are removed?
