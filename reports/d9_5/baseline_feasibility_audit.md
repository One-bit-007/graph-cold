# D9.5 Baseline Feasibility Audit

## Decoupling
- include_in_this_patch: True
- target: tabular two-classifier disagreement update
- faithfulness: faithful to standard disagreement-update mechanism
- reason: classic noisy-label disagreement-update baseline; feasible on tabular CICIDS/CESNET

## FINE
- include_in_this_patch: False
- target: none
- faithfulness: excluded full method
- reason: full original implementation is not reproduced here

## FINE-style
- include_in_this_patch: True
- target: standardized feature PCA plus class-wise eigenvector filtering
- faithfulness: style/approximate
- reason: representation/eigenvector filtering baseline using standardized feature projections
- warning: not full FINE unless exact original implementation is reproduced

## MCRe
- include_in_this_patch: False
- target: excluded
- faithfulness: not implemented
- reason: no verified real-data compatible implementation in current artifact

## MORSE
- include_in_this_patch: False
- target: excluded
- faithfulness: not implemented
- reason: no verified real-data compatible implementation in current artifact

## Flash
- include_in_this_patch: False
- target: excluded
- faithfulness: not implemented
- reason: no verified real-data compatible implementation in current artifact

## Argus
- include_in_this_patch: False
- target: excluded
- faithfulness: not implemented
- reason: no verified real-data compatible implementation in current artifact
