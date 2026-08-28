# Protocol History

This file separates historical experiment settings from the frozen P3 protocol.

| Protocol family | Train/test audit window | Seeds | Status |
| --- | ---: | ---: | --- |
| D5.x / P2d | 1,500 / 4,000 | mixed early subsets | historical only; not a final-result source |
| P2f / P3 | 10,000 / 5,000 | 0--9 | frozen main protocol |

The three-seed set `[0,1,2]` is retained only for screening. Results promoted to
the final matrix must use seeds 0--9 and the frozen 0.4/0.6/0.8 asymmetric-noise
rates. Historical outputs must not be pooled with P3 results.
