import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.metrics_canonical import canonical_tail_f1_from_perclass as canonical
from src.paper import p2e_salvage as p2e, p3_proxy_audit as p3
from src.paper.p9_tail_metric_unification import validate_grid


def test_full_set_false_positives_are_counted():
    y = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([1, 1, 0, 0, 1, 1])
    scores = dict(zip(map(str, np.unique(y)), f1_score(y, pred, average=None)))
    assert canonical(json.dumps(scores), [1]) == pytest.approx(2/3)
    assert canonical(scores, [1]) == p3._tail_f1(y, pred, np.array([1]))
    assert p2e._tail_macro_f1(y, pred, np.array([1])) == 1.0


def test_absent_tail_matches_p5():
    assert np.isnan(canonical({"0": 0.9}, [1]))
    assert canonical({"0": 0.9, "1": 0.2}, [1, 1, 2]) == 0.2


@pytest.mark.parametrize("values", [{"1": float("nan")}, {"1": 1.01}, {"1": -0.1}, [1, 2], {"1": 0.5, "01": 0.7}])
def test_invalid_scores_are_rejected(values):
    with pytest.raises(ValueError):
        canonical(values, [1])


def test_incomplete_grid_cannot_be_averaged():
    frame = pd.DataFrame([{"dataset": "cicids2017", "seed": 0, "tail_flip_rate": 0.4, "method": "M"}])
    with pytest.raises(AssertionError, match="Grid mismatch"):
        validate_grid(frame, ["M"])
