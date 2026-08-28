import json

import numpy as np
import pandas as pd

from src.paper import p4_cl_graph_combo as p4


def test_primary_protocols_are_masked_and_use_strongest_available_split():
    assert p4.PROTOCOLS["cicids2017"]["mask"] == "tier1"
    assert p4.PROTOCOLS["cicids2017"]["split"] == "temporal"
    assert p4.PROTOCOLS["cesnet_tls_year22"]["mask"] == "tier2"
    assert p4.PROTOCOLS["cesnet_tls_year22"]["split"] == "temporal_entity"
    assert p4.PROTOCOLS["unsw_nb15"]["mask"] == "tier2"


def test_issue_fusion_set_semantics():
    cl_issue = np.asarray([True, True, False, False])
    graph_issue = np.asarray([True, False, True, False])
    intersection_keep = ~(cl_issue & graph_issue)
    union_keep = ~(cl_issue | graph_issue)
    np.testing.assert_array_equal(intersection_keep, [False, True, True, True])
    np.testing.assert_array_equal(union_keep, [False, False, False, True])


def test_g2_primary_and_guardrail_are_preregistered():
    assert p4.PRIMARY_METHOD == "CL+Graph-soft"
    assert p4.REFERENCE_METHOD == "Confident-Learning"
    assert p4.MACRO_NI_DELTA == 0.01
    assert p4.CLASS_F1_GUARD == 0.05
    assert p4.FAMILY_B_SIZE == 18


def test_class_guard_uses_classwise_seed_means():
    rows = []
    for seed in range(3):
        rows.extend([
            {
                "seed": seed,
                "method": p4.REFERENCE_METHOD,
                "per_class_f1_json": json.dumps({"0": 0.8, "1": 0.6}),
            },
            {
                "seed": seed,
                "method": p4.PRIMARY_METHOD,
                "per_class_f1_json": json.dumps({"0": 0.76, "1": 0.7}),
            },
        ])
    result = p4._class_guard(pd.DataFrame(rows), p4.PRIMARY_METHOD, p4.REFERENCE_METHOD, 0.05)
    assert result["passed"] is True
    assert result["worst_class"] == "0"
    assert np.isclose(result["worst_mean_delta"], -0.04)
