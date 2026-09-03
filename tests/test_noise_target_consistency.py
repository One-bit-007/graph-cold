"""Cross-runner protocol-consistency guards (root cause of the F-3 CESNET bug).

The number audit and claim gates verify transcription; these verify that every
runner injects noise through the *same* frozen F-3 target function, so a runner
can never silently diverge (as p5 did with `benign_class or 0`, which fell back
to label 0 on CESNET instead of the frozen 11).
"""
from __future__ import annotations

import inspect

import numpy as np

from src.paper import p3_proxy_audit as p3
from src.paper import p4_cl_graph_combo as p4
from src.paper import p5_factor_ablation as p5


def test_frozen_cesnet_target_is_11() -> None:
    y = np.arange(25)  # label 11 present
    target, _ = p3._noise_target("cesnet_tls_year22", y, {"class_names": [str(i) for i in range(25)]})
    assert target == 11, "frozen F-3 CESNET target must be 11 (microsoft-diagnostic)"


def test_cesnet_target_raises_if_absent() -> None:
    y = np.arange(10)  # label 11 missing
    try:
        p3._noise_target("cesnet_tls_year22", y, {})
    except ValueError:
        return
    raise AssertionError("CESNET target must raise when label 11 is absent, not fall back")


def test_p5_and_p4_use_the_frozen_target_function() -> None:
    # p5/p4 scenario builders must call p3._noise_target, not a local `benign_class or 0` fallback.
    p5_src = inspect.getsource(p5._prepare_scenario)
    p4_src = inspect.getsource(p4._sample)
    for name, src in (("p5", p5_src), ("p4", p4_src)):
        assert "_noise_target(" in src, f"{name} must inject via p3._noise_target"
        assert "benign_class) or 0" not in src and "benign_class or 0" not in src, \
            f"{name} still contains the buggy `benign_class or 0` fallback"


def test_ids_target_matches_declared_benign() -> None:
    y = np.arange(11)
    target, _ = p3._noise_target("cicids2017", y, {"benign_class": 0, "class_names": [str(i) for i in range(11)]})
    assert target == 0


def test_cell_row_persists_noise_target() -> None:
    # noise_target must land in the output row so the frozen target is auditable in the CSV.
    p5_src = inspect.getsource(p5._cell_row)
    guard_src = inspect.getsource(p5)  # module-level; guard runner lives in p5_perclass_guard
    assert '"noise_target"' in p5_src, "p5._cell_row must persist noise_target"
    import src.paper.p5_perclass_guard as guard
    assert '"noise_target"' in inspect.getsource(guard.run), "guard run() must persist noise_target"


def test_guard_calls_prepare_scenario_with_dataset_name() -> None:
    import src.paper.p5_perclass_guard as guard
    src = inspect.getsource(guard.run)
    assert "_prepare_scenario(ds, dataset," in src, "guard must pass dataset_name to _prepare_scenario"
