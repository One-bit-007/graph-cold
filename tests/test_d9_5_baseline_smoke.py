import numpy as np

from src.baselines.base import BaselineResult, array_hash
from src.experiments.d9_5_baseline_smoke import _write_feasibility_audit
from src.experiments.d9_5_baseline_common import pass_smoke_row


def test_feasibility_audit_includes_only_decoupling_and_fine_style(tmp_path):
    audit = _write_feasibility_audit(tmp_path)
    methods = {item["method"]: item for item in audit["methods"]}

    assert methods["Decoupling"]["include_in_this_patch"] is True
    assert methods["FINE-style"]["include_in_this_patch"] is True
    assert methods["FINE"]["include_in_this_patch"] is False
    assert methods["MCRe"]["include_in_this_patch"] is False
    assert (tmp_path / "baseline_feasibility_audit.json").exists()
    assert (tmp_path / "baseline_feasibility_audit.md").exists()


def test_smoke_gate_blocks_failed_or_leaky_rows():
    noisy = np.array([0, 1, 1, 0], dtype=np.int64)
    result = BaselineResult(
        method="Decoupling",
        method_family="decoupling",
        implementation_status="implemented_smoke_passed",
        y_pred=np.array([0, 1]),
        proba=np.array([[0.8, 0.2], [0.1, 0.9]]),
        weights=np.ones(4),
        retained_mask=np.ones(4, dtype=bool),
        details={"training_label_hash": array_hash(noisy), "disagreement_fraction": 0.2, "update_fraction": 0.1},
    )
    row = {
        "macro_f1": 0.7,
        "fpr": 0.1,
        "fnr": 0.2,
        "err": 0.5,
        "err_tail": 0.5,
        "err_final": 0.5,
        "compression_ratio": 0.8,
        "runtime_sec": 0.1,
        "memory_mb": 1.0,
        "retained_fraction": 0.5,
    }

    ok, reasons = pass_smoke_row(row, result, noisy)
    assert ok is True
    assert reasons == []

    result.details["training_label_hash"] = array_hash(np.array([1, 1, 1, 1], dtype=np.int64))
    ok, reasons = pass_smoke_row(row, result, noisy)
    assert ok is False
    assert "not_trained_on_noisy_y_train" in reasons
