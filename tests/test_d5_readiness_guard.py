import json
from pathlib import Path

import pytest

from src.experiments.d5 import _readiness_guard, run_d5_experiments


def test_d5_guard_blocks_before_creating_result_outputs(tmp_path: Path):
    configs = tmp_path / "configs"
    reports = tmp_path / "reports"
    configs.mkdir()
    reports.mkdir()
    (reports / "second_dataset_selection_gate.json").write_text(
        json.dumps({"d5_allowed": False, "d5_scope": [], "blocking_reasons": ["No verified second dataset is ready"]}),
        encoding="utf-8",
    )
    out = tmp_path / "results"

    with pytest.raises(RuntimeError, match="D5 readiness guard blocked"):
        run_d5_experiments(out_dir=out, configs_dir=configs)

    assert not out.exists()


def test_d5_guard_rejects_maltls_and_optc_scope(tmp_path: Path):
    configs = tmp_path / "configs"
    reports = tmp_path / "reports"
    configs.mkdir()
    reports.mkdir()
    (reports / "second_dataset_selection_gate.json").write_text(
        json.dumps({"d5_allowed": True, "d5_scope": ["cicids2017", "maltls22", "optc"]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="MALTLS-22 and OpTC are not allowed"):
        _readiness_guard(configs)
