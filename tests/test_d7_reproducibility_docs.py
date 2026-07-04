import hashlib
from pathlib import Path

from src.paper.d7_assemble import run_d7_assembly


def _ensure_outputs() -> None:
    if not Path("reproducibility/README_realdata.md").exists():
        run_d7_assembly()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_d7_reproducibility_readme_records_data_roots_and_hashes():
    _ensure_outputs()
    text = Path("reproducibility/README_realdata.md").read_text(encoding="utf-8")

    assert "data/cicids2017" in text
    assert r"E:\graphcold-data\tls_alternative\cesnet_tls_year22" in text
    assert "CICIDS-2017 postfilter11" in text
    assert "CESNET-TLS-Year22 postfilter25" in text
    assert "MALTLS-22 and OpTC are not part of the formal" in text

    for path in [
        "results/table_main_expanded.csv",
        "results/table_baseline_expansion.csv",
        "results/stat_tests_baseline_expansion.json",
        "reports/realdata_readiness_report.json",
    ]:
        assert _sha256(path) in text


def test_d7_reproducibility_scripts_use_gates_and_do_not_download_data():
    _ensure_outputs()
    d5 = Path("reproducibility/run_d5_realdata.ps1").read_text(encoding="utf-8")
    d6 = Path("reproducibility/run_d6_tables_figures.ps1").read_text(encoding="utf-8")
    combined = f"{d5}\n{d6}"

    assert "python -m src.data.audit" in d5
    assert "python scripts/check_data_ready.py" in d5
    assert "python -m src.experiments.d5 --out results --configs configs" in d5
    assert "python -m src.experiments.d5_baseline_expansion" in d5
    assert "python -m src.paper.d6_prep" in d6
    assert "python -m src.paper.d7_assemble" in d6
    assert "build_elsevier.ps1" in d6

    for term in ["aria2c", "wget", "curl ", "download=1", "MachineLearningCSV.zip"]:
        assert term.lower() not in combined.lower()


def test_d7_readiness_reconciliation_resolves_d6_d7_semantics():
    _ensure_outputs()
    text = Path("reports/d7/d7_readiness_reconciliation.md").read_text(encoding="utf-8")
    assert "Source of truth" in text
    assert "D7 assembly allowed: true" in text
    assert "D7 artifacts existed before this D7 step: false" in text
    assert "Submission ready: false" in text
