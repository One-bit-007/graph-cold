from pathlib import Path

from src.data.contracts import CESNET_TLS_YEAR22_CONTRACT, DATASET_CONTRACTS


def test_cesnet_contract_uses_tls_alternative_path_not_maltls():
    assert CESNET_TLS_YEAR22_CONTRACT.name == "cesnet_tls_year22"
    assert CESNET_TLS_YEAR22_CONTRACT.root == "data/tls_alternative/cesnet_tls_year22"
    assert "maltls22" not in CESNET_TLS_YEAR22_CONTRACT.root.lower()
    assert CESNET_TLS_YEAR22_CONTRACT.replacement_for == "maltls22"
    assert CESNET_TLS_YEAR22_CONTRACT.replacement_name_must_be_reported is True
    assert DATASET_CONTRACTS["cesnet_tls_year22"] is CESNET_TLS_YEAR22_CONTRACT


def test_cesnet_view_policy_disables_process_and_threat_intel():
    support = CESNET_TLS_YEAR22_CONTRACT.expected_view_support

    assert support["ip"] is True
    assert support["temporal"] is True
    assert support["process"] is False
    assert support["threat_intel"] is False


def test_download_docs_report_true_cesnet_name_not_maltls_alias():
    docs = (Path(__file__).resolve().parents[1] / "docs" / "DATASETS.md").read_text(encoding="utf-8")

    assert "CESNET-TLS-Year22" in docs
    assert "cesnet_tls_year22" in docs
    assert "never as MALTLS-22" in docs
