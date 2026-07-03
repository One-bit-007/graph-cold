from pathlib import Path

from src.data.audit import audit_dataset, build_readiness
from src.data.contracts import CICIDS2017_CONTRACT, MALTLS22_CONTRACT, OPTC_CONTRACT


def test_cicids_contract_lists_the_eight_required_csv_files():
    assert CICIDS2017_CONTRACT.name == "cicids2017"
    assert CICIDS2017_CONTRACT.label_column == "Label"
    assert len(CICIDS2017_CONTRACT.expected_files) == 8
    assert "Monday-WorkingHours.pcap_ISCX.csv" in CICIDS2017_CONTRACT.expected_files
    assert "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" in CICIDS2017_CONTRACT.expected_files
    assert CICIDS2017_CONTRACT.expected_view_support["process"] is False
    assert CICIDS2017_CONTRACT.expected_view_support["threat_intel"] is False


def test_cicids_missing_default_files_is_blocked():
    result = audit_dataset(CICIDS2017_CONTRACT)

    if Path(CICIDS2017_CONTRACT.root).exists() and result.expected_files_present:
        return
    assert result.ready_for_d5 is False
    assert result.missing_files or any("root does not exist" in reason for reason in result.blocking_reasons)


def test_maltls22_source_unverified_blocks_d5_even_if_contract_exists():
    result = audit_dataset(MALTLS22_CONTRACT)

    assert MALTLS22_CONTRACT.source_verified is False
    assert result.ready_for_d5 is False
    assert any("source is not verified" in reason for reason in result.blocking_reasons)


def test_optc_missing_events_csv_is_blocked_without_replacement_path():
    result = audit_dataset(OPTC_CONTRACT)

    if Path("data/optc/events.csv").exists():
        return
    assert result.ready_for_d5 is False
    assert "events.csv" in result.missing_files or any("root does not exist" in reason for reason in result.blocking_reasons)


def test_readiness_blocks_when_maltls_source_is_unverified():
    audits = {
        "cicids2017": audit_dataset(CICIDS2017_CONTRACT),
        "maltls22": audit_dataset(MALTLS22_CONTRACT),
        "optc": audit_dataset(OPTC_CONTRACT),
    }
    readiness = build_readiness(audits)

    assert readiness["d5_allowed"] is False
    assert readiness["d6_d7_allowed"] is False
    assert readiness["datasets"]["maltls22"]["source_verified"] is False
    assert readiness["datasets"]["maltls22"]["ready_for_d5"] is False

