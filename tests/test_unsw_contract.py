from src.data.contracts import DATASET_CONTRACTS, UNSW_NB15_CONTRACT, USTC_TFC2016_CONTRACT


def test_unsw_contract_is_verified_backup_dataset():
    assert DATASET_CONTRACTS["unsw_nb15"] is UNSW_NB15_CONTRACT
    assert UNSW_NB15_CONTRACT.source_verified is True
    assert UNSW_NB15_CONTRACT.reported_as == "UNSW-NB15"
    assert UNSW_NB15_CONTRACT.required_any_columns["label"]
    assert UNSW_NB15_CONTRACT.expected_view_support["ip"] is True
    assert UNSW_NB15_CONTRACT.expected_view_support["temporal"] is True
    assert UNSW_NB15_CONTRACT.expected_view_support["process"] is False


def test_ustc_contract_is_candidate_only():
    assert DATASET_CONTRACTS["ustc_tfc2016"] is USTC_TFC2016_CONTRACT
    assert USTC_TFC2016_CONTRACT.source_verified is False
    assert USTC_TFC2016_CONTRACT.reported_as == "USTC-TFC2016"
