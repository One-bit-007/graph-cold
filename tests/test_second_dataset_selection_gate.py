from src.experiments.second_dataset_selection import select_second_dataset


def test_selection_prefers_cesnet_when_ready():
    report = select_second_dataset(
        {
            "datasets": {
                "cicids2017": {"ready_for_d5_component": True},
                "cesnet_tls_year22": {"ready_for_d5_component": True},
                "unsw_nb15": {"ready_for_d5_component": True},
            }
        }
    )

    assert report["d5_allowed"] is True
    assert report["selected_second_dataset"] == "cesnet_tls_year22"
    assert report["d5_scope"] == ["cicids2017", "cesnet_tls_year22"]


def test_selection_uses_unsw_when_cesnet_not_ready():
    report = select_second_dataset(
        {
            "datasets": {
                "cicids2017": {"ready_for_d5_component": True},
                "cesnet_tls_year22": {"ready_for_d5_component": False, "blocking_reasons": ["download incomplete"]},
                "unsw_nb15": {"ready_for_d5_component": True},
            }
        }
    )

    assert report["d5_allowed"] is True
    assert report["selected_second_dataset"] == "unsw_nb15"
    assert report["paper_claims_changed"] is True


def test_selection_blocks_when_no_second_dataset_ready():
    report = select_second_dataset({"datasets": {"cicids2017": {"ready_for_d5_component": True}}})

    assert report["d5_allowed"] is False
    assert report["d5_scope"] == []
    assert "No verified second dataset is ready" in report["blocking_reasons"]


def test_selection_does_not_use_ustc_while_candidate_only():
    report = select_second_dataset(
        {
            "datasets": {
                "cicids2017": {"ready_for_d5_component": True},
                "ustc_tfc2016": {"ready_for_d5_component": True, "candidate_only": True},
            }
        }
    )

    assert report["d5_allowed"] is False
    assert report["selected_second_dataset"] is None
