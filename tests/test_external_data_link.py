from pathlib import Path

from src.data.paths import apply_data_root_to_config


def test_apply_data_root_rewrites_only_requested_dataset():
    cfg = {
        "cesnet_tls_year22": {"path": "data/tls_alternative/cesnet_tls_year22", "label_col": "service"},
        "cicids2017": {"path": "data/cicids2017", "label_col": "Label"},
    }

    out = apply_data_root_to_config(cfg, "cesnet_tls_year22", "E:/graphcold-data")

    assert out["cesnet_tls_year22"]["path"] == str(Path("E:/graphcold-data") / "tls_alternative" / "cesnet_tls_year22")
    assert out["cicids2017"]["path"] == "data/cicids2017"
