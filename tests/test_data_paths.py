from pathlib import Path

from src.data.paths import get_data_root, get_download_cache, resolve_dataset_path


def test_data_root_priority_cli_over_config(tmp_path: Path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "paths.yaml").write_text(
        'data_root: "Z:/ignored"\ndownload_cache: "Z:/ignored/_downloads"\nexternal_data_enabled: true\n',
        encoding="utf-8",
    )

    assert get_data_root("E:/graphcold-data", configs) == Path("E:/graphcold-data")
    assert get_download_cache("E:/graphcold-data", None, configs) == Path("Z:/ignored/_downloads")


def test_resolve_cesnet_external_dataset_path():
    path = resolve_dataset_path("cesnet_tls_year22", "E:/graphcold-data")

    assert path == Path("E:/graphcold-data") / "tls_alternative" / "cesnet_tls_year22"
