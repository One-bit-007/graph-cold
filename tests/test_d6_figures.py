import json
from pathlib import Path

from src.paper.d6_prep import run_d6_realdata_prep


def _ensure_d6_outputs():
    if not Path("reports/d6/d6_generation_manifest.json").exists():
        run_d6_realdata_prep()


def test_required_figures_exist_as_png_and_pdf():
    _ensure_d6_outputs()
    expected = [
        "figures/fig2_macro_f1_vs_noise_rate.png",
        "figures/fig2_macro_f1_vs_noise_rate.pdf",
        "figures/fig3_err_retention.png",
        "figures/fig3_err_retention.pdf",
        "figures/fig4_ablation.png",
        "figures/fig4_ablation.pdf",
        "figures/fig5_runtime_cost.png",
        "figures/fig5_runtime_cost.pdf",
    ]
    for file_name in expected:
        path = Path(file_name)
        assert path.exists(), file_name
        assert path.stat().st_size > 1000, file_name


def test_figures_are_declared_from_real_expanded_results():
    _ensure_d6_outputs()
    manifest = json.loads(Path("reports/d6/d6_generation_manifest.json").read_text(encoding="utf-8"))

    assert manifest["source_csv"] == "results/table_main_expanded.csv"
    assert "results/table_main_expanded.csv" in Path("reports/d6/d6_statistical_narrative.md").read_text(encoding="utf-8")
    for figure in manifest["figures"]:
        assert Path(figure).exists(), figure
