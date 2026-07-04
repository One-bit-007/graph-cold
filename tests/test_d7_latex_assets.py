import json
from pathlib import Path

from src.paper.d7_assemble import run_d7_assembly


def _ensure_outputs() -> None:
    if not Path("reports/d7/d7_generation_manifest.json").exists():
        run_d7_assembly()


def test_d7_elsevier_assets_exist_and_are_nonempty():
    _ensure_outputs()
    required = [
        "paper/elsevier/graph_cold_cas_realdata.tex",
        "paper/elsevier/graph_cold_cas_realdata.pdf",
        "paper/elsevier/references.bib",
        "paper/elsevier/elsarticle.cls",
        "paper/elsevier/build_elsevier.ps1",
        "paper/elsevier/build_elsevier.sh",
        "paper/elsevier/cover_letter_draft.md",
    ]
    for path in required:
        item = Path(path)
        assert item.exists(), path
        assert item.stat().st_size > 0, path

    assert Path("paper/elsevier/graph_cold_cas_realdata.pdf").stat().st_size > 1000


def test_d7_figures_and_tables_are_embedded_from_d6_assets():
    _ensure_outputs()
    for stem in [
        "fig2_macro_f1_vs_noise_rate",
        "fig3_err_retention",
        "fig4_ablation",
        "fig5_runtime_cost",
    ]:
        assert Path(f"paper/elsevier/figures/{stem}.pdf").exists()
        assert Path(f"paper/elsevier/figures/{stem}.png").exists()

    for name in [
        "table_1_dataset_protocol.tex",
        "table_2_main_summary.tex",
        "table_3_high_noise.tex",
        "table_4_ablation.tex",
        "table_5_statistical_tests.tex",
    ]:
        assert Path(f"paper/elsevier/tables/{name}").exists()


def test_d7_final_audit_matches_required_booleans():
    _ensure_outputs()
    audit = json.loads(Path("reports/d7/d7_final_audit.json").read_text(encoding="utf-8"))
    expected_keys = {
        "real_data_only",
        "uses_table_main_expanded",
        "no_maltls22_results",
        "no_optc_results",
        "no_synthetic_fallback_claims",
        "no_fake_baselines",
        "cesnet_reported_as_cesnet",
        "cesnet_subset_declared",
        "co_teaching_lite_named_correctly",
        "baseline_exclusions_declared",
        "figures_exist",
        "tables_exist",
        "pdf_compiles",
        "references_no_known_fake_entries",
        "submission_ready",
    }
    assert set(audit) == expected_keys
    for key, value in audit.items():
        if key == "submission_ready":
            assert value is False
        else:
            assert value is True, key


def test_d7_references_have_no_todo_or_unknown_entries():
    _ensure_outputs()
    refs = Path("paper/elsevier/references.bib").read_text(encoding="utf-8", errors="ignore")
    forbidden = ["TODO", "TBD", "unknown", "fake"]
    for term in forbidden:
        assert term.lower() not in refs.lower()
    assert "yang2026cold" in refs
    assert "cesnettlsyear22" in refs
