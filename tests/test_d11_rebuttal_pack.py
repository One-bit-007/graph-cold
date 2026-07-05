import json
from pathlib import Path


def test_d11_rebuttal_pack_exists_and_covers_required_concerns():
    report_dir = Path("reports/d11")
    full = (report_dir / "rebuttal_prewrite_full.md").read_text(encoding="utf-8")
    structured = json.loads((report_dir / "rebuttal_prewrite_structured.json").read_text(encoding="utf-8"))

    assert structured["count"] >= 15
    required = [
        "Why Graph-CDM is not just generic reweighting.",
        "Why label-space diagnostic is used instead of embedding distance.",
        "Why FINE-style is excluded from formal results.",
        "Why Decoupling underperforms.",
        "Why Co-Teaching-lite is named lite.",
        "Why MALTLS-22 is omitted.",
        "Why OpTC is omitted.",
        "Why CESNET subset is acceptable.",
        "Why ERR=1.0 is not suspicious.",
        "Why compression ratio is an operational proxy.",
        "Why Graph-CoLD still matters when CESNET Macro-F1 is near ceiling.",
        "Why excluded baselines are not reported.",
        "Why statistical tests are paired and scenario-level.",
        "Why active view masks avoid artificial process/TI views.",
        "What future work covers.",
    ]
    concerns = {item["reviewer_concern"] for item in structured["concerns"]}
    assert set(required).issubset(concerns)
    assert "Whether new experiment is needed: no" in full
