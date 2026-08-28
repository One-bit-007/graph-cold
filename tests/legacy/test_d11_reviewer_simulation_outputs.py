import json
from pathlib import Path


def test_d11_editor_and_reviewer_simulations_exist():
    report_dir = Path("reports/d11")
    editor = json.loads((report_dir / "editor_desk_reject_simulation.json").read_text(encoding="utf-8"))
    assert editor["probability_is_internal_heuristic_only"] is True
    assert editor["decision_simulation"] in {"desk_reject", "send_to_review", "borderline"}
    assert editor["top_acceptance_signals"]
    assert editor["top_desk_reject_risks"]

    patch_plan = json.loads((report_dir / "desk_reject_patch_plan.json").read_text(encoding="utf-8"))
    assert patch_plan["trigger"] == "pre_patch_desk_reject_probability_above_20_percent_internal_heuristic"
    assert len(patch_plan["patches"]) >= 5
    assert {item["status_after_generation"] for item in patch_plan["patches"]} == {"applied"}

    for name in [
        "reviewer_1_method_ml.md",
        "reviewer_2_security_soc.md",
        "reviewer_3_data_experiments.md",
    ]:
        text = (report_dir / name).read_text(encoding="utf-8")
        assert "Fatal risk:" in text
        assert "Major Concerns" in text
        assert "Questions to Authors" in text

    summary = json.loads((report_dir / "reviewer_attack_summary.json").read_text(encoding="utf-8"))
    assert summary["overall_likely_outcome"] == "send_to_review_with_revision_risk"
    assert summary["fatal_risks_detected"] == []


def test_d11_consensus_risk_map_has_required_clusters():
    risk_map = json.loads(Path("reports/d11/cross_reviewer_consensus_risk_map.json").read_text(encoding="utf-8"))
    clusters = {item["risk_cluster"] for item in risk_map["risks"]}
    required = {
        "baseline coverage",
        "FINE-style exclusion",
        "CESNET subset / ceiling effect",
        "SOC operational validity",
        "ERR interpretation",
        "Graph-CDM novelty",
        "Co-Teaching-lite naming",
        "Decoupling weakness",
        "references / related work",
        "submission declarations",
    }
    assert required.issubset(clusters)
    assert risk_map["safe_to_submit_after_patches"] is False


def test_d11_input_inventory_reads_required_d9_risk_pack():
    inventory = json.loads(Path("reports/d11/input_evidence_inventory.json").read_text(encoding="utf-8"))
    assert inventory["missing_required_inputs"] == []
    assert inventory["d9_risk_pack_v2_read"] is True
    assert "reports/d9/reviewer_risk_pack_v2.md" in inventory["result_hashes"]
