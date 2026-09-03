"""P10 preservation primitives. Historical results are immutable inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "reproducibility/history/p10_before_unsw_label_20260903"
MANIFEST = ROOT / "reports/p10_preservation_manifest.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preserve():
    if MANIFEST.exists():
        result = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert_preserved(result)
        return result
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")
    protected = {p: sha256(ROOT/p) for p in paths if p and (ROOT/p).is_file()
                 and p.startswith(("results/", "tables/", "reports/", "paper/", "reproducibility/"))
                 and "p10_" not in p}
    sources = ["configs/datasets.yaml", "src/data/loaders.py", "src/paper/p3_proxy_audit.py",
               "src/paper/p4_cl_graph_combo.py", "src/paper/p5_factor_ablation.py",
               "src/paper/p6_leakage_ablation.py", "src/metrics_canonical.py",
               "results/p4_cl_graph_combo_full_family_b_complete.csv",
               "results/p5_factor_ablation_full.csv", "results/p5_perclass_guard.csv",
               "results/p6_leakage_ablation_fixedtail_all.csv", "results/p9_canonical_tail_f1.csv",
               "results/p9_fine_final_protocol.partial.csv",
               "reports/p4_cl_graph_combo_full_family_b_complete.json",
               "reports/p5_factor_ablation_full.json", "reports/p9_tail_metric_unification.json",
               "tables/table_p5_family_a.csv", "tables/table_p5_family_c.csv",
               "tables/table_p6_leakage_ablation_fixedtail_all.csv"]
    for source in sources:
        if not (ROOT/source).is_file():
            raise FileNotFoundError(source)
    if HISTORY.exists():
        raise RuntimeError("History directory exists without manifest; do not overwrite")
    archive = {}
    for source in sources:
        destination = HISTORY/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/source, destination)
        archive[source] = sha256(destination)
    result = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "protected_sha256": protected, "archive_sha256": archive, "submission_ready": False}
    MANIFEST.write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert_preserved(result)
    return result


def assert_preserved(manifest=None):
    if manifest is None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed = [p for p, digest in manifest["protected_sha256"].items()
               if not (ROOT/p).exists() or sha256(ROOT/p) != digest]
    archive_changed = [p for p, digest in manifest["archive_sha256"].items()
                       if not (HISTORY/p).exists() or sha256(HISTORY/p) != digest]
    if changed or archive_changed:
        raise AssertionError({"protected_changed": changed, "archive_changed": archive_changed})


if __name__ == "__main__":
    m = preserve()
    print(json.dumps({"protected": len(m["protected_sha256"]), "archived": len(m["archive_sha256"])}))
