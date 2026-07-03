"""UNSW-NB15 preparation helper.

The helper prints/verifies acquisition instructions only. It does not fabricate
rows, convert unrelated datasets, or run D5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


OFFICIAL_PAGE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"


def run(mode: str, out: str | Path = "data/unsw_nb15") -> dict:
    out_path = Path(out)
    result = {
        "dataset": "unsw_nb15",
        "reported_as": "UNSW-NB15",
        "mode": mode,
        "official_page": OFFICIAL_PAGE,
        "target_path": str(out_path),
        "download_success": False,
        "audit_passed": False,
        "d5_full_matrix_invoked": False,
        "instructions": [
            "Download the official UNSW-NB15 CSV files and feature list from the official UNSW page.",
            "Place CSV files under the target path without committing raw data.",
            "Run: python -m src.data.audit --dataset unsw_nb15",
            "Run: python -m src.data.unsw_policy --out reports",
        ],
    }
    if mode == "manifest":
        out_path.mkdir(parents=True, exist_ok=True)
        manifest = out_path / "UNSW_NB15_MANIFEST.txt"
        manifest.write_text("\n".join(result["instructions"]) + "\n", encoding="utf-8")
        result["manifest"] = str(manifest)
    elif mode != "instructions":
        raise ValueError("UNSW-NB15 helper supports only 'instructions' and 'manifest' modes.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["instructions", "manifest"], default="instructions")
    parser.add_argument("--out", default="data/unsw_nb15")
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.out), indent=2))


if __name__ == "__main__":
    main()
