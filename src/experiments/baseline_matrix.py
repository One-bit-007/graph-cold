"""Neutral CLI wrapper for the expanded baseline matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiments.d5_baseline_expansion import run_d5_baseline_expansion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--configs", default="configs")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    result = run_d5_baseline_expansion(
        out_dir=Path(args.out),
        configs_dir=Path(args.configs),
        reports_dir=Path(args.reports),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
