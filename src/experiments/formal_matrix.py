"""Neutral CLI wrapper for the formal real-data matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiments.d5 import run_d5_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--configs", default="configs")
    args = parser.parse_args()
    result = run_d5_experiments(out_dir=Path(args.out), configs_dir=Path(args.configs))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
