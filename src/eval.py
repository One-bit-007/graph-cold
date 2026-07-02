"""Evaluation / results aggregation entry point.

Runs the full experiment matrix and writes tables/figures to experiments/results.

Usage (to be wired by Codex):
    python -m src.eval --matrix full   # 3 datasets x noise x baselines + ablations
"""
from __future__ import annotations

import argparse

from src.experiments.d5 import run_d5_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="full")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    if args.matrix not in {"full", "d5"}:
        raise ValueError("--matrix must be one of {'full', 'd5'} for the D5 runner.")
    summary = run_d5_experiments(args.out)
    print(summary)


if __name__ == "__main__":
    main()
