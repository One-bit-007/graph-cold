"""Evaluation / results aggregation entry point.

Runs the full experiment matrix and writes tables/figures to experiments/results.

Usage (to be wired by Codex):
    python -m src.eval --matrix full   # 3 datasets x noise x baselines + ablations
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="full")
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args()
    raise NotImplementedError("TODO(Codex): run matrix, dump csv + png, t-tests.")


if __name__ == "__main__":
    main()
