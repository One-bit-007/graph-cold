"""Training entry point.

Pipeline
--------
1. load dataset + inject noise (configs/noise.yaml)
2. build five-view heterogeneous temporal graph
3. stage-1: self-supervised multi-view representation learning
4. stage-2: Graph-CDM -> evidence-preserving soft weights
5. train weighted robust classifier
6. produce alert priority ranking

Usage (to be wired by Codex):
    python -m src.train --dataset maltls22 --noise symmetric --ratio 0.4 \
        --config configs/model.yaml --seed 0
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--noise", default="symmetric")
    parser.add_argument("--ratio", type=float, default=0.0)
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--method", default="graph_cold",
                        choices=["graph_cold", "cold", "ablation_hard"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError("TODO(Codex): wire the 6-step pipeline.")


if __name__ == "__main__":
    main()
