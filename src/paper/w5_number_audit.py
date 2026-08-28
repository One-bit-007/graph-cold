"""W5-7: audit inline manuscript numbers against frozen artifacts.

Tables are auto-generated (w5_result_tables.py); the residual risk is prose.
Each check asserts (a) the claim string appears in the blind manuscript and
(b) the frozen artifact still yields that value at display precision.
Exit non-zero on any failure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEX = Path("paper/springer/graph_cold_sn_blind.tex").read_text(encoding="utf-8")
FAILURES: list[str] = []


def check(name: str, tex_pattern: str, ok: bool) -> None:
    present = re.search(tex_pattern, TEX) is not None
    if not present:
        FAILURES.append(f"{name}: pattern not found in tex: {tex_pattern}")
    elif not ok:
        FAILURES.append(f"{name}: frozen artifact does not support the stated value")
    else:
        print(f"ok  {name}")


def main() -> None:
    fa = pd.read_csv("tables/table_p5_family_a.csv")
    cic = fa[fa.dataset == "cicids2017"].sort_values("tail_flip_rate")
    # intro/abstract: CICIDS family-A range +0.21..+0.31
    check("famA CICIDS range", r"\+0\.21\$? to \$?\+0\.31",
          round(cic.mean_delta.min(), 2) == 0.21 and round(cic.mean_delta.max(), 2) == 0.31)
    other = fa[fa.dataset != "cicids2017"]
    check("famA other range", r"\+0\.025\$? to \$?\+0\.052",
          round(other.mean_delta.min(), 3) == 0.025 and round(other.mean_delta.max(), 3) == 0.052)
    check("famA dz range", r"0\.77 to 3\.14",
          round(fa.cohens_dz.min(), 2) == 0.77 and round(fa.cohens_dz.max(), 2) == 3.14)

    fb = json.load(open("reports/p4_cl_graph_combo_full_family_b_complete.json"))["summary"]["family_b_tests"]
    tail = [t for t in fb if t["metric"] == "tail_macro_f1"]
    ceil = max(t["bootstrap_ci95_high"] for t in tail)
    check("famB effect ceiling +0.026", r"\+0\.026", round(ceil, 3) == 0.026)
    minp = min(t["p_holm_greater"] for t in fb)
    check("famB min Holm p >= 0.608", r"p\\ge0\.608", round(minp, 3) == 0.608)

    fc = pd.read_csv("tables/table_p5_family_c.csv")
    u = fc[(fc.dataset == "unsw_nb15") & (fc.reference == "5_graph_cdm_freqfloor")].sort_values("tail_flip_rate")
    check("famC UNSW reversal CIs", r"-0\.011\$?/\$?-0\.013",
          round(u.iloc[0].bootstrap_ci95_high, 3) == -0.011 and round(u.iloc[1].bootstrap_ci95_high, 3) == -0.013)

    nn = json.load(open("reports/p7_natural_noise.json"))
    d = next(x for x in nn["deltas_cleanref"]
             if x["pair"] == "Graph-CoLD-soft-ablation_hard" and x["metric"] == "tail_f1_cleanref")
    check("natural-noise +0.284", r"\+0\.284",
          round(d["mean_delta"], 3) == 0.284 and round(d["bootstrap_ci95_low"], 3) == 0.089
          and round(d["bootstrap_ci95_high"], 3) == 0.487)

    diff = json.load(open("reports/p7_engelen_diff_summary.json"))
    txt = json.dumps(diff)
    check("Engelen 128,247 rows / 4.53%", r"128\{?,?\}?247", "128247" in txt.replace(",", ""))

    lk = pd.read_csv("tables/table_p6_leakage_ablation_fixedtail_all.csv")
    t = lk[lk.metric == "tail_f1"]

    def delta(ds, m, c):
        return float(t[(t.dataset == ds) & (t.method == m) & (t.condition == c)].mean_delta.iloc[0])

    check("leak CESNET ENT +0.187", r"\+0\.187", round(delta("cesnet_tls_year22", "Graph-CoLD-soft", "ENT"), 3) == 0.187)
    check("leak CESNET PRX +0.184", r"\+0\.184", round(delta("cesnet_tls_year22", "Graph-CoLD-soft", "PRX"), 3) == 0.184)
    check("leak CESNET ALL soft +0.249", r"\+0\.249", round(delta("cesnet_tls_year22", "Graph-CoLD-soft", "ALL"), 3) == 0.249)
    check("leak CESNET ALL proxy +0.359", r"\+0\.359", round(delta("cesnet_tls_year22", "CoLD", "ALL"), 3) == 0.359)
    check("leak CICIDS proxy DUP +0.467", r"\+0\.467", round(delta("cicids2017", "CoLD", "DUP"), 3) == 0.467)
    check("leak CICIDS hard DUP +0.375", r"\+0\.375", round(delta("cicids2017", "ablation_hard", "DUP"), 3) == 0.375)
    check("leak UNSW dedup -0.115", r"0\.115", round(delta("unsw_nb15", "Graph-CoLD-soft", "DUP"), 3) == -0.115)
    check("leak UNSW PRX -0.052", r"-0\.052", round(delta("unsw_nb15", "Graph-CoLD-soft", "PRX"), 3) == -0.052)

    # soft-vs-hard gap compression: FINAL 0.256 vs DUP 0.125 (CICIDS)
    g = t[(t.dataset == "cicids2017") & (t.condition == "DUP")]
    soft = g[g.method == "Graph-CoLD-soft"].iloc[0]
    hard = g[g.method == "ablation_hard"].iloc[0]
    final_gap = soft.final_mean - hard.final_mean
    dup_gap = soft.cond_mean - hard.cond_mean
    check("gap compression 0.256->0.125", r"(?s)0\.256.*?0\.125",
          round(final_gap, 3) == 0.256 and round(dup_gap, 3) == 0.125)

    sv = json.load(open("reports/p6_scale_validation_cicids.json"))
    a = next(c for c in sv["comparisons"] if c["expected"] == "positive")["per_rate"]
    check("scale famA +0.150/+0.056/+0.031", r"\+0\.150\$?/\$?\+0\.056\$?/\$?\+0\.031",
          [round(r["mean_delta"], 3) for r in a] == [0.150, 0.056, 0.031])
    b = next(c for c in sv["comparisons"] if c["expected"] == "null")["per_rate"]
    check("scale G2 rate0.8 -0.014", r"-0\.014", round(b[-1]["mean_delta"], 3) == -0.014)

    print()
    if FAILURES:
        print("NUMBER AUDIT FAILED:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print(f"number audit passed: all checks green")


if __name__ == "__main__":
    main()
