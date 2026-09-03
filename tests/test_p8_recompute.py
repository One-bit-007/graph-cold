"""Read-only acceptance checks for the P8 artifact handoff; no model training."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

ARCHIVE = Path("reproducibility/history/p8_before_f3_20260903")
RESULT = Path("results/p5_factor_ablation_full.csv")


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


@pytest.mark.parametrize("dataset", ["cicids2017", "unsw_nb15"])
def test_unaffected_shard_bytes_and_original_main_tokens(dataset):
    rel = Path(f"results/p5_factor_ablation_full_{dataset}.csv")
    assert rel.read_bytes() == (ARCHIVE / rel).read_bytes()
    with RESULT.open(newline="", encoding="utf-8") as f:
        new = [r for r in csv.DictReader(f) if r["dataset"] == dataset]
    with (ARCHIVE / RESULT).open(newline="", encoding="utf-8") as f:
        old = [r for r in csv.DictReader(f) if r["dataset"] == dataset]
    assert len(new) == len(old) == 120
    assert old == [{k: r[k] for k in old[0]} for r in new]


def test_complete_grid_and_corrected_target():
    data = read(RESULT)
    assert len(data) == 360
    keys = ["dataset", "cell", "seed", "tail_flip_rate"]
    assert not data.duplicated(keys).any()
    for _, group in data.groupby(["dataset", "cell"]):
        assert set(zip(group.seed, group.tail_flip_rate)) == {(s, r) for s in range(10) for r in [.4, .6, .8]}
    new = data[data.dataset == "cesnet_tls_year22"]
    assert len(new) == 120 and new.noise_target.eq(11).all()
    # Never backfill run-time metadata into old runs as if it had been recorded.
    assert data[data.dataset != "cesnet_tls_year22"].noise_target.isna().all()


@pytest.mark.parametrize("family,size", [("a", 9), ("c", 18)])
def test_full_family_against_independent_scipy_statsmodels(family, size):
    data = read(RESULT)
    table = read(f"tables/table_p5_family_{family}.csv")
    assert len(table) == size
    raw_ps = []
    for row in table.itertuples(index=False):
        part = data[(data.dataset == row.dataset) & (data.tail_flip_rate == row.tail_flip_rate)]
        paired = part.pivot(index="seed", columns="cell", values="tail_macro_f1")
        difference = (paired[row.method] - paired[row.reference]).to_numpy()
        assert len(difference) == 10 and np.isfinite(difference).all()
        np.testing.assert_allclose(difference.mean(), row.mean_delta, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(difference.mean() / difference.std(ddof=1), row.cohens_dz, rtol=1e-12)
        p = max(float(stats.ttest_1samp(difference, 0, alternative="greater").pvalue), 1e-12)
        raw_ps.append(p)
        np.testing.assert_allclose(p, row.p_value_greater, rtol=1e-10, atol=1e-15)
        boot = np.random.default_rng(42).choice(difference, size=(2000, 10), replace=True).mean(axis=1)
        np.testing.assert_allclose(np.quantile(boot, [.025, .975]),
                                   [row.bootstrap_ci95_low, row.bootstrap_ci95_high], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(multipletests(raw_ps, method="holm")[1], table.p_holm_greater, rtol=1e-10, atol=1e-15)


def test_guard_complete_and_old_ids_rows_preserved():
    old = read(ARCHIVE / "results/p5_perclass_guard.csv")
    new = read("results/p5_perclass_guard.csv")
    assert len(new) == 180
    assert not new.duplicated(["dataset", "method", "seed", "tail_flip_rate"]).any()
    assert new[new.dataset == "cesnet_tls_year22"].noise_target.eq(11).all()
    pd.testing.assert_frame_equal(old[old.dataset != "cesnet_tls_year22"].reset_index(drop=True),
                                  new.loc[new.dataset != "cesnet_tls_year22", old.columns].reset_index(drop=True),
                                  check_exact=True)
    assert len(read("tables/table_p5_perclass_guard.csv")) == 9


def test_handoff_records_decisions_and_submission_hold():
    audit = json.loads(Path("reports/p8_cesnet_f3_recompute.json").read_text(encoding="utf-8"))
    assert audit["family_a_test_count"] == 9
    assert audit["family_c_test_count"] == 18
    assert audit["submission_ready"] is False
    assert "decision_changes" in audit and "guard_decision_changes" in audit
    if audit["decision_changes"] or audit["guard_decision_changes"]:
        assert audit["status"].startswith("STOP_")
