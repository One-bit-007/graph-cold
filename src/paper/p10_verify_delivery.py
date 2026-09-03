"""Read-only P10 evidence checks; write only a new validation receipt."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.metrics_canonical import canonical_tail_f1_from_perclass
from src.paper.p10_support import ROOT, assert_preserved, sha256


def _retained_records(path):
    lines = path.read_bytes().splitlines(keepends=True)
    header = next(csv.reader([lines[0].decode('utf-8-sig')]))
    return [line for line in lines[1:] if dict(zip(header, next(csv.reader([line.decode('utf-8')]))))['dataset'] != 'unsw_nb15']


def run():
    assert_preserved()
    root = ROOT/'results/p10'
    report = json.loads((ROOT/'reports/p10/family_ac_complete.json').read_text())
    frame = pd.read_csv(root/'p5_full_merged.csv')
    guard = pd.read_csv(root/'p5_guard_full_merged.csv')
    windows = {(w['dataset'], w['seed']): w for w in json.loads((ROOT/'reports/p9_tail_metric_unification.json').read_text())['windows']}
    differences = []
    for r in guard.itertuples(index=False):
        if r.dataset != 'unsw_nb15':
            continue
        actual = canonical_tail_f1_from_perclass(r.per_class_f1_json, windows[(r.dataset, r.seed)]['tail_labels'])
        expected = frame[(frame.dataset == r.dataset) & (frame.seed == r.seed) &
                         np.isclose(frame.tail_flip_rate, r.tail_flip_rate) & (frame.cell == r.method)].iloc[0].tail_macro_f1
        differences.append(abs(actual-expected))
        assert r.noise_target == 0
    assert len(differences) == 60 and max(differences) < 1e-12
    checked = 0
    for family, size in [('family_a', 9), ('family_c', 18)]:
        tests = report['new'][family]['tests']
        assert len(tests) == size
        pvals = []
        for t in tests:
            part = frame[(frame.dataset == t['dataset']) & np.isclose(frame.tail_flip_rate, t['tail_flip_rate'])]
            pivot = part.pivot(index='seed', columns='cell', values='tail_macro_f1')
            diff = (pivot[t['method']]-pivot[t['reference']]).to_numpy()
            assert len(diff) == 10
            p = max(float(stats.ttest_1samp(diff, popmean=0, alternative='greater').pvalue), 1e-12)
            pvals.append(p)
            np.testing.assert_allclose(p, t['p_value_greater'], atol=1e-12, rtol=1e-10)
            np.testing.assert_allclose(diff.mean(), t['mean_delta'], atol=1e-12)
            rng = np.random.default_rng(42)
            means = diff[rng.integers(0, 10, size=(2000, 10))].mean(axis=1)
            ci = np.quantile(means, [.025, .975])
            np.testing.assert_allclose(ci, [t['bootstrap_ci95_low'], t['bootstrap_ci95_high']], atol=1e-12)
            checked += 1
        holm = multipletests(pvals, alpha=.05, method='holm')[1]
        np.testing.assert_allclose(holm, [t['p_holm_greater'] for t in tests], atol=1e-12)
    for before, after in [('results/p5_factor_ablation_full.csv', root/'p5_full_merged.csv'),
                          ('results/p5_perclass_guard.csv', root/'p5_guard_full_merged.csv')]:
        assert _retained_records(ROOT/before) == _retained_records(after)
    old = json.loads((ROOT/'reports/p5_factor_ablation_full.json').read_text())
    for f in ('family_a', 'family_c'):
        for a, b in zip(old[f]['tests'], report['old'][f]['tests']):
            for field in ('mean_delta', 'p_holm_greater', 'bootstrap_ci95_low', 'bootstrap_ci95_high'):
                np.testing.assert_allclose(a[field], b[field], atol=1e-12)
    result = {'validation': 'Share with caveats: P10 stopped before B/leakage', 'submission_ready': False,
              'independently_recomputed_tests': checked, 'holm_families': [9, 18],
              'paired_n_per_test': 10, 'guard_main_max_abs_difference': max(differences),
              'guard_rows_checked': len(differences), 'old_ac_matches_frozen_report': True,
              'other_dataset_raw_record_bytes_unchanged': True, 'historical_file_hashes_unchanged': True,
              'new_artifact_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in root.glob('*.csv')}}
    (ROOT/'reports/p10/validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
