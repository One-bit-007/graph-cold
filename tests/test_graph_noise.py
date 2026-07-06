import numpy as np

from src.analysis.graph_noise_validation import beta0_matches_symmetric, beta_sweep_report


def test_beta0_matches_symmetric():
    y = np.arange(240) % 4

    report = beta0_matches_symmetric(y, ratio=0.25, num_classes=4, seed=42)

    assert report["mask_equal"] is True
    assert report["labels_equal"] is True
    assert report["flip_count"] == report["target_flip_count"]
    assert report["transition_l1"] == 0.0


def test_beta_sweep_concentrates_transitions_without_fabricating_flips(tmp_path):
    figure = tmp_path / "beta_sweep.pdf"

    report = beta_sweep_report(figure_path=figure, n_per_class=40)

    assert report["beta0_matches_symmetric"] is True
    assert report["concentration_increases"] is True
    assert figure.exists()
