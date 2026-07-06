import pytest

from src.ranking.prioritize import compression_at_fixed_recall, queue_load_curve, ranking_metrics


def test_ranking_metrics_measure_topk_and_fixed_recall_compression():
    scores = [0.9, 0.8, 0.2, 0.7, 0.1]
    y_true = [1, 0, 1, 2, 0]

    metrics = ranking_metrics(scores, y_true, {"ranking": {"top_k": 2, "review_budget": 0.4, "benign_class": 0}})

    assert metrics["topk_precision"] == pytest.approx(0.5)
    assert metrics["topk_recall"] == pytest.approx(1 / 3)
    assert metrics["precision_at_budget"] == pytest.approx(0.5)
    assert metrics["compression_at_recall_90"] == pytest.approx(0.8)
    assert compression_at_fixed_recall(scores, y_true, 0.95, benign_class=0) == pytest.approx(0.8)


def test_queue_load_curve_is_monotonic_in_recall():
    curve = queue_load_curve([0.9, 0.8, 0.2, 0.7, 0.1], [1, 0, 1, 2, 0], budgets=[0.2, 0.4, 0.8], benign_class=0)

    recalls = [row["topk_recall"] for row in curve]
    assert recalls == sorted(recalls)
    assert curve[-1]["topk_recall"] == pytest.approx(1.0)
