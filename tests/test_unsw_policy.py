from src.data.unsw_policy import postfilter_counts


def test_unsw_postfilter_removes_low_count_and_downsamples_dominant():
    kept, removed, rule = postfilter_counts({"Normal": 10, "Exploits": 4, "Tiny": 1}, min_count=2)

    assert removed == {"Tiny": 1}
    assert kept == {"Exploits": 4, "Normal": 4}
    assert "Downsample dominant class" in rule
