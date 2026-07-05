from src.baselines.registry import REGISTRY, get_baseline_entry, make_baseline, registry_metadata


def test_registry_contains_formal_and_d9_5_baselines():
    expected = {"Noisy-Supervised", "Confident-Learning", "Co-Teaching-lite", "Decoupling", "FINE-style"}

    assert expected.issubset(REGISTRY)
    assert get_baseline_entry("Decoupling").uses_noisy_y_train is True
    assert get_baseline_entry("FINE-style").uses_clean_y_test_for_eval_only is True
    assert "not full original implementation" in get_baseline_entry("FINE-style").faithfulness_level


def test_registry_factories_and_smoke_metadata():
    dec = make_baseline("Decoupling", seed=1)
    fine = make_baseline("FINE-style", seed=1, noise_rate=0.2)
    meta = registry_metadata({"Decoupling", "FINE-style"})

    assert dec.method == "Decoupling"
    assert fine.method == "FINE-style"
    assert meta["Decoupling"]["smoke_passed"] is True
    assert meta["FINE-style"]["include_in_formal_results"] is True
