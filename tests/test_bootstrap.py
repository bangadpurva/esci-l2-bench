import numpy as np
import pytest

from recl2bench.eval import bootstrap as B


def test_holm_hand_computed():
    # p = [0.01, 0.04, 0.03], m = 3
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone max -> 0.06
    assert B.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    assert B.holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])


def test_paired_bootstrap_identical_models_has_zero_diff():
    x = np.random.default_rng(1).random(300)
    mean, boot = B.paired_bootstrap(x, x, n_resamples=500)
    assert mean == 0.0 and np.all(boot == 0.0)


def test_clear_winner_is_significant_and_noise_is_not():
    rng = np.random.default_rng(7)
    base = rng.random(500)
    better = np.clip(base + 0.1, 0, 1)
    noise = np.clip(base + rng.normal(0, 0.05, 500), 0, 1)
    res = {r.name: r for r in B.compare_to_baseline(
        {"better": better, "noise": noise}, base, n_resamples=2000)}
    assert res["better"].significant and res["better"].ci_low > 0
    assert not res["noise"].significant


def test_misaligned_arrays_raise():
    with pytest.raises(ValueError):
        B.paired_bootstrap(np.zeros(3), np.zeros(4))
