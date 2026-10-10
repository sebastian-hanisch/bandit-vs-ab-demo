"""Welch-Test, Konfidenzintervall und Normalformel: Handrechnung und Gegenprobe gegen scipy."""

import math

import numpy as np
import pytest
from scipy import stats

import bva_stats as S


def test_welch_by_hand_on_a_tiny_example():
    # A = [1, 2, 3] (Mittel 2, s^2 = 1), B = [2, 4, 6] (Mittel 4, s^2 = 4): SE = sqrt(1/3 + 4/3) = 1,2910, t = 2/1,2910 = 1,5492
    # df = (5/3)^2 / ((1/3)^2/2 + (4/3)^2/2) = 2,7778 / (0,05556 + 0,88889) = 2,9412
    r = S.welch([1, 2, 3], [2, 4, 6])
    assert r["diff"] == pytest.approx(2.0)
    assert r["se"] == pytest.approx(math.sqrt(5 / 3))
    assert r["t"] == pytest.approx(2 / math.sqrt(5 / 3))
    assert r["df"] == pytest.approx((5 / 3) ** 2 / ((1 / 3) ** 2 / 2 + (4 / 3) ** 2 / 2))


def test_welch_matches_scipy_including_the_confidence_interval():
    rng = np.random.default_rng(1)
    for _ in range(40):
        a = rng.normal(0, 1, rng.integers(3, 60))
        b = rng.normal(0.3, 2, rng.integers(3, 60))
        mine = S.welch(a, b)
        ref = stats.ttest_ind(b, a, equal_var=False)
        ci = ref.confidence_interval(0.95)
        assert mine["p"] == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-12)
        assert mine["t"] == pytest.approx(ref.statistic, rel=1e-9)
        assert mine["df"] == pytest.approx(ref.df, rel=1e-9)
        assert mine["lo"] == pytest.approx(ci.low, rel=1e-9, abs=1e-12)
        assert mine["hi"] == pytest.approx(ci.high, rel=1e-9, abs=1e-12)


def test_batch_equals_the_scalar_version_row_by_row():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, (25, 12))
    b = rng.normal(0.5, 1.5, (25, 12))
    batch = S.welch_batch(a, b)
    for i in range(25):
        one = S.welch(a[i], b[i])
        for key in ("diff", "se", "df", "t", "p", "lo", "hi"):
            assert batch[key][i] == pytest.approx(one[key], rel=1e-9, abs=1e-12)


def test_identical_constant_arms_give_no_signal():
    r = S.welch([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert r["diff"] == 0 and r["p"] == 1.0
    batch = S.welch_batch(np.ones((3, 4)), np.ones((3, 4)))
    assert np.all(batch["p"] == 1.0)


def test_normal_formula_by_hand():
    # (1,95996 + 0,84162)^2 = 7,8489; Wirkung 0,5, sd 1 und 1: 7,8489 * 2 / 0,25 = 62,79 -> 63
    assert S.n_per_arm_normal(0.5, 1.0, 1.0) == 63
    assert S.n_per_arm_normal(0.0, 1.0, 1.0) == math.inf


def test_confidence_level_changes_the_width():
    a, b = [1, 2, 3, 4, 5], [2, 3, 4, 5, 6.5]
    narrow = S.welch(a, b, conf=0.80)
    wide = S.welch(a, b, conf=0.99)
    assert wide["hi"] - wide["lo"] > narrow["hi"] - narrow["lo"]
