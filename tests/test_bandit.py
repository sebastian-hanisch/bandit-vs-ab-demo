"""Zuteilungsstrategien: jede Einheit an Mini-Instanzen von Hand, dazu Eigenschaften (Kausalität, Gewichte, Reproduzierbarkeit)."""

import numpy as np
import pytest
from scipy import stats

import bva_bandit as BB
import bva_evaluation as E


def test_arm_factors_by_hand():
    assert BB.arm_z(1.4) == (1.0, 1.2, 1.4)
    assert BB.arm_z(1.0) == (1.0, 1.0, 1.0)
    assert BB.arm_z(1.8) == (1.0, 1.4, 1.8)


def test_potential_outcomes_are_pool_values_deterministic_and_trend_shifted():
    pools = [np.array([0.5, 0.6, 0.7]), np.array([0.8, 0.9, 0.95])]
    a = BB.potential_outcomes(pools, 20, 5, 1)
    b = BB.potential_outcomes(pools, 20, 5, 1)
    assert a.shape == (5, 20, 2) and np.array_equal(a, b)
    assert set(np.unique(a[:, :, 0])) <= {0.5, 0.6, 0.7} and set(np.unique(a[:, :, 1])) <= {0.8, 0.9, 0.95}
    shifted = BB.potential_outcomes(pools, 20, 5, 1, trend=0.04)
    shift = 0.04 * ((np.arange(20) + 0.5) / 20 - 0.5)
    expected = np.minimum(a + shift[None, :, None], 1.0)
    assert np.allclose(shifted, expected)
    assert shift[0] < 0 < shift[-1] and abs(shift.sum()) < 1e-12


def test_ts_probabilities_by_hand():
    mu = np.array([[0.9, 0.9, 0.9], [0.0, 0.0, 10.0]])
    sd = np.array([[0.1, 0.1, 0.1], [1.0, 1.0, 1.0]])
    p = BB.ts_probabilities(mu, sd)
    assert p[0] == pytest.approx([1 / 3] * 3, abs=1e-3)
    assert p[1][2] > 0.999
    assert p.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_run_uniform_hits_each_arm_equally_and_reads_the_potential_outcome():
    po = np.random.default_rng(0).random((200, 60, 3))
    r = BB.run_uniform(po, 5)
    assert abs((r["arms"] == 0).mean() - 1 / 3) < 0.01 and abs((r["arms"] == 2).mean() - 1 / 3) < 0.01
    assert np.array_equal(r["reward"], np.take_along_axis(po, r["arms"][:, :, None].astype(int), axis=2)[:, :, 0])
    assert r["uniform"].all() and np.allclose(r["prop"], 1 / 3)


def test_abn_sends_everything_after_the_test_to_the_arm_with_the_best_test_mean():
    po = np.zeros((1, 100, 3))
    po[0, :, 0] = 0.5
    po[0, :, 1] = 0.9
    po[0, :, 2] = 0.7
    r = BB.run_abn(po, 0.3, 11)
    assert r["n_test"] == 30 and int(r["choice"][0]) == 1
    assert (r["arms"][0, 30:] == 1).all()
    assert r["reward"][0, 30:].tolist() == [0.9] * 70
    assert r["uniform"][0, :30].all() and not r["uniform"][0, 30:].any()


def test_thompson_warmup_is_uniform_and_outcomes_match_the_potential_outcomes():
    po = np.random.default_rng(1).random((50, 120, 3)) * 0.1 + np.array([0.80, 0.85, 0.90])
    r = BB.run_thompson(po, 10, 0.0, 3, want_propensity=True)
    assert np.array_equal(r["reward"], np.take_along_axis(po, r["arms"][:, :, None].astype(int), axis=2)[:, :, 0])
    assert r["uniform"][:, :10].all()                        # erste Runde ohne Daten: Anlauf
    assert (r["prop"] > 0).all() and (r["prop"] <= 1).all()
    assert np.allclose(r["prop"][:, :10], 1 / 3)
    counts = BB.arm_counts(r["arms"])
    assert (counts.sum(axis=1) == 120).all()


def test_thompson_moves_windows_to_the_better_rule():
    po = np.random.default_rng(2).normal(0, 0.03, (300, 300, 3)) + np.array([0.90, 0.93, 0.96])
    r = BB.run_thompson(po, 10, 0.0, 4)
    share = BB.arm_counts(r["arms"]).mean(axis=0) / 300
    assert share[2] > 0.6 > share[0] and share[0] < 0.15


def test_the_allocation_of_a_round_depends_only_on_earlier_rounds():
    rng = np.random.default_rng(5)
    po = rng.random((20, 80, 3)) * 0.1 + 0.85
    po2 = po.copy()
    po2[:, 50:, :] = rng.random((20, 30, 3))                  # ab Fenster 50 andere mögliche Ergebnisse
    a = BB.run_thompson(po, 10, 0.1, 7)
    b = BB.run_thompson(po2, 10, 0.1, 7)
    assert np.array_equal(a["arms"][:, :60], b["arms"][:, :60])        # Runde 5 (Fenster 50-59) sieht nur Fenster 0-49
    assert not np.array_equal(a["arms"][:, 60:], b["arms"][:, 60:])


def test_floor_one_is_uniform_with_logged_propensity_one_third():
    po = np.random.default_rng(3).random((30, 60, 3))
    r = BB.run_thompson(po, 10, 1.0, 9, want_propensity=True)
    assert r["uniform"].all()
    assert np.allclose(r["prop"], 1 / 3)


def test_runs_are_pure_functions_of_the_seed():
    po = np.random.default_rng(4).random((10, 50, 3))
    a = BB.run_thompson(po, 10, 0.1, 7, want_propensity=True)
    b = BB.run_thompson(po, 10, 0.1, 7, want_propensity=True)
    assert np.array_equal(a["arms"], b["arms"]) and np.array_equal(a["prop"], b["prop"])


def test_welch_moments_equals_scipy():
    rng = np.random.default_rng(6)
    x, y = rng.normal(0.9, 0.04, 40), rng.normal(0.93, 0.03, 55)
    res = E.welch_moments(np.array([40.0]), np.array([x.sum()]), np.array([(x * x).sum()]),
                          np.array([55.0]), np.array([y.sum()]), np.array([(y * y).sum()]))
    ref = stats.ttest_ind(y, x, equal_var=False)
    assert res["p"][0] == pytest.approx(ref.pvalue, rel=1e-6)
    assert res["diff"][0] == pytest.approx(y.mean() - x.mean())


def test_welch_moments_needs_two_windows_per_arm():
    res = E.welch_moments(np.array([1.0]), np.array([0.9]), np.array([0.81]), np.array([5.0]), np.array([4.5]), np.array([4.1]))
    assert res["p"][0] == 1.0 and np.isnan(res["diff"][0])


def test_ipw_means_by_hand():
    arms = np.array([[0, 0, 2, 2, 2]], dtype=np.int8)
    y = np.array([[0.8, 0.9, 0.9, 1.0, 0.95]])
    prop = np.array([[0.5, 0.25, 0.5, 0.5, 0.25]])
    m = E.ipw_means(arms, y, prop)[0]
    # Regel 0: Gewichte 2, 4 -> (0.8*2 + 0.9*4)/6; Regel 2: Gewichte 2, 2, 4 -> (0.9*2 + 1.0*2 + 0.95*4)/8
    assert m[0] == pytest.approx(5.2 / 6) and m[2] == pytest.approx(7.6 / 8)


def test_weighted_test_with_constant_propensity_is_the_plain_difference_of_means():
    rng = np.random.default_rng(8)
    arms = rng.integers(0, 3, (1, 90)).astype(np.int8)
    y = rng.random((1, 90))
    res = E.weighted_test(arms, y, np.full((1, 90), 1 / 3))
    ya, yc = y[0][arms[0] == 0], y[0][arms[0] == 2]
    assert res["diff"][0] == pytest.approx(yc.mean() - ya.mean())


def test_simulation_is_a_pure_function_of_settings(pools):
    a = E.summarize(pools, 1.4, 150, 10, 0.0, reps=60)
    b = E.summarize(pools, 1.4, 150, 10, 0.0, reps=60)
    assert a["rows"] == b["rows"]
