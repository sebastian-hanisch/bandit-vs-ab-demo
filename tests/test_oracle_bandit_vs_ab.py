"""Orakel-Tests: der Kern gegen einen unabhängigen Rechenweg (ORACLE-PLAYBOOK §2).

1. Zuteilungswahrscheinlichkeit des Thompson-Sampling gegen die geschlossene Formel (zwei Regeln) und gegen scipy.integrate.
2. Welch aus Momenten gegen scipy.stats.ttest_ind.
3. Gewichteter Schätzer: bei bekannten, ungleichen Zuteilungswahrscheinlichkeiten (nicht adaptiv) konsistent, Niveau 5 % im A/A.
4. Der naive Test auf adaptiv gesammelten Daten ist bei Gleichheit anti-konservativ (Thompson-Zuteilung auf Normaldaten)."""

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import ndtr

import bva_bandit as BB
import bva_evaluation as E


def test_two_arm_probability_matches_the_closed_form():
    mu = np.array([[0.90, 0.93], [0.50, 0.50], [0.0, 1.0]])
    sd = np.array([[0.02, 0.03], [0.10, 0.30], [0.5, 0.5]])
    # K = 3 braucht eine dritte Regel; eine praktisch ausgeschlossene (sechs Standardabweichungen tiefer) lässt zwei übrig
    low = mu.min(axis=1) - 7 * sd.max(axis=1)
    mu3 = np.concatenate([mu, low[:, None]], axis=1)
    sd3 = np.concatenate([sd, (sd.min(axis=1) / 2)[:, None]], axis=1)
    p = BB.ts_probabilities(mu3, sd3)
    for i in range(3):
        exact = ndtr((mu[i, 1] - mu[i, 0]) / np.sqrt(sd[i, 0] ** 2 + sd[i, 1] ** 2))
        assert p[i, 1] == pytest.approx(exact, abs=2e-3)
        assert p[i, 2] == pytest.approx(0.0, abs=1e-9)


def test_three_arm_probability_matches_adaptive_quadrature():
    mu = np.array([0.91, 0.95, 0.93])
    sd = np.array([0.03, 0.04, 0.02])
    p = BB.ts_probabilities(mu[None, :], sd[None, :])[0]
    for k in range(3):
        others = [j for j in range(3) if j != k]

        def f(x):
            return stats.norm.pdf(x, mu[k], sd[k]) * np.prod([stats.norm.cdf(x, mu[j], sd[j]) for j in others])

        exact, _ = integrate.quad(f, 0.5, 1.5, limit=200)
        assert p[k] == pytest.approx(exact, abs=3e-3)


def test_welch_from_run_equals_scipy_on_a_real_run(pools):
    po, runs = E.simulate_all(pools, 1.4, 150, 10, 0.0, 4, 77, with_ipw=False)
    run = runs["ts"]
    res = E.welch_from_run(run["arms"], run["reward"], np.ones(run["arms"].shape, dtype=bool))
    for i in range(4):
        a = run["reward"][i][run["arms"][i] == 0]
        c = run["reward"][i][run["arms"][i] == 2]
        ref = stats.ttest_ind(c, a, equal_var=False)
        assert res["p"][i] == pytest.approx(ref.pvalue, rel=1e-6)


def test_weighted_estimator_is_consistent_under_known_unequal_probabilities():
    rng = np.random.default_rng(10)
    R, T = 2000, 200
    pi = np.array([0.2, 0.3, 0.5])
    arms = rng.choice(3, size=(R, T), p=pi).astype(np.int8)
    truth = np.array([0.90, 0.92, 0.95])
    y = truth[arms] + rng.normal(0, 0.05, (R, T))
    m = E.ipw_means(arms, y, pi[arms])
    assert m.mean(axis=0) == pytest.approx(truth, abs=2e-3)
    # A/A mit ungleichen Wahrscheinlichkeiten: der gewichtete Test hält das Niveau
    y0 = 0.93 + rng.normal(0, 0.05, (R, T))
    res = E.weighted_test(arms, y0, pi[arms])
    assert float((res["p"] < 0.05).mean()) == pytest.approx(0.05, abs=0.02)


def test_naive_test_after_thompson_is_anti_conservative_under_the_null():
    rng = np.random.default_rng(11)
    po = rng.normal(0.93, 0.04, (2000, 600, 3))
    run = BB.run_thompson(po, 10, 0.0, 13)
    naive = E.welch_from_run(run["arms"], run["reward"], np.ones(run["arms"].shape, dtype=bool))
    clean = E.welch_from_run(run["arms"], run["reward"], run["uniform"])
    assert float((naive["p"] < 0.05).mean()) > 0.065
    assert float((clean["p"] < 0.05).mean()) == pytest.approx(0.05, abs=0.015)
    # der Standardfehler ist zu klein: mittlerer Standardfehler gegen Streuung der Schätzung
    assert float(np.nanmean(naive["se"])) / float(np.nanstd(naive["diff"])) < 0.8
