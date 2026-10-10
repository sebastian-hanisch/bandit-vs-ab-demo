"""Jede Zahl des README (Abschnitt "Befunde") wird hier nachgerechnet.

Die Zahlen stammen aus `precomputed_sweeps.json` (3 000 Wiederholungen je Zelle, SplitMix64, Seeds fest); zwei Zellen werden
zusätzlich mit denselben Funktionen wie in der App neu gerechnet, damit die Datei nicht von der Rechnung abweicht."""

import json
from pathlib import Path

import pytest

import bva_evaluation as E

ROOT = Path(__file__).resolve().parent.parent


def pct(x):
    return 100 * x


@pytest.fixture(scope="module")
def cells():
    data = json.loads((ROOT / "precomputed_sweeps.json").read_text(encoding="utf-8"))
    return {(c["sweep"], c["z"], c["T"], c["m"], round(c["trend"], 2)): c for c in data["cells"]}


def row(cells, key, strategy):
    return cells[key]["rows"][strategy]


def test_the_sweep_file_matches_a_fresh_computation(cells, pools):
    for z, t, m, trend in ((1.4, 600, 10, 0.0), (1.0, 600, 10, 0.03)):
        fresh = E.summarize(pools, z, t, m, trend, reps=3000, tag=3)
        stored = cells[("effect" if trend == 0 else "trend", z, t, m, trend)]
        for strategy in E.STRATEGIES:
            for key in ("reached", "reject", "est", "est_sd"):
                assert fresh["rows"][strategy][key] == pytest.approx(stored["rows"][strategy][key], abs=2e-3), (strategy, key)
        assert fresh["rows"]["ts_boden"]["ipw_reject"] == pytest.approx(stored["rows"]["ts_boden"]["ipw_reject"], abs=3e-3)


def test_the_bandit_loses_less_than_the_alternatives(cells):
    k = ("effect", 1.4, 600, 10, 0.0)
    best = max(cells[k]["means"])
    loss = {s: pct(best - row(cells, k, s)["reached"]) for s in E.STRATEGIES}
    assert loss == pytest.approx({"gleich": 1.68, "abn": 0.53, "ts": 0.21, "ts_boden": 0.34}, abs=0.02)
    assert pct(row(cells, k, "ts")["share"][2]) == pytest.approx(90.1, abs=0.2)
    assert pct(row(cells, k, "ts_boden")["share"][2]) == pytest.approx(85.1, abs=0.2)
    assert pct(row(cells, k, "abn")["share"][2]) == pytest.approx(78.7, abs=0.2)
    assert pct(row(cells, k, "abn")["choice_best"]) == pytest.approx(98.2, abs=0.2)


def test_false_alarm_without_difference(cells):
    k = ("effect", 1.0, 600, 10, 0.0)
    expected = {"gleich": 4.5, "abn": 4.8, "ts": 7.8, "ts_boden": 4.7}
    for s, v in expected.items():
        assert pct(row(cells, k, s)["reject"]) == pytest.approx(v, abs=0.2), s
    assert pct(row(cells, k, "ts_boden")["clean_reject"]) == pytest.approx(4.6, abs=0.2)
    assert pct(row(cells, k, "ts_boden")["ipw_reject"]) == pytest.approx(3.3, abs=0.2)
    assert row(cells, k, "ts")["se_fidelity"] == pytest.approx(0.58, abs=0.01)
    assert row(cells, k, "ts_boden")["se_fidelity"] == pytest.approx(0.84, abs=0.01)
    assert row(cells, k, "gleich")["se_fidelity"] == pytest.approx(1.01, abs=0.01)


def test_estimate_of_the_effect_with_a_clear_difference(cells):
    k = ("effect", 1.4, 600, 10, 0.0)
    assert pct(cells[k]["truth"]) == pytest.approx(3.49, abs=0.01)
    est = {s: (pct(row(cells, k, s)["est"]), pct(row(cells, k, s)["est_sd"])) for s in E.STRATEGIES}
    assert est["gleich"] == pytest.approx((3.50, 0.45), abs=0.02)
    assert est["abn"] == pytest.approx((3.49, 0.81), abs=0.02)
    assert est["ts"] == pytest.approx((3.90, 1.42), abs=0.02)
    assert est["ts_boden"] == pytest.approx((3.56, 0.91), abs=0.02)
    assert pct(row(cells, k, "ts_boden")["ipw_est"]) == pytest.approx(3.48, abs=0.02)
    assert pct(row(cells, k, "ts_boden")["ipw_sd"]) == pytest.approx(1.10, abs=0.02)
    assert pct(row(cells, k, "ts_boden")["clean_est"]) == pytest.approx(3.50, abs=0.02)


def test_power_with_a_clear_and_with_a_small_difference(cells):
    k14 = ("effect", 1.4, 600, 10, 0.0)
    got = {s: pct(row(cells, k14, s)["reject"]) for s in E.STRATEGIES}
    assert got == pytest.approx({"gleich": 100.0, "abn": 99.1, "ts": 97.0, "ts_boden": 99.9}, abs=0.2)
    assert pct(row(cells, k14, "ts_boden")["clean_reject"]) == pytest.approx(80.8, abs=0.2)
    assert pct(row(cells, k14, "ts_boden")["ipw_reject"]) == pytest.approx(95.0, abs=0.2)
    k12 = ("effect", 1.2, 600, 10, 0.0)
    assert pct(cells[k12]["truth"]) == pytest.approx(1.96, abs=0.01)
    got = {s: pct(row(cells, k12, s)["reject"]) for s in E.STRATEGIES}
    assert got == pytest.approx({"gleich": 98.6, "abn": 61.9, "ts": 68.0, "ts_boden": 80.6}, abs=0.2)
    assert pct(row(cells, k12, "ts_boden")["clean_reject"]) == pytest.approx(31.0, abs=0.2)
    assert pct(row(cells, k12, "ts_boden")["ipw_reject"]) == pytest.approx(53.7, abs=0.2)
    assert pct(row(cells, k12, "ts")["est"]) == pytest.approx(2.36, abs=0.02)
    assert pct(row(cells, k12, "ts_boden")["est"]) == pytest.approx(2.08, abs=0.02)
    best = max(cells[k12]["means"])
    assert pct(best - row(cells, k12, "gleich")["reached"]) == pytest.approx(0.85, abs=0.02)
    assert pct(best - row(cells, k12, "ts")["reached"]) == pytest.approx(0.26, abs=0.02)


def test_power_grows_with_the_number_of_windows(cells):
    expected = {150: (51.9, 18.4, 22.3, 28.0), 300: (81.6, 34.1, 42.3, 52.0), 600: (98.6, 61.9, 68.0, 80.6), 900: (99.9, 77.5, 84.0, 92.6)}
    for t, values in expected.items():
        got = tuple(pct(row(cells, ("horizon", 1.2, t, 10, 0.0), s)["reject"]) for s in E.STRATEGIES)
        assert got == pytest.approx(values, abs=0.2), t


def test_a_common_trend_breaks_the_naive_test_after_the_bandit(cells):
    expected_ts = {0.0: 7.8, 0.01: 15.7, 0.02: 28.7, 0.03: 46.4, 0.04: 59.1}
    expected_tb = {0.0: 4.7, 0.01: 10.9, 0.02: 21.3, 0.03: 32.0, 0.04: 46.2}
    for trend, v in expected_ts.items():
        assert pct(row(cells, ("trend", 1.0, 600, 10, trend), "ts")["reject"]) == pytest.approx(v, abs=0.2), trend
        assert pct(row(cells, ("trend", 1.0, 600, 10, trend), "ts_boden")["reject"]) == pytest.approx(expected_tb[trend], abs=0.2), trend
    for trend in expected_ts:
        k = ("trend", 1.0, 600, 10, trend)
        assert 4.0 < pct(row(cells, k, "gleich")["reject"]) < 6.0
        assert 4.0 < pct(row(cells, k, "ts_boden")["clean_reject"]) < 6.0
        assert 2.5 < pct(row(cells, k, "ts_boden")["ipw_reject"]) < 5.0
    # ohne Boden bleibt auch die Gewichtung ungültig
    assert pct(row(cells, ("trend", 1.0, 600, 10, 0.03), "ts")["ipw_reject"]) == pytest.approx(17.9, abs=0.2)


def test_a_trend_inflates_the_estimate_of_the_effect(cells):
    k = ("trend", 1.4, 600, 10, 0.03)
    assert pct(row(cells, k, "ts")["est"]) == pytest.approx(5.15, abs=0.02)
    assert pct(row(cells, k, "ts_boden")["est"]) == pytest.approx(4.06, abs=0.02)
    assert pct(row(cells, k, "gleich")["est"]) == pytest.approx(3.41, abs=0.02)
    assert pct(row(cells, k, "ts_boden")["clean_est"]) == pytest.approx(3.43, abs=0.02)
    assert pct(row(cells, k, "ts_boden")["ipw_est"]) == pytest.approx(3.45, abs=0.02)


def test_round_size_changes_the_adaptivity(cells):
    got = [pct(row(cells, ("round", 1.0, 600, m, 0.0), "ts")["reject"]) for m in (5, 10, 25, 50)]
    assert got == pytest.approx([7.6, 7.8, 8.2, 5.7], abs=0.2)
    loss5 = max(cells[("round", 1.4, 600, 5, 0.0)]["means"]) - row(cells, ("round", 1.4, 600, 5, 0.0), "ts")["reached"]
    loss50 = max(cells[("round", 1.4, 600, 50, 0.0)]["means"]) - row(cells, ("round", 1.4, 600, 50, 0.0), "ts")["reached"]
    assert pct(loss5) == pytest.approx(0.21, abs=0.02) and pct(loss50) == pytest.approx(0.25, abs=0.02)


def test_small_difference_losses_and_the_choice_of_abn(cells):
    k = ("effect", 1.2, 600, 10, 0.0)
    best = max(cells[k]["means"])
    assert pct(best - row(cells, k, "ts_boden")["reached"]) == pytest.approx(0.31, abs=0.02)
    assert pct(best - row(cells, k, "abn")["reached"]) == pytest.approx(0.36, abs=0.02)
    assert pct(row(cells, k, "abn")["choice_best"]) == pytest.approx(75.0, abs=0.2)


def test_remedies_hold_the_level_at_every_trend(cells):
    trends = (0.0, 0.01, 0.02, 0.03, 0.04)
    clean = [pct(row(cells, ("trend", 1.0, 600, 10, tr), "ts_boden")["clean_reject"]) for tr in trends]
    weighted = [pct(row(cells, ("trend", 1.0, 600, 10, tr), "ts_boden")["ipw_reject"]) for tr in trends]
    equal = [pct(row(cells, ("trend", 1.0, 600, 10, tr), "gleich")["reject"]) for tr in trends]
    assert (min(clean), max(clean)) == pytest.approx((4.6, 5.6), abs=0.2)
    assert (min(weighted), max(weighted)) == pytest.approx((3.3, 4.8), abs=0.2)
    assert (min(equal), max(equal)) == pytest.approx((4.5, 5.2), abs=0.2)
    assert pct(row(cells, ("trend", 1.0, 600, 10, 0.03), "ts_boden")["ipw_reject"]) == pytest.approx(3.6, abs=0.2)


def test_default_seed_experiment(pools):
    one = E.run_one(pools, {"z": 1.4, "T": 600, "m": 10, "trend": 0.0, "seed": 172})
    est = {e["key"]: pct(e["diff"]) for e in E.single_estimates(one)}
    assert est["ts"] == pytest.approx(4.12, abs=0.02)
    assert est["ts_boden"] == pytest.approx(3.83, abs=0.02)
    assert est["ts_boden_ipw"] == pytest.approx(3.57, abs=0.02)
