"""Auswertung: Zuteilungsstrategien gegeneinander, mit bekannter Wahrheit (Pools = Population der Standort-Fenster).

Die Frage des Stücks: Ein Bandit verteilt die Fenster so, dass weniger davon auf schlechte Regeln fallen. Was kostet das die
Auswertung? Je Strategie: Servicegrad, der im Experiment tatsächlich erreicht wird; Anteil der Fenster auf der besten Regel;
und für die Frage "C gegen A": Fehlalarm bzw. Teststärke des naiven Welch-Tests auf den gesammelten Daten, Verzerrung der
Schätzung, sowie zwei Gegenmittel (Test nur auf den gleichverteilt zugeteilten Fenstern, Horvitz-Thompson-Schätzung)."""

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, stdtr

import bva_bandit as BB
import bva_constants as C

PRECOMPUTED = Path(__file__).resolve().parent / "precomputed_pools.json"
STRATEGIES = ("gleich", "abn", "ts", "ts_boden")
PROP_MIN = 1e-3           # Untergrenze der Zuteilungswahrscheinlichkeit in den Gewichten (gekappte Gewichtung)


def load_pools(path=PRECOMPUTED):
    """{z: Array der Fenster-Servicegrade}; z auf eine Nachkommastelle gerundet."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {round(float(k), 1): np.asarray(v, dtype=float) for k, v in data["pools"].items()}


def nearest_z(z):
    return min(C.Z_GRID, key=lambda g: abs(g - z))


def arm_pools(pools, z_best):
    return [pools[nearest_z(z)] for z in BB.arm_z(nearest_z(z_best))]


def true_means(pools, z_best):
    """Wahre Mittelwerte der drei Regeln (Poolmittel); mit Trend bleiben die Unterschiede zwischen den Regeln gleich."""
    return np.array([p.mean() for p in arm_pools(pools, z_best)])


def welch_moments(n1, s1, q1, n2, s2, q2):
    """Welch-Test C (Gruppe 2) gegen A (Gruppe 1) aus Zählern, Summen und Quadratsummen (vektorisiert über die Läufe).
    Rückgabe: dict(diff, se, p). Zu wenige Fenster (n < 2) oder keine Streuung: p = 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        m1, m2 = s1 / n1, s2 / n2
        v1 = np.maximum(q1 - s1 * s1 / n1, 0.0) / (n1 - 1)
        v2 = np.maximum(q2 - s2 * s2 / n2, 0.0) / (n2 - 1)
        a, b = v1 / n1, v2 / n2
        se = np.sqrt(a + b)
        diff = m2 - m1
        t = diff / se
        df = (a + b) ** 2 / (a**2 / (n1 - 1) + b**2 / (n2 - 1))
    ok = (n1 >= 2) & (n2 >= 2) & np.isfinite(t) & (se > 0) & np.isfinite(df)
    p = np.where(ok, 2 * stdtr(np.where(ok, df, 1.0), -np.abs(np.where(ok, t, 0.0))), 1.0)
    return {"diff": np.where(ok, diff, np.nan), "se": se, "p": p}


def _moments(arms, reward, mask, k):
    sel = (arms == k) & mask
    y = np.where(sel, reward, 0.0)
    return sel.sum(axis=1).astype(float), y.sum(axis=1), (y * y).sum(axis=1)


def welch_from_run(arms, reward, mask, a=0, c=2):
    n1, s1, q1 = _moments(arms, reward, mask, a)
    n2, s2, q2 = _moments(arms, reward, mask, c)
    return welch_moments(n1, s1, q1, n2, s2, q2)


def ipw_means(arms, reward, prop):
    """Gewichtetes Mittel je Regel mit den Kehrwerten der Zuteilungswahrscheinlichkeiten als Gewichte (normiert, Hajek):
    Summe(y / pi) / Summe(1 / pi) über die Fenster, die die Regel bekamen; pi wird nach unten bei PROP_MIN gekappt. Die Horvitz-Thompson-Summe ohne Normierung ist
    unverzerrt, streut aber bei kleinen Wahrscheinlichkeiten extrem."""
    w = 1.0 / np.maximum(prop, PROP_MIN)
    out = []
    for k in range(BB.K):
        sel = arms == k
        out.append(np.where(sel, reward * w, 0.0).sum(axis=1) / np.maximum(np.where(sel, w, 0.0).sum(axis=1), 1e-300))
    return np.stack(out, axis=1)


def weighted_test(arms, reward, prop, a=0, c=2):
    """Test C gegen A mit den Kehrwerten der Zuteilungswahrscheinlichkeiten als Gewichten (alle Fenster). Standardfehler aus der
    Linearisierung des normierten gewichteten Mittels; p-Wert aus der Normalverteilung. Rückgabe: dict(diff, se, p)."""
    w = 1.0 / np.maximum(prop, PROP_MIN)
    mu_var = []
    for k in (a, c):
        sel = arms == k
        sw = np.where(sel, w, 0.0).sum(axis=1)
        mu = np.where(sel, reward * w, 0.0).sum(axis=1) / np.maximum(sw, 1e-300)
        var = np.where(sel, (w * (reward - mu[:, None])) ** 2, 0.0).sum(axis=1) / np.maximum(sw, 1e-300) ** 2
        mu_var.append((mu, var))
    diff = mu_var[1][0] - mu_var[0][0]
    se = np.sqrt(mu_var[0][1] + mu_var[1][1])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, diff / se, 0.0)
    return {"diff": diff, "se": se, "p": 2.0 * ndtr(-np.abs(z))}


def simulate_all(pools, z_best, T, m, trend, reps, seed, floor=C.FLOOR, test_frac=C.TEST_FRAC, with_ipw=True):
    """Läuft alle vier Strategien auf denselben möglichen Ergebnissen. Rückgabe: (po, {Strategie: run})."""
    po = BB.potential_outcomes(arm_pools(pools, z_best), T, reps, seed, trend)
    runs = {
        "gleich": BB.run_uniform(po, seed + 101),
        "abn": BB.run_abn(po, test_frac, seed + 211),
        "ts": BB.run_thompson(po, m, 0.0, seed + 307, want_propensity=with_ipw),
        "ts_boden": BB.run_thompson(po, m, floor, seed + 401, want_propensity=with_ipw),
    }
    return po, runs


def summarize_runs(po, runs, means, trend=0.0):
    """Kennzahlen je Strategie aus den Läufen. means: wahre Mittel der drei Regeln (ohne Trend)."""
    r_, t_, _ = po.shape
    best = int(np.argmax(means))
    truth = float(means[2] - means[0])                     # C minus A
    # Mittel über den Horizont: der Trend verschiebt alle Regeln gleich; die Abkürzung gilt, solange nichts gekappt wird.
    rows = {}
    for name, run in runs.items():
        arms, reward = run["arms"], run["reward"]
        counts = BB.arm_counts(arms)
        row = {
            "reached": float(reward.mean()),
            "best_share": float((arms == best).mean()),
            "share": (counts.mean(axis=0) / t_).tolist(),
        }
        if name == "abn":
            nt = run["n_test"]
            mask = np.zeros(arms.shape, dtype=bool)
            mask[:, :nt] = True
            w = welch_from_run(arms, reward, mask)
        else:
            w = welch_from_run(arms, reward, np.ones(arms.shape, dtype=bool))
        row["reject"] = float((w["p"] < C.ALPHA).mean())
        row["est"] = float(np.nanmean(w["diff"]))
        row["est_sd"] = float(np.nanstd(w["diff"], ddof=1))
        row["bias"] = row["est"] - truth
        ok = np.isfinite(w["diff"])
        row["wrong_sign"] = float(((w["p"] < C.ALPHA) & (w["diff"] * truth < 0)).mean()) if truth else float("nan")
        row["se_fidelity"] = float(np.nanmean(w["se"]) / row["est_sd"]) if row["est_sd"] > 0 else float("nan")
        if name in ("ts", "ts_boden"):
            u = welch_from_run(arms, reward, run["uniform"])
            row["clean_reject"] = float((u["p"] < C.ALPHA).mean())
            row["clean_est"] = float(np.nanmean(u["diff"]))
            row["clean_n"] = float(run["uniform"].sum(axis=1).mean())
            if run["prop"] is not None:
                wt = weighted_test(arms, reward, run["prop"])
                row["ipw_est"] = float(wt["diff"].mean())
                row["ipw_sd"] = float(wt["diff"].std(ddof=1))
                row["ipw_reject"] = float((wt["p"] < C.ALPHA).mean())
                row["ipw_se_fidelity"] = float(wt["se"].mean() / row["ipw_sd"]) if row["ipw_sd"] > 0 else float("nan")
        if name == "abn":
            row["choice_best"] = float((run["choice"] == best).mean())
        rows[name] = row
    return rows


def run_seed(z_best, T, m, trend, tag=0):
    return 17_000_000 + 10_007 * int(round(z_best * 10)) + 131 * T + 17 * m + 997 * int(round(trend * 1000)) + 13 * tag


def summarize(pools, z_best, T, m, trend, reps=C.LIVE_REPS, tag=0):
    z_best = nearest_z(z_best)
    po, runs = simulate_all(pools, z_best, T, m, trend, reps, run_seed(z_best, T, m, trend, tag))
    means = true_means(pools, z_best)
    return {"z": z_best, "T": T, "m": m, "trend": trend, "reps": reps, "means": means.tolist(),
            "truth": float(means[2] - means[0]), "rows": summarize_runs(po, runs, means, trend)}


def run_one(pools, settings):
    """Das eine Experiment der Seitenleiste: alle vier Strategien auf denselben möglichen Ergebnissen (ein Lauf, R = 1)."""
    s = settings
    z_best = nearest_z(s["z"])
    seed = 1_000_003 * (s["seed"] + 1) + 5
    po, runs = simulate_all(pools, z_best, s["T"], s["m"], s["trend"], 1, seed)
    return {"po": po, "runs": runs, "means": true_means(pools, z_best), "z": z_best}


def regret_curves(one):
    """Kumulierter erwarteter Verlust je Strategie (Prozentpunkte x Fenster) gegenüber der besten Regel."""
    means = one["means"]
    gap = 100 * (means.max() - means)
    return {name: np.cumsum(gap[run["arms"][0].astype(int)]) for name, run in one["runs"].items()}


def single_estimates(one):
    """Die Schätzung C minus A samt 95-%-Intervall je Auswertung für das eine Experiment: Liste von dicts
    (key, strategy, diff, lo, hi, p). Intervalle: Normalnäherung (mindestens 50 Fenster je Regel)."""
    out = []

    def add(key, strategy, w):
        d, se, p = float(w["diff"][0]), float(w["se"][0]), float(w["p"][0])
        out.append({"key": key, "strategy": strategy, "diff": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se, "p": p})

    runs = one["runs"]
    everything = {k: np.ones(r["arms"].shape, dtype=bool) for k, r in runs.items()}
    add("gleich", "gleich", welch_from_run(runs["gleich"]["arms"], runs["gleich"]["reward"], everything["gleich"]))
    nt = runs["abn"]["n_test"]
    mask = np.zeros(runs["abn"]["arms"].shape, dtype=bool)
    mask[:, :nt] = True
    add("abn", "abn", welch_from_run(runs["abn"]["arms"], runs["abn"]["reward"], mask))
    add("ts", "ts", welch_from_run(runs["ts"]["arms"], runs["ts"]["reward"], everything["ts"]))
    add("ts_boden", "ts_boden", welch_from_run(runs["ts_boden"]["arms"], runs["ts_boden"]["reward"], everything["ts_boden"]))
    add("ts_boden_clean", "ts_boden", welch_from_run(runs["ts_boden"]["arms"], runs["ts_boden"]["reward"], runs["ts_boden"]["uniform"]))
    add("ts_boden_ipw", "ts_boden", weighted_test(runs["ts_boden"]["arms"], runs["ts_boden"]["reward"], runs["ts_boden"]["prop"]))
    return out
