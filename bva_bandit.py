"""Zuteilungsstrategien für Standort-Fenster: gleichverteilt, A/B/n mit Testphase, Thompson-Sampling (mit und ohne Boden).

Einheit = ein Standort-Fenster (vier Wochen, Servicegrad Typ 2 aus dem Pool der Regel). Die Fenster treffen in Runden zu
je `m` Fenstern ein; die Zuteilung einer Runde sieht nur die Ergebnisse der früheren Runden. Alle Strategien sehen
dieselben möglichen Ergebnisse (gemeinsame Zufallszahlen): Fenster t hat für jede der K Regeln ein festes Ergebnis, und die
Strategie bestimmt nur, welches davon sie beobachtet. Aller Zufall kommt aus SplitMix64 (`bva_rng.uniform_block`).

Thompson-Sampling für stetige Ergebnisse im Normalmodell: je Regel ein Normal-Posterior um das Stichprobenmittel mit
Standardfehler s_k / sqrt(n_k); ein Fenster geht an die Regel mit der größten Ziehung. Anlauf: bis jede Regel mindestens
`WARMUP` Fenster hat, wird gleichverteilt zugeteilt."""

import numpy as np
from scipy.special import ndtr, ndtri

from bva_rng import uniform_block

K = 3                     # Zahl der Regeln (A = bestehende Regel, B, C)
WARMUP = 5                # Fenster je Regel, bevor Thompson-Sampling entscheiden darf
GRID = 160                # Stützstellen der Quadratur für die Zuteilungswahrscheinlichkeit


def arm_z(z_best):
    """Sicherheitsfaktoren der drei Regeln: A = 1,0, C = z_best, B in der Mitte (z_best im Raster 0,2)."""
    return (1.0, round(1.0 + (z_best - 1.0) / 2.0, 1), round(z_best, 1))


def potential_outcomes(pools_by_arm, T, R, seed, trend=0.0):
    """Mögliche Ergebnisse (R, T, K): je Fenster und Regel ein Servicegrad aus dem Pool der Regel (Ziehen mit Zurücklegen).
    trend: gemeinsame Verschiebung aller Regeln über den Horizont (Prozentpunkte als Anteil), von -trend/2 bis +trend/2;
    Werte werden bei 1 gekappt."""
    out = np.empty((R, T, len(pools_by_arm)))
    for k, pool in enumerate(pools_by_arm):
        idx = np.minimum((uniform_block(seed + 7_919 * (k + 1), R * T) * len(pool)).astype(np.int64), len(pool) - 1)
        out[:, :, k] = pool[idx].reshape(R, T)
    if trend:
        shift = trend * ((np.arange(T) + 0.5) / T - 0.5)
        out = np.minimum(out + shift[None, :, None], 1.0)
    return out


def ts_probabilities(mu, sd):
    """Wahrscheinlichkeit, dass jede Regel die größte Thompson-Ziehung hat (Quadratur): mu, sd Form (R, K) -> (R, K)."""
    r, k = mu.shape
    lo = (mu - 6 * sd).min(axis=1)
    hi = (mu + 6 * sd).max(axis=1)
    x = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, GRID)[None, :]            # (R, G)
    z = (x[:, :, None] - mu[:, None, :]) / sd[:, None, :]                                    # (R, G, K)
    cdf = ndtr(z)
    pdf = np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sd[:, None, :])
    dx = (hi - lo) / (GRID - 1)
    p = np.empty((r, k))
    for j in range(k):
        others = np.prod(np.delete(cdf, j, axis=2), axis=2)
        integrand = pdf[:, :, j] * others
        p[:, j] = (integrand.sum(axis=1) - 0.5 * (integrand[:, 0] + integrand[:, -1])) * dx
    return p / p.sum(axis=1, keepdims=True)


def run_thompson(po, m, floor, seed, want_propensity=False):
    """Thompson-Sampling in Runden zu je m Fenstern. po: mögliche Ergebnisse (R, T, K). floor: Anteil der Fenster, die
    gleichverteilt zugeteilt werden (0 = reines Thompson-Sampling). Rückgabe: dict mit arms (R, T), reward (R, T),
    uniform (R, T) (Fenster, deren Zuteilung unabhängig von allen Ergebnissen war: Anlauf und Boden) und, wenn verlangt,
    prop (R, T) = Wahrscheinlichkeit, mit der die tatsächlich gewählte Regel gewählt wurde."""
    r_, t_, k_ = po.shape
    arms = np.empty((r_, t_), dtype=np.int8)
    reward = np.empty((r_, t_))
    uniform = np.zeros((r_, t_), dtype=bool)
    prop = np.empty((r_, t_)) if want_propensity else None
    n = np.zeros((r_, k_))
    s1 = np.zeros((r_, k_))
    s2 = np.zeros((r_, k_))
    rows = np.arange(r_)[:, None]
    for rnd, t0 in enumerate(range(0, t_, m)):
        t1 = min(t0 + m, t_)
        mm = t1 - t0
        u = uniform_block(seed + 1_000_003 * (rnd + 1), r_ * mm * (k_ + 2)).reshape(r_, mm, k_ + 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(n > 0, s1 / np.maximum(n, 1), 0.0)
            var = np.where(n > 1, np.maximum(s2 - s1 * s1 / np.maximum(n, 1), 0.0) / np.maximum(n - 1, 1), 0.0)
        sd = np.sqrt(var / np.maximum(n, 1)) + 1e-9
        warm = n.min(axis=1) < WARMUP                                                         # (R,)
        theta = mean[:, None, :] + sd[:, None, :] * ndtri(np.clip(u[:, :, :k_], 1e-12, 1 - 1e-12))
        arm_ts = theta.argmax(axis=2)
        arm_un = np.minimum((u[:, :, k_] * k_).astype(np.int64), k_ - 1)
        floor_hit = u[:, :, k_ + 1] < floor
        use_un = warm[:, None] | floor_hit
        arm = np.where(use_un, arm_un, arm_ts)
        val = np.take_along_axis(po[:, t0:t1, :], arm[:, :, None], axis=2)[:, :, 0]
        arms[:, t0:t1] = arm
        reward[:, t0:t1] = val
        uniform[:, t0:t1] = use_un
        if want_propensity:
            p_ts = ts_probabilities(mean, sd)                                                 # (R, K)
            p_all = np.where(warm[:, None], 1.0 / k_, floor / k_ + (1.0 - floor) * p_ts)
            prop[:, t0:t1] = np.take_along_axis(np.broadcast_to(p_all[:, None, :], (r_, mm, k_)), arm[:, :, None], axis=2)[:, :, 0]
        onehot = (arm[:, :, None] == np.arange(k_)[None, None, :])
        n += onehot.sum(axis=1)
        s1 += (onehot * val[:, :, None]).sum(axis=1)
        s2 += (onehot * (val * val)[:, :, None]).sum(axis=1)
    return {"arms": arms, "reward": reward, "uniform": uniform, "prop": prop}


def run_uniform(po, seed):
    """Dauerhaft gleichverteilt: jedes Fenster unabhängig mit gleicher Wahrscheinlichkeit auf eine Regel."""
    r_, t_, k_ = po.shape
    u = uniform_block(seed, r_ * t_).reshape(r_, t_)
    arms = np.minimum((u * k_).astype(np.int64), k_ - 1).astype(np.int8)
    reward = np.take_along_axis(po, arms[:, :, None].astype(np.int64), axis=2)[:, :, 0]
    return {"arms": arms, "reward": reward, "uniform": np.ones((r_, t_), dtype=bool), "prop": np.full((r_, t_), 1.0 / k_)}


def run_abn(po, test_frac, seed):
    """A/B/n mit Testphase: die ersten round(test_frac * T) Fenster gleichverteilt, danach alle Fenster an die Regel mit dem
    höchsten Stichprobenmittel der Testphase."""
    r_, t_, k_ = po.shape
    n_test = max(k_ * 2, int(round(test_frac * t_)))
    out = run_uniform(po, seed)
    arms = out["arms"]
    reward = out["reward"]
    onehot = arms[:, :n_test, None] == np.arange(k_)[None, None, :]
    cnt = onehot.sum(axis=1)
    mean = (onehot * reward[:, :n_test, None]).sum(axis=1) / np.maximum(cnt, 1)
    mean = np.where(cnt > 0, mean, -np.inf)
    choice = mean.argmax(axis=1).astype(np.int8)
    arms = arms.copy()
    arms[:, n_test:] = choice[:, None]
    reward = np.take_along_axis(po, arms[:, :, None].astype(np.int64), axis=2)[:, :, 0]
    uniform = np.zeros((r_, t_), dtype=bool)
    uniform[:, :n_test] = True
    return {"arms": arms, "reward": reward, "uniform": uniform, "prop": None, "n_test": n_test, "choice": choice}


def arm_counts(arms, k_=K, upto=None):
    a = arms if upto is None else arms[:, :upto]
    return np.stack([(a == k).sum(axis=1) for k in range(k_)], axis=1)
