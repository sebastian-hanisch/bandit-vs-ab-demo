"""Exakte Wahrheit des Vehikels (nur im i.i.d.-Fall ohne Saison): der Servicegrad einer Order-up-to-Regel bei täglicher
Überprüfung und i.i.d. negativ-binomialer Nachfrage ist
    beta = 1 - (E[(D_L - S)+] - E[(D_{L-1} - S)+]) / mu,
mit D_n = Nachfrage von n Tagen = NegBin(Mittel n*mu, Größe n*disp), L = Wiederbeschaffungszeit, S = Bestellniveau."""

import math

import numpy as np
from scipy.stats import nbinom

import bva_constants as C
from bva_vehicle import order_up_to_level


def expected_excess(n_days, level, mean=C.BASE_MEAN, disp=C.DISP):
    """E[(D_n - S)+] für die Nachfrage von n_days Tagen."""
    if n_days == 0:
        return max(0.0, -float(level))
    size, mu = n_days * disp, n_days * mean
    p = size / (size + mu)
    sd = math.sqrt(n_days * (mean + mean * mean / disp))
    k = np.arange(int(level), int(level + 40 * sd) + 50)
    return float(nbinom.sf(k, size, p).sum())


def fill_rate(z, mean=C.BASE_MEAN, disp=C.DISP, lead=C.LEAD):
    """Servicegrad Typ 2 (Quote der Summen im Gleichgewicht) bei Sicherheitsfaktor z."""
    level = order_up_to_level(z, mean, disp, lead)
    return 1.0 - (expected_excess(lead, level, mean, disp) - expected_excess(lead - 1, level, mean, disp)) / mean


def shortage_rate(z):
    """Fehlmengenanteil = 1 - Servicegrad."""
    return 1.0 - fill_rate(z)
