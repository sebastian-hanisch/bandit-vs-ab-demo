"""Konstanten: Vehikel, Regler-Grenzen, Farben, Presets. Das Vehikel (ein Standort-Fenster) ist das der ganzen Linie."""

# Vehikel: ein Standort, tägliche Nachbestellung (Order-up-to), Wiederbeschaffungszeit 2 Tage, Zentrallager unbegrenzt.
BASE_MEAN = 20.0          # mittlere Tagesnachfrage des Standorts (Stück)
DISP = 5                  # Dispersion der negativ-binomialen Tagesnachfrage (ganzzahlig: Gamma(5) = Summe von 5 Exponentialzahlen)
LEAD = 2                  # Wiederbeschaffungszeit in Tagen
BURN_DAYS = 28            # Einschwingtage vor dem Auswertefenster (gleiche Regel wie im Fenster)
WINDOW_DAYS = 28          # Auswertefenster: vier Wochen = ein "Standort-Fenster" = eine Einheit des Experiments
Z_A = 1.0                 # Sicherheitsfaktor der bestehenden Regel A

Z_MIN, Z_MAX, DEFAULT_Z = 1.0, 2.0, 1.4      # Pool-Raster (Regel B beim Erzeugen der Pools)
Z_STEP = 0.1
Z_GRID = tuple(round(Z_MIN + i * Z_STEP, 1) for i in range(int(round((Z_MAX - Z_MIN) / Z_STEP)) + 1))
POOL_SIZE = 10000         # Einheiten je Pool (je Sicherheitsfaktor)
ALPHA = 0.05

# Stück 9: Bandit statt A/B
ZB_MIN, ZB_MAX, ZB_STEP = 1.0, 1.8, 0.2                   # Sicherheitsfaktor der besten Regel C (B liegt in der Mitte)
T_MIN, T_MAX, DEFAULT_T, T_STEP = 150, 900, 600, 150      # Standort-Fenster im Experiment insgesamt
M_MIN, M_MAX, DEFAULT_M, M_STEP = 5, 50, 10, 5            # Fenster je Runde (Rückmeldung erst nach der Runde)
TREND_PP_MIN, TREND_PP_MAX, DEFAULT_TREND_PP = 0, 4, 0       # gemeinsame Verschiebung aller Regeln über den Horizont (Prozentpunkte)
FLOOR = 0.10              # Anteil der Fenster, die auch beim Bandit gleichverteilt bleiben (Boden)
TEST_FRAC = 0.30          # A/B/n: Anteil der Fenster in der Testphase
SEED_MAX, DEFAULT_SEED = 999999, 172
LIVE_REPS = 1000          # Wiederholungen der Live-Auswertung

STRATEGY_LABELS = {
    "gleich": "Immer gleichverteilt (A/B/n ohne Ende)",
    "abn": "A/B/n: 30 % Test, dann die beste Regel",
    "ts": "Bandit (Thompson-Sampling)",
    "ts_boden": "Bandit mit 10 % Boden",
}
STRATEGY_SHORT = {"gleich": "Gleichverteilt", "abn": "A/B/n", "ts": "Bandit", "ts_boden": "Bandit + Boden"}
STRATEGY_COLORS = {"gleich": "#7f7f7f", "abn": "#4c78a8", "ts": "#c0392b", "ts_boden": "#2F6B65"}
ARM_LABELS = ("A (bestehend)", "B", "C")
ARM_COLORS = ("#4c78a8", "#8c564b", "#2F6B65")

COLOR_A = "#4c78a8"
COLOR_B = "#2F6B65"
COLOR_TRUTH = "#d95f02"
COLOR_MISS = "#c0392b"
COLOR_OK = "rgba(120,120,120,0.55)"

PRESETS = {
    "Bandit gegen A/B/n (Voreinstellung)": {"z": DEFAULT_Z, "T": DEFAULT_T, "m": DEFAULT_M, "trend": 0, "seed": DEFAULT_SEED},
    "Kein Unterschied (A/A/A)": {"z": 1.0, "T": DEFAULT_T, "m": DEFAULT_M, "trend": 0, "seed": DEFAULT_SEED},
    "Kleiner Unterschied": {"z": 1.2, "T": DEFAULT_T, "m": DEFAULT_M, "trend": 0, "seed": DEFAULT_SEED},
    "Die Lage verschiebt sich": {"z": 1.0, "T": DEFAULT_T, "m": DEFAULT_M, "trend": 3, "seed": DEFAULT_SEED},
}
PRESET_ORDER = tuple(PRESETS)
PRESET_HELP = {
    "Bandit gegen A/B/n (Voreinstellung)": "Regel C (Sicherheitsfaktor 1,4) ist die beste, 600 Fenster, zehn je Runde: wer verliert wie viel, und was kann der Test danach noch?",
    "Kein Unterschied (A/A/A)": "Alle drei Regeln gleich: jeder Befund ist ein Fehlalarm. Wie oft meldet der Test nach adaptiver Zuteilung trotzdem einen Unterschied?",
    "Kleiner Unterschied": "Regel C mit Sicherheitsfaktor 1,2: der Unterschied ist klein, und die Teststärke leidet am stärksten.",
    "Die Lage verschiebt sich": "Alle Regeln gleich, aber die Gesamtlage steigt über das Experiment um 3 Prozentpunkte: der Bandit sieht spätere Fenster öfter und meldet einen Unterschied, den es nicht gibt.",
}


def fmt_pct(x, digits=1):
    """Anteil (0..1) als Prozentzahl mit Dezimalkomma, z. B. 0,934 -> '93,4 %'."""
    return f"{100 * x:.{digits}f}".replace(".", ",") + " %"


def fmt_pp(x, digits=2):
    """Differenz in Prozentpunkten (aus Anteilen 0..1) mit Vorzeichen und Dezimalkomma."""
    return f"{100 * x:+.{digits}f}".replace(".", ",") + " Prozentpunkte"


def fmt_pkt(x, digits=1):
    """Kurzform für enge Kacheln: '+4,0 Pkt.' (Differenz in Prozentpunkten, aus Anteilen 0..1)."""
    return f"{100 * x:+.{digits}f}".replace(".", ",") + " Pkt."


def fmt_num(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_int(x):
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(x):,}".replace(",", " ")
