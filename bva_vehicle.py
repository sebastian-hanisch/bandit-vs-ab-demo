"""Das Vehikel: ein Standort-Fenster. Ein Standort mit negativ-binomialer Tagesnachfrage bestellt täglich nach einer
Order-up-to-Regel mit Sicherheitsfaktor z nach (Wiederbeschaffungszeit 2 Tage, Zentrallager unbegrenzt). Eine Einheit des
Experiments ist der Servicegrad (Typ 2: Anteil der Nachfrage, der sofort aus dem Lagerbestand bedient wird) in einem
Auswertefenster von 28 Tagen nach 28 Einschwingtagen mit derselben Regel.

Zeitablauf je Tag: (1) Wareneingang (am Ende von Tag t-LEAD bestellt), (2) Nachfrage, sofort bedient aus dem Bestand,
Rest wird Rückstand (negativer Netto-Bestand), (3) Bestellung bis zum Niveau S = round(LEAD*mu + z*sd) über der
Bestandsposition (Netto-Bestand + Unterwegs). Gleiche Modellgleichungen wie das Vehikel der Messreihen dieser Linie."""

import math

import bva_constants as C
from bva_rng import SplitMix64


def order_up_to_level(z, mean=C.BASE_MEAN, disp=C.DISP, lead=C.LEAD):
    """Bestellniveau S: Nachfrage der Schutzzeit plus z Standardabweichungen (i.i.d.-Näherung der Tagesnachfrage)."""
    sd = math.sqrt(lead * (mean + mean * mean / disp))
    return round(lead * mean + z * sd)


def initial_net_inventory(z, mean=C.BASE_MEAN, disp=C.DISP, lead=C.LEAD):
    """Startbestand: Bedarf der Schutzzeit plus Sicherheitsbestand (die Einschwingtage gleichen den Rest aus)."""
    sd = math.sqrt(lead * (mean + mean * mean / disp))
    return round(lead * mean) + round(z * sd)


def draw_demand(rng, mean=C.BASE_MEAN):
    """Eine Tagesnachfrage: Gamma-Poisson. Die Rate ist mean * G / 5 mit G ~ Gamma(5, 1) (Summe von 5 Exponentialzahlen),
    die Nachfrage Poisson(Rate) (Inversion). Mittel `mean`, Varianz mean + mean^2 / 5."""
    g = rng.exponential() + rng.exponential() + rng.exponential() + rng.exponential() + rng.exponential()
    lam = mean * g / C.DISP
    u = rng.uniform()
    p = math.exp(-lam)
    cdf = p
    k = 0
    while u > cdf and k < 10_000:
        k += 1
        p *= lam / k
        cdf += p
    return k


def inventory_step(net, pipeline, demand, level):
    """Ein Tag nach dem Wareneingang: Nachfrage bedienen und bestellen.
    Rückgabe: (neuer Netto-Bestand, sofort bedient, neue Bestellung)."""
    served = min(max(net, 0), demand)
    net_after = net - demand
    position = net_after + sum(pipeline)
    order = max(0, level - position)
    return net_after, served, order


def window_from_demands(z, demands, burn, mean=C.BASE_MEAN, lead=C.LEAD):
    """Das Fenster aus vorgegebenen Tagesnachfragen (die ersten `burn` Tage schwingen ein, der Rest ist das Fenster).
    Rückgabe: (Nachfrage im Fenster, Fehlmenge im Fenster). Reine Funktion: gleiche Nachfragen, gleiches Ergebnis."""
    level = order_up_to_level(z, mean, C.DISP, lead)
    net = initial_net_inventory(z, mean, C.DISP, lead)
    pipe = [0] * lead                      # pipe[0] kommt am Anfang des nächsten Tages an
    demand_sum = short_sum = 0
    for day, d in enumerate(demands):
        net += pipe[0]
        net, served, order = inventory_step(net, pipe[1:], d, level)
        pipe = pipe[1:] + [order]
        if day >= burn:
            demand_sum += d
            short_sum += d - served
    return demand_sum, short_sum


def simulate_window(z, rng, mean=C.BASE_MEAN, lead=C.LEAD, burn=C.BURN_DAYS, window=C.WINDOW_DAYS):
    """Ein Standort-Fenster mit Sicherheitsfaktor z. Rückgabe: (Nachfrage im Fenster, Fehlmenge im Fenster)."""
    demands = [draw_demand(rng, mean) for _ in range(burn + window)]
    return window_from_demands(z, demands, burn, mean, lead)


def window_service_level(z, rng, **kwargs):
    """Servicegrad des Fensters = 1 - Fehlmenge / Nachfrage (die Einheit des Experiments)."""
    d, s = simulate_window(z, rng, **kwargs)
    return 1.0 - s / d if d > 0 else 1.0


def run_units(z, count, seed):
    """`count` unabhängige Standort-Fenster aus einem gemeinsamen Zufallsstrom `SplitMix64(seed)`."""
    rng = SplitMix64(seed)
    return [window_service_level(z, rng) for _ in range(count)]
