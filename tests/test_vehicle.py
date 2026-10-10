"""Das Vehikel: jede Einheit an der Mini-Instanz von Hand, dazu Momente der Nachfrage und Reproduzierbarkeit."""

import math

import numpy as np

import bva_constants as C
import bva_vehicle as V
from bva_rng import SplitMix64


def test_order_up_to_level_and_start_inventory_by_hand():
    # sd = sqrt(2 * (20 + 400/5)) = sqrt(200) = 14,142; S = round(40 + 14,142) = 54; Start = 40 + 14
    assert V.order_up_to_level(1.0) == 54
    assert V.initial_net_inventory(1.0) == 54
    # z = 2,0: 40 + 28,28 = 68; Start 40 + 28
    assert V.order_up_to_level(2.0) == 68
    assert V.initial_net_inventory(2.0) == 68


def test_inventory_step_serves_from_stock_and_orders_up_to_the_level():
    # Bestand 34, unterwegs [20], Nachfrage 60, Niveau 54: bedient 34, Bestand -26, Position -6, Bestellung 60
    assert V.inventory_step(34, [20], 60, 54) == (-26, 34, 60)
    # Nachfrage kleiner als Bestand: alles bedient
    assert V.inventory_step(34, [0], 20, 54) == (14, 20, 40)
    # Rückstand (negativer Bestand) bedient nichts mehr
    assert V.inventory_step(-6, [60], 10, 54) == (-16, 0, 10)


def test_window_from_demands_matches_the_hand_calculation(mini_demands):
    demand, short = V.window_from_demands(1.0, mini_demands, burn=1)
    assert (demand, short) == (95, 36)


def test_window_before_the_window_is_not_counted(mini_demands):
    demand, short = V.window_from_demands(1.0, mini_demands, burn=4)
    assert (demand, short) == (0, 0)


def test_higher_safety_factor_never_adds_shortage_on_the_same_demands(mini_demands):
    rng = SplitMix64(7)
    for _ in range(50):
        demands = [V.draw_demand(rng) for _ in range(56)]
        _, s_low = V.window_from_demands(1.0, demands, burn=28)
        _, s_high = V.window_from_demands(2.0, demands, burn=28)
        assert s_high <= s_low


def test_demand_draw_has_the_stated_mean_and_variance():
    rng = SplitMix64(2026)
    draws = np.array([V.draw_demand(rng) for _ in range(40000)], dtype=float)
    assert abs(draws.mean() - C.BASE_MEAN) < 0.15                       # Standardfehler etwa 0,05
    assert abs(draws.var() - (C.BASE_MEAN + C.BASE_MEAN**2 / C.DISP)) < 4.0  # Soll 100


def test_window_is_a_pure_function_of_the_seed():
    assert V.run_units(1.4, 5, seed=11) == V.run_units(1.4, 5, seed=11)
    assert V.run_units(1.4, 5, seed=11) != V.run_units(1.4, 5, seed=12)


def test_service_level_is_a_share_between_zero_and_one():
    values = V.run_units(1.0, 30, seed=3)
    assert all(0.0 <= v <= 1.0 for v in values)
    assert not math.isnan(sum(values))


def test_safety_factor_branch_changes_the_result():
    """Zweig-Test: der Sicherheitsfaktor wirkt (kein toter Regler) - bei gleichen Zufallszahlen weicht z = 2 von z = 1 ab."""
    a = V.run_units(1.0, 40, seed=5)
    b = V.run_units(2.0, 40, seed=5)
    assert np.mean(b) > np.mean(a) + 0.03
