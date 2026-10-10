"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bva_evaluation as E  # noqa: E402


@pytest.fixture
def mini_demands():
    """Mini-Instanz: Tagesnachfragen 20, 60, 10, 25 mit einem Einschwingtag (Tag 0), Mittel 20, z = 1,0, Wiederbeschaffungszeit 2.

    Handrechnung (S = round(2 * 20 + 1,0 * sqrt(2 * (20 + 400/5))) = round(40 + 14,14) = 54, Start-Bestand 40 + 14 = 54):
      Tag 0: Bestand 54, Nachfrage 20 bedient, Rest 34; Position 34; Bestellung 20; unterwegs [0, 20]
      Tag 1: Eingang 0 -> 34; Nachfrage 60, bedient 34, Fehlmenge 26; Bestand -26; Position -26 + 20 = -6; Bestellung 60; unterwegs [20, 60]
      Tag 2: Eingang 20 -> -6; Nachfrage 10, bedient 0, Fehlmenge 10; Bestand -16; Position -16 + 60 = 44; Bestellung 10; unterwegs [60, 10]
      Tag 3: Eingang 60 -> 44; Nachfrage 25, bedient 25; Bestand 19
    Fenster (Tage 1 bis 3): Nachfrage 95, Fehlmenge 36, Servicegrad 1 - 36/95."""
    return [20, 60, 10, 25]


@pytest.fixture(scope="session")
def pools():
    return E.load_pools()
