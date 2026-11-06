import numpy as np
import pandas as pd

from bolan.models import purged_walk_forward
from bolan.simulate import forvantad_bana
from bolan.strategies import Bank, Del, Lan, Strategi, ranteskillnadsersattning, utvardera

TN = ["3M", "1Y", "2Y", "3Y", "5Y", "10Y"]


def _platt(n=50, M=121, nivå=3.0):
    return np.full((n, M, 6), nivå)


def _bank(rabatt=0.0):
    return Bank("A", {t: 3.0 for t in TN}, {t: rabatt for t in TN}, nuvarande=True)


def test_purged_walk_forward_ingen_lacka():
    for tr, te in purged_walk_forward(300, 5, gap=24):
        assert tr.max() < te.min() - 24 + 1


def test_platt_kurva_ger_samma_kostnad_for_alla_tenorer():
    R = _platt()
    lan = Lan(belopp=1_000_000, ranteavdrag=False, diskonteringsranta=0.0)
    kost = [utvardera(Strategi(t, [Del(1, t)]), _bank(), lan, R)["pv"].mean() for t in TN]
    assert np.allclose(kost, kost[0], rtol=1e-6)
    assert np.isclose(kost[0], 1_000_000 * 0.03 * 10, rtol=1e-6)


def test_rabatt_sanker_kostnad_och_passivitet_kostar():
    R = _platt()
    lan = Lan(belopp=1_000_000, ranteavdrag=False, diskonteringsranta=0.0)
    aktiv = utvardera(Strategi("r", [Del(1, "3M")]), _bank(0.5), lan, R)["snittranta"].mean()
    passiv = utvardera(Strategi("r", [Del(1, "3M")], forhandla=False), _bank(0.5), lan, R)["snittranta"].mean()
    assert np.isclose(aktiv, 2.5)
    assert passiv > aktiv


def test_ranteavdrag():
    R = _platt()
    lan = Lan(belopp=1_000_000, ranteavdrag=True, diskonteringsranta=0.0)
    pv = utvardera(Strategi("r", [Del(1, "3M")]), _bank(), lan, R)["pv"].mean()
    assert np.isclose(pv, 300_000 * 0.7, rtol=1e-6)  # 30 000 kr/år < 100 000 -> 30 %


def test_rse():
    assert ranteskillnadsersattning(1_000_000, 5.0, 3.0, 24, paslag=1.0) == 20_000
    assert ranteskillnadsersattning(1_000_000, 3.0, 3.0, 24) == 0


def test_forvantad_bana_interpolerar():
    b = forvantad_bana(3.0, {12: 1.0, 24: 1.0}, 24)
    assert b[0] == 3.0 and np.isclose(b[6], 3.5) and np.isclose(b[24], 4.0)
