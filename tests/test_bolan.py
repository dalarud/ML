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


def test_realtidsprognos_anvands_forst_efter_publicering():
    from bolan.features import realtidsprognoser
    idx = pd.date_range("2020-01-01", "2020-06-01", freq="MS")
    p = pd.DataFrame({"styrranta": 1.0, "kpif_yoy": 2.0, "stat_2y": 1.0}, index=idx)
    rader = []
    for serie in ("rb_styr", "mkt_styr", "rb_kpif"):
        for d, v in [("2020-03-31", 1.0), ("2020-06-30", 1.5), ("2021-03-31", 2.0), ("2021-06-30", 2.0)]:
            rader.append((serie, "2020:1", pd.Timestamp("2020-03-20"), pd.Timestamp(d), 3.0 if serie == "rb_kpif" else v))
    prog = pd.DataFrame(rader, columns=["serie", "omgang", "publicerad", "datum", "varde"])
    ut = realtidsprognoser(p, prog)
    # Före publicering (jan–feb): ingen Riksbanksprognos, proxy används och flaggan är 0
    assert (ut.loc[:"2020-02-01", "har_rb_prognos"] == 0).all()
    # Från mars: prognosen används
    assert (ut.loc["2020-03-01":, "har_rb_prognos"] == 1).all()
    assert np.isclose(ut.loc["2020-03-01", "rb_dstyr_12"], 1.0)  # 2021-03 -> 2.0 minus 1.0
    assert np.isclose(ut.loc["2020-03-01", "rb_dkpif_12"], 1.0)


def test_realtids_stack_ingen_framtidsinformation():
    from bolan.models import realtids_stack
    rng = np.random.default_rng(0)
    n = 120
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    X = pd.DataFrame(rng.normal(size=(n, 3)), index=idx)
    Y = pd.DataFrame({"dstyr_12": rng.normal(size=n)}, index=idx)
    w = pd.Series(1.0, index=idx)
    a = realtids_stack(X, Y, w, ["dstyr_12"])
    # Ändra framtida utfall – tidigare prognoser får inte påverkas
    Y2 = Y.copy()
    Y2.iloc[100:] = 99.0
    b = realtids_stack(X, Y2, w, ["dstyr_12"])
    assert np.allclose(a.iloc[:100], b.iloc[:100])


def test_klassa_granser():
    from bolan.risk import klassa
    assert list(klassa(np.array([-2, -0.5, 0, 0.5, 2]))) == [0, 1, 2, 3, 4]


def test_omviktning_traffar_malsannolikheter():
    from bolan.risk import klassa, omvikta
    rng = np.random.default_rng(1)
    n, M = 4000, 40
    R = np.zeros((n, M, 6))
    R[:, :, 0] = 3.0 + np.cumsum(rng.normal(0, 0.25, (n, M)), axis=1) - rng.normal(0, 0.25, (n, 1)) * 0
    R[:, 0, 0] = 3.0
    mal = {12: np.array([0.05, 0.15, 0.2, 0.3, 0.3]), 36: np.array([0.2, 0.2, 0.2, 0.2, 0.2])}
    w, ess = omvikta(R, mal)
    for h, p in mal.items():
        akt = np.bincount(klassa(R[:, h, 0] - 3.0), weights=w, minlength=5)
        assert np.allclose(akt, p, atol=0.02)
    assert 0 < ess <= n and np.isclose(w.sum(), 1)


def test_viktade_riskmatt():
    from bolan.risk import vcvar, vkvantil, vmedel
    x = np.arange(100, dtype=float)
    w = np.full(100, 0.01)
    assert np.isclose(vmedel(x, w), 49.5)
    assert np.isclose(vcvar(x, w, 0.95), np.mean(x[95:]), atol=0.6)
    assert vkvantil(x, w, 0.5) in (49.0, 50.0)
    # Mer vikt på höga utfall höjer väntevärdet
    w2 = np.where(x >= 50, 0.015, 0.005)
    assert vmedel(x, w2 / w2.sum()) > 49.5


def test_amorteringskrav():
    from bolan.strategies import amorteringsplan
    plan = amorteringsplan(Lan(belopp=3_000_000, bostadsvarde=4_000_000), 24)   # 75 % -> 2 %
    assert np.isclose(plan["amortering"].iloc[0] * 12, 60_000)
    assert np.isclose(amorteringsplan(Lan(belopp=2_400_000, bostadsvarde=4_000_000), 1)["amortering"].iloc[0] * 12,
                      24_000)                                                    # 60 % -> 1 %
    assert amorteringsplan(Lan(belopp=1_800_000, bostadsvarde=4_000_000), 1)["amortering"].iloc[0] == 0


def test_ranteavdrag_tva_lantagare():
    R = _platt(nivå=4.0)
    s = Strategi("r", [Del(1, "3M")])
    b = Bank("A", {t: 4.0 for t in TN}, {t: 0.0 for t in TN}, nuvarande=True)
    en = Lan(belopp=3_500_000, diskonteringsranta=0.0, amorteringskrav=False)
    tva = Lan(belopp=3_500_000, diskonteringsranta=0.0, amorteringskrav=False, antal_lantagare=2)
    # 140 000 kr ränta/år: en låntagare får bara 21 % på 40 000 kr, två får 30 % på allt
    assert utvardera(s, b, tva, R)["pv"].mean() < utvardera(s, b, en, R)["pv"].mean()
