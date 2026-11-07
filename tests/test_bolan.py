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


def test_netto_hyra_schablon():
    from bolan.hushall import netto_hyra
    assert netto_hyra(3_000) == 3_000                       # 36 000 kr/år < schablon -> skattefritt
    # 120 000 kr/år: överskott 120 000 − 40 000 − 24 000 = 56 000, skatt 30 % = 16 800
    assert np.isclose(netto_hyra(10_000), (120_000 - 16_800) / 12)


def test_riskprofil_klassning_foljer_marginal():
    from bolan.hushall import riskprofil
    bas = dict(belopp=3_000_000, bostadsvarde=4_000_000, antal_lantagare=2, nettoinkomst_manad=55_000,
               ovriga_boendekostnader_manad=5_000, levnadskostnader_manad=25_000)
    rik = riskprofil(Lan(**bas | {"nettoinkomst_manad": 90_000}), 3.0)
    fattig = riskprofil(Lan(**bas | {"nettoinkomst_manad": 40_000}), 3.0)
    assert rik["lambda"] < fattig["lambda"]
    assert rik["klass"] == "Hög" and fattig["klass"] == "Mycket låg"
    assert rik["brytranta"]["Med hyra"] > fattig["brytranta"]["Med hyra"]


def test_brytkostnad_bara_for_bundet_och_vid_fallande_rantor():
    n, M = 20, 61
    R = np.full((n, M, 6), 4.0)
    R[:, 12:, :] = 2.0                 # räntan faller efter ett år -> RSE om man löser bundet lån
    b = Bank("A", {t: 4.0 for t in TN}, {t: 0.0 for t in TN}, nuvarande=True)
    lan = Lan(belopp=2_000_000, ranteavdrag=False, horisont_ar=5, amorteringskrav=False, sannolikhet_losen_ar=0.10)
    bobl = {t: 3.0 for t in TN}
    rorlig = utvardera(Strategi("r", [Del(1, "3M")]), b, lan, R, bobl)
    bunden = utvardera(Strategi("b", [Del(1, "5Y")]), b, lan, R, bobl)
    assert rorlig["brytkostnad"].max() == 0
    assert bunden["brytkostnad"].mean() > 0
    lan0 = Lan(belopp=2_000_000, ranteavdrag=False, horisont_ar=5, amorteringskrav=False, sannolikhet_losen_ar=0.0)
    assert utvardera(Strategi("b", [Del(1, "5Y")]), b, lan0, R, bobl)["brytkostnad"].max() == 0


def test_formogenhetsvarde_okar_med_alternativ_avkastning():
    R = _platt()
    b = _bank()
    s = Strategi("r", [Del(1, "3M")])
    lag = utvardera(s, b, Lan(belopp=1_000_000, ranteavdrag=False, alternativ_avkastning=0.0), R)["fv_kostnad"].mean()
    hog = utvardera(s, b, Lan(belopp=1_000_000, ranteavdrag=False, alternativ_avkastning=5.0), R)["fv_kostnad"].mean()
    assert np.isclose(lag, 300_000) and hog > lag


def test_vantekalkyl_tecken():
    from bolan.timing import vantekalkyl
    tim = {f"dfix_{t}_{h}": {"prognos": 0.0, "RMSE": 0.1, "anvands": True} for t in ["2Y", "3Y", "5Y"] for h in [1, 3]}
    tim["dfix_3Y_1"]["prognos"] = 0.30          # bunden ränta väntas stiga kraftigt -> vänta inte
    pers = {"3M": 3.0, "2Y": 3.2, "3Y": 3.2, "5Y": 3.4}
    v = vantekalkyl(tim, pers, 1_000_000)
    assert v.loc["3Y: vänta 1 mån", "Förväntad vinst av att vänta kr"] < 0
    # oförändrad bunden ränta men rörligt billigare -> att vänta ger en liten vinst
    assert v.loc["2Y: vänta 1 mån", "Förväntad vinst av att vänta kr"] > 0
