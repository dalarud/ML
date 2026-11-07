"""Feature engineering och målvariabler för räntemodellerna."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TENORS

# Horisonter (månader) för prognos av den rörliga räntans förändring
PROGNOS_H = [3, 6, 12, 24, 36, 60]
# Bindningstider (månader) där vi predikterar "bindningspremien i efterhand"
BINDNING_K = {"1Y": 12, "2Y": 24, "3Y": 36, "5Y": 60}


def bygg_features(p: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=p.index)
    r3 = p["bolan_3M"]
    for t in TENORS:
        f[f"bolan_{t}"] = p[f"bolan_{t}"]
        if t != "3M":
            f[f"spread_{t}"] = p[f"bolan_{t}"] - r3          # kurvans lutning för bolån
    for c in ["styrranta", "ssvx_3m", "stat_2y", "stat_5y", "stat_10y", "bobl_2y", "bobl_5y"]:
        f[c] = p[c]
    # Bankmarginaler och marknadsspreadar
    f["marginal_3M"] = r3 - p["ssvx_3m"]
    f["marginal_5Y"] = p["bolan_5Y"] - p["bobl_5y"]
    f["kurva_10_2"] = p["stat_10y"] - p["stat_2y"]
    f["kredit_5y"] = p["bobl_5y"] - p["stat_5y"]
    f["forvantan_2y"] = p["stat_2y"] - p["styrranta"]    # marknadens förväntade styrränteändring
    f["ssvx_styr"] = p["ssvx_3m"] - p["styrranta"]
    # Momentum
    for c in ["bolan_3M", "bolan_5Y", "styrranta", "stat_2y", "bobl_5y", "de_10y", "us_10y"]:
        for h in (3, 6, 12):
            f[f"d{h}_{c}"] = p[c] - p[c].shift(h)
    # Inflation och realränta
    f["kpif_yoy"] = p["kpif_yoy"]
    f["kpif_3m_ann"] = p["kpif_3m_ann"]
    f["inflationsgap"] = p["kpif_yoy"] - 2.0
    f["realranta"] = p["styrranta"] - p["kpif_yoy"]
    # Volatilitet (månadsspann i dagliga noteringar, glidande)
    f["vol_bobl5"] = p["bobl_5y_spann"].rolling(6).mean()
    f["vol_stat2"] = p["stat_2y_spann"].rolling(6).mean()
    # Nivå relativt lång historik (mean reversion-signal)
    f["r3_vs_5arsmedel"] = r3 - r3.rolling(60, min_periods=24).mean()
    # Realekonomi och kärninflation (Riksbankens utfallsdata, med publiceringsfördröjning)
    if "kpifxe" in p:
        f["kpifxe_yoy"] = p["kpifxe"]
        f["karninfl_gap"] = p["kpifxe"] - 2.0
        f["realranta_karna"] = p["styrranta"] - p["kpifxe"]
        f["d6_kpifxe"] = p["kpifxe"] - p["kpifxe"].shift(6)
    if "bnp" in p:
        f["bnp_yoy"] = p["bnp"]
        f["bnpgap"] = p["bnpgap"]
        f["d12_bnpgap"] = p["bnpgap"] - p["bnpgap"].shift(12)
    if "arbl" in p:
        # Arbetslöshet finns från 2001; före dess fylls nivån med första värdet och förändringen med 0
        arbl = p["arbl"].fillna(p["arbl"].dropna().iloc[0])
        f["arbetsloshet"] = arbl
        f["d12_arbetsloshet"] = (p["arbl"] - p["arbl"].shift(12)).fillna(0.0)
    if "kix" in p:
        f["d12_kix_pct"] = (p["kix"] / p["kix"].shift(12) - 1) * 100   # + = svagare krona
    return f


def bygg_mal(p: pd.DataFrame) -> pd.DataFrame:
    """Framåtblickande mål. NaN där framtiden ännu inte är känd."""
    r3 = p["bolan_3M"]
    y = pd.DataFrame(index=p.index)
    for h in PROGNOS_H:
        y[f"d3m_{h}"] = r3.shift(-h) - r3
    for t, k in BINDNING_K.items():
        # Genomsnittlig rörlig ränta kommande k månader minus bunden ränta idag.
        # > 0  => att binda hade varit billigare.
        snitt = r3[::-1].rolling(k).mean()[::-1].shift(-1)
        y[f"premie_{t}"] = snitt - p[f"bolan_{t}"]
    return y


def dataset(p: pd.DataFrame):
    X = bygg_features(p)
    Y = bygg_mal(p)
    ok = X.notna().all(axis=1)
    vikt = np.where(p["proxy"] == 1, 0.5, 1.0)  # proxydata väger mindre
    return X[ok], Y[ok], pd.Series(vikt, index=p.index)[ok]


# ---------------------------------------------------------------------------
# Makro: styrränta och inflation
# ---------------------------------------------------------------------------
STYR_H = [3, 6, 12, 24]   # horisonter för styrränteprognoser (månader)
KPIF_H = [6, 12, 24]      # horisonter för inflationsprognoser (månader)


def _manadsbana(g: pd.DataFrame) -> pd.Series:
    """En prognosomgång -> månadsserie (kvartalsdata interpoleras linjärt)."""
    s = g.set_index(g["datum"].dt.to_period("M").dt.to_timestamp())["varde"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(idx).interpolate()


def realtidsprognoser(p: pd.DataFrame, prog: pd.DataFrame) -> pd.DataFrame:
    """Riksbankens och marknadens prognoser så som de såg ut vid varje månad t.

    Vid månad t används den senaste prognosomgång som publicerats senast i slutet av t.
    Före 2015 (inga sparade omgångar) fylls värdena med en marknadsproxy och
    `har_rb_prognos` = 0, så att modellerna kan skilja på dem.
    """
    ut = pd.DataFrame(index=p.index)
    banor = {}
    for (serie, omg), g in prog.groupby(["serie", "omgang"]):
        banor.setdefault(serie, []).append((g["publicerad"].iloc[0], _manadsbana(g)))
    for serie in banor:
        banor[serie].sort(key=lambda x: x[0])

    def varde(serie, t, h):
        manadsslut = t + pd.offsets.MonthEnd(0)
        kand = [b for pub, b in banor.get(serie, []) if pub <= manadsslut]
        if not kand:
            return np.nan
        b = kand[-1]
        mal = t + pd.DateOffset(months=h)
        return float(b.loc[:mal].iloc[-1]) if mal >= b.index[0] else np.nan

    styr, kpif = p["styrranta"], p["kpif_yoy"]
    for h in STYR_H:
        ut[f"rb_dstyr_{h}"] = [varde("rb_styr", t, h) for t in p.index] - styr
        ut[f"mkt_dstyr_{h}"] = [varde("mkt_styr", t, h) for t in p.index] - styr
    for h in KPIF_H:
        ut[f"rb_dkpif_{h}"] = [varde("rb_kpif", t, h) for t in p.index] - kpif
    # Riksbankens syn på kärninflation, resursutnyttjande och arbetsmarknad 12 mån framåt
    if "kpifxe" in p:
        ut["rb_dkpifxe_12"] = [varde("rb_kpifxe", t, 12) for t in p.index] - p["kpifxe"]
    if "bnpgap" in p:
        ut["rb_bnpgap_12"] = [varde("rb_bnpgap", t, 12) for t in p.index]
    if "arbl" in p:
        ut["rb_darbl_12"] = [varde("rb_arbl", t, 12) for t in p.index] - p["arbl"]
    ut["har_rb_prognos"] = ut["rb_dstyr_12"].notna().astype(float)
    # Proxy före 2015: marknadens 2-årsränta minus styrränta, skalad till horisonten
    for h in STYR_H:
        proxy = (p["stat_2y"] - styr) * min(h, 24) / 24
        ut[f"rb_dstyr_{h}"] = ut[f"rb_dstyr_{h}"].fillna(proxy)
        ut[f"mkt_dstyr_{h}"] = ut[f"mkt_dstyr_{h}"].fillna(proxy)
    for h in KPIF_H:
        ut[f"rb_dkpif_{h}"] = ut[f"rb_dkpif_{h}"].fillna(0.0)
    for c in ("rb_dkpifxe_12", "rb_darbl_12"):
        if c in ut:
            ut[c] = ut[c].fillna(0.0)
    if "rb_bnpgap_12" in ut:
        ut["rb_bnpgap_12"] = ut["rb_bnpgap_12"].fillna(p["bnpgap"])   # proxy: dagens gap
    return ut


def bygg_makromal(p: pd.DataFrame) -> pd.DataFrame:
    y = pd.DataFrame(index=p.index)
    for h in STYR_H:
        y[f"dstyr_{h}"] = p["styrranta"].shift(-h) - p["styrranta"]
    for h in KPIF_H:
        y[f"dkpif_{h}"] = p["kpif_yoy"].shift(-h) - p["kpif_yoy"]
    return y


def vikter(p: pd.DataFrame, halveringstid_ar: float | None = None) -> pd.Series:
    """Observationsvikter: proxydata väger hälften; valfritt avtagande vikt för äldre data."""
    v = pd.Series(np.where(p["proxy"] == 1, 0.5, 1.0), index=p.index)
    if halveringstid_ar:
        alder = (p.index[-1] - p.index).days / 365.25
        v = v * 0.5 ** (alder / halveringstid_ar)
    return v


UTOKADE = ("kpifxe", "bnp", "bnpgap", "arbl", "kix")


def dataset_makro(p: pd.DataFrame, prog: pd.DataFrame, halveringstid_ar: float | None = None,
                  utokade=True):
    """Features inkl. Riksbankens/marknadens realtidsprognoser, plus alla mål.

    `utokade`: True = alla realekonomiska serier, False = inga, eller en lista med de
    serier (ur UTOKADE) som ska användas.
    """
    behall = UTOKADE if utokade is True else (() if utokade is False else tuple(utokade))
    q = p.drop(columns=[c for c in UTOKADE if c in p and c not in behall])
    X = bygg_features(q).join(realtidsprognoser(q, prog))
    Y = bygg_mal(p).join(bygg_makromal(p))
    ok = X.notna().all(axis=1)
    return X[ok], Y[ok], vikter(p, halveringstid_ar)[ok]
