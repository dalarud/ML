"""Feature engineering och målvariabler för räntemodellerna."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TENORS

# Horisonter (månader) för prognos av den rörliga räntans förändring
PROGNOS_H = [3, 6, 12, 24, 36, 60]
# Bindningstider (månader) där vi predikterar "bindningspremien i efterhand"
BINDNING_K = {"1Y": 12, "2Y": 24, "3Y": 36, "5Y": 60}
# Timing: förändring i bunden ränta de närmaste månaderna ("binda nu eller vänta?")
FIX_TENORER = ["2Y", "3Y", "5Y"]
FIX_H = [1, 3, 6]
GENOMSLAG = 0.93   # historiskt genomslag styrränta -> rörlig bolåneränta (skattat i styrräntekanalen)


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
    for c in ["bolan_3M", "bolan_5Y", "styrranta", "stat_2y", "bobl_2y", "bobl_5y", "de_10y", "us_10y"]:
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
    # --- Timingfeatures ---
    # Bankernas marginal på bundna lån mot finansieringskostnaden (bostadsobligationer)
    f["marg_fix_2Y"] = p["bolan_2Y"] - p["bobl_2y"]
    f["marg_fix_3Y"] = p["bolan_3Y"] - (2 * p["bobl_2y"] + p["bobl_5y"]) / 3
    for c in ("marg_fix_2Y", "marg_fix_3Y", "marginal_5Y"):
        f[f"{c}_vs_12m"] = f[c] - f[c].rolling(12, min_periods=6).mean()   # avvikelse från normal marginal
    # Kortsiktig momentum: bundna räntor följer obligationsräntorna med eftersläpning
    for c in ("bobl_2y", "bobl_5y", "stat_2y", "stat_5y", "bolan_2Y", "bolan_3Y", "bolan_5Y"):
        f[f"d1_{c}"] = p[c] - p[c].shift(1)
    f["eftersl_5Y"] = (p["bobl_5y"] - p["bobl_5y"].shift(3)) - (p["bolan_5Y"] - p["bolan_5Y"].shift(3))
    f["eftersl_2Y"] = (p["bobl_2y"] - p["bobl_2y"].shift(3)) - (p["bolan_2Y"] - p["bolan_2Y"].shift(3))
    # Räntecykel: tid sedan senaste styrränteändring och dess riktning
    d = p["styrranta"].diff().fillna(0)
    andrad = d.abs() > 0.05
    grupp = andrad.cumsum()
    f["man_sedan_styrandring"] = grupp.groupby(grupp).cumcount().astype(float).clip(upper=60)
    f["senaste_styrandring"] = d.where(andrad).ffill().fillna(0.0)
    f["styr_12m_riktning"] = np.sign(p["styrranta"] - p["styrranta"].shift(12)).fillna(0.0)
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
    for t in FIX_TENORER:
        for h in FIX_H:
            y[f"dfix_{t}_{h}"] = p[f"bolan_{t}"].shift(-h) - p[f"bolan_{t}"]
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

    # Ex ante-bindningspremie: förväntad snittränta rörligt under bindningstiden (via styrräntebanan
    # och genomslaget) minus dagens bundna ränta. > 0 = bunden ränta billig relativt förväntningarna.
    if "bolan_3M" in p and all(f"bolan_{t}" in p for t in BINDNING_K):
        aktiv = {}
        for serie in ("rb_styr", "mkt_styr"):
            lista = banor.get(serie, [])
            pubs = [pub for pub, _ in lista]
            aktiv[serie] = [lista[np.searchsorted(pubs, t + pd.offsets.MonthEnd(0), side="right") - 1][1]
                            if pubs and pubs[0] <= t + pd.offsets.MonthEnd(0) else None for t in p.index]
        for i_t, t in enumerate(p.index):
            r3, s0 = p["bolan_3M"].iloc[i_t], styr.iloc[i_t]
            for serie, kort in (("rb_styr", "rb"), ("mkt_styr", "mkt")):
                b = aktiv[serie][i_t]
                for tn, k in BINDNING_K.items():
                    if b is not None:
                        idx = pd.date_range(t + pd.DateOffset(months=1), periods=k, freq="MS")
                        bana = b.reindex(b.index.union(idx)).ffill().bfill().reindex(idx)
                        snitt_styr = float(bana.mean())
                    else:   # proxy före 2015: linjär bana mot 2-årsräntan över 24 mån, sedan flat
                        mal = p["stat_2y"].iloc[i_t]
                        j = np.arange(1, k + 1)
                        snitt_styr = float(np.mean(s0 + (mal - s0) * np.minimum(j, 24) / 24))
                    ut.loc[t, f"exante_{kort}_{tn}"] = r3 + GENOMSLAG * (snitt_styr - s0) - p[f"bolan_{tn}"].iloc[i_t]
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
