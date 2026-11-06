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
