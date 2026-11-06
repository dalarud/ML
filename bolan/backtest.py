"""Historisk backtest av bindningsstrategier på faktiska genomsnittsräntor.

1. Statiska strategier ("förnya alltid samma bindningstid") över rullande
   5-årsfönster: vilken bindningstid hade varit billigast, och hur ofta?
2. ML-timingstrategi, helt walk-forward: vid varje tidpunkt då lånet är fritt
   tränas modellen ENBART på data som var känd då (målet för bindningstid k är
   känt först k månader senare), och binder den tid med högst förväntad
   bindningspremie – annars rörligt i 3 månader.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .data import TENORS
from .features import BINDNING_K, bygg_features, bygg_mal

STAT = ["3M", "1Y", "2Y", "3Y", "5Y"]


def kor_statisk(p: pd.DataFrame, tenor: str, start: int, slut: int) -> np.ndarray:
    """Månadsräntor från index start till slut när man alltid förnyar `tenor`."""
    r = np.empty(slut - start)
    k = TENORS[tenor]
    t = start
    while t < slut:
        r[t - start:min(t + k, slut) - start] = p[f"bolan_{tenor}"].iloc[t]
        t += k
    return r


def rullande_fonster(p: pd.DataFrame, ar: int = 5, fran: str = "2005-09-01") -> pd.DataFrame:
    i0 = p.index.get_loc(p.index[p.index >= fran][0])
    L = ar * 12
    rader = []
    for s in range(i0, len(p) - L + 1):
        rader.append({t: kor_statisk(p, t, s, s + L).mean() for t in STAT} | {"start": p.index[s]})
    return pd.DataFrame(rader).set_index("start")


def kor_ml_timing(p: pd.DataFrame, fran: str = "2008-01-01", troskel: float = 0.0,
                  omtraning_manader: int = 12, seed: int = 0):
    X = bygg_features(p)
    Y = bygg_mal(p)
    w = np.where(p["proxy"] == 1, 0.5, 1.0)
    ok_x = X.notna().all(axis=1).values
    i0 = p.index.get_loc(p.index[p.index >= fran][0])
    n = len(p)
    rantor = np.empty(n - i0)
    val = []
    modeller, senast = {}, -10**9
    t = i0
    while t < n:
        if t - senast >= omtraning_manader:
            modeller = {}
            for tn, k in BINDNING_K.items():
                y = Y[f"premie_{tn}"].values
                # endast rader vars utfall var känt vid tid t
                tr = np.where(ok_x & ~np.isnan(y) & (np.arange(n) <= t - k - 1))[0]
                m = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.4,
                                          random_state=seed, n_jobs=-1)
                m.fit(X.values[tr], y[tr], sample_weight=w[tr])
                modeller[tn] = m
            senast = t
        pred = {tn: float(m.predict(X.values[[t]])[0]) for tn, m in modeller.items()}
        bast = max(pred, key=pred.get)
        tenor = bast if pred[bast] > troskel else "3M"
        k = TENORS[tenor]
        rantor[t - i0:min(t + k, n) - i0] = p[f"bolan_{tenor}"].iloc[t]
        val.append({"datum": p.index[t], "val": tenor, **{f"pred_{a}": b for a, b in pred.items()}})
        t += k
    stat = {tn: kor_statisk(p, tn, i0, n) for tn in STAT}
    serier = pd.DataFrame({"ML-timing": rantor, **{f"Förnya {a}": b for a, b in stat.items()}},
                          index=p.index[i0:])
    return serier, pd.DataFrame(val).set_index("datum")


def sammanfattning(serier: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "snittränta %": serier.mean(),
        "std %": serier.std(),
        "max %": serier.max(),
        "största 12m-ökning pe": (serier - serier.shift(12)).max(),
    }).sort_values("snittränta %")
