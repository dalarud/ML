"""Scenariogenerator för framtida bolåneräntekurvor.

Kurvan beskrivs som rörlig ränta r(t) plus spreadar s_k(t) = bolan_k − bolan_3M.

* Spreadarna följer en AR(1)-process (mean reversion) skattad på historiken.
* Den rörliga räntans *förväntade* bana kommer från ML-ensemblen (eller
  marknadens prissättning eller ingen förändring, för jämförelse).
* Osäkerheten kommer från block-bootstrap av historiska månadschocker för hela
  kurvan samtidigt (12-månadersblock), vilket bevarar samvariation mellan
  bindningstider och trender/regimer inom ett år.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TENORS

TN = list(TENORS)
GOLV = 0.5  # lägsta tänkbara bolåneränta (%), mjukt golv


@dataclass
class SpreadModell:
    kappa: np.ndarray   # mean reversion-hastighet per månad
    mu: np.ndarray      # långsiktig nivå
    chocker: np.ndarray  # (T, 1+K) residualer: [dr, e_1Y, ..., e_10Y]


def skatta_dynamik(p: pd.DataFrame, fran: str = "1996-01-01") -> SpreadModell:
    d = p[p.index >= fran]
    r = d["bolan_3M"].values
    S = np.column_stack([d[f"bolan_{t}"].values - r for t in TN[1:]])
    kappa, mu, res = [], [], []
    for j in range(S.shape[1]):
        s0, ds = S[:-1, j], np.diff(S[:, j])
        A = np.column_stack([np.ones_like(s0), s0])
        (a, b), *_ = np.linalg.lstsq(A, ds, rcond=None)
        kap = float(np.clip(-b, 0.01, 0.5))
        m = float(a / kap) if kap > 0 else float(s0.mean())
        kappa.append(kap)
        mu.append(m)
        res.append(ds - kap * (m - s0))
    dr = np.diff(r)
    dr = dr - dr.mean()
    return SpreadModell(np.array(kappa), np.array(mu), np.column_stack([dr] + res))


def forvantad_bana(nu: float, prognos: dict[int, float], manader: int, lang_niva: float | None = None,
                   halveringstid_efter: int = 60) -> np.ndarray:
    """Interpolerar förväntad rörlig ränta månad för månad från horisontprognoser.

    Efter sista prognoshorisonten dras banan långsamt mot `lang_niva`.
    """
    h = np.array([0] + sorted(prognos))
    v = np.array([0.0] + [prognos[k] for k in sorted(prognos)])
    m = np.arange(manader + 1)
    bana = nu + np.interp(m, h, v)
    if lang_niva is not None:
        sist = h[-1]
        efter = m > sist
        decay = 0.5 ** ((m[efter] - sist) / halveringstid_efter)
        bana[efter] = lang_niva + (bana[sist] - lang_niva) * decay
    return bana


def marknadens_bana(kurva_nu: dict[str, float], premie_hist: dict[str, float], manader: int) -> np.ndarray:
    """Förväntad rörlig ränta implicit i dagens bundna räntor (förväntningshypotesen).

    Snitt av rörlig ränta över k mån ≈ bunden_k − historisk bindningspremie_k.
    Snitt -> punktbana via differens av kumulativa summor.
    """
    pts = {0: kurva_nu["3M"]}
    for t in ["1Y", "2Y", "3Y", "5Y"]:
        pts[TENORS[t]] = kurva_nu[t] + premie_hist.get(t, 0.0)  # premie = snitt_rörlig − bunden
    ks = sorted(pts)
    cum = {k: pts[k] * k for k in ks}
    nivå = {}
    for a, b in zip(ks[:-1], ks[1:]):
        nivå[(a + b) / 2] = (cum[b] - cum[a]) / (b - a) if a > 0 else pts[b]
    x = np.array([0] + list(nivå))
    y = np.array([kurva_nu["3M"]] + list(nivå.values()))
    return np.interp(np.arange(manader + 1), x, y)


def simulera(kurva_nu: dict[str, float], bana: np.ndarray, dyn: SpreadModell, n: int = 3000,
             block: int = 12, aterforing: float = 0.01, seed: int = 0) -> np.ndarray:
    """Returnerar array (n, M+1, 6) med simulerade marknadssnitträntor per tenor."""
    rng = np.random.default_rng(seed)
    M = len(bana) - 1
    T = len(dyn.chocker)
    # Block-bootstrap av chockindex
    nblock = int(np.ceil(M / block))
    starter = rng.integers(0, T - block, size=(n, nblock))
    idx = (starter[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :M]
    ch = dyn.chocker[idx]  # (n, M, 1+K)

    ut = np.empty((n, M + 1, len(TN)))
    avv = np.zeros(n)                       # avvikelse från förväntad bana
    s = np.array([kurva_nu[t] - kurva_nu["3M"] for t in TN[1:]])[None, :].repeat(n, 0)
    r = np.full(n, bana[0])
    ut[:, 0, 0] = r
    ut[:, 0, 1:] = r[:, None] + s
    for m in range(1, M + 1):
        avv = (1 - aterforing) * avv + ch[:, m - 1, 0]
        r = bana[m] + avv
        # mjukt golv: reflektera under golvet
        r = np.where(r < GOLV, GOLV + 0.3 * (GOLV - r), r)
        s = s + dyn.kappa * (dyn.mu - s) + ch[:, m - 1, 1:]
        ut[:, m, 0] = r
        ut[:, m, 1:] = np.maximum(r[:, None] + s, GOLV)
    return ut
