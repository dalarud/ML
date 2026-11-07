"""Hushållets ekonomi och riskprofil (riskbärande förmåga).

Riskprofilen bygger på hur mycket marginal hushållet har kvar efter boende- och
levnadskostnader i ett antal stresscenarier – samma tanke som bankernas
kvar-att-leva-på-kalkyl (KALP), men med flera samtidiga påfrestningar. Marginalen
översätts till en rekommenderad riskaversion λ för strategivalet:

    marginal i huvudstresstestet (7 % ränta + ingen hyresintäkt), som andel av levnadskostnaderna
      ≥ 30 %   -> Hög riskbärande förmåga     λ = 0,15
      10–30 %  -> Medel                       λ = 0,35
      0–10 %   -> Låg                         λ = 0,60
      < 0      -> Mycket låg                  λ = 1,00

λ kan alltid överstyras i profilen om din riskvilja skiljer sig från förmågan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategies import Lan, amorteringsplan

KLASSNING = [(0.30, "Hög", 0.15), (0.10, "Medel", 0.35), (0.0, "Låg", 0.60), (-np.inf, "Mycket låg", 1.00)]


def netto_hyra(brutto_manad: float, kapitalskatt: float = 0.30) -> float:
    """Privatuthyrning av del av egen villa: schablonavdrag 40 000 kr + 20 % av hyran."""
    ar = brutto_manad * 12
    overskott = max(ar - 40_000 - 0.20 * ar, 0.0)
    return (ar - overskott * kapitalskatt) / 12


def total_nettoinkomst(lan: Lan, med_hyra: bool = True, inkomstbortfall: float = 0.0) -> float:
    """Lön efter skatt + barnbidrag + hyra efter skatt. `inkomstbortfall` = andel av lönerna som faller bort."""
    lon = (lan.nettoinkomst_manad or 0.0) * (1 - inkomstbortfall)
    hyra = netto_hyra(lan.hyresintakt_manad) if (med_hyra and lan.hyresintakt_manad) else 0.0
    return lon + lan.barnbidrag_manad + hyra


def boendekostnad(lan: Lan, ranta: float, saldo: float, amort: float) -> tuple[float, float]:
    rta_ar = saldo * ranta / 100
    tak = 100_000 * max(lan.antal_lantagare, 1)
    avdrag = (0.30 * min(rta_ar, tak) + 0.21 * max(rta_ar - tak, 0)) if lan.ranteavdrag else 0.0
    netto_rta = (rta_ar - avdrag) / 12
    return netto_rta, netto_rta + amort + lan.ovriga_boendekostnader_manad + lan.andra_lan_manad


def stresstabell(lan: Lan, nuvarande_ranta: float) -> pd.DataFrame:
    plan = amorteringsplan(lan, 12)
    saldo, amort = plan["saldo"].mean(), plan["amortering"].mean()
    scen = [
        ("Idag", nuvarande_ranta, True, 0.0),
        ("Ränta 5 %", 5.0, True, 0.0),
        ("Ränta 7 % (bankernas kalkylränta)", 7.0, True, 0.0),
        ("Ränta 7 % + ingen hyresintäkt", 7.0, False, 0.0),
        ("Ränta 7 % + ingen hyra + 20 % lägre löner", 7.0, False, 0.20),
        ("Ränta 5 % + ingen hyra + 20 % lägre löner", 5.0, False, 0.20),
    ]
    rader = {}
    for namn, r, hyra, bort in scen:
        netto_rta, boende = boendekostnad(lan, r, saldo, amort)
        ink = total_nettoinkomst(lan, hyra, bort)
        kvar = ink - boende - lan.levnadskostnader_manad
        rader[namn] = {"Bolåneränta %": r, "Inkomst netto kr/mån": ink, "Ränta efter avdrag kr/mån": netto_rta,
                       "Boendekostnad kr/mån": boende, "Levnadskostnader kr/mån": lan.levnadskostnader_manad,
                       "Kvar att leva på kr/mån": kvar,
                       "Marginal (andel av levnadskostn.)": kvar / lan.levnadskostnader_manad
                       if lan.levnadskostnader_manad else np.nan}
    return pd.DataFrame(rader).T


def riskprofil(lan: Lan, nuvarande_ranta: float) -> dict:
    tab = stresstabell(lan, nuvarande_ranta)
    huvud = tab.loc["Ränta 7 % + ingen hyresintäkt", "Marginal (andel av levnadskostn.)"]
    for grans, klass, lam in KLASSNING:
        if huvud >= grans:
            break
    plan = amorteringsplan(lan, 12)
    per_pe = boendekostnad(lan, nuvarande_ranta + 1, plan["saldo"].mean(), 0)[0] - \
        boendekostnad(lan, nuvarande_ranta, plan["saldo"].mean(), 0)[0]
    # Hur hög kan räntan bli innan kvar att leva på blir noll (med respektive utan hyra)?
    def brytranta(med_hyra, bort=0.0):
        for r in np.arange(0, 20.01, 0.05):
            _, b = boendekostnad(lan, r, plan["saldo"].mean(), plan["amortering"].mean())
            if total_nettoinkomst(lan, med_hyra, bort) - b - lan.levnadskostnader_manad < 0:
                return float(r)
        return np.inf
    sv = tab["Kvar att leva på kr/mån"]
    kr = f"{sv.loc['Ränta 7 % + ingen hyresintäkt']:,.0f}".replace(",", " ")
    motivering = (f"Huvudstresstestet (7 % ränta och ingen hyresintäkt) lämnar {kr} kr/mån, "
                  f"{huvud:.0%} av levnadskostnaderna → riskbärande förmåga {klass.lower()}.")
    return {"tabell": tab, "klass": klass, "lambda": lam, "marginal": float(huvud), "motivering": motivering,
            "kostnad_per_pe": per_pe,
            "brytranta": {"Med hyra": brytranta(True), "Utan hyra": brytranta(False),
                          "Utan hyra, 20 % lägre löner": brytranta(False, 0.20)}}
