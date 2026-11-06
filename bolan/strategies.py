"""Bolånestrategier och kostnadsutvärdering över simulerade räntebanor.

Den personliga räntan modelleras som

    din ränta(τ, k) = marknadssnitt_k(τ) + (listränta_k − marknadssnitt_k idag) − rabatt_k · f

där marknadssnittet är SCB:s genomsnittsränta (vad kunder faktiskt betalar),
`listränta − marknadssnitt` är bankens listpåslag (antas bestå över tid) och
`f` är hur stor del av rabatten du får vid varje ny räntesättning: 1 om du
förhandlar/är beredd att flytta, annars bankens "befintlig kund"-andel. På så
sätt syns att rabatter skiljer sig mellan bindningstider och att passivitet
kostar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import TENORS

TN = list(TENORS)
TM = np.array([TENORS[t] for t in TN], dtype=float)


@dataclass
class Bank:
    namn: str
    listrantor: dict[str, float]
    rabatter: dict[str, float]
    nuvarande: bool = False
    flyttkostnad: float = 0.0
    befintlig_kund_rabattandel: float = 0.6
    rabatt_garanti_manader: int = 12   # hur länge rörlig rabatt gäller innan den kan sänkas


@dataclass
class Lan:
    belopp: float
    horisont_ar: int = 10
    amortering_per_ar: float = 0.0
    nuvarande_tenor: str = "3M"
    nuvarande_ranta: float | None = None
    bundet_kvar_manader: int = 0
    ranteavdrag: bool = True
    diskonteringsranta: float = 2.0
    budget_manad: float | None = None
    rse_paslag: float = 1.0           # påslag (procentenheter) i jämförelseräntan vid RSE


@dataclass
class Del:
    andel: float
    tenor: str
    vanta_manader: int = 0            # ligg rörligt så här länge innan bindning


@dataclass
class Strategi:
    namn: str
    delar: list[Del]
    forhandla: bool = True            # omförhandlar/hotar flytta vid varje räntesättning
    bryt_bindning: bool = False       # lös nuvarande bundna lån direkt (betala RSE)
    kategori: str = "statisk"


def standardstrategier() -> list[Strategi]:
    S = [Strategi("Rörlig 3 mån", [Del(1, "3M")])]
    for t, txt in [("1Y", "1 år"), ("2Y", "2 år"), ("3Y", "3 år"), ("5Y", "5 år"), ("10Y", "10 år")]:
        S.append(Strategi(f"Bind {txt} (förnya)", [Del(1, t)]))
    for v in (6, 12):
        for t, txt in [("2Y", "2 år"), ("3Y", "3 år"), ("5Y", "5 år")]:
            S.append(Strategi(f"Vänta {v} mån, bind {txt}", [Del(1, t, v)], kategori="timing"))
    S += [
        Strategi("Mix 50% rörlig / 50% 2 år", [Del(.5, "3M"), Del(.5, "2Y")], kategori="mix"),
        Strategi("Mix 50% rörlig / 50% 3 år", [Del(.5, "3M"), Del(.5, "3Y")], kategori="mix"),
        Strategi("Mix 50% rörlig / 50% 5 år", [Del(.5, "3M"), Del(.5, "5Y")], kategori="mix"),
        Strategi("Trappa 1/3: rörlig, 2 år, 5 år", [Del(1/3, "3M"), Del(1/3, "2Y"), Del(1/3, "5Y")], kategori="mix"),
        Strategi("Trappa 4×25%: 1, 2, 3, 5 år", [Del(.25, "1Y"), Del(.25, "2Y"), Del(.25, "3Y"), Del(.25, "5Y")],
                 kategori="mix"),
        Strategi("Rörlig 3 mån – passiv (förhandlar ej)", [Del(1, "3M")], forhandla=False, kategori="passiv"),
        Strategi("Bind 3 år – passiv (förhandlar ej)", [Del(1, "3Y")], forhandla=False, kategori="passiv"),
    ]
    return S


def _tenorranta(kurva: np.ndarray, manader: float) -> np.ndarray:
    """Interpolera kurva (n, 6) till given löptid i månader."""
    m = float(np.clip(manader, TM[0], TM[-1]))
    j = int(np.searchsorted(TM, m))
    if j == 0:
        return kurva[:, 0]
    w = (m - TM[j - 1]) / (TM[j] - TM[j - 1])
    return (1 - w) * kurva[:, j - 1] + w * kurva[:, j]


def ranteskillnadsersattning(belopp: float, ranta: float, jamforelse: float, kvar_manader: int,
                             paslag: float = 1.0) -> float:
    """Förenklad RSE: (avtalsränta − (jämförelseränta + påslag)) × kvarvarande tid × belopp."""
    diff = ranta - (jamforelse + paslag)
    return max(0.0, diff) / 100 * belopp * kvar_manader / 12


def utvardera(strategi: Strategi, bank: Bank, lan: Lan, R: np.ndarray, bobl_nu: dict[str, float] | None = None,
              nuvarande_bank: Bank | None = None) -> dict:
    """Kostnad för strategi hos `bank` över simulerade marknadsbanor R (n, M+1, 6)."""
    n, M1, _ = R.shape
    M = min(lan.horisont_ar * 12, M1 - 1)
    R0 = R[0, 0, :]
    listpaslag = np.array([bank.listrantor[t] - R0[i] for i, t in enumerate(TN)])
    rabatt = np.array([bank.rabatter.get(t, 0.0) for t in TN])

    manad = np.arange(M)
    saldo = np.maximum(lan.belopp - lan.amortering_per_ar * manad / 12, 0.0)

    # Nuvarande bundna lån ligger kvar tills villkorsändringsdagen (om vi inte bryter)
    start = 0 if (strategi.bryt_bindning or lan.nuvarande_tenor == "3M") else min(lan.bundet_kvar_manader, M)
    engang = 0.0
    if strategi.bryt_bindning and lan.nuvarande_tenor != "3M" and lan.bundet_kvar_manader > 0:
        jf = float(_tenorranta(np.array([[bobl_nu.get(t, R0[i]) for i, t in enumerate(TN)]]),
                               lan.bundet_kvar_manader)[0]) if bobl_nu else float(R0[0])
        engang += ranteskillnadsersattning(lan.belopp, lan.nuvarande_ranta, jf, lan.bundet_kvar_manader,
                                           lan.rse_paslag)
    byter_bank = nuvarande_bank is not None and bank.namn != nuvarande_bank.namn
    if byter_bank:
        engang += bank.flyttkostnad

    ranta = np.zeros((n, M))
    terminal_vinst = np.zeros(n)
    if start > 0:
        ranta[:, :start] = lan.nuvarande_ranta

    for d in strategi.delar:
        r_del = np.zeros((n, M - start))
        t = start
        senast_forhandlad = start
        # Fas 1: rörligt under väntetid (och hela tiden för 3M-delar)
        fixar = []  # (tid, tenor-index)
        while t < M:
            k = 0 if (d.tenor == "3M" or t < start + d.vanta_manader) else TN.index(d.tenor)
            fixar.append((t, k))
            t += int(TM[k])
        for i, (tau, k) in enumerate(fixar):
            slut = fixar[i + 1][0] if i + 1 < len(fixar) else M
            if strategi.forhandla:
                f = 1.0
            elif k == 0:
                # rörlig: full rabatt under garantitiden, därefter befintlig-kund-andel
                f = 1.0 if tau - senast_forhandlad < bank.rabatt_garanti_manader else bank.befintlig_kund_rabattandel
            else:
                f = 1.0 if tau == start else bank.befintlig_kund_rabattandel
            r = R[:, tau, k] + listpaslag[k] - rabatt[k] * f
            r_del[:, tau - start:slut - start] = r[:, None]
            # Värdet av låst ränta som löper efter horisonten (mark-to-market)
            if tau + TM[k] > M and k > 0:
                kvar = tau + TM[k] - M
                ersattning = _tenorranta(R[:, M, :], kvar) + np.interp(kvar, TM, listpaslag) \
                    - np.interp(kvar, TM, rabatt)
                terminal_vinst += d.andel * saldo[-1] * (ersattning - r) / 100 * kvar / 12
        ranta[:, start:] += d.andel * r_del

    kostnad_m = saldo[None, :] * ranta / 1200          # kr per månad, före skatt
    # Ränteavdrag per kalenderår: 30 % upp till 100 000 kr, 21 % därutöver
    if lan.ranteavdrag:
        ar = manad // 12
        netto = kostnad_m.copy()
        for a in np.unique(ar):
            sel = ar == a
            arsrta = kostnad_m[:, sel].sum(1)
            avdrag = 0.30 * np.minimum(arsrta, 100_000) + 0.21 * np.maximum(arsrta - 100_000, 0)
            andel = np.divide(avdrag, arsrta, out=np.zeros_like(arsrta), where=arsrta > 0)
            netto[:, sel] = kostnad_m[:, sel] * (1 - andel[:, None])
    else:
        netto = kostnad_m
    df = (1 + lan.diskonteringsranta / 100) ** (-(manad + 1) / 12)
    pv = netto @ df + engang - terminal_vinst * (1 - 0.3 * lan.ranteavdrag) * df[-1]
    snittranta = (kostnad_m.sum(1) / (saldo.sum() / 1200)) if saldo.sum() > 0 else np.zeros(n)
    return {
        "pv": pv,
        "snittranta": snittranta,
        "max_manad": kostnad_m.max(1),
        "forsta_ar_manad": kostnad_m[:, :12].mean(1),
        "engang": engang,
        "snitt_bana": ranta.mean(0),
    }
