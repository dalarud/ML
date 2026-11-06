"""Rekommendationsmotor: binder ihop data, modeller, scenarier och strategier.

Poäng för varje (bank, strategi):

    poäng = E[kostnad] + λ · (CVaR_α[kostnad] − E[kostnad])

där kostnad är nuvärdet av räntekostnaden efter ränteavdrag (inkl. flytt-/
brytkostnader och värdet av låst ränta efter horisonten), CVaR_α är
genomsnittet av de (1−α) sämsta scenarierna och λ din riskaversion. Lägst poäng
vinner. Rekommendationen kontrolleras även mot alternativa scenarier
(marknadens prissättning, oförändrad ränta) för robusthet.
"""
from __future__ import annotations

import pickle
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, TENORS, bygg_panel
from .features import BINDNING_K, PROGNOS_H, dataset
from .models import MalResultat, feature_importance, kor_alla
from .simulate import forvantad_bana, marknadens_bana, simulera, skatta_dynamik
from .strategies import Bank, Lan, standardstrategier, utvardera

CACHE = ROOT / "data" / "modellcache.pkl"


@dataclass
class Profil:
    lan: Lan
    banker: list[Bank]
    riskaversion: float = 0.25
    cvar_niva: float = 0.95
    scenarier: int = 3000


def las_profil(path: str | Path) -> Profil:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    lan = Lan(**c["lan"])
    banker = [Bank(**b) for b in c["bank"]]
    r = c.get("risk", {})
    return Profil(lan, banker, r.get("riskaversion", 0.25), r.get("cvar_niva", 0.95), r.get("scenarier", 3000))


def tranade_modeller(panel: pd.DataFrame, tvinga: bool = False, k: int = 5) -> dict:
    """Tränar (eller läser cache för) alla modeller för senaste datamånad."""
    nyckel = (str(panel.index[-1].date()), len(panel), k)
    if CACHE.exists() and not tvinga:
        with open(CACHE, "rb") as f:
            c = pickle.load(f)
        if c.get("nyckel") == nyckel:
            return c
    print("Tränar och korsvaliderar modeller (purged walk-forward k-fold) ...")
    X, Y, w = dataset(panel)
    res = kor_alla(X, Y, w, k=k)
    fi = feature_importance(X, Y["d3m_12"], w)
    c = {"nyckel": nyckel, "resultat": res, "feature_importance": fi}
    with open(CACHE, "wb") as f:
        pickle.dump(c, f)
    return c


def _cvar(x: np.ndarray, a: float) -> float:
    q = np.quantile(x, a)
    return float(x[x >= q].mean())


def huvudbana(res: dict[str, MalResultat], kurva: dict, manader: int):
    """Skicklighetsviktad förväntad bana för rörlig ränta.

    För varje horisont vägs ML-ensemblen mot marknadens implicita bana med
    vikten w_h = clip(R²_oos / 0.25, 0, 1): där modellen visat prediktiv
    förmåga out-of-sample får den styra, annars gäller marknadens prissättning.
    """
    premie_hist = {t: res[f"premie_{t}"].prognos["naiv"] for t in BINDNING_K}
    marknad = marknadens_bana(kurva, premie_hist, manader)
    ml, vikt = {}, {}
    for h in PROGNOS_H:
        r = res[f"d3m_{h}"]
        r2 = float(r.cv.loc["ENSEMBLE", "R2_oos"])
        w = float(np.clip(r2 / 0.25, 0, 1))
        m_h = marknad[h] - kurva["3M"]
        ml[h] = r.ensemble
        vikt[h] = w
    blandad = {h: vikt[h] * ml[h] + (1 - vikt[h]) * (marknad[h] - kurva["3M"]) for h in PROGNOS_H}
    lang = marknad[min(manader, 60)]
    bana = forvantad_bana(kurva["3M"], blandad, manader, lang_niva=lang)
    ml_bana = forvantad_bana(kurva["3M"], ml, manader, lang_niva=None)
    return bana, marknad, ml_bana, vikt, premie_hist


def obligationskurva(p: pd.Series) -> dict:
    return {"3M": p["ssvx_3m"], "1Y": (p["ssvx_6m"] + p["bobl_2y"]) / 2, "2Y": p["bobl_2y"],
            "3Y": (2 * p["bobl_2y"] + p["bobl_5y"]) / 3, "5Y": p["bobl_5y"],
            "10Y": p["stat_10y"] + (p["bobl_5y"] - p["stat_5y"])}


def analysera(profil: Profil, tvinga_traning: bool = False, seed: int = 0) -> dict:
    panel = bygg_panel()
    modeller = tranade_modeller(panel, tvinga_traning)
    res = modeller["resultat"]
    sista = panel.iloc[-1]
    kurva = {t: float(sista[f"bolan_{t}"]) for t in TENORS}
    M = profil.lan.horisont_ar * 12 + 1
    bana, marknad, ml_bana, vikt, premie_hist = huvudbana(res, kurva, M)

    dyn = skatta_dynamik(panel)
    scen = {
        "Huvudscenario (ML + marknad)": bana,
        "Marknadens prissättning": marknad,
        "Oförändrad rörlig ränta": np.full(M + 1, kurva["3M"]),
    }
    sims = {k: simulera(kurva, v, dyn, n=profil.scenarier, seed=seed) for k, v in scen.items()}
    huvud = "Huvudscenario (ML + marknad)"

    nuv = next((b for b in profil.banker if b.nuvarande), profil.banker[0])
    bobl = obligationskurva(sista)
    strategier = standardstrategier()
    if profil.lan.nuvarande_tenor != "3M" and profil.lan.bundet_kvar_manader > 0:
        from copy import deepcopy
        for s in list(strategier[:6]):
            s2 = deepcopy(s)
            s2.namn += " – bryt bindning nu"
            s2.bryt_bindning = True
            s2.kategori = "bryt"
            strategier.append(s2)

    rader, banor = [], {}
    for bank in profil.banker:
        for s in strategier:
            if s.kategori == "passiv" and not bank.nuvarande:
                continue
            rad = {"bank": bank.namn, "strategi": s.namn, "kategori": s.kategori,
                   "flytt": bank.namn != nuv.namn}
            for sn, R in sims.items():
                u = utvardera(s, bank, profil.lan, R, bobl, nuv)
                if sn == huvud:
                    pv = u["pv"]
                    rad.update({
                        "forvantad_kostnad": float(pv.mean()),
                        "median": float(np.median(pv)),
                        "std": float(pv.std()),
                        "p5": float(np.quantile(pv, 0.05)),
                        "p95": float(np.quantile(pv, 0.95)),
                        "cvar": _cvar(pv, profil.cvar_niva),
                        "snittranta": float(u["snittranta"].mean()),
                        "snittranta_p95": float(np.quantile(u["snittranta"], 0.95)),
                        "max_manad_p95": float(np.quantile(u["max_manad"], 0.95)),
                        "forsta_ar_manad": float(u["forsta_ar_manad"].mean()),
                        "engangskostnad": u["engang"],
                    })
                    if profil.lan.budget_manad:
                        rad["p_over_budget"] = float(np.mean(u["max_manad"] > profil.lan.budget_manad))
                    banor[(bank.namn, s.namn)] = u["snitt_bana"]
                else:
                    rad[f"kostnad | {sn}"] = float(u["pv"].mean())
            rader.append(rad)
    df = pd.DataFrame(rader)
    lam = profil.riskaversion
    df["poang"] = df["forvantad_kostnad"] + lam * (df["cvar"] - df["forvantad_kostnad"])
    # Robusthet: rang i varje scenario, sämsta rang
    rangkol = []
    for sn in scen:
        kol = "forvantad_kostnad" if sn == huvud else f"kostnad | {sn}"
        df[f"rang | {sn}"] = df[kol].rank()
        rangkol.append(f"rang | {sn}")
    df["samsta_rang"] = df[rangkol].max(axis=1)
    df = df.sort_values("poang").reset_index(drop=True)

    # Personliga räntor idag per bank och tenor
    erbjud = pd.DataFrame({b.namn: {t: b.listrantor[t] - b.rabatter.get(t, 0) for t in TENORS}
                           for b in profil.banker}).T
    erbjud.loc["Marknadssnitt (SCB)"] = kurva

    signaler = pd.DataFrame({
        t: {"förväntad bindningspremie (pe)": res[f"premie_{t}"].ensemble,
            "P(bindning billigare än rörlig)": res[f"premie_{t}"].p_positiv,
            "modell": "/".join(res[f"premie_{t}"].vikter)}
        for t in BINDNING_K}).T
    prognos = pd.DataFrame({
        f"{h} mån": {"ML-ensemble": kurva["3M"] + res[f"d3m_{h}"].ensemble,
                     "Marknad": marknad[h], "Huvudscenario": bana[h],
                     "ML-vikt": vikt[h],
                     "P10 (sim)": float(np.quantile(sims[huvud][:, h, 0], 0.10)),
                     "P90 (sim)": float(np.quantile(sims[huvud][:, h, 0], 0.90))}
        for h in PROGNOS_H if h < M}).T

    return {
        "panel": panel, "modeller": res, "feature_importance": modeller["feature_importance"],
        "kurva": kurva, "banor": {"huvud": bana, "marknad": marknad, "ml": ml_bana},
        "sims": sims, "huvud": huvud, "resultat": df, "erbjudanden": erbjud,
        "signaler": signaler, "prognos": prognos, "premie_hist": premie_hist,
        "profil": profil, "nuvarande_bank": nuv.namn, "strategibanor": banor,
    }


def sammanfatta(a: dict) -> list[str]:
    """Rekommendation i klartext."""
    df, prof = a["resultat"], a["profil"]
    bast = df.iloc[0]
    nuv = a["nuvarande_bank"]
    ut = []
    ut.append(f"REKOMMENDATION (riskaversion λ={prof.riskaversion}, horisont {prof.lan.horisont_ar} år):")
    ut.append(f"  → {bast['strategi']} hos {bast['bank']}"
              f"{' (FLYTTA)' if bast['flytt'] else ''}")
    ut.append(f"    Förväntad snittränta {bast['snittranta']:.2f} % (95:e percentil {bast['snittranta_p95']:.2f} %),"
              f" nuvärde räntekostnad efter avdrag {_kr(bast['forvantad_kostnad'])} kr"
              f" (CVaR{int(prof.cvar_niva*100)} {_kr(bast['cvar'])} kr)")
    ut.append(f"    Robusthet: sämsta rang över scenarier {int(bast['samsta_rang'])} av {len(df)}")
    billigast_mu = df["forvantad_kostnad"].min()
    if bast["forvantad_kostnad"] > billigast_mu + 1:
        ut.append(f"    Försäkringskostnad: {_kr(bast['forvantad_kostnad'] - billigast_mu)} kr högre förväntad kostnad än "
                  f"billigaste strategin, mot {_kr(df.loc[df['forvantad_kostnad'].idxmin(), 'cvar'] - bast['cvar'])}"
                  f" kr lägre kostnad i de sämsta scenarierna.")
    alt = df.drop_duplicates("strategi").iloc[1:4]
    ut.append("  Närmaste alternativ: " + "; ".join(
        f"{r['strategi']} ({r['bank']}, +{_kr(r['poang'] - bast['poang'])} kr)" for _, r in alt.iterrows()))

    # Bästa hos nuvarande bank -> flyttvinst
    hos_nuv = df[df["bank"] == nuv].iloc[0]
    if bast["bank"] != nuv:
        vinst = hos_nuv["poang"] - bast["poang"]
        ut.append(f"  Flytt: bästa alternativ hos {nuv} är '{hos_nuv['strategi']}'. Att flytta till "
                  f"{bast['bank']} sparar ≈ {_kr(vinst)} kr (riskjusterat nuvärde) efter flyttkostnad.")
        ut.append("    Tips: använd erbjudandet för att förhandla med nuvarande bank innan du flyttar.")
    else:
        ut.append(f"  Flytt: ingen annan bank slår {nuv} efter flyttkostnader i nuläget.")

    # Rörlig vs bunden – riskneutral jämförelse
    rorlig = df[(df["strategi"] == "Rörlig 3 mån") & (df["bank"] == bast["bank"])].iloc[0]
    billigast = df.sort_values("forvantad_kostnad").iloc[0]
    tryggast = df.sort_values("cvar").iloc[0]
    ut.append(f"  Lägst förväntad kostnad: {billigast['strategi']} ({billigast['bank']}),"
              f" lägst risk (CVaR): {tryggast['strategi']} ({tryggast['bank']}).")
    ut.append(f"  Rörlig 3 mån hos {bast['bank']}: snitt {rorlig['snittranta']:.2f} %, "
              f"95:e percentil {rorlig['snittranta_p95']:.2f} %.")

    # Timing
    tim = df[df["kategori"] == "timing"].iloc[0] if (df["kategori"] == "timing").any() else None
    if tim is not None:
        motsv = tim["strategi"].split("bind ")[-1]
        direkt = df[(df["strategi"] == f"Bind {motsv} (förnya)") & (df["bank"] == tim["bank"])]
        if len(direkt):
            d = direkt.iloc[0]
            battre = "vänta" if tim["poang"] < d["poang"] else "binda direkt"
            ut.append(f"  Tidpunkt: för {motsv} är det bättre att {battre} "
                      f"({tim['strategi']}: {_kr(tim['poang'])} kr vs direkt: {_kr(d['poang'])} kr).")
    sig = a["signaler"]
    for t, r in sig.iterrows():
        ut.append(f"  Signal {t}: P(bindning billigare än rörlig) = {r['P(bindning billigare än rörlig)']:.0%}, "
                  f"förväntad premie {r['förväntad bindningspremie (pe)']:+.2f} pe")
    return ut


def _kr(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")
