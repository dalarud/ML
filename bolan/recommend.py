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

from .data import ROOT, TENORS, bygg_panel, las_prognoser
from .features import BINDNING_K, KPIF_H, PROGNOS_H, STYR_H, dataset_makro, _manadsbana
from .models import MalResultat, feature_importance, kor_alla, kor_makro, realtids_stack, riktmarken
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


STACK_MAL = ["dstyr_6", "dstyr_12", "dstyr_24", "dkpif_12"]


def tranade_modeller(panel: pd.DataFrame, tvinga: bool = False, k: int = 5) -> dict:
    """Tränar (eller läser cache för) hela modellkedjan för senaste datamånad.

    Steg 1  Makromodeller: styrränta (3–24 mån) och KPIF-inflation (6–24 mån), jämförda
            med Riksbankens egna prognoser och marknadens prissättning.
    Steg 2  Walk-forward-prognoser från steg 1 läggs till som features, tillsammans med
            Riksbankens räntebana/KPIF-prognos och marknadens förväntningar (realtid).
    Steg 3  Bolånemodellerna (rörlig ränta 3–60 mån, bindningspremier) tränas på detta.
    Steg 4  Styrräntekanalen: genomslag styrränta -> rörlig bolåneränta skattas, och
            kanalen vägs mot den direkta ML-prognosen efter out-of-sample-träffsäkerhet.
    """
    prog = las_prognoser()
    nyckel = (str(panel.index[-1].date()), len(panel), k, str(prog["publicerad"].max().date()), "v2")
    if CACHE.exists() and not tvinga:
        with open(CACHE, "rb") as f:
            c = pickle.load(f)
        if c.get("nyckel") == nyckel:
            return c
    print("Steg 1: makromodeller för styrränta och inflation ...")
    X, Y, w = dataset_makro(panel, prog)
    makro, jmf = kor_makro(X, Y, w, k=k)
    print("Steg 2: realtidsprognoser från makromodellerna som features ...")
    stack = realtids_stack(X, Y, w, STACK_MAL)
    X2 = X.join(stack)
    print("Steg 3: bolånemodeller (purged walk-forward k-fold) ...")
    res = kor_alla(X2, Y, w, k=k)
    print("Steg 4: styrräntekanalen ...")
    kanal = styrrantekanal(X2, Y, res, makro, jmf)
    fi = feature_importance(X2, Y["d3m_12"], w)
    c = {"nyckel": nyckel, "resultat": res, "makro": makro, "makro_jmf": jmf, "kanal": kanal,
         "feature_importance": fi, "x_nu": X2.iloc[-1], "X": X2}
    with open(CACHE, "wb") as f:
        pickle.dump(c, f)
    return c


def _komb_styr(makro, jmf, X, h, rader=None):
    """Kombinerad styrränteprognos (ML + Riksbanken + marknad) med jmf-vikter."""
    namn = f"dstyr_{h}"
    v = jmf[namn]["_vikter"]
    if rader is None:   # prognos idag
        delar = {"ML-ensemble": makro[namn].ensemble,
                 **{b: float(s.iloc[-1]) for b, s in riktmarken(X, namn).items()}}
        return sum(v[b] * delar[b] for b in v), delar
    oos = makro[namn].oos
    delar = {"ML-ensemble": oos["ensemble"].reindex(rader),
             **{b: s.reindex(rader) for b, s in riktmarken(X, namn).items()}}
    return sum(v[b] * delar[b] for b in v)


def styrrantekanal(X, Y, res, makro, jmf) -> dict:
    """Rörlig bolåneränta via styrräntan: d3m_h ≈ β_h · Δstyrränta_h.

    β_h (genomslaget) skattas på historiken. Kanalens out-of-sample-fel jämförs med den
    direkta ML-prognosen på samma månader (2015–, då Riksbankens prognoser finns), och
    de två vägs ihop med invers MSE.
    """
    ut = {}
    for h in STYR_H:
        d = pd.concat([Y[f"d3m_{h}"], Y[f"dstyr_{h}"]], axis=1).dropna()
        beta = float(np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0])
        direkt = res[f"d3m_{h}"].oos
        rader = direkt.index.intersection(makro[f"dstyr_{h}"].oos.index)
        rader = rader[X.loc[rader, "har_rb_prognos"] == 1]
        if len(rader) < 12:
            ut[h] = {"beta": beta, "vikt_kanal": 0.0, "r2": np.nan}
            continue
        y = direkt.loc[rader, "faktiskt"]
        p_dir = direkt.loc[rader, "ensemble"]
        p_kan = beta * _komb_styr(makro, jmf, X, h, rader)
        mse_d, mse_k, mse_0 = [float(np.mean((y - p) ** 2)) for p in (p_dir, p_kan, 0.0)]
        vk = (1 / mse_k) / (1 / mse_k + 1 / mse_d)
        p_komb = vk * p_kan + (1 - vk) * p_dir
        mse_c = float(np.mean((y - p_komb) ** 2))
        ut[h] = {"beta": beta, "vikt_kanal": vk, "n": len(rader),
                 "RMSE naiv": np.sqrt(mse_0), "RMSE direkt ML": np.sqrt(mse_d),
                 "RMSE styrräntekanal": np.sqrt(mse_k), "RMSE kombination": np.sqrt(mse_c),
                 "r2": 1 - mse_c / mse_0}
    return ut


def _cvar(x: np.ndarray, a: float) -> float:
    q = np.quantile(x, a)
    return float(x[x >= q].mean())


def huvudbana(mod: dict, kurva: dict, manader: int):
    """Förväntad bana för rörlig bolåneränta.

    1. För horisonter ≤ 24 mån kombineras den direkta ML-prognosen med styrräntekanalen
       (β · kombinerad styrränteprognos från ML, Riksbankens räntebana och marknaden),
       viktat efter out-of-sample-träffsäkerhet.
    2. Resultatet vägs mot marknadens implicita bana med w_h = clip(R²_oos / 0.25, 0, 1):
       där modellerna bevisat träffsäkerhet styr de, annars marknadens prissättning.
    """
    res, kanal, makro, jmf, X = mod["resultat"], mod["kanal"], mod["makro"], mod["makro_jmf"], mod["X"]
    premie_hist = {t: res[f"premie_{t}"].prognos["naiv"] for t in BINDNING_K}
    marknad = marknadens_bana(kurva, premie_hist, manader)
    ml, vikt, rb = {}, {}, {}
    for h in PROGNOS_H:
        r = res[f"d3m_{h}"]
        r2 = float(r.cv.loc["ENSEMBLE", "R2_oos"])
        pred = r.ensemble
        if h in kanal and kanal[h].get("n"):
            k = kanal[h]
            styr, _ = _komb_styr(makro, jmf, X, h)
            pred = k["vikt_kanal"] * k["beta"] * styr + (1 - k["vikt_kanal"]) * r.ensemble
            r2 = max(r2, k["r2"])
            rb[h] = k["beta"] * float(X[f"rb_dstyr_{h}"].iloc[-1])
        ml[h] = pred
        vikt[h] = float(np.clip(r2 / 0.25, 0, 1))
    blandad = {h: vikt[h] * ml[h] + (1 - vikt[h]) * (marknad[h] - kurva["3M"]) for h in PROGNOS_H}
    lang = marknad[min(manader, 60)]
    bana = forvantad_bana(kurva["3M"], blandad, manader, lang_niva=lang)
    ml_bana = forvantad_bana(kurva["3M"], {h: ml[h] for h in PROGNOS_H if h <= 60}, manader, lang_niva=None)
    rb_bana = forvantad_bana(kurva["3M"], rb, manader, lang_niva=lang)
    return bana, marknad, ml_bana, rb_bana, vikt, premie_hist


def makroprognos(mod: dict, panel: pd.DataFrame) -> dict:
    """Tabeller och banor för styrränta och inflation till rapporten."""
    makro, jmf, X = mod["makro"], mod["makro_jmf"], mod["X"]
    styr0, kpif0 = float(panel["styrranta"].iloc[-1]), float(panel["kpif_yoy"].iloc[-1])
    styr = {}
    for h in STYR_H:
        komb, delar = _komb_styr(makro, jmf, X, h)
        styr[f"{h} mån"] = {**{k: styr0 + v for k, v in delar.items()}, "Kombination": styr0 + komb}
    kpif = {}
    for h in KPIF_H:
        n = f"dkpif_{h}"
        v = jmf[n]["_vikter"]
        delar = {"ML-ensemble": makro[n].ensemble, "Riksbanken": float(X[f"rb_dkpif_{h}"].iloc[-1])}
        kpif[f"{h} mån"] = {**{k: kpif0 + d for k, d in delar.items()},
                            "Kombination": kpif0 + sum(v[b] * delar[b] for b in v)}
    prog = las_prognoser()
    banor = {}
    for serie in ("rb_styr", "mkt_styr", "rb_kpif"):
        g = prog[prog["serie"] == serie]
        senast = g[g["publicerad"] == g["publicerad"].max()]
        banor[serie] = _manadsbana(senast)
        banor[serie + "_omgang"] = senast["omgang"].iloc[0]
    validering = pd.DataFrame({k: {a: b for a, b in v.items() if a != "_vikter"} for k, v in jmf.items()}).T
    vikter = pd.DataFrame({k: v["_vikter"] for k, v in jmf.items()}).T
    return {"styr": pd.DataFrame(styr).T, "kpif": pd.DataFrame(kpif).T, "banor": banor,
            "validering": validering, "vikter": vikter, "styr0": styr0, "kpif0": kpif0,
            "kanal": pd.DataFrame({f"{h} mån": v for h, v in mod["kanal"].items()}).T}


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
    bana, marknad, ml_bana, rb_bana, vikt, premie_hist = huvudbana(modeller, kurva, M)

    dyn = skatta_dynamik(panel)
    scen = {
        "Huvudscenario (ML + marknad)": bana,
        "Marknadens prissättning": marknad,
        "Riksbankens räntebana": rb_bana,
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
                     "Marknad": marknad[h], "Riksbankens bana": rb_bana[h], "Huvudscenario": bana[h],
                     "ML-vikt": vikt[h],
                     "P10 (sim)": float(np.quantile(sims[huvud][:, h, 0], 0.10)),
                     "P90 (sim)": float(np.quantile(sims[huvud][:, h, 0], 0.90))}
        for h in PROGNOS_H if h < M}).T

    return {
        "panel": panel, "modeller": res, "feature_importance": modeller["feature_importance"],
        "kurva": kurva, "banor": {"huvud": bana, "marknad": marknad, "ml": ml_bana, "riksbank": rb_bana},
        "makro": makroprognos(modeller, panel),
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
    mk = a["makro"]
    st, kp = mk["styr"], mk["kpif"]
    ut.append(f"  Styrränta nu {mk['styr0']:.2f} %. Om 12 mån: ML {st.loc['12 mån', 'ML-ensemble']:.2f} %, "
              f"Riksbanken {st.loc['12 mån', 'Riksbanken']:.2f} %, marknaden {st.loc['12 mån', 'Marknaden']:.2f} % "
              f"→ kombinerat {st.loc['12 mån', 'Kombination']:.2f} %.")
    ut.append(f"  KPIF-inflation nu {mk['kpif0']:.2f} %. Om 12 mån: ML {kp.loc['12 mån', 'ML-ensemble']:.2f} %, "
              f"Riksbanken {kp.loc['12 mån', 'Riksbanken']:.2f} % → kombinerat {kp.loc['12 mån', 'Kombination']:.2f} %.")
    sig = a["signaler"]
    for t, r in sig.iterrows():
        ut.append(f"  Signal {t}: P(bindning billigare än rörlig) = {r['P(bindning billigare än rörlig)']:.0%}, "
                  f"förväntad premie {r['förväntad bindningspremie (pe)']:+.2f} pe")
    return ut


def _kr(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")
