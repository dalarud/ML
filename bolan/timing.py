"""Timing av bundet mot rörligt: snåla, teoristyrda modeller och beslutsbacktest.

Varför egna modeller? De generella ML-modellerna (~100 features) överanpassar på ~360
månader och har ingen påvisad förmåga att förutsäga bindningspremien. Timing kräver
få, ekonomiskt motiverade förklaringsvariabler:

* Bindningspremie (är bunden ränta billig?): ex ante-premie enligt marknadens och
  Riksbankens styrräntebana, kurvans lutning, bankens marginal mot bostadsobligationer,
  läge i räntecykeln, realränta och inflationsgap.
* Bunden ränta nästa 1–3 månader (binda nu eller vänta?): obligationsräntornas senaste
  rörelse, eftersläpning (obligationer har rört sig men bolåneräntan inte ännu),
  marginalens avvikelse från normalt och styrräntans riktning.

Modeller: Ridge, grunt Gradient Boosting och Random Forest med stora löv – medelvärdet
används. Utvärdering med purged walk-forward k-fold (R², träffsäkerhet i riktning och
utfall givet signal) och ett beslutsbacktest som visar vad reglerna hade kostat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import TENORS
from .features import BINDNING_K, FIX_TENORER
from .models import purged_walk_forward

MARG = {"1Y": "marg_fix_2Y", "2Y": "marg_fix_2Y", "3Y": "marg_fix_3Y", "5Y": "marginal_5Y"}
BOBL = {"2Y": "bobl_2y", "3Y": "bobl_2y", "5Y": "bobl_5y"}
EFTERSL = {"2Y": "eftersl_2Y", "3Y": "eftersl_2Y", "5Y": "eftersl_5Y"}
TIMING_H = [1, 3]


def features_premie(t: str) -> list[str]:
    return [f"exante_mkt_{t}", f"exante_rb_{t}", f"spread_{t}", MARG[t], "man_sedan_styrandring",
            "senaste_styrandring", "r3_vs_5arsmedel", "realranta", "inflationsgap", "forvantan_2y"]


def features_fix(t: str) -> list[str]:
    b = BOBL[t]
    return [f"d1_{b}", f"d3_{b}", EFTERSL[t], f"{MARG[t]}_vs_12m", f"d1_bolan_{t}", "d3_styrranta"]


def _modeller():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=30.0)),
        "gb": HistGradientBoostingRegressor(max_depth=2, max_iter=150, learning_rate=0.03, min_samples_leaf=25,
                                            random_state=0),
        "rf": RandomForestRegressor(n_estimators=300, min_samples_leaf=15, max_features=0.5, random_state=0, n_jobs=-1),
    }


def _fit_pred(Xtr, ytr, wtr, Xte):
    preds = []
    for n, m in _modeller().items():
        if n == "ridge":
            m.fit(Xtr, ytr, ridge__sample_weight=wtr)
        else:
            m.fit(Xtr, ytr, sample_weight=wtr)
        preds.append(m.predict(Xte))
    return np.mean(preds, axis=0)


def _mal_lista():
    return ([(f"premie_{t}", k, features_premie(t)) for t, k in BINDNING_K.items()] +
            [(f"dfix_{t}_{h}", h, features_fix(t)) for t in FIX_TENORER for h in TIMING_H])


def utvardera(X: pd.DataFrame, Y: pd.DataFrame, w: pd.Series, k: int = 5) -> dict:
    """CV-resultat och dagens prognos för varje timingmål."""
    ut = {}
    for mal, h, F in _mal_lista():
        y = Y[mal]
        ok = y.notna()
        Xo, yo, wo = X.loc[ok, F].values, y[ok].values, w[ok].values
        pred, naiv = np.full(len(yo), np.nan), np.full(len(yo), np.nan)
        for tr, te in purged_walk_forward(len(yo), k, gap=h):
            naiv[te] = 0.0 if mal.startswith("dfix") else np.average(yo[tr], weights=wo[tr])
            pred[te] = _fit_pred(Xo[tr], yo[tr], wo[tr], Xo[te])
        t_ = ~np.isnan(pred)
        mse, mse0 = np.mean((yo[t_] - pred[t_]) ** 2), np.mean((yo[t_] - naiv[t_]) ** 2)
        r2 = 1 - mse / mse0
        nu_ml = float(_fit_pred(Xo, yo, wo, X[F].iloc[[-1]].values)[0])
        lang = float(np.mean(yo))                       # oviktat långsiktigt snitt
        # Använd modellen bara om den bevisat träffsäkerhet; annars långsiktigt snitt (premie) / 0 (dfix)
        anvand = r2 > 0
        nu = nu_ml if anvand else (0.0 if mal.startswith("dfix") else lang)
        upp = pred[t_] > (0 if mal.startswith("dfix") else 0)
        ut[mal] = {
            "horisont": h, "R2_oos": r2, "RMSE": float(np.sqrt(mse)), "traff_riktning": float(np.mean(np.sign(pred[t_]) == np.sign(yo[t_]))),
            "utfall_om_signal_positiv": float(yo[t_][upp].mean()) if upp.any() else np.nan,
            "utfall_om_signal_negativ": float(yo[t_][~upp].mean()) if (~upp).any() else np.nan,
            "prognos_ml": nu_ml, "prognos": nu, "anvands": anvand, "langsiktigt_snitt": lang, "n": int(t_.sum()),
        }
    return ut


def vantekalkyl(tim: dict, kurva_pers: dict[str, float], saldo: float) -> pd.DataFrame:
    """Binda nu eller ligga rörligt w månader och sedan binda?

    Förväntad vinst av att vänta (kr) = saldo × [ w/12 × (bunden_nu − rörlig_nu)
                                                − k/12 × E(förändring i bunden ränta under w mån) ] / 100
    Osäkerheten tas från prognosens out-of-sample-fel.
    """
    from scipy.stats import norm
    rader = {}
    for t in FIX_TENORER:
        k = TENORS[t]
        for h in TIMING_H:
            r = tim[f"dfix_{t}_{h}"]
            mu_fix = r["prognos"]
            sd_fix = r["RMSE"] if r["anvands"] else r["RMSE"]
            carry = h / 12 * (kurva_pers[t] - kurva_pers["3M"])
            mu = saldo * (carry - k / 12 * mu_fix) / 100
            sd = saldo * (k / 12 * sd_fix) / 100
            rader[f"{t}: vänta {h} mån"] = {
                "Bunden nu %": kurva_pers[t], "Förväntad ändring i bunden ränta pe": mu_fix,
                "Modell används": "ja" if r["anvands"] else "nej (ingen påvisad träffsäkerhet)",
                "Ränteskillnad under väntan kr": saldo * carry / 100,
                "Förväntad vinst av att vänta kr": mu, "P(väntan lönar sig)": float(norm.sf(0, mu, max(sd, 1e-9))),
            }
    return pd.DataFrame(rader).T


# ---------------------------------------------------------------------------
# Beslutsbacktest
# ---------------------------------------------------------------------------
def beslutsbacktest(p: pd.DataFrame, X: pd.DataFrame, Y: pd.DataFrame, w: pd.Series, fran: str = "2008-01-01",
                    omtraning: int = 12, troskel: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward-test av beslutsregler på faktiska SCB-genomsnittsräntor.

    Regler (beslut tas när lånet är fritt, dvs. inte bundet):
      * Förnya k                : alltid samma bindningstid
      * Marknadsregel           : bind den tid med högst ex ante-premie (marknadens bana) om > tröskel
      * ML-timing               : bind den tid med högst predikterad premie om > tröskel, annars rörligt 3 mån
      * ML-timing + vänta       : som ML-timing, men skjut upp bindningen en månad om modellen
                                  förutser att den bundna räntan sjunker nästa månad
      * Perfekt framförhållning : facit – övre gräns för vad timing kan ge
    Modellerna tränas om var 12:e månad, enbart på data vars utfall var känt vid beslutet.
    """
    idx = X.index
    n = len(idx)
    i0 = int(np.searchsorted(idx, pd.Timestamp(fran)))
    wv = w.values
    pos = {t: k for t, k in BINDNING_K.items()}
    rates = {t: p.loc[idx, f"bolan_{t}"].values for t in ["3M", *BINDNING_K]}

    # Walk-forward-prognoser för premier och dfix_1
    pred = {}
    for mal, h, F in _mal_lista():
        if mal.startswith("dfix") and not mal.endswith("_1"):
            continue
        y = Y[mal].values
        Xv = X[F].values
        out = np.full(n, np.nan)
        for T in range(i0, n, omtraning):
            tr = np.where(~np.isnan(y) & (np.arange(n) <= T - h - 1))[0]
            if len(tr) < 60:
                continue
            sl = slice(T, min(T + omtraning, n))
            out[sl] = _fit_pred(Xv[tr], y[tr], wv[tr], Xv[sl])
        pred[mal] = out

    def kor(valj):
        r = np.full(n - i0, np.nan)
        t, val = i0, []
        while t < n:
            tenor, steg = valj(t)
            k = 3 if tenor == "3M" else pos[tenor]
            steg = steg or k
            r[t - i0:min(t + steg, n) - i0] = rates[tenor][t]
            val.append((idx[t], tenor))
            t += steg
        return r, val

    def facit(t):
        best, bv = "3M", 0.0
        for tn, k in pos.items():
            v = Y[f"premie_{tn}"].iloc[t]
            if not np.isnan(v) and v > bv:
                best, bv = tn, v
        return best, None

    def marknad(t):
        v = {tn: X[f"exante_mkt_{tn}"].iloc[t] for tn in pos}
        b = max(v, key=v.get)
        return (b, None) if v[b] > troskel else ("3M", None)

    def ml(t):
        v = {tn: pred[f"premie_{tn}"][t] for tn in pos}
        b = max(v, key=v.get)
        return (b, None) if v[b] > troskel else ("3M", None)

    def vanta(regel):
        def f(t):
            b, _ = regel(t)
            if b in FIX_TENORER and pred[f"dfix_{b}_1"][t] < -0.05:
                return "3M", 1      # bunden ränta väntas sjunka – vänta en månad på rörligt
            return b, None
        return f

    ml_vanta = vanta(ml)

    serier, val_ml = {}, None
    for tn in ["3M", *BINDNING_K]:
        serier[f"Förnya {tn}"] = kor(lambda t, tn=tn: (tn, None))[0]
    serier["Marknadsregel (ex ante-premie)"], val_mkt = kor(marknad)
    serier["Marknadsregel + vänta-signal"] = kor(vanta(marknad))[0]
    serier["ML-timing"], val_ml = kor(ml)
    serier["ML-timing + vänta-signal"] = kor(ml_vanta)[0]
    serier["Perfekt framförhållning (facit)"] = kor(facit)[0]
    df = pd.DataFrame(serier, index=idx[i0:])
    # Facit kräver kända framtida utfall; jämför alla regler på samma period
    df = df.dropna()
    perioder = {"2008–2014": ("2008", "2014"), "2015–2021": ("2015", "2021"), "2022–": ("2022", "2030")}
    per = pd.DataFrame({k: df.loc[a:b].mean() for k, (a, b) in perioder.items()})
    sam = pd.DataFrame({
        "snittränta %": df.mean(),
        "vs förnya 3M (pe)": df.mean() - df["Förnya 3M"].mean(),
        "vs bästa statiska (pe)": df.mean() - df[[c for c in df if c.startswith("Förnya")]].mean().min(),
        "max %": df.max(),
        "största 12m-ökning pe": (df - df.shift(12)).max(),
    }).sort_values("snittränta %")
    sam = sam.join(per.add_prefix("snitt "))
    val = pd.DataFrame(val_mkt, columns=["datum", "val"]).set_index("datum")
    return sam, val
