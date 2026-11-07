"""Sannolikhetsbaserad riskbedömning.

1. Sannolikhetsmodell: ML-klassificerare (Random Forest + regulariserad logistisk
   regression) skattar sannolikheten för fem utfall för den rörliga bolåneräntan om
   12 och 36 månader – kraftig nedgång, nedgång, oförändrad, uppgång, kraftig uppgång.
   De utvärderas med purged walk-forward k-fold och Brier-poäng mot två riktmärken:
   historisk basfrekvens och simuleringens egen fördelning.
2. Omviktning: de simulerade ränteutvecklingarna viktas om (raking / iterativ
   proportionell anpassning) så att andelen scenarier i varje utfallsklass matchar
   sannolikheterna. Modellen får bara påverka vikterna i den mån den bevisat
   träffsäkerhet (Brier skill score > 0); annars behålls simuleringens fördelning.
3. Riskmått: viktat väntevärde, kvantiler och CVaR, scenariomatris (kostnad per
   utfallsklass), förväntad ånger (regret) och sannolikhet att slå rörlig ränta.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .models import purged_walk_forward

KLASSER = ["Kraftig nedgång", "Nedgång", "Oförändrad", "Uppgång", "Kraftig uppgång"]
GRANSER = np.array([-1.0, -0.25, 0.25, 1.0])          # procentenheter
SANNOLIKHET_H = [12, 36]


def klassa(d: np.ndarray) -> np.ndarray:
    return np.searchsorted(GRANSER, d, side="right")


def _klassificerare(seed=0):
    return {
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features=0.4,
                                                random_state=seed, n_jobs=-1),
        "logistisk": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
    }


def _proba(m, X, K=5):
    """Sannolikheter för alla K klasser även om någon saknades i träningen."""
    p = np.zeros((len(X), K))
    p[:, m.classes_ if hasattr(m, "classes_") else m.steps[-1][1].classes_] = m.predict_proba(X)
    return p


def brier(p: np.ndarray, y: np.ndarray) -> float:
    oh = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


@dataclass
class Sannolikhet:
    horisont: int
    nu: np.ndarray            # slutlig sannolikhet per klass (efter skicklighetsviktning)
    modell_nu: np.ndarray     # ren ML-sannolikhet
    bas: np.ndarray           # historisk basfrekvens
    validering: dict          # Brier-poäng per metod
    vikt_modell: float        # andel ML i slutlig sannolikhet


def skatta_sannolikheter(X: pd.DataFrame, Y: pd.DataFrame, w: pd.Series,
                         k: int = 5, seed: int = 0) -> dict[int, Sannolikhet]:
    """Tränar och validerar sannolikhetsmodellerna. `nu` sätts senare av `kombinera`."""
    ut = {}
    for h in SANNOLIKHET_H:
        y_full = Y[f"d3m_{h}"]
        ok = y_full.notna()
        Xo, wo = X[ok].values, w[ok].values
        yo = klassa(y_full[ok].values)
        mods = _klassificerare(seed)
        oos = {n: np.full((len(yo), 5), np.nan) for n in mods}
        bas_oos = np.full((len(yo), 5), np.nan)
        for tr, te in purged_walk_forward(len(yo), k, gap=h):
            f = np.bincount(yo[tr], weights=wo[tr], minlength=5)
            bas_oos[te] = (f + 0.5) / (f + 0.5).sum()
            for n, m in mods.items():
                mm = clone(m)
                if n == "random_forest":
                    mm.fit(Xo[tr], yo[tr], sample_weight=wo[tr])
                else:
                    mm.fit(Xo[tr], yo[tr], logisticregression__sample_weight=wo[tr])
                oos[n][te] = _proba(mm, Xo[te])
        test = ~np.isnan(bas_oos[:, 0])
        ens = np.mean([oos[n] for n in mods], axis=0)
        b_bas = brier(bas_oos[test], yo[test])
        val = {"Historisk basfrekvens": b_bas}
        for n in mods:
            val[f"ML: {n}"] = brier(oos[n][test], yo[test])
        val["ML-ensemble"] = brier(ens[test], yo[test])
        # Bästa blandning ML/basfrekvens (en parameter, vald på OOS-data)
        grid = np.linspace(0, 1, 11)
        bs = [brier(a * ens[test] + (1 - a) * bas_oos[test], yo[test]) for a in grid]
        a = float(grid[int(np.argmin(bs))])
        val["Kombination"] = float(min(bs))
        val["Brier skill score (mot basfrekvens)"] = 1 - val["Kombination"] / b_bas
        val["n_manader"] = int(test.sum())

        # Slutlig modell på all känd data, sannolikhet för senaste månaden
        x_nu = X.iloc[[-1]].values
        p_nu = []
        for n, m in mods.items():
            mm = clone(m)
            if n == "random_forest":
                mm.fit(Xo, yo, sample_weight=wo)
            else:
                mm.fit(Xo, yo, logisticregression__sample_weight=wo)
            p_nu.append(_proba(mm, x_nu)[0])
        p_ml = np.mean(p_nu, axis=0)
        f = np.bincount(yo, weights=wo, minlength=5)
        bas = (f + 0.5) / (f + 0.5).sum()
        skill = val["Brier skill score (mot basfrekvens)"]
        ut[h] = Sannolikhet(h, None, p_ml, bas, val, a if skill > 0 else 0.0)
    return ut


def kombinera(s: Sannolikhet, sim_p: np.ndarray) -> np.ndarray:
    """Slutlig sannolikhet: ML där den bevisat skicklighet, annars simuleringens fördelning.

    Simuleringen är redan centrerad på huvudbanan (ML + Riksbanken + marknad), så utan
    påvisad skicklighet i klassificeraren behålls den – i stället för att ersättas av en
    historisk basfrekvens som speglar 30 års fallande räntor.
    """
    p = s.vikt_modell * s.modell_nu + (1 - s.vikt_modell) * sim_p
    return p / p.sum()


def simulerad_fordelning(R: np.ndarray, h: int) -> np.ndarray:
    d = R[:, h, 0] - R[:, 0, 0]
    return np.bincount(klassa(d), minlength=5) / len(d)


def omvikta(R: np.ndarray, mal: dict[int, np.ndarray], iterationer: int = 100,
            klipp: float = 10.0) -> tuple[np.ndarray, float]:
    """Raking: vikter per scenario så att klassandelarna för varje horisont matchar målen.

    Returnerar normerade vikter och effektiv stickprovsstorlek (ESS).
    """
    n = R.shape[0]
    w = np.full(n, 1.0 / n)
    klass = {h: klassa(R[:, h, 0] - R[:, 0, 0]) for h in mal}
    for _ in range(iterationer):
        for h, p in mal.items():
            akt = np.bincount(klass[h], weights=w, minlength=5)
            # Klasser som saknas i simuleringen kan inte få vikt; fördela om målet
            finns = akt > 0
            p2 = np.where(finns, p, 0)
            p2 = p2 / p2.sum()
            fak = np.divide(p2, akt, out=np.ones(5), where=finns)
            w = w * fak[klass[h]]
            w = np.clip(w, 1 / (n * klipp), klipp / n)
            w = w / w.sum()
    ess = 1.0 / np.sum(w ** 2)
    return w, float(ess)


# ---------------------------------------------------------------------------
# Viktade riskmått
# ---------------------------------------------------------------------------
def vmedel(x, w):
    return float(np.sum(x * w))


def vkvantil(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, q * c[-1])])


def vcvar(x, w, a):
    """Viktat CVaR: viktat medel av utfallen i den sämsta (1−a)-andelen."""
    o = np.argsort(x)[::-1]
    c = np.cumsum(w[o])
    svans = c <= (1 - a) * c[-1] + 1e-12
    svans[0] = True
    return float(np.sum(x[o][svans] * w[o][svans]) / np.sum(w[o][svans]))


def scenariomatris(pv: dict[str, np.ndarray], snitt: dict[str, np.ndarray], R: np.ndarray, w: np.ndarray,
                   h: int = 36) -> tuple[pd.DataFrame, pd.Series]:
    """Snittränta per strategi i varje utfallsklass (ränteförändring om h mån)."""
    k = klassa(R[:, h, 0] - R[:, 0, 0])
    prob = pd.Series({KLASSER[i]: float(w[k == i].sum()) for i in range(5)})
    rader = {}
    for namn, s in snitt.items():
        rader[namn] = {KLASSER[i]: (float(np.sum(s[k == i] * w[k == i]) / w[k == i].sum()) if w[k == i].sum() > 0
                                    else np.nan) for i in range(5)}
    return pd.DataFrame(rader).T, prob


def anger_och_jamforelse(pv: dict[str, np.ndarray], w: np.ndarray, referens: str) -> pd.DataFrame:
    """Förväntad ånger (kostnad minus bästa strategin i efterhand) och P(slå referensen)."""
    namn = list(pv)
    M = np.vstack([pv[n] for n in namn])
    bast = M.min(axis=0)
    ref = pv[referens]
    return pd.DataFrame({
        "forvantad_anger": [vmedel(M[i] - bast, w) for i in range(len(namn))],
        "anger_p95": [vkvantil(M[i] - bast, w, 0.95) for i in range(len(namn))],
        "p_billigare_an_rorlig": [np.nan if namn[i] == referens else float(np.sum(w[M[i] < ref]))
                                  for i in range(len(namn))],
    }, index=namn)
