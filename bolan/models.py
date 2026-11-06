"""Prediktiva modeller: Ridge, Random Forest, Gradient Boosting och neuralt nät.

Validering sker med *purged walk-forward k-fold*: datat delas i k
kronologiska block; varje testblock förutsägs av modeller som enbart tränats på
data före blocket, med ett glapp (purge) lika långt som prognoshorisonten så
att överlappande målvariabler inte läcker framtida information. Som jämförelse
körs även vanlig (blandad) k-fold, vilken ger en kraftigt överoptimistisk bild
för tidsserier – det visar varför den inte ska användas här.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import BINDNING_K, PROGNOS_H


def modellkatalog(seed: int = 0) -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=5, max_features=0.4, random_state=seed, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_depth=3, min_samples_leaf=15,
            l2_regularization=1.0, random_state=seed),
        "neuralt_nat": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(16,), alpha=5.0, learning_rate_init=1e-3,
            max_iter=3000, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=50, random_state=seed)),
    }


def _fit(model, X, y, w):
    m = clone(model)
    try:
        if hasattr(m, "steps"):
            last = m.steps[-1][0]
            # MLPRegressor saknar sample_weight i äldre versioner
            try:
                m.fit(X, y, **{f"{last}__sample_weight": w})
            except TypeError:
                m.fit(X, y)
        else:
            m.fit(X, y, sample_weight=w)
    except TypeError:
        m.fit(X, y)
    return m


def purged_walk_forward(n: int, k: int, gap: int, min_train: int = 60):
    """Ger (train_idx, test_idx) för k kronologiska testblock."""
    start = min_train + gap
    gränser = np.linspace(start, n, k + 1).astype(int)
    for i in range(k):
        test = np.arange(gränser[i], gränser[i + 1])
        train = np.arange(0, max(gränser[i] - gap, 0))
        if len(train) >= min_train and len(test):
            yield train, test


@dataclass
class MalResultat:
    mal: str
    horisont: int
    cv: pd.DataFrame                      # metrik per modell
    cv_blandad: pd.DataFrame              # vanlig k-fold (läckande) för jämförelse
    vikter: dict
    prognos: dict                         # modell -> prognos idag
    ensemble: float
    sigma: float                          # OOS-standardavvikelse för ensemblen
    oos: pd.DataFrame = field(repr=False, default=None)  # out-of-sample prediktioner

    @property
    def p_positiv(self) -> float:
        return float(norm.sf(0, loc=self.ensemble, scale=max(self.sigma, 1e-6)))


def _metrik(y, pred, naiv):
    fel = y - pred
    mse, mse_n = np.mean(fel ** 2), np.mean((y - naiv) ** 2)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(fel))),
        "R2_oos": float(1 - mse / mse_n) if mse_n > 0 else np.nan,
        "traffsakerhet_tecken": float(np.mean(np.sign(pred) == np.sign(y))),
    }


def utvardera_mal(X: pd.DataFrame, y: pd.Series, w: pd.Series, horisont: int,
                  k: int = 5, seed: int = 0) -> MalResultat:
    ok = y.notna()
    Xo, yo, wo = X[ok].values, y[ok].values, w[ok].values
    modeller = modellkatalog(seed)
    preds = {m: np.full(len(yo), np.nan) for m in modeller}
    naiv_pred = np.full(len(yo), np.nan)
    # "Naiv" = ingen förändring för d3m, historiskt medel för premien
    naiv_noll = y.name.startswith("d3m")

    for tr, te in purged_walk_forward(len(yo), k, gap=horisont):
        naiv_pred[te] = 0.0 if naiv_noll else np.average(yo[tr], weights=wo[tr])
        for namn, m in modeller.items():
            preds[namn][te] = _fit(m, Xo[tr], yo[tr], wo[tr]).predict(Xo[te])

    test = ~np.isnan(naiv_pred)
    rader = {"naiv": _metrik(yo[test], naiv_pred[test], naiv_pred[test])}
    for namn in modeller:
        rader[namn] = _metrik(yo[test], preds[namn][test], naiv_pred[test])
    cv = pd.DataFrame(rader).T

    # Ensemblevikter: invers MSE bland modeller som slår naiv; annars naiv.
    mse = {n: np.mean((yo[test] - preds[n][test]) ** 2) for n in modeller}
    mse_naiv = np.mean((yo[test] - naiv_pred[test]) ** 2)
    bra = {n: 1 / v for n, v in mse.items() if v < mse_naiv}
    if bra:
        s = sum(bra.values())
        vikter = {n: v / s for n, v in bra.items()}
    else:
        vikter = {"naiv": 1.0}
    ens_oos = sum(vikter.get(n, 0) * preds[n] for n in modeller) + vikter.get("naiv", 0) * naiv_pred
    sigma = float(np.sqrt(np.mean((yo[test] - ens_oos[test]) ** 2)))
    cv.loc["ENSEMBLE"] = _metrik(yo[test], ens_oos[test], naiv_pred[test])

    # Vanlig blandad k-fold – för att visa läckaget
    blandad = {}
    for namn in ("random_forest", "ridge"):
        p_b = np.zeros(len(yo))
        for tr, te in KFold(k, shuffle=True, random_state=seed).split(Xo):
            p_b[te] = _fit(modeller[namn], Xo[tr], yo[tr], wo[tr]).predict(Xo[te])
        blandad[namn] = _metrik(yo, p_b, np.full(len(yo), yo.mean() if not naiv_noll else 0.0))
    cv_blandad = pd.DataFrame(blandad).T

    # Slutlig träning på all känd data och prognos för senaste månaden
    x_nu = X.iloc[[-1]].values
    prognos = {}
    for namn, m in modeller.items():
        prognos[namn] = float(_fit(m, Xo, yo, wo).predict(x_nu)[0])
    prognos["naiv"] = 0.0 if naiv_noll else float(np.average(yo, weights=wo))
    ens = sum(vikter.get(n, 0) * prognos[n] for n in prognos)

    oos = pd.DataFrame({"faktiskt": yo, "ensemble": ens_oos, **{n: preds[n] for n in modeller}},
                       index=y[ok].index)[test]
    return MalResultat(y.name, horisont, cv, cv_blandad, vikter, prognos, float(ens), sigma, oos)


def kor_alla(X, Y, w, k: int = 5, seed: int = 0, verbose: bool = True) -> dict[str, MalResultat]:
    res = {}
    mal = [(f"d3m_{h}", h) for h in PROGNOS_H] + [(f"premie_{t}", m) for t, m in BINDNING_K.items()]
    for namn, h in mal:
        if verbose:
            print(f"  tränar/validerar {namn} ...", flush=True)
        res[namn] = utvardera_mal(X, Y[namn].rename(namn), w, h, k=k, seed=seed)
    return res


def feature_importance(X, y, w, seed: int = 0, top: int = 15) -> pd.Series:
    ok = y.notna()
    rf = _fit(modellkatalog(seed)["random_forest"], X[ok], y[ok], w[ok])
    # Permutationsvikt på senaste tredjedelen (mindre känslig än impurity-vikt)
    n = ok.sum()
    sl = slice(int(n * 2 / 3), n)
    pi = permutation_importance(rf, X[ok].iloc[sl], y[ok].iloc[sl], n_repeats=10, random_state=seed)
    return pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(top)
