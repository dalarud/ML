"""HTML-rapport med diagram (matplotlib, inbäddade som PNG)."""
from __future__ import annotations

import base64
import html
import io
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import kor_ml_timing, rullande_fonster, sammanfattning  # noqa: E402
from .recommend import sammanfatta  # noqa: E402

# Validerad kategorisk palett (fast ordning)
FARG = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT2 = "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8, "lines.linewidth": 2,
    "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _kr(x):
    return f"{x:,.0f}".replace(",", " ")


def _tabell(df: pd.DataFrame, fmt: dict | None = None, index=True) -> str:
    fmt = fmt or {}
    d = df.copy()
    for c in d.columns:
        f = fmt.get(c)
        if f:
            d[c] = d[c].map(f)
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.2f}")
    return d.to_html(index=index, escape=True, border=0, classes="tbl")


def diagram_historik(p):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    p = p[p.index >= "2000-01-01"]
    for i, (c, n) in enumerate([("bolan_3M", "Rörlig 3 mån"), ("bolan_2Y", "2 år"), ("bolan_5Y", "5 år"),
                                ("styrranta", "Styrränta")]):
        ax.plot(p.index, p[c], color=FARG[i], label=n, lw=1.6 if c != "styrranta" else 1.2,
                ls="-" if c != "styrranta" else "--")
    prox = p.index[p["proxy"] == 1]
    if len(prox):
        ax.axvspan(prox.min(), prox.max(), color="#ecebe7", alpha=0.6, lw=0)
        ax.text(prox.min(), ax.get_ylim()[1] * 0.95, " proxy (rekonstruerad)", color=TEXT2, fontsize=8, va="top")
    ax.set_ylabel("%")
    ax.legend(ncol=4, loc="upper right")
    return _png(fig)


def diagram_prognos(a):
    S = a["sims"][a["huvud"]][:, :, 0]
    M = S.shape[1]
    hist = a["panel"]["bolan_3M"].iloc[-60:]
    t0 = hist.index[-1]
    fram = pd.date_range(t0, periods=M, freq="MS")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(hist.index, hist.values, color="#0b0b0b", lw=1.6, label="Utfall (SCB)")
    q = np.quantile(S, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    ax.fill_between(fram, q[0], q[4], color=FARG[0], alpha=0.12, lw=0, label="5–95 % scenarier")
    ax.fill_between(fram, q[1], q[3], color=FARG[0], alpha=0.25, lw=0, label="25–75 %")
    ax.plot(fram, a["banor"]["huvud"][:M], color=FARG[0], label="Huvudscenario (förväntat)")
    ax.plot(fram, a["banor"]["marknad"][:M], color=FARG[1], ls="--", lw=1.4, label="Marknadens prissättning")
    ax.plot(fram, a["banor"]["riksbank"][:M], color=FARG[3], ls="-.", lw=1.4, label="Via Riksbankens räntebana")
    ml = a["banor"]["ml"][:min(M, 61)]
    ax.plot(fram[:len(ml)], ml, color=FARG[2], ls=":", lw=1.6, label="Ren ML-ensemble")
    ax.set_ylabel("Rörlig bolåneränta, %")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    return _png(fig)


def diagram_styr(a):
    mk, p = a["makro"], a["panel"]
    hist = p["styrranta"].iloc[-72:]
    t0 = hist.index[-1]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(hist.index, hist.values, color="#0b0b0b", lw=1.6, label="Styrränta (utfall)")
    b = mk["banor"]
    rb = b["rb_styr"][b["rb_styr"].index >= t0]
    mkt = b["mkt_styr"][b["mkt_styr"].index >= t0]
    ax.plot(rb.index, rb.values, color=FARG[1], ls="--", lw=1.6, label=f"Riksbankens räntebana ({b['rb_styr_omgang']})")
    ax.plot(mkt.index, mkt.values, color=FARG[3], ls=":", lw=1.8, label="Marknadens prissättning")
    h = [0] + [int(i.split()[0]) for i in mk["styr"].index]
    x = [t0 + pd.DateOffset(months=i) for i in h]
    for kol, f, mark in [("ML-ensemble", FARG[2], "o"), ("Kombination", FARG[0], "D")]:
        y = [mk["styr0"]] + list(mk["styr"][kol])
        ax.plot(x, y, color=f, marker=mark, ms=5, lw=1.6, label=kol)
    ax.set_ylabel("Styrränta, %")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    return _png(fig)


def diagram_kpif(a):
    mk, p = a["makro"], a["panel"]
    hist = p["kpif_yoy"].iloc[-72:]
    t0 = hist.index[-1]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axhline(2.0, color="#c9c8c2", lw=1, ls="--")
    ax.text(hist.index[0], 2.05, " inflationsmål 2 %", color=TEXT2, fontsize=8, va="bottom")
    ax.plot(hist.index, hist.values, color="#0b0b0b", lw=1.6, label="KPIF, årstakt (utfall)")
    rb = mk["banor"]["rb_kpif"]
    rb = rb[rb.index >= t0]
    ax.plot(rb.index, rb.values, color=FARG[1], ls="--", lw=1.6, label=f"Riksbankens prognos ({mk['banor']['rb_kpif_omgang']})")
    h = [0] + [int(i.split()[0]) for i in mk["kpif"].index]
    x = [t0 + pd.DateOffset(months=i) for i in h]
    for kol, f, mark in [("ML-ensemble", FARG[2], "o"), ("Kombination", FARG[0], "D")]:
        ax.plot(x, [mk["kpif0"]] + list(mk["kpif"][kol]), color=f, marker=mark, ms=5, lw=1.6, label=kol)
    ax.set_ylabel("KPIF, %")
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    return _png(fig)


def diagram_sannolikhet(a):
    tab = a["sannolikhet"]["tabeller"]
    fig, axes = plt.subplots(1, len(tab), figsize=(9, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (h, t) in zip(axes, tab.items()):
        x = np.arange(len(t))
        b = 0.26
        for i, (kol, f) in enumerate([("Historisk basfrekvens", "#b9b8b2"), ("Simulering (före viktning)", FARG[1]),
                                      ("Använd sannolikhet", FARG[0])]):
            ax.bar(x + (i - 1) * b, t[kol].values * 100, width=b - 0.03, color=f, label=kol)
        ax.set_xticks(x, [k.replace(" ", "\n") for k in t.index], fontsize=7.5)
        ax.set_title(f"Rörlig bolåneränta om {h} mån", fontsize=9.5, color=TEXT2)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Sannolikhet, %")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    return _png(fig)


def diagram_feature(fi: pd.Series):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    fi = fi.iloc[::-1]
    ax.barh(fi.index, fi.values, color=FARG[0], height=0.6)
    ax.set_xlabel("Permutationsvikt (minskning i R²) – rörlig ränta om 12 mån")
    ax.grid(axis="y", visible=False)
    return _png(fig)


def diagram_risk(df: pd.DataFrame, bank: str):
    d = df[df["bank"] == bank]
    kat = {"statisk": FARG[0], "timing": FARG[1], "mix": FARG[2], "passiv": FARG[7], "bryt": FARG[6]}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for k, c in kat.items():
        s = d[d["kategori"] == k]
        if len(s):
            ax.scatter(s["forvantad_kostnad"] / 1000, s["cvar"] / 1000, s=46, color=c, label=k,
                       edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    for _, r in d.nsmallest(6, "poang").iterrows():
        ax.annotate(r["strategi"], (r["forvantad_kostnad"] / 1000, r["cvar"] / 1000), fontsize=7.5,
                    color=TEXT2, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("Förväntad kostnad, tkr (nuvärde efter avdrag)")
    ax.set_ylabel("CVaR – snitt av sämsta scenarier, tkr")
    ax.legend(title="Strategityp", fontsize=8)
    return _png(fig)


def diagram_fordelning(df, a):
    top = df.drop_duplicates("strategi").head(8)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    data, namn = [], []
    from .strategies import standardstrategier  # noqa
    for _, r in top.iterrows():
        data.append([r["p5"] / 1000, r["median"] / 1000, r["p95"] / 1000, r["forvantad_kostnad"] / 1000])
        namn.append(f"{r['strategi']}\n({r['bank']})")
    y = np.arange(len(data))[::-1]
    for yi, (p5, med, p95, mu) in zip(y, data):
        ax.plot([p5, p95], [yi, yi], color=FARG[0], lw=2, solid_capstyle="round")
        ax.scatter([med], [yi], color=FARG[0], s=40, zorder=3, label="Median" if yi == y[0] else None)
        ax.scatter([mu], [yi], color=FARG[1], marker="D", s=30, zorder=3, label="Förväntat" if yi == y[0] else None)
    ax.set_yticks(y, namn, fontsize=7.5)
    ax.set_xlabel("Kostnad tkr, 5:e–95:e percentil")
    ax.legend(fontsize=8)
    return _png(fig)


def diagram_backtest(rf: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, c in enumerate(rf.columns):
        ax.plot(rf.index, rf[c], color=FARG[i], lw=1.5, label=f"Förnya {c}")
    ax.set_ylabel("Snittränta kommande 5 år, %")
    ax.set_xlabel("Startmånad")
    ax.legend(ncol=5, fontsize=8)
    return _png(fig)


CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--line:#e4e3de;--acc:#2a78d6;--card:#f4f3ef}
body{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg);
max-width:980px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px;border-top:1px solid var(--line);padding-top:20px}
.sub{color:var(--fg2);margin:0 0 20px}.rek{background:var(--card);border-left:4px solid var(--acc);
padding:14px 18px;border-radius:6px;white-space:pre-wrap;font:13.5px/1.55 ui-monospace,Menlo,monospace;overflow-x:auto}
img{max-width:100%;height:auto}.tbl{border-collapse:collapse;font-size:12.5px;width:100%;margin:8px 0}
.tbl th,.tbl td{padding:5px 8px;white-space:nowrap;border-bottom:1px solid var(--line);text-align:right}
.tbl th:first-child,.tbl td:first-child{text-align:left}.wrap{overflow-x:auto}
.note{color:var(--fg2);font-size:13px}
"""


def ekonomi_sektion(a, df) -> str:
    ek = a.get("ekonomi")
    if not ek:
        return ""
    lan = a["profil"].lan
    pct = lambda v: f"{v:.0%}"
    plan = ek["plan"].copy()
    plan.columns = ["Lån vid årets början kr", "Amortering under året kr", "Belåningsgrad", "Amorteringskrav"]
    kalp_kol = [c for c in ["boende_ar1", "boende_max_p95", "boendeandel_max_p95", "kvar_min_p5", "p_underskott"] if c in df]
    kalp = df.head(10)[["strategi"] + kalp_kol].rename(columns={
        "strategi": "Strategi", "boende_ar1": "Boendekostnad år 1 kr/mån", "boende_max_p95": "Högsta boendekostnad P95 kr/mån",
        "boendeandel_max_p95": "Andel av netto P95", "kvar_min_p5": "Minsta kvar att leva på P5 kr/mån",
        "p_underskott": "P(underskott)"})
    fmt = {"Boendekostnad år 1 kr/mån": _kr, "Högsta boendekostnad P95 kr/mån": _kr, "Andel av netto P95": pct,
           "Minsta kvar att leva på P5 kr/mån": _kr, "P(underskott)": lambda v: f"{v:.1%}"}
    stress = ek["stress"]
    sfmt = {c: _kr for c in stress.columns if "kr" in c}
    sfmt["Andel av nettoinkomst"] = pct
    return "".join([
        "<h2>Din ekonomi: belåningsgrad, amortering och betalningsförmåga</h2>",
        f"<p class='note'>Bostad {_kr(lan.bostadsvarde)} kr, lån {_kr(lan.belopp)} kr → belåningsgrad "
        f"{ek['belaningsgrad']:.0%}. Skuldkvot {ek['skuldkvot']:.1f} gånger hushållets bruttoinkomst (under 4,5 – "
        "inget extra amorteringskrav). Amorteringskravet: 2 % av lånet per år över 70 % belåning, 1 % mellan 50 och "
        f"70 %. Bostadsvärdet antas oförändrat ({lan.bostad_prisutveckling:.0f} %/år) och omvärderas vart 5:e år. "
        "Boendekostnad = ränta efter avdrag + amortering + övriga boendekostnader. Kvar att leva på = "
        "nettoinkomst − boendekostnad − levnadskostnader.</p>",
        "<h3>Amorteringsplan</h3><div class='wrap'>",
        _tabell(plan, {"Lån vid årets början kr": _kr, "Amortering under året kr": _kr, "Belåningsgrad": pct,
                       "Amorteringskrav": pct}), "</div>",
        "<h3>Betalningsförmåga per strategi (sannolikhetsviktad)</h3><div class='wrap'>",
        _tabell(kalp, fmt, index=False), "</div>",
        "<h3>Stresstest: månadskostnad år 1 vid olika räntor</h3>",
        "<p class='note'>Bankerna prövar normalt låntagare mot en kalkylränta runt 7 %.</p><div class='wrap'>",
        _tabell(stress, sfmt), "</div>",
    ])


def bygg_rapport(a: dict, ut: Path) -> Path:
    df = a["resultat"]
    res = a["modeller"]
    prof = a["profil"]
    bast_bank = df.iloc[0]["bank"]

    cv = pd.DataFrame({k: v.cv["R2_oos"] for k, v in res.items()}).T
    cv_bl = pd.DataFrame({k: v.cv_blandad["R2_oos"] for k, v in res.items()}).T.add_suffix(" (blandad k-fold)")
    cvtab = cv.join(cv_bl)
    trf = pd.DataFrame({k: v.cv["traffsakerhet_tecken"] for k, v in res.items()}).T

    p = a["panel"]
    rf = rullande_fonster(p)
    vinnare = rf.idxmin(axis=1).value_counts(normalize=True).rename("andel fönster där billigast")
    serier, _ = kor_ml_timing(p)
    bt = sammanfattning(serier)

    visa = df.head(20)[["bank", "strategi", "snittranta", "snittranta_p95", "forvantad_kostnad", "cvar",
                        "max_manad_p95", "forsta_ar_manad", "samsta_rang", "poang"]].copy()
    visa.columns = ["Bank", "Strategi", "Snittränta %", "Snittränta P95 %", "Förväntad kostnad kr",
                    "CVaR kr", "Max mån.kostn P95 kr", "Mån.kostn år 1 kr", "Sämsta rang", "Poäng"]
    krf = {c: _kr for c in ["Förväntad kostnad kr", "CVaR kr", "Max mån.kostn P95 kr", "Mån.kostn år 1 kr", "Poäng"]}
    krf["Sämsta rang"] = lambda v: f"{int(v)}"

    scen_kol = [c for c in df.columns if c.startswith("kostnad | ")]
    robust = df.head(12)[["bank", "strategi", "forvantad_kostnad"] + scen_kol].copy()
    robust.columns = ["Bank", "Strategi", "Huvudscenario"] + [c.replace("kostnad | ", "") for c in scen_kol]

    rabatt = pd.DataFrame({b.namn: b.rabatter for b in prof.banker}).T
    lista = pd.DataFrame({b.namn: b.listrantor for b in prof.banker}).T

    sig = a["signaler"].copy()
    sig["P(bindning billigare än rörlig)"] = sig["P(bindning billigare än rörlig)"].map(lambda v: f"{v:.0%}")
    sig["förväntad bindningspremie (pe)"] = sig["förväntad bindningspremie (pe)"].map(lambda v: f"{v:+.2f}")

    sektioner = [
        f"<h1>Bolåneplan</h1><p class='sub'>Genererad {date.today()} · data t.o.m. "
        f"{p.index[-1]:%Y-%m} · lån {_kr(prof.lan.belopp)} kr · horisont {prof.lan.horisont_ar} år · "
        f"riskaversion λ={prof.riskaversion}</p>",
        "<div class='rek'>" + html.escape("\n".join(sammanfatta(a))) + "</div>",
        "<h2>1. Dina erbjudanden: listränta, rabatt och effektiv ränta</h2>",
        "<p class='note'>Effektiv ränta = listränta − rabatt. Marknadssnitt = SCB:s genomsnittliga ränta för nya "
        "avtal (vad kunder i snitt faktiskt betalar). Rabatten skiljer sig mellan bindningstider – modellen räknar "
        "med rätt rabatt per tid, och med att passiva kunder får en mindre del av rabatten vid nästa "
        "räntesättning.</p>",
        "<h3>Effektiv ränta %</h3><div class='wrap'>" + _tabell(a["erbjudanden"]) + "</div>",
        "<h3>Listräntor %</h3><div class='wrap'>" + _tabell(lista) + "</div>",
        "<h3>Rabatter pe</h3><div class='wrap'>" + _tabell(rabatt) + "</div>",
        ekonomi_sektion(a, df),
        "<h2>2. Historik</h2>", diagram_historik(p),
        "<h2>3. Inflation och styrränta (steg 1 i ML-kedjan)</h2>",
        "<p class='note'>Styrräntan och KPIF-inflationen prognostiseras med samma ML-modeller (Ridge, Random "
        "Forest, Gradient Boosting, neuralt nät) och jämförs med Riksbankens egna publicerade prognoser och "
        "marknadens prissättning. Alla tre kombineras med vikter efter träffsäkerhet out-of-sample sedan 2015. "
        "Prognoserna förs sedan vidare som förklaringsvariabler till bolånemodellerna och via "
        "styrräntekanalen (genomslag styrränta → rörlig bolåneränta).</p>",
        diagram_styr(a), "<div class='wrap'>" + _tabell(a["makro"]["styr"]) + "</div>",
        diagram_kpif(a), "<div class='wrap'>" + _tabell(a["makro"]["kpif"]) + "</div>",
        "<h3>Träffsäkerhet: ML mot Riksbanken och marknaden (RMSE, procentenheter, lägre är bättre)</h3>",
        "<p class='note'>Samma out-of-sample-månader (2015–) för alla. dstyr = förändring i styrränta, "
        "dkpif = förändring i KPIF-inflation, siffran = horisont i månader. Där ML inte slår naiv prognos "
        "(t.ex. styrränta 24 mån) är ML-prognosen = oförändrat, och vikten flyttas i praktiken till "
        "Riksbanken och marknaden.</p>",
        "<div class='wrap'>" + _tabell(a["makro"]["validering"], {"n_manader": lambda v: f"{int(v)}"}) + "</div>",
        "<h3>Kombinationsvikter</h3><div class='wrap'>" + _tabell(a["makro"]["vikter"].fillna(0)) + "</div>",
        "<h3>Styrräntekanalen: genomslag till rörlig bolåneränta</h3>",
        "<p class='note'>beta = hur mycket den rörliga bolåneräntan historiskt rört sig per procentenhet "
        "styrränteförändring. vikt_kanal = andel av bolåneprognosen som går via styrräntan (resten är den "
        "direkta ML-prognosen).</p>",
        "<div class='wrap'>" + _tabell(a["makro"]["kanal"], {"n": lambda v: "" if pd.isna(v) else f"{int(v)}"}) + "</div>",
        "<h2>4. Sannolikheter för ränteutvecklingen</h2>",
        "<p class='note'>ML-klassificerare (Random Forest och logistisk regression) skattar sannolikheten för fem "
        "utfall: kraftig nedgång (&lt; −1 pe), nedgång (−1 till −0,25), oförändrad (±0,25), uppgång (0,25–1) och "
        "kraftig uppgång (&gt; 1 pe). De utvärderas med Brier-poäng mot historisk basfrekvens på samma "
        "out-of-sample-månader. Modellen får påverka sannolikheterna i den mån den bevisat träffsäkerhet; "
        "scenarierna viktas sedan om så att de matchar sannolikheterna, och alla kostnads- och riskmått "
        "nedan är sannolikhetsviktade.</p>",
        diagram_sannolikhet(a),
        "".join("<h3>Om " + str(h) + " månader</h3><div class='wrap'>" + _tabell(t, {c: (lambda v: f"{v:.0%}") for c in t.columns}) + "</div>"
                for h, t in a["sannolikhet"]["tabeller"].items()),
        "<h3>Validering av sannolikhetsmodellerna (Brier-poäng, lägre är bättre)</h3>",
        "<div class='wrap'>" + _tabell(a["sannolikhet"]["validering"].T, {"n_manader": lambda v: f"{int(v)}"}) + "</div>",
        f"<p class='note'>Effektivt antal scenarier efter omviktning: {a['sannolikhet']['ess']:.0f} av "
        f"{len(a['sannolikhet']['vikter'])}.</p>",
        "<h2>5. Prognos för rörlig bolåneränta</h2>",
        "<p class='note'>Huvudscenariot kombinerar den direkta ML-prognosen med styrräntekanalen och väger sedan mot marknadens implicita bana per horisont, med vikt efter bevisad träffsäkerhet out-of-sample (ML-vikt = 1 vid R² ≥ 0,25). Banden visar "
        f"{prof.scenarier} block-bootstrapscenarier.</p>",
        diagram_prognos(a), "<div class='wrap'>" + _tabell(a["prognos"]) + "</div>",
        "<h3>Bindningssignaler</h3><p class='note'>Bindningspremie = genomsnittlig rörlig ränta under "
        "bindningstiden minus bunden ränta idag. Positiv = att binda förväntas bli billigare.</p>",
        "<div class='wrap'>" + _tabell(sig) + "</div>",
        "<h2>6. Strategier – kostnad mot risk</h2>",
        f"<p class='note'>Varje punkt är en strategi hos {html.escape(bast_bank)}. Längre ner till vänster är "
        "bättre. Poäng = förväntad kostnad + λ·(CVaR − förväntad kostnad), där både förväntad kostnad och CVaR är "
        "sannolikhetsviktade.</p>",
        diagram_risk(df, bast_bank),
        "<div class='wrap'>" + _tabell(visa, krf, index=False) + "</div>",
        "<h3>Kostnadsspridning för de bästa strategierna</h3>", diagram_fordelning(df, a),
        "<h3>Scenariomatris: snittränta % per ränteutveckling (rörlig ränta om 3 år)</h3>",
        "<p class='note'>Rubrikerna visar sannolikheten för varje utfall. Läs raden för en strategi: så här "
        "hade din snittränta blivit om räntan utvecklas så. Hos " + html.escape(df.iloc[0]["bank"]) + ".</p>",
        "<div class='wrap'>" + _tabell(a["scenariomatris"].loc[[s for s in df[df["bank"] == df.iloc[0]["bank"]]["strategi"].head(12)]]
                                       .rename(columns={k: f"{k} ({v:.0%})" for k, v in a["scenariomatris_p"].items()})) + "</div>",
        "<h3>Ånger och sannolikhet att slå rörligt</h3>",
        "<p class='note'>Förväntad ånger = genomsnittlig merkostnad jämfört med den strategi som i efterhand hade "
        "varit billigast i samma scenario. P(billigare än rörligt) = sannolikheten att strategin blir billigare än "
        "rörlig ränta hos samma bank.</p>",
        "<div class='wrap'>" + _tabell(df.head(12)[["bank", "strategi", "p_billigare_an_rorlig", "forvantad_anger", "anger_p95"]]
                                       .rename(columns={"bank": "Bank", "strategi": "Strategi",
                                                        "p_billigare_an_rorlig": "P(billigare än rörligt)",
                                                        "forvantad_anger": "Förväntad ånger kr", "anger_p95": "Ånger P95 kr"}),
                                       {"P(billigare än rörligt)": lambda v: "–" if pd.isna(v) else f"{v:.0%}", "Förväntad ånger kr": _kr,
                                        "Ånger P95 kr": _kr}, index=False) + "</div>",
        "<h3>Robusthet: förväntad kostnad (kr) under alternativa scenarier</h3>",
        "<div class='wrap'>" + _tabell(robust, {c: _kr for c in robust.columns[2:]}, index=False) + "</div>",
        "<h2>7. Modellvalidering (purged walk-forward k-fold, k=5)</h2>",
        "<p class='note'>R² out-of-sample relativt naiv prognos (ingen förändring / historiskt snitt). "
        "&gt; 0 betyder att modellen slår den naiva prognosen på data den aldrig sett. Kolumnerna "
        "”blandad k-fold” visar vad vanlig slumpmässig k-fold felaktigt hade rapporterat – läckage "
        "från överlappande framtida mål ger skenbart höga värden.</p>",
        "<div class='wrap'>" + _tabell(cvtab) + "</div>",
        "<h3>Träffsäkerhet i riktning (andel rätt tecken)</h3><div class='wrap'>" + _tabell(trf) + "</div>",
        "<h3>Viktigaste förklaringsvariabler</h3>", diagram_feature(a["feature_importance"]),
        "<h2>8. Historisk backtest</h2>",
        "<p class='note'>Faktiska genomsnittsräntor (SCB). Rullande 5-årsfönster sedan 2005: snittränta om man "
        "alltid förnyat samma bindningstid.</p>", diagram_backtest(rf),
        "<div class='wrap'>" + _tabell(vinnare.to_frame(), {vinnare.name: lambda v: f"{v:.0%}"}) + "</div>",
        "<h3>ML-timing (walk-forward) mot statiska strategier, 2008–</h3>",
        "<div class='wrap'>" + _tabell(bt) + "</div>",
        "<h2>9. Metod och begränsningar</h2>",
        "<ul class='note'><li>Räntor är svåra att förutsäga: modellerna har påvisbar förmåga på 3–12 månader men "
        "inte på flera års sikt. Därför bygger rekommendationen på riskjusterad kostnad över tusentals scenarier, "
        "inte på en enskild prognos.</li><li>Före 2005 (och för vissa bindningstider före 2014) är "
        "bolåneräntorna rekonstruerade från marknadsräntor och väger hälften i träningen.</li>"
        "<li>”10 år” bygger på SCB:s kategori ”över 5 år”.</li>"
        "<li>Sannolikheterna kommer från ML-klassificerare med påvisad träffsäkerhet på 12 mån (Brier skill "
        "≈ 0,08) men inte på 36 mån; där bygger de i huvudsak på simuleringen. Utfallsdata för BNP och "
        "arbetslöshet är reviderade siffror.</li><li>Ränteskillnadsersättning är förenklad – "
        "begär alltid bankens exakta beräkning.</li><li>Verktyget är ett beslutsstöd, inte finansiell "
        "rådgivning.</li></ul>",
    ]
    doc = (f"<!DOCTYPE html><html lang='sv'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>Bolåneplan</title><style>{CSS}</style>"
           f"</head><body>{''.join(sektioner)}</body></html>")
    ut.parent.mkdir(parents=True, exist_ok=True)
    ut.write_text(doc, encoding="utf-8")
    return ut
