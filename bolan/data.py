"""Datainsamling: Riksbanken (SWEA API) och SCB (PxWeb API).

Allt mellanlagras som CSV i data/raw så att verktyget fungerar offline och
så att körningar är reproducerbara. `bygg_panel()` slår ihop allt till en
månadspanel som resten av pipelinen använder.

Källor
------
* SCB FM5001 RantaT04N  – genomsnittliga bolåneräntor (nya och omförhandlade
  avtal, hushåll) per räntebindningstid, 2005M09–. Detta är vad kunderna
  faktiskt betalar, dvs. listränta minus genomsnittlig rabatt.
* SCB PR0101 KPIF2020    – KPIF-index, för inflation.
* Riksbanken SWEA        – styrränta, statsskuldväxlar, stats- och
  bostadsobligationer samt utländska räntor (månadsmedel).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# Räntebindningstider som verktyget arbetar med: namn -> månader
TENORS: dict[str, int] = {"3M": 3, "1Y": 12, "2Y": 24, "3Y": 36, "5Y": 60, "10Y": 120}

# SCB-kod för ursprunglig räntebindningstid -> vår tenor.
# ">5 år" används som proxy för 10 år (i praktiken 7–10-årslån).
SCB_TENOR_CODES = {
    "1.1.1": "3M",          # t.o.m. 3 mån (rörligt)
    "1.1.2.1": "1Y",        # över 3 mån – 1 år
    "1.1.2.2.1.1": "2Y",    # 1–2 år
    "1.1.2.2.1.2": "3Y",    # 2–3 år
    "1.1.2.2.2": "5Y",      # 3–5 år
    "1.1.2.3": "10Y",       # över 5 år
}

RIKSBANK_SERIES = {
    "SECBREPOEFF": "styrranta",
    "SETB3MBENCH": "ssvx_3m",
    "SETB6MBENCH": "ssvx_6m",
    "SEGVB2YC": "stat_2y",
    "SEGVB5YC": "stat_5y",
    "SEGVB10YC": "stat_10y",
    "SEMB2YCACOMB": "bobl_2y",
    "SEMB5YCACOMB": "bobl_5y",
    "DEGVB10Y": "de_10y",
    "USGVB10Y": "us_10y",
    "EMGVB5Y": "eu_5y",
}

SWEA = "https://api.riksbank.se/swea/v1"
SCB = "https://api.scb.se/OV0104/v1/doris/sv/ssd"


def _get(url: str, *, method: str = "GET", json_body=None, tries: int = 6):
    wait = 5.0
    for i in range(tries):
        r = requests.request(method, url, json=json_body, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(wait)
            wait *= 2
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Kunde inte hämta {url} (status {r.status_code})")


def hamta_riksbanken(start: str = "1985-01-01", slut: str | None = None) -> pd.DataFrame:
    """Månadsmedel + månadsintervall (max-min, ett volatilitetsmått) per serie."""
    slut = slut or pd.Timestamp.today().strftime("%Y-%m-%d")
    frames = []
    for sid, namn in RIKSBANK_SERIES.items():
        rows = _get(f"{SWEA}/ObservationAggregates/{sid}/M/{start}/{slut}")
        df = pd.DataFrame(rows)
        df["datum"] = pd.to_datetime(df["from"]).dt.to_period("M").dt.to_timestamp()
        df = df.set_index("datum")
        frames.append(df["average"].rename(namn))
        frames.append((df["max"] - df["min"]).rename(f"{namn}_spann"))
        time.sleep(1.5)  # Riksbankens öppna API har låg anropskvot
    out = pd.concat(frames, axis=1).sort_index()
    out.to_csv(RAW / "riksbanken_manad.csv")
    return out


def _scb_post(tabell: str, query: list) -> dict:
    return _get(f"{SCB}/{tabell}", method="POST",
                json_body={"query": query, "response": {"format": "json"}})


def hamta_scb_bolan() -> pd.DataFrame:
    q = [
        {"code": "Referenssektor", "selection": {"filter": "item", "values": ["1"]}},
        {"code": "Motpartssektor", "selection": {"filter": "item", "values": ["2c"]}},
        {"code": "Avtal", "selection": {"filter": "item", "values": ["0100"]}},
        {"code": "Rantebindningstid", "selection": {"filter": "item", "values": list(SCB_TENOR_CODES)}},
    ]
    d = _scb_post("FM/FM5001/FM5001C/RantaT04N", q)
    rows = []
    for item in d["data"]:
        kod, period = item["key"][3], item["key"][4]
        val = item["values"][0]
        if val in ("..", ""):
            continue
        rows.append((pd.Timestamp(period.replace("M", "-") + "-01"), SCB_TENOR_CODES[kod], float(val)))
    df = pd.DataFrame(rows, columns=["datum", "tenor", "ranta"])
    df = df.pivot(index="datum", columns="tenor", values="ranta")[list(TENORS)].sort_index()
    df.columns = [f"bolan_{c}" for c in df.columns]
    df.to_csv(RAW / "scb_bolan.csv")
    return df


def hamta_scb_kpif() -> pd.DataFrame:
    d = _scb_post("PR/PR0101/PR0101G/KPIF2020", [])
    rows = [(pd.Timestamp(i["key"][-1].replace("M", "-") + "-01"), float(i["values"][0]))
            for i in d["data"] if i["values"][0] not in ("..", "")]
    df = pd.DataFrame(rows, columns=["datum", "kpif"]).set_index("datum").sort_index()
    df.to_csv(RAW / "scb_kpif.csv")
    return df


MPD = "https://api.riksbank.se/monetary_policy_data/v1/forecasts"
PROGNOS_SERIER = {
    "SEQRATENAYNA": "rb_styr",     # Riksbankens räntebana (kvartal)
    "SEMCPIFNAYNA": "rb_kpif",     # Riksbankens KPIF-prognos, årstakt (månad)
    "SEDMENAYNA": "mkt_styr",      # Marknadens prissatta styrränteförväntningar (kvartal)
}


def hamta_riksbank_prognoser() -> pd.DataFrame:
    """Alla prognosomgångar (vintages) sedan 2015 för räntebana, KPIF och marknadsförväntningar.

    Varje rad: serie, prognosomgång, publiceringsdag, datum som avses, värde. Genom att
    bara använda omgångar som var publicerade vid en viss tidpunkt blir de historiska
    features "realtidskorrekta" (ingen framtida information läcker in).
    """
    rader = []
    for sid, namn in PROGNOS_SERIER.items():
        d = _get(f"{MPD}?series={sid}")["data"][0]
        for v in d["vintages"]:
            md = v["metadata"]
            for o in v["observations"]:
                if o["value"] is None:
                    continue
                rader.append((namn, md["policy_round"], md["policy_round_end_dtm"][:10],
                              md["forecast_cutoff_date"], o["dt"], float(o["value"])))
        time.sleep(1.0)
    df = pd.DataFrame(rader, columns=["serie", "omgang", "publicerad", "cutoff", "datum", "varde"])
    df.to_csv(RAW / "riksbank_prognoser.csv", index=False)
    return df


def las_prognoser() -> pd.DataFrame:
    p = RAW / "riksbank_prognoser.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} saknas – kör `python -m bolan hamta` först.")
    return pd.read_csv(p, parse_dates=["publicerad", "cutoff", "datum"])


def hamta_allt() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    print("Hämtar SCB bolåneräntor ...")
    hamta_scb_bolan()
    print("Hämtar SCB KPIF ...")
    hamta_scb_kpif()
    print("Hämtar Riksbankens prognoser (räntebana, KPIF, marknadsförväntningar) ...")
    hamta_riksbank_prognoser()
    print("Hämtar Riksbanken (tar ett par minuter p.g.a. anropskvot) ...")
    hamta_riksbanken()
    print(f"Klart. Data sparad i {RAW}")


def _las(namn: str) -> pd.DataFrame:
    p = RAW / namn
    if not p.exists():
        raise FileNotFoundError(f"{p} saknas – kör `python -m bolan hamta` först.")
    return pd.read_csv(p, index_col=0, parse_dates=True)


def bygg_panel(med_proxy: bool = True) -> pd.DataFrame:
    """Månadspanel med bolåneräntor, marknadsräntor och inflation.

    Om `med_proxy` förlängs bolåneräntorna bakåt före 2005 med en linjär
    modell mot marknadsräntor (skattad på överlappet). Raderna markeras med
    `proxy=1` så att modeller kan vikta/utesluta dem.
    """
    rb = _las("riksbanken_manad.csv")
    bol = _las("scb_bolan.csv")
    kpif = _las("scb_kpif.csv")

    panel = rb.join(bol, how="outer").join(kpif, how="outer").sort_index()
    panel = panel[panel.index >= "1994-01-01"]
    panel["kpif_yoy"] = panel["kpif"].pct_change(12, fill_method=None) * 100
    panel["kpif_3m_ann"] = ((panel["kpif"] / panel["kpif"].shift(3)) ** 4 - 1) * 100
    panel["proxy"] = 0

    if med_proxy:
        # Förklarande marknadsräntor per tenor (bankernas finansiering)
        drivare = {
            "3M": ["ssvx_3m", "styrranta"],
            "1Y": ["ssvx_6m", "bobl_2y"],
            "2Y": ["bobl_2y"],
            "3Y": ["bobl_2y", "bobl_5y"],
            "5Y": ["bobl_5y"],
            "10Y": ["bobl_5y", "stat_10y"],
        }
        for t, x in drivare.items():
            y = panel[f"bolan_{t}"]
            X = panel[x]
            ok = y.notna() & X.notna().all(axis=1)
            A = np.column_stack([np.ones(ok.sum()), X[ok].values])
            beta, *_ = np.linalg.lstsq(A, y[ok].values, rcond=None)
            # Fyll bara bakåt i tiden (före första SCB-observationen)
            fyll = y.isna() & (panel.index < y.first_valid_index()) & X.notna().all(axis=1)
            panel.loc[fyll, f"bolan_{t}"] = beta[0] + X[fyll].values @ beta[1:]
            panel.loc[fyll, "proxy"] = 1
        # Månader efter senaste SCB-publicering tas bort
        panel = panel[panel.index <= bol.index.max()]

    # Bara månader där vi har den rörliga räntan
    panel = panel[panel["bolan_3M"].notna()]
    # Fyll enstaka luckor i marknadsserier framåt (t.ex. senaste månad ej publicerad)
    panel = panel.ffill(limit=2)
    return panel


if __name__ == "__main__":
    hamta_allt()
