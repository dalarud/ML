# Bolåneplan – ML-verktyg för bolåneräntor

Verktyg för att planera bolånet: **bindningstid, tidpunkt för bindning, mix/trappa och byte av bank**, med
hänsyn till **rabatter mot listräntan per bindningstid**, ränteavdrag, amortering, flyttkostnader och
ränteskillnadsersättning. Målet: lägsta långsiktiga kostnad med begränsad risk.

Utredning, analys och modellval: se [docs/UTREDNING.md](docs/UTREDNING.md).

## Så fungerar det

```
SCB + Riksbanken ─► features (52 st) ─► Ridge / Random Forest / Gradient Boosting / Neuralt nät
                                         └─ purged walk-forward k-fold, ensemble
                                                   │
               marknadens implicita bana ──► skicklighetsviktad förväntad bana
                                                   │
                         block-bootstrap ──► 3 000 scenarier för hela räntekurvan
                                                   │
        din profil (lån, banker, listräntor, rabatter, risk) ──► ~50 strategier × banker
                                                   │
                     E[kostnad] + λ·(CVaR95 − E[kostnad])  ──► rekommendation + HTML-rapport
```

## Kom igång

```bash
pip install -r requirements.txt
python -m bolan hamta            # hämta senaste data (SCB + Riksbanken), ~1 min
# fyll i config/profil.toml med ditt lån, dina banker, listräntor och rabatter
python -m bolan analysera        # rekommendation + rapport/bolaneplan.html
python -m bolan analysera --riskaversion 0.5
python -m bolan backtest         # historisk backtest
python -m bolan validera         # korsvalideringsresultat
python -m pytest tests
```

Data finns redan cachad i `data/raw/` (t.o.m. 2026-09). Modellerna tränas vid första körningen (~3 min) och
cachas sedan tills nya data hämtas.

## Din profil (`config/profil.toml`)

* `[lan]` – belopp, horisont, amortering, nuvarande bindning (och månader kvar), ränteavdrag, budget.
* `[risk]` – `riskaversion` λ (0 riskneutral · 0,25 balanserad · 0,5 försiktig · 1+ mycket försiktig).
* `[[bank]]` – ett block per bank/offert: listräntor och **rabatter per bindningstid**, flyttkostnad och
  `befintlig_kund_rabattandel` (hur mycket av rabatten en passiv kund behåller).

Exemplets banker och räntor är fiktiva – ersätt med dina riktiga erbjudanden.

## Viktigaste insikterna från analysen

* ML-modellerna förutsäger rörlig ränta **3–12 månader** framåt bättre än naiv prognos (R² ≈ 0,2–0,28
  out-of-sample) – men **inte** på 2–5 års sikt. Vanlig k-fold ger skenbart R² ≈ 0,9 p.g.a. läckage.
* Historiskt (2005–) har rörligt och 1–2 år varit billigast; 5 år har i snitt kostat ≈ 0,5 pe mer men
  skyddar mot de värsta utfallen. Valet är därför främst en fråga om **riskaversion**, och verktyget visar
  exakt vad "försäkringen" kostar.
* Rabatten och att förhandla/flytta är ofta värd lika mycket som rätt bindningstid.

*Beslutsstöd, inte finansiell rådgivning.*
