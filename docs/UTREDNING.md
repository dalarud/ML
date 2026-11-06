# Utredning, analys och plan: ML-stöd för bolåneräntebeslut

## 1. Problemformulering

Målet är att minimera den **långsiktiga räntekostnaden** för ett bolån och samtidigt **begränsa risken**.
Beslutet består av fyra delar:

| Beslut | Fråga |
|---|---|
| **Bindningstid** | Rörligt (3 mån) eller 1, 2, 3, 5, 10 år – eller en mix/trappa? |
| **Tidpunkt** | Binda nu eller ligga rörligt några månader först? |
| **Bank / flytt** | Ger en annan bank (efter flyttkostnad och ev. ränteskillnadsersättning) lägre riskjusterad kostnad? |
| **Rabatt** | Rabatten mot listräntan skiljer sig per bindningstid och krymper ofta för passiva kunder. |

Det här är i grunden ett **beslut under osäkerhet**, inte ett rent prognosproblem. Därför består verktyget av
tre lager: (1) prediktiva modeller för räntan, (2) en scenariogenerator för hela räntekurvan och
(3) en rekommendationsmotor som utvärderar strategier över scenarierna med ett riskmått.

## 2. Data ("stordata"-delen)

| Källa | Innehåll | Period |
|---|---|---|
| SCB FM5001 (RantaT04N) | Genomsnittlig bolåneränta, nya avtal, per bindningstid (3M, 3M–1 år, 1–2, 2–3, 3–5, >5 år) | 2005– (vissa tider 2014–) |
| SCB KPIF | Inflation | 1987– |
| Riksbanken SWEA | Styrränta, statsskuldväxlar, stats-/bostadsobligationer 2/5/10 år, Tyskland/USA 10 år – månadsmedel och månadsspann (volatilitet) från dagliga noteringar | 1985– |

SCB:s snittränta är vad kunder **faktiskt betalar**, dvs. listränta minus genomsnittlig rabatt. För att få
längre historik (inkl. 90-talets högräntemiljö) rekonstrueras bolåneräntorna före första SCB-observationen
med en linjär modell mot bankernas finansieringsräntor, skattad på överlappet. Dessa rader markeras som
proxy och väger 50 % i träningen. Resultat: 388 månader × 32 variabler, ~52 features.

## 3. Analys – vad historiken säger

* **Bindningspremien är i snitt negativ för långa tider.** Den som bundit 5 år har i genomsnitt betalat
  ≈ 0,5 procentenheter mer än rörligt (premie_5Y, naivt medel −0,49 pe). Banker tar betalt för att bära risken.
* **Backtest 2005– (rullande 5-årsfönster, faktiska SCB-räntor):** rörligt och 1 år var billigast i 30 % av
  fönstren vardera, 2 år 18 %, 5 år 16 %, 3 år 6 %. Snittränta: 1Y 2,41 %, 3M 2,45 %, 2Y 2,43 %, 5Y 3,01 %.
* **Men långa bindningar skyddar mot de värsta utfallen** – lägst max-ränta och minst 12-månadersökning.
* **Rabatter:** rabatten är oftast störst på rörligt och minskar med bindningstiden. Passiva kunder får ofta
  en lägre rabatt vid nästa räntesättning – i modellen kostar passivitet tydligt.

## 4. Modeller och validering

**Mål:**
* `d3m_h` – förändring i rörlig bolåneränta om h = 3, 6, 12, 24, 36, 60 månader.
* `premie_k` – genomsnittlig rörlig ränta under kommande k månader minus bunden k-årsränta idag
  (> 0 ⇒ att binda blir billigare). k = 1, 2, 3, 5 år.

**Modeller:** Ridge (linjär baslinje), Random Forest, Gradient Boosting (HistGB), neuralt nät (MLP) och en
ensemble viktad med invers out-of-sample-MSE bland de modeller som slår en naiv prognos.

**Validering: purged walk-forward k-fold (k = 5).** Data delas i kronologiska block; varje block prognostiseras
av modeller tränade enbart på äldre data, med ett glapp lika långt som horisonten (annars läcker överlappande
mål framtid in i träningen). Vanlig blandad k-fold körs som jämförelse för att visa läckaget.

**Resultat (R² out-of-sample mot naiv prognos):**

| Mål | Bästa modeller | Ensemble R² | Blandad k-fold (läckande) |
|---|---|---|---|
| 3 mån | RF, Ridge | ≈ 0,2 | ≈ 0,6 |
| 6 mån | RF, GB | ≈ 0,28 | ≈ 0,7 |
| 12 mån | RF, GB | ≈ 0,27 | ≈ 0,8 |
| 24–60 mån | – (ingen slår naiv stabilt) | ≈ 0 | ≈ 0,9 |
| bindningspremie 1–5 år | – (svag/ingen) | ≈ 0–0,06 | ≈ 0,7–0,95 |

Slutsats: **modellerna har verklig prognosförmåga 3–12 månader framåt, men inte på flera års sikt.** Den
som utlovar mer har sannolikt validerat fel (se kolumnen blandad k-fold). En walk-forward-backtest av en
ren ML-timingstrategi (bind den tid med högst predikterad premie) slog inte enkla 1–2-årsbindningar 2008–.

## 5. Design som följer av analysen

1. **Skicklighetsviktad förväntad bana:** för varje horisont vägs ML-ensemblen mot marknadens implicita
   bana (dagens kurva minus historisk bindningspremie) med vikt `clip(R²_oos/0,25, 0, 1)`. ML styr alltså
   där den bevisat fungerar, marknaden annars.
2. **Scenariogenerator:** rörlig ränta = förväntad bana + block-bootstrappade historiska chocker (12-mån
   block, hela kurvan samtidigt) med svag återgång; spreadar per bindningstid följer skattad AR(1). Spridningen
   är kalibrerad mot historiken (5-årsförändringar: sim ±2,6 pe, historiskt −2,5/+2,75 pe).
3. **Personlig ränta:** `marknadssnitt + (listränta − snitt idag) − rabatt·f` per bindningstid, där `f` = 1
   om du förhandlar och bankens "befintlig kund"-andel annars.
4. **Kostnad:** nuvärde av räntekostnad efter ränteavdrag (30 %/21 %), amortering, flyttkostnad,
   ränteskillnadsersättning vid brytning, och marknadsvärdet av låst ränta som löper efter horisonten.
5. **Poäng:** `E[kostnad] + λ·(CVaR95 − E[kostnad])`. λ = 0 riskneutral, 0,25 balanserad, 0,5 försiktig, 1+ mycket försiktig.
6. **Robusthet:** alla strategier räknas även under "marknadens prissättning" och "oförändrad ränta".

## 6. Strategier som utvärderas

Rörligt · Bind 1/2/3/5/10 år (förnya) · Vänta 6/12 mån och bind 2/3/5 år · Mix 50/50 rörligt + 2/3/5 år ·
Trappa rörligt/2/5 år · Trappa 1/2/3/5 år · Passiva varianter · (Om bundet idag:) bryt bindning nu + RSE.
Varje strategi körs hos varje bank i profilen → flyttanalys.

## 7. Begränsningar

* Ränteprognoser på flera års sikt är i praktiken inte möjliga – verktyget hanterar det med scenarier och riskmått.
* Proxy-data före 2005/2014; "10 år" bygger på SCB:s ">5 år".
* RSE-beräkningen är förenklad – begär bankens beräkning.
* Beslutsstöd, inte finansiell rådgivning.

## 8. Vidareutveckling

* Historiska listräntor per bank (t.ex. bankernas egna snitträntor) för att modellera rabatter över tid.
* Riksbankens räntebana och STIBOR/SWESTR-terminer som features.
* Månatlig automatisk körning med notis när signalen för bindning ändras.
