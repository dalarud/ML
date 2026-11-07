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

## 4b. Inflation och Riksbankens räntebana – ML-kedja i två steg

Bolåneräntan styrs i grunden av styrräntan, som i sin tur styrs av inflationen. Modellen följer den kedjan:

**Data.** Riksbankens API för penningpolitiska prognoser ger *alla prognosomgångar sedan 2015* (60 st):
räntebanan, KPIF-prognosen och marknadens prissatta styrränteförväntningar. Vid varje historisk månad
används bara den omgång som då var publicerad (realtidsdata, ingen framtida information). Före 2015
används en marknadsproxy (2-årig statsränta − styrränta), markerad med en flagga.

**Steg 1 – makromodeller.** Samma modellfamilj (Ridge, Random Forest, Gradient Boosting, neuralt nät) och
samma purged walk-forward k-fold används för att förutsäga
* styrräntans förändring om 3, 6, 12, 24 månader (`dstyr_h`)
* KPIF-inflationens förändring om 6, 12, 24 månader (`dkpif_h`).

Varje ML-prognos jämförs med Riksbankens egen prognos och marknadens prissättning **på exakt samma
out-of-sample-månader (2015–)**. De tre kombineras sedan med vikter efter invers MSE. Kombinationen har lägst
fel på alla horisonter (t.ex. styrränta 12 mån: naiv 1,07, ML 0,81, kombination 0,66 pe RMSE).

**Steg 2 – bolånemodellerna** får som features: walk-forward-prognoserna från steg 1 (`ml_dstyr_6/12/24`,
`ml_dkpif_12`, framtagna med modeller som bara sett data fram till respektive månad), Riksbankens räntebana
och KPIF-prognos (`rb_dstyr_h`, `rb_dkpif_h`) samt marknadens förväntningar (`mkt_dstyr_h`).

**Styrräntekanalen.** Genomslaget β_h från styrränta till rörlig bolåneränta skattas historiskt. Prognosen
för rörlig ränta på 3–24 mån blir en viktning av den direkta ML-prognosen och β_h × kombinerad
styrränteprognos, med vikter efter out-of-sample-fel. Ett extra scenario, "Riksbankens räntebana", används i
robusthetskontrollen.

## 4c. Förfining av dataanvändningen – ablationsstudie

Nya data: realekonomi och kärninflation ur Riksbankens prognos-API (KPIF exkl. energi, BNP-tillväxt,
BNP-gap, arbetslöshet; utfall med publiceringsfördröjning + Riksbankens realtidsprognoser för dem) och
kronans växelkurs (KIX). Nya metodval: PCA före det neurala nätet och tidsviktning av träningsdata.

Varje förändring prövades för sig med samma purged walk-forward-validering (R² out-of-sample, ensemble):

| Konfiguration | rörlig 6m | rörlig 12m | styr 6m | styr 12m | KPIF 12m | premie 2år | **snitt** |
|---|---|---|---|---|---|---|---|
| A: ursprungliga features | 0,35 | 0,26 | 0,39 | 0,32 | 0,19 | 0,00 | 0,251 |
| B: + realekonomi/kärninfl./KIX | 0,27 | 0,18 | 0,17 | 0,12 | 0,02 | 0,04 | 0,130 |
| C: B + PCA i neuralt nät | 0,24 | 0,26 | 0,17 | 0,27 | 0,27 | 0,04 | 0,204 |
| **D: C + tidsvikt (halveringstid 10 år)** | 0,32 | **0,36** | 0,35 | **0,38** | 0,21 | 0,00 | **0,268** |
| E: C + tidsvikt 5 år | 0,27 | 0,33 | 0,26 | 0,32 | 0,07 | 0,04 | 0,216 |
| F: A + PCA + tidsvikt 10 år | 0,28 | 0,25 | 0,25 | 0,24 | 0,12 | 0,00 | 0,190 |
| G: F + bara kärninflation | 0,32 | 0,30 | 0,29 | 0,22 | 0,11 | 0,08 | 0,220 |

Slutsatser:
* **Mer data är inte automatiskt bättre.** Fler makroserier utan tidsviktning (B) försämrade alla mål – med
  ~360 månader blir extra brusiga variabler dyra.
* **Tidsviktning (10 år) är den viktigaste förbättringen**: modellerna lär mer av de senaste
  räntecyklerna. D valdes (bäst i snitt och på de viktigaste målen, rörlig ränta och styrränta 12 mån).
* Skillnaderna mellan konfigurationerna ligger nära bruset, och att välja den bästa av sju ger en viss
  optimism. Valet bör omprövas när mer data finns.
* **Tidsviktningen gäller inte den långsiktiga bindningspremien** som marknadens implicita bana bygger på –
  den skattas oviktat över 1996–, eftersom kortare perioder domineras av enstaka episoder (2014– innehåller
  bara en höjningscykel och ger en missvisande positiv premie).
* Utfallsdata för BNP/arbetslöshet är *reviderade* siffror, inte de som fanns i realtid → viss optimism i
  valideringen för de variablerna.

## 4d. Riskbedömning med sannolikheter för ränteutvecklingen

Tidigare viktades alla simulerade scenarier lika. Nu:

1. **Sannolikhetsmodell.** ML-klassificerare (Random Forest + regulariserad logistisk regression) skattar
   sannolikheten för fem utfall för rörlig bolåneränta om 12 och 36 mån: kraftig nedgång (< −1 pe),
   nedgång (−1 till −0,25), oförändrad (±0,25), uppgång (0,25–1), kraftig uppgång (> 1 pe).
   Validering med Brier-poäng på purged walk-forward:
   * 12 mån: Brier 0,84 mot 0,91 för historisk basfrekvens → **skill score 0,08** (påvisad träffsäkerhet).
   * 36 mån: skill score ≈ 0 → ingen påvisad träffsäkerhet.
2. **Skicklighetsviktning.** Andelen ML i sannolikheten väljs på out-of-sample-data (12 mån: 80 %,
   36 mån: 20 %). Resten kommer från simuleringens egen fördelning (centrerad på huvudbanan) – inte från
   historisk basfrekvens, som speglar 30 år av fallande räntor.
3. **Omviktning av scenarier** (raking): varje simulerad ränteutveckling får en vikt så att andelen scenarier
   i varje utfallsklass, på både 12 och 36 mån, matchar sannolikheterna. Effektivt antal scenarier efter
   omviktning redovisas (≈ 1 900 av 3 000, dvs. ingen extrem omviktning).
4. **Sannolikhetsviktade riskmått:** förväntad kostnad, kvantiler, CVaR, sannolikhet att överskrida budget.
   Nya mått: **scenariomatris** (snittränta per strategi i varje utfall, med sannolikhet),
   **förväntad ånger** (merkostnad mot det val som i efterhand var bäst) och **P(billigare än rörligt)**.

## 4e. Vad som fortfarande saknas

| Saknas | Betydelse | Hur det kan läggas till |
|---|---|---|
| Din belåningsgrad (bostadens värde) | Banker ger lägre ränta under 50–60 % belåning; amorteringskrav | `bostadsvarde` i profilen + räntetrappa per belåningsgrad |
| Din inkomst / kvar-att-leva-på | Betalningsrisk: klarar du 6–7 % ränta? | Inkomst och utgifter i profilen → sannolikhet för likviditetsproblem |
| Bankernas historiska listräntor och rabatter | Rabattens utveckling över tid modelleras nu med antaganden | Handelsbankens API ger list- och snitträntor per månad – kan samlas in löpande |
| Hushållens/företagens inflationsförväntningar (Prospera) | Ledande indikator för inflation och styrränta | Riksbankens/Kantar Sifos publicerade serier |
| SWESTR-terminer | Marknadens styrränteförväntningar dagligen | Används nu kvartalsvis via Riksbankens marknadsprisserie |
| Realtidsversioner av BNP/arbetslöshet | Undviker revideringsoptimism | Riksbankens prognosomgångar innehåller dem från 2015 |

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

## 9. Timing av bundet mot rörligt och alternativkostnad

### Varför de generella modellerna inte räckte
Med ~100 features och ~360 månader överanpassar de generella ML-modellerna. Bindningspremien
(snitt rörligt under bindningstiden − bunden ränta idag) gick inte att förutsäga (R² ≤ 0), och i
beslutsbacktestet var ML-timingen sämre än att alltid förnya 1 år.

### Nya timingfeatures (realtid, utan framtidsinformation)
* **Ex ante-bindningspremie**: förväntad snittränta på rörligt under bindningstiden enligt
  marknadens respektive Riksbankens styrräntebana (via genomslaget 0,93) minus dagens bundna ränta.
* **Bankens marginal på bundet** mot bostadsobligationer och dess avvikelse från de senaste 12 månaderna.
* **Eftersläpning**: obligationsräntans rörelse som ännu inte slagit igenom i bolåneräntan.
* **Räntecykel**: månader sedan senaste styrränteändring, dess storlek och riktning.

### Snåla modeller (bolan/timing.py)
Ridge, grunt Gradient Boosting och Random Forest med stora löv på 6–10 teoristyrda variabler.

| Mål | R² OOS | Rätt riktning | Kommentar |
|---|---|---|---|
| Bunden ränta 2/3/5 år, nästa månad | 0,14–0,16 | 58–63 % | Verklig, måttlig förmåga |
| Bunden ränta, 3 mån | 0,03–0,08 | 57–59 % | Svag |
| Bindningspremie 1–5 år | ≤ 0,05 | – | Ingen påvisad förmåga – används inte |

### Beslutsbacktest 2008– (walk-forward, faktiska SCB-räntor)

| Regel | Snittränta | 2008–2014 | 2015–2021 | 2022– | Största 12m-ökning |
|---|---|---|---|---|---|
| Perfekt framförhållning (tak) | 2,12 % | 2,97 | 1,50 | 1,78 | 1,16 |
| **Marknadsregel (ex ante-premie > 0)** | **2,32 %** | 3,48 | 1,62 | **1,63** | **0,11** |
| Förnya 1 år (bästa statiska) | 2,50 % | 3,12 | 1,51 | 3,03 | 2,66 |
| Rörligt | 2,60 % | 3,13 | 1,55 | 3,36 | 2,60 |
| ML-timing (premiemodell) | 2,69 % | 3,44 | 1,67 | 3,09 | 2,66 |
| Marknadsregel + vänta-signal | 2,81 % | 3,54 | 1,83 | 3,18 | 2,09 |

Tolkning:
* Marknadsregeln är den enda regel som slår statiska strategier (−0,18 pe/år, stabilt för trösklar
  0–0,25 pe) och dämpar räntechocker kraftigt. **Men vinsten kommer nästan helt från ett beslut**
  (5 år i januari 2022); 2008–2021 förlorade den. Med 8 beslut på 18 år är den vägledande, inte bevisad.
* Mekanismen är rimlig: bundna räntor släpar efter marknadens förväntningar. När marknaden prisar in
  höjningar som bolåneräntorna ännu inte speglar är bunden ränta "billig".
* "Vänta en månad om den bundna räntan väntas sjunka" försämrade resultatet – prognosen redovisas som
  information, inte som regel.

### Alternativkostnad
* **Försäkringspremie**: förväntad merkostnad (nuvärde) mot rörligt hos samma bank.
* **Brytkostnad (flexibilitet)**: förväntad ränteskillnadsersättning om lånet måste lösas i förtid –
  `sannolikhet_losen_ar` per år; jämförelseränta = marknadsränta för återstående tid − bankens marginal
  mot bostadsobligationer + `rse_paslag`. Ingår i kostnaden och poängen.
* **Kapitalkostnad**: alla kostnader diskonteras med `alternativ_avkastning`, och skillnaden mot rörligt
  redovisas som förmögenhet vid horisonten om pengarna i stället placerats.
* **Väntekostnad**: förväntad vinst av att ligga rörligt 1–3 månader och sedan binda =
  ränteskillnad under väntan − bindningstid × förväntad förändring i bunden ränta, med sannolikhet
  från prognosens out-of-sample-fel.
