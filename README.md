# Bandit statt A/B-Test: was kostet das Lernen beim Zuteilen? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-bandit-vs-ab-demo.streamlit.app/)** -->

Interaktive Demo zur **Inferenz nach adaptiver Zuteilung**. **Neuntes Stück der Konzepte-Linie „Experimente und
Wirkungsmessung“** im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine
Learning). Das Vehikel ist dasselbe wie in den Vorgängern (Standort-Fenster mit täglicher Nachbestellung, Servicegrad Typ 2),
mit drei Regeln statt zwei. [bandit-demo](https://github.com/sebastian-hanisch/bandit-demo) fragt, wie viel ein Bandit
verliert; hier geht es um die Frage danach: was gilt der Test noch, wenn die Zuteilung von den Ergebnissen abhängt?

## Kernfrage

Ein Bandit (Thompson-Sampling) verschiebt Standort-Fenster zur besser aussehenden Regel und verliert weniger als ein
gleichverteilter Test. Was kostet das die Auswertung (Fehlalarm, Teststärke, Schätzung), und was davon lässt sich reparieren?

## Modell und Methodik

- **Vehikel:** je Sicherheitsfaktor ein Pool von 10 000 Standort-Fenstern (`precomputed_pools.json`, dieselben Pools wie
  [ab-test-demo](https://github.com/sebastian-hanisch/ab-test-demo)). Drei Regeln: A (z = 1,0, bestehend), C (z einstellbar),
  B in der Mitte. Wahrheit = Poolmittel.
- **Gemeinsame Zufallszahlen:** jedes Fenster hat für jede Regel ein festes mögliches Ergebnis; die Strategie bestimmt nur, welches
  sie sieht (`bva_bandit.potential_outcomes`). Aller Zufall aus SplitMix64.
- **Runden:** die Fenster treffen in Runden zu *m* Fenstern ein; die Zuteilung einer Runde sieht nur frühere Runden.
- **Vier Strategien** (`bva_bandit.py`): *gleichverteilt*, *A/B/n* (30 % Testphase, danach alles auf das beste Mittel), *Bandit*
  (Thompson-Sampling im Normalmodell, Anlauf gleichverteilt bis jede Regel fünf Fenster hat) und *Bandit mit Boden* (10 % der
  Fenster gleichverteilt).
- **Auswertung** (`bva_evaluation.py`): Welch-Test C gegen A auf allen gesammelten Fenstern (naiv; bei A/B/n nur die Testphase);
  Welch-Test nur auf den gleichverteilt zugeteilten Fenstern; gewichteter Test mit den protokollierten Zuteilungs-
  wahrscheinlichkeiten (normiert, Hájek; Standardfehler aus der Linearisierung; Gewichte bei 10⁻³ gekappt).
- **Trend:** optional eine gemeinsame Verschiebung aller Regeln über die Dauer des Experiments (Saison); bei 1 gekappt.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Sweeps in `precomputed_sweeps.json`, je Zelle 3 000 Experimente; Fehler der Raten
höchstens 0,9 Prozentpunkte). 600 Fenster, 10 je Runde, wenn nicht anders genannt. Verlust = Abstand des mittleren Servicegrads
im Experiment zur besten Regel.

| Frage | Befund |
|---|---|
| Was verliert der Bandit? | Bei Faktor 1,4 (C: 96,35 %) im Mittel **0,21** Prozentpunkte Servicegrad; A/B/n **0,53**, gleichverteilt **1,68**, Bandit mit Boden **0,34**. **90,1 %** der Fenster gehen auf C (mit Boden 85,1 %, A/B/n 78,7 %). |
| Wie oft wählt A/B/n die beste Regel? | Bei Faktor 1,4 in **98,2 %**; bei Faktor 1,2 in **75,0 %**. |
| Fehlalarm ohne Unterschied (A/A/A)? | Gleichverteilt **4,5 %**, A/B/n **4,8 %**, Bandit naiv **7,8 %**, Bandit mit Boden naiv **4,7 %**. Der Standardfehler des Bandit-Tests ist nur **0,58-fach** der tatsächlichen Streuung (mit Boden **0,84**). |
| Wird die Wirkung richtig geschätzt? | Bei wahren **3,49 Punkten**: gleichverteilt **3,50 ± 0,45**, A/B/n **3,49 ± 0,81**, Bandit **3,90 ± 1,42** (zu hoch, **3-fache** Streuung), mit Boden **3,56 ± 0,91**; gewichtet (mit Boden) **3,48 ± 1,10**. |
| Wie steht es um die Teststärke bei klarem Unterschied (1,4)? | Gleichverteilt **100,0 %**, A/B/n **99,1 %**, Bandit **97,0 %**, mit Boden **99,9 %**. Nur die gleichverteilten Fenster allein: **80,8 %**; gewichtet: **95,0 %**. |
| Und bei kleinem Unterschied (1,2, wahr 1,96 Punkte)? | Gleichverteilt **98,6 %**, A/B/n **61,9 %**, Bandit **68,0 %** (Schätzung **2,36**, 20 % zu hoch), mit Boden **80,6 %**; gleichverteilte Fenster allein **31,0 %**, gewichtet **53,7 %**. Verlust: gleichverteilt **0,85**, Bandit **0,26** Punkte. |
| Hilft mehr Zeit? | Teststärke beim kleinen Unterschied (gleichverteilt / A/B/n / Bandit / mit Boden) bei 150 Fenstern **51,9 / 18,4 / 22,3 / 28,0 %**, bei 300 **81,6 / 34,1 / 42,3 / 52,0 %**, bei 600 **98,6 / 61,9 / 68,0 / 80,6 %**, bei 900 **99,9 / 77,5 / 84,0 / 92,6 %**. |
| Was macht ein gemeinsamer Trend? | Ohne jeden Unterschied: Fehlalarm des naiven Tests nach dem Bandit **7,8 %** ohne Trend, **15,7 / 28,7 / 46,4 / 59,1 %** bei 1 / 2 / 3 / 4 Punkten Trend; mit Boden **4,7 → 10,9 / 21,3 / 32,0 / 46,2 %**; gleichverteilt bleibt bei **4,5 bis 5,2 %**. |
| Was repariert das? | Test auf den gleichverteilten Fenstern **4,6 bis 5,6 %** bei allen Trends, gewichtet (mit Boden) **3,3 bis 4,8 %**. **Ohne Boden** bleibt auch die Gewichtung ungültig (bei 3 Punkten Trend **17,9 %**). |
| Trend bei echtem Unterschied (1,4, Trend 3 Punkte)? | Bandit schätzt **5,15** statt 3,49, mit Boden **4,06**, gleichverteilt **3,41**; gleichverteilte Fenster allein **3,43**, gewichtet **3,45**. |
| Rundengröße? | Fehlalarm des naiven Bandit-Tests ohne Unterschied bei 5 / 10 / 25 / 50 Fenstern je Runde **7,6 / 7,8 / 8,2 / 5,7 %**; Verlust bei Faktor 1,4: **0,21** (5 je Runde) gegen **0,25** (50 je Runde). |

## Befunde und Korrekturen gegenüber dem Plan

- **Der Schaden ist kleiner und anders als in der Vorab-Messreihe.** Auf dem Terminal-Gate (Ja/Nein-Ergebnis „pünktlich“,
  Beta-Thompson) fanden sich 12 bis 13 % Fehlalarm und eine Teststärke von 0,22 statt 0,87. Hier, mit dem stetigen Servicegrad
  und Normal-Thompson, ist der Fehlalarm ohne Trend **7,8 %** und die Teststärke bei klarem Unterschied **97,0 %**. Der Schaden
  zeigt sich anders: zu hohe Schätzung, dreifache Streuung, Einbruch der Teststärke beim kleinen Unterschied und Kippen bei Trend.
- **„Regret ja, Inferenz nein“ stimmt nur halb.** Der Bandit gewinnt den Verlust deutlich, der naive Test bleibt bei klarem
  Unterschied brauchbar. Verloren gehen die **Schätzung** (zu hoch, weil Regeln, die früh gut aussehen, mehr Fenster bekommen) und
  die **Gültigkeit bei Drift**: schon **1 Punkt** gemeinsamer Trend verdoppelt den Fehlalarm.
- **Boden und Gewichtung gehören zusammen.** Die Gewichtung mit Zuteilungswahrscheinlichkeiten braucht, dass jede Regel
  Wahrscheinlichkeit über null behält. Ohne Boden meldet der gewichtete Test bei 3 Punkten Trend **17,9 %** Fehlalarm; mit
  Boden **3,6 %**. Der unnormierte Horvitz-Thompson-Schätzer ist unverzerrt, streut aber bei kleinen Wahrscheinlichkeiten zu
  stark; die Demo nutzt die normierte Fassung.
- **Teststärke und Verlust konkurrieren um dieselben Fenster.** A/B/n verliert mehr als der Bandit und ist beim kleinen
  Unterschied trotzdem nicht besser (61,9 gegen 68,0 %): die Testphase ist kurz. Der Bandit mit Boden ist beim kleinen
  Unterschied sowohl bei der Teststärke (80,6 gegen 61,9 %) als auch beim Verlust (0,31 gegen 0,36 Punkte) vor A/B/n.
- **Die Rundengröße ändert wenig.** Größere Runden machen den Bandit etwas vorsichtiger (Fehlalarm 5,7 % bei 50 je Runde), kosten
  aber kaum Verlust.
- **Default-Seed 172** zeigt ein Experiment mit typischer Lage (Bandit schätzt 4,12 statt 3,49, mit Boden 3,83, gewichtet 3,57);
  wie oft das so ausgeht, zeigt die Wiederholung darunter.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Ziel ist das Ergebnis während des Experiments** | Soll das Experiment vor allem entscheiden, welche Regel an weitere Standorte geht, zählt die Inferenz, nicht der Verlust. | (dieses Stück) |
| **Die Gesamtlage ist stabil** | Ein gemeinsamer Trend macht den naiven Test nach adaptiver Zuteilung unzuverlässig. | (dieses Stück) |
| **Die Regeln beeinflussen sich nicht** | Teilen Standorte Kundschaft oder Lager, misst der Vergleich etwas anderes. | Interferenz und Switchback |
| **Ergebnisse liegen rechtzeitig vor** | Ein Fenster dauert vier Wochen; Runden verzögern das Lernen. | (dieses Stück) |
| **Ein Ergebnis, ein Test** | Mehrere Kennzahlen oder neue Regeln vervielfachen die Tests. | Viele Kennzahlen: Mehrfachtests und Gewinnerfluch |
| **Die Zuteilung hängt nur vom Verlauf ab** | Hängt sie von Merkmalen des Standorts ab, braucht die Auswertung Logdaten und Wahrscheinlichkeiten. | Auswertung aus Logdaten (Off-Policy) |

Die Zahlen gelten für dieses erfundene Vehikel (drei Regeln, ein Servicegrad, Normal-Thompson); der Trend ist eine
Modellannahme, kein Betriebsbefund.

## Bezug zu OR

Jedes Pilotprogramm teilt Regeln auf Standorte auf und möchte unterwegs nicht mehr verlieren als nötig. Ein Bandit löst das
Verlustproblem; wer danach „ist die neue Regel besser?“ beantworten muss, sollte die Zuteilung so protokollieren, dass der Test
gültig bleibt (Zuteilungswahrscheinlichkeit mitschreiben, einen Teil der Fenster gleichverteilt lassen).

## Verwandte Demos im Portfolio

- [ab-test-demo](https://github.com/sebastian-hanisch/ab-test-demo), [test-power-demo](https://github.com/sebastian-hanisch/test-power-demo):
  Vehikel, Welch-Test und Teststärke bei fester Zuteilung.
- [sequential-test-demo](https://github.com/sebastian-hanisch/sequential-test-demo): dieselbe Gefahr von der anderen Seite (der
  Zeitpunkt des Stoppens hängt von den Ergebnissen ab).
- [bandit-demo](https://github.com/sebastian-hanisch/bandit-demo): Regret und Strategien des Bandits.

## Verifikation

- **Strategien** (`tests/test_bandit.py`): von Hand an Mini-Instanzen (Regelfaktoren, A/B/n sendet nach der Testphase alles auf
  das beste Mittel, Anlauf gleichverteilt); die Zuteilung einer Runde hängt nur von früheren Runden ab (andere mögliche
  Ergebnisse ab Fenster 50 ändern die Fenster davor nicht); Boden = 1 liefert gleichverteilte Zuteilung mit Wahrscheinlichkeit 1/3;
  Reproduzierbarkeit aus dem Seed.
- **Orakel** (`tests/test_oracle_bandit_vs_ab.py`): Zuteilungswahrscheinlichkeit des Thompson-Sampling gegen die geschlossene
  Formel für zwei Regeln und gegen `scipy.integrate.quad` für drei; Welch aus Momenten gegen `scipy.stats.ttest_ind`; gewichteter
  Schätzer bei bekannten ungleichen Wahrscheinlichkeiten konsistent und im A/A auf 5 %; naiver Test nach Thompson auf Normaldaten
  anti-konservativ, Test auf gleichverteilten Fenstern bei 5 %.
- **Vehikel:** Mini-Instanz, Zufall, Pools reproduzierbar (`generate_precomputed.py` erzeugt dieselbe Datei wie im ab-test-demo).
- **Befunde:** jede Zahl dieses READMEs in `tests/test_claims.py`; zwei Zellen werden live nachgerechnet.
- **Presets, Oberfläche, Footer:** vollständig, Permalink eingerastet, AppTest-Rauchtest, Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: ein Experiment, tausend Wiederholungen, Trend, Horizont, Grenzen |
| `bva_constants.py` | Vehikel-Konstanten, Regler-Grenzen, Strategien, Presets |
| `bva_rng.py`, `bva_vehicle.py`, `bva_formulas.py`, `bva_stats.py` | Zufall, Standort-Fenster, exakter Servicegrad, Welch-Test |
| `bva_bandit.py` | mögliche Ergebnisse, die vier Strategien, Zuteilungswahrscheinlichkeit |
| `bva_evaluation.py` | Tests (naiv, gleichverteilte Fenster, gewichtet), Zusammenfassungen, ein Experiment |
| `bva_presets.py`, `bva_visualization.py` | Permalink, Presets; Plotly-Abbildungen (Achsen gesperrt) |
| `generate_precomputed.py`, `precomputed_pools.json`, `precomputed_sweeps.json` | Pools und Sweeps (35 Zellen à 3 000 Experimente) |
| `tests/` | Strategien, Orakel, Vehikel, Presets, Befunde, AppTest, Footer |

## Bewusst nicht umgesetzt

- **ε-gierig, UCB, Beta-Thompson, kontextuelle Bandits:** nicht gebaut (Strategien im bandit-demo).
- **Nicht-stationäre Bandits** (Abzinsung, gleitende Fenster): nicht gebaut.
- **Frühes Stoppen** des Experiments: nicht gebaut (siehe sequential-test-demo).
- **Vorab-Wissen und Varianzreduktion** im Bandit (CUPED): nicht gebaut.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy (Verteilungsfunktionen).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
