# Verteidigungsvortrag — Ablaufplan

Grundlage: `main.pdf`. Der Vortrag ist **20 Minuten** lang: 46 Folien im
Hauptteil (inklusive sieben Abschnittstrenner) plus 25 Zusatzfolien für die
Diskussion.

| Datei | Inhalt |
|---|---|
| `vortrag.tex` | die Folien; `vortrag_stil.tex` trägt alles Sichtbare |
| `vortrag.pdf` | Präsentationsfassung — `make vortrag` |
| `vortrag_notizen.pdf` | Folie links, Sprechnotiz rechts — `make vortrag-notizen` |
| — | `make vortrag-pruefen` nennt jede Folie, die zu voll ist |

## Das Entwurfsprinzip

**Die Folien tragen Stichpunkte, die `\note{}`-Blöcke den Sprechtext.** Wer
nur die Folien liest, bekommt das Gerüst; die Argumente stehen in den Notizen
und die Belege auf den Zusatzfolien. Das ist Absicht: Eine Folie, die man
vorlesen kann, hat zu viel Text.

Konkret gilt für jede Folie im Hauptteil:

- **Eine Aussage.** Sagt eine Folie zwei Dinge, sind es zwei Folien.
- **Höchstens ~40 Wörter**, im Schnitt deutlich darunter.
- **Grundschrift 14 pt**, Aufzählungen `\large`, Kernzahlen 62 pt. Kein
  `\scriptsize` und kein `\footnotesize` im Hauptteil — die einzige Ausnahme
  ist `\fuss`, und `\fuss` trägt nie ein Argument, nur eine Einschränkung.
- **Keine zwei Textspalten.** Zwei Spalten nur, wenn eine davon ein Bild ist.
- **Tabellen nur, wenn der Vergleich die Aussage IST** — dann höchstens vier
  Zeilen und vier Spalten.

Es gibt genau sechs erlaubte Bauformen, alle in `vortrag_stil.tex` definiert:
`\botschaft` (ein Satz, ganze Folie), `\grosszahl` / `\zweizahl` /
`\dreizahl` / `\sprungzahl` (die Zahl ist der Inhalt), `\schlagworte` (Begriffe
ohne Erklärung), `\bild` / `\bildgross` (Abbildung mit Höhendeckel), `\fuss`
(die eine Einschränkung) und `\trenner` (Abschnitt).

**Das Höhenbudget ist an der Klasse gemessen, nicht geschätzt** — die Rechnung
steht als Kommentarblock in `vortrag_stil.tex`. Kurzfassung: mit Untertitel
passen vier Aufzählungspunkte oder drei plus `\fuss`, ohne Untertitel fünf
oder vier plus `\fuss`. Wer mehr braucht, nimmt zwei Folien und nicht eine
kleinere Schrift. `make vortrag-pruefen` meldet jeden Bruch mit Folientitel
und Zeilennummer.

**Alle Messzahlen kommen aus `kapitel/zahlen.tex`** — derselben Datei, die
`main.tex` einliest und die `src/report_neu.py` schreibt (170 Makros, 288
Vorkommen). Die Folien können deshalb nicht von der Arbeit abweichen. Keine
Zahl ist abgeschrieben.

---

## Der rote Faden

> Für diese Detektionsaufgabe gibt es keine Trainingsdaten — also erzeugt die
> Arbeit sie. Und weil dann die Pipeline *der Datensatz* ist, muss jeder ihrer
> 43 Parameterwerte begründet sein statt nur eingestellt. Genau diese
> Begründungspflicht macht sichtbar, welche Begründung noch nie geprüft wurde
> — und die wird geprüft.

Drei Sätze sollen hängen bleiben. Jeder hat seine eigene, textfreie
`\botschaft`-Folie:

1. **Die Pipeline *ist* der Datensatz** (Seite 7) → daraus die
   Begründungspflicht (Seite 8).
2. **Nicht die Herkunft der eingefügten Pixel entscheidet, sondern ihre
   Statistik — und das gilt auch für alles, was davor liegt** (Seite 37). Das
   ist der übertragbare Beitrag der Arbeit.
3. **Bei programmatisch erzeugten Labels wird die Labelkonvention selbst zu
   einer messbaren Größe** (Seite 41).

---

## Zeitbudget (exakt 20:00)

Seitenzahl = PDF-Seite. Abschnittstrenner verbrauchen kein Budget — sie werden
im Übergang mitgesprochen.

| Seite | Folie | Zeit | kum. |
|---:|---|---:|---:|
| 1 | Titel | 0:20 | 0:20 |
| 2 | *Teil 1 — Das Problem* | — | |
| 3 | Zu detektieren: die Stirnflächen | 0:30 | 0:50 |
| 4 | Ein unangenehmer Spezialfall | 0:30 | 1:20 |
| 5 | Der Engpass ist nicht das Netz — **0** | 0:25 | 1:45 |
| 6 | Verfügbar war das — **9 / 7 / 2** | 0:30 | 2:15 |
| 7 | **Die Pipeline ist der Datensatz** | 0:25 | 2:40 |
| 8 | Also: Begründungspflicht | 0:25 | 3:05 |
| 9 | Die Bilanz — **14 von 43 sind Annahmen** | 0:25 | 3:30 |
| 10 | *Teil 2 — Was Augmentierung leistet* | — | |
| 11 | Statt des empirischen Risikos | 0:30 | 4:00 |
| 12 | $\tau_t$ ist der kritische Teil | 0:30 | 4:30 |
| 13 | Was hier *ein* $t$ ist — **6000 Ziehungen** | 0:30 | 5:00 |
| 14 | Sechs Wege, sich zu verraten | 0:30 | 5:30 |
| 15 | **S1–S6 gelten auch für die Verdecker** | 0:30 | 6:00 |
| 16 | Sechs Anforderungen an die Pipeline | 0:25 | 6:25 |
| 17 | *Teil 3 — Die Pipeline* | — | |
| 18 | Aus 16 Aufnahmen wird der ganze Datensatz | 0:30 | 6:55 |
| 19 | Stufe für Stufe am selben Bild | 0:30 | 7:25 |
| 20 | Kein Generator darf Labels blind vergeben | 0:35 | 8:00 |
| 21 | Der Nachweis ist nicht der optische Eindruck | 0:35 | 8:35 |
| 22 | Trainingsaufbau | 0:30 | 9:05 |
| 23 | *Teil 4 — Das Ergebnis* | — | |
| 24 | **0,947 / 0,969** mAP@0,5 | 0:35 | 9:40 |
| 25 | Aufgabe gelernt — oder Artefakte? | 0:35 | 10:15 |
| 26 | Eine Vorhersage aus Kapitel 3 | 0:30 | 10:45 |
| 27 | Und so sieht sie gemessen aus | 0:30 | 11:15 |
| 28 | Der größte Einzeleffekt — **+63,3 %-Punkte** | 0:35 | 11:50 |
| 29 | *Teil 5 — Die Prüfung* | — | |
| 30 | **Das ist ein Argument. Keine Messung.** | 0:35 | 12:25 |
| 31 | Drei Bibliotheken, eine Leiter | 0:35 | 13:00 |
| 32 | Dieselbe Szene, drei Bibliotheken | 0:25 | 13:25 |
| 33 | Warum der Versuch *gepaart* ist | 0:35 | 14:00 |
| 34 | Das Ergebnis (die Tabelle) | 0:40 | 14:40 |
| 35 | Der Verlust sitzt dort, wo verdeckt wird | 0:35 | 15:15 |
| 36 | Aber anders als erwartet | 0:35 | 15:50 |
| 37 | **Nicht die Herkunft, sondern die Statistik** | 0:35 | 16:25 |
| 38 | *Teil 6 — Das Messinstrument* | — | |
| 39 | Ein Wert, der nicht passt | 0:35 | 17:00 |
| 40 | Am Modell nichts geändert | 0:35 | 17:35 |
| 41 | **Die Labelkonvention ist messbar** | 0:20 | 17:55 |
| 42 | *Teil 7 — Bilanz* | — | |
| 43 | Was ist belegt — **1 von 43** | 0:35 | 18:30 |
| 44 | Grenzen | 0:35 | 19:05 |
| 45 | Nächste Schritte | 0:20 | 19:25 |
| 46 | Drei Einsichten | 0:35 | 20:00 |
| 47 | Vielen Dank | — | |

**46 Folien in 20 Minuten heißt rund 30 Sekunden pro Folie.** Das ist für
textarme Folien richtig, erzwingt aber flüssiges Blättern. Vier Folien liegen
bei 20 Sekunden (1, 41, 45 und die Begründungspflicht) — die als reine
Durchlauffolien einüben.

**Wenn die Zeit reißt**, in dieser Reihenfolge kürzen: Seite 19 (Stufenbilder),
26 (Vorhersage), 21 (Kalibrierung), 45 (nächste Schritte).
**Nicht kürzen:** 7, 15, 33, 34, 36, 37, 43. Seite 43 ist der Grund, warum die
Arbeit als redlich gelesen werden kann; 34 und 37 tragen den Beitrag.

**Ein Satz, der zwingend gesprochen werden muss:** auf Seite 28 der Nachsatz
*„und es ist keine Augmentierungsentscheidung"*. Isoliert liest sich die
Zahl +63,3 %-Punkte wie ein Erfolg der Arbeit; sie ist aber das Gegenargument
gegen deren eigenen Aufwand.

---

## Zusatzfolien (Seiten 48–72)

Ein Thema je Folie, damit man beim Springen findet, was man sucht. Im
PDF-Betrachter Seitenzahl eingeben, nicht scrollen.

| Seite | Folie | greift bei der Frage |
|---:|---|---|
| 48 | Z1 Hauptergebnis, vollständig | alle Metriken über alle Splits |
| 49 | Z2 Metriken und Bewertungsprotokoll | Warum mAP@0,5 als Leitgröße? Eigene Implementierung? |
| 50 | Z3 Die sechs Signaturen S1–S6 | Wie genau schließen Sie die Abkürzungen? |
| 51 | Z4 Die sechs Anforderungen A1–A6 | Wie ist die Pipeline begründet? |
| 52 | Z5 Effektive Objektgröße und P2-Kopf | Warum diese Architektur? Warum bricht 640 px ein? |
| 53 | Z6 Räumlicher Block-Bootstrap | Wie belastbar ist ein Bild pro Split? |
| 54 | Z7 Fehleranalyse | Wo genau irrt das Modell? |
| 55 | Z8 Ablation B, vollständig | die ganze Tabelle, alle Metriken |
| 56 | Z9 Ablation B nach Objektgröße | Wo sitzt der Verlust? Ist er konsistent? |
| 57 | Z10 Ablation A (Okklusionsanteil) | Warum nur eine Ablation? |
| 58 | Z11 Die Schranke von Ben-David | theoretische Fundierung |
| 59 | Z12 Die Schranke, mit Werten besetzt | Wie belegen Sie „kein Artefaktlernen"? |
| 60 | Z13 Kalibrierung, vollständig | Warum gerade diese drei Kenngrößen? |
| 61 | Z14 Der systematische Größenversatz | Warum ist mAP@0,75 so niedrig? |
| 62 | Z15 Sensitivität gegenüber dem Versatz | Wie groß ist der Metrikanteil? |
| 63 | Z16 Belegstand der Entscheidungen | Was ist belegt, was begründet? |
| 64 | Z17 Parameterbilanz | Aufschlüsselung der 43 Werte |
| 65 | Z18 Farbangleich (S2), Rauschangleich (S4) | Wie funktioniert die Angleichung? |
| 66 | Z19 Layoutmodell und Maßstabskopplung | Woher kommen die Gitterparameter? |
| 67 | Z20 Trainingszeit-Augmentierung | Warum `degrees=0`? Warum kein mixup? |
| 68 | Z21 Die beiden Trainingsbudgets | Sind die Zahlen vergleichbar? |
| 69 | Z22 Optimierung und Modellauswahl | Optimierer, Early Stopping, Vortraining |
| 70 | Z23 Vollständiger Auflösungssweep | Wie wurde `imgsz` gewählt? |
| 71 | Z24 Trainingsverlauf | Konvergenz, Nebenbefund zu B2 |
| 72 | Z25 Qualitativergebnis | Wie sehen die Detektionen aus? |

Die Zusatzfolien sind in 7,5 pt gesetzt — sie werden aus der Nähe gelesen,
nicht projiziert vorgetragen. Das ist die einzige Stelle, an der bewusst klein
gesetzt wird.

---

## Vorbereitete Antworten

Alles unten steht in der Arbeit; die Klammer nennt Zusatzfolie oder Abschnitt.

**„Ein Bild pro Split — was sagen diese Zahlen dann überhaupt aus?"**
Das ist die schwerwiegendste methodische Schwäche, und sie steht in der Arbeit
so drin (Abschnitt 8.2). Räumlicher Block-Bootstrap: ±0,011 auf `val`, ±0,012
auf `test`. Unterschiede darunter sind nicht entscheidbar — deshalb ist die
Ablation *nicht* über eine Einzeldifferenz argumentiert, sondern über
**Konsistenz**: dieselbe Richtung über drei Arme, vier Metriken, zwei Splits
und beide Größenklassen. Der B2-Effekt übersteigt das Intervall um mehr als
das Doppelte, der Auflösungseffekt um ein Vielfaches. (Z6, Z9)

**„Warum nur eine Ablation?"**
Rechenzeit — und eine bewusste Wahl. Von den zwei Fragen an der
Okklusionsstufe ist die eine keine Entwurfsfreiheit: *Dass* verdeckt wird, ist
physikalische Randbedingung. Die andere war die **teuerste** Einzelentscheidung
der Pipeline und ausschließlich durch ein Argument gedeckt. Und sie ließ sich
als **gepaarter** Versuch anlegen — byte-identische Labels, keine zweite
Variable. Die Arme zum Verdeckungs*grad* liegen trainiert vor (`runs/A0`,
`A1`, `A2`) und zeigen einen Effekt innerhalb des Bootstrap-Intervalls.
(Seite 33, Z10, Z16)

**„Wie schließen Sie aus, dass das Netz nur die Einfügeartefakte gelernt hat?"**
Ich schließe es nicht aus, ich zeige, dass das Muster nicht auftritt. Bei
reinem Artefaktlernen wäre `sval` nahezu perfekt und die Realbilder
katastrophal; gemessen sind 3,6 bzw. 1,5 %-Punkte Abstand. In die
Ben-David-Schranke eingesetzt: $d+\lambda \geq 0{,}015$ gegen $\geq 0{,}94$ im
Artefaktfall. Auf die Richtung achten — die Ungleichung nagelt die Summe nach
*unten* fest. Es bleibt ein **Indiz**, kein Beweis, und so steht es auch in
der Arbeit. (Seite 25, Z11, Z12)

**„mAP@0,75 von 0,368 auf `test` — das Modell lokalisiert doch schlecht?"**
Nein, es *findet* praktisch jedes Objekt und trifft die Boxgröße systematisch
zu klein. Beweiskette: Auf dem synthetischen Split trifft es die Boxgröße auf
Promille (1,003), auf den Realbildern 10 % darunter, der *Zentrums*versatz
liegt aber im Rauschen — ein Größen-, kein Positionsfehler. Ursache ist die
Labelkonvention: Generatorboxen folgen der $\alpha > 0{,}5$-Kontur, die
Handannotation umschließt großzügiger. Aufweiten um 1,00 px hebt mAP@0,75 von
0,368 auf 0,805, ohne dass sich am Modell etwas ändert. Anforderung A5 ist
damit nur *annähernd* erfüllt — und das Angleichen ist der billigste offene
Gewinn. (Seiten 39–40, Z14, Z15)

**„Warum ist die Architektur nicht abliert?"**
Weil sie die **Konstante** der Experimente ist, nicht ihre Variable. Eine
Ablation der Datenerzeugung soll auf dem Modell laufen, das man auch einsetzen
würde — sonst kompensiert ein gemessener Datengewinn nur die Schwäche eines
unterlegenen Detektors. Die Wahl ist dennoch begründet, nicht geraten: Größe
„s" wegen neun zugrunde liegender Objektinstanzen, P2-Kopf, weil er die
Zuweisungsgrenze $S\sqrt2$ von 11,3 auf 5,7 px senkt — unter alle
auftretenden Objektgrößen. (Seite 22, Z5)

**„Sie haben `imgsz` auf `val` gewählt — ist `test` dann noch unabhängig?"**
`test` ja, `val` nein. Das Kriterium (mAP@0,5:0,95 auf `val`) stand *vor* den
Läufen fest, und der gefundene Wert 1920 wird unverändert auf **beide** Splits
angewandt — eine Wahl je Split wäre Hyperparameterselektion auf dem Testsatz.
`val` wird allerdings doppelt verwendet, für Modellauswahl *und*
Auflösungswahl; es ist damit kein unabhängiger Bewertungssplit mehr. Das steht
in Abschnitt 8.2, und die belastbaren Zahlen sind die `test`-Zahlen. (Z23)

**„B1 ist so gut wie B0 — ist das nicht einfach ein Nullergebnis?"**
Es ist eine **Nicht-Ablehnung**, kein Nachweis der Gleichheit, und ich nenne
es auch so. Bei einem Lauf je Arm und widersprüchlicher Richtung über die
Splits ist die Differenz nicht entscheidbar. Der schwache Hinweis liegt im
Rückruf, der auf beiden Splits leicht unter B0 liegt. Belastbar ist die andere
Hälfte des Befunds: B2 verliert 4,7 %-Punkte mAP@0,5, und das übersteigt das
Bootstrap-Intervall deutlich. (Seite 34, Z8)

**„Wäre ein besseres prozedurales Modell nicht doch schlechter als B0?"**
Möglich — B1 ist *ein* sorgfältig kalibriertes Modell, nicht das bestmögliche.
Kalibriert sind fünf **skalare** Kennwerte; nicht getroffen ist die räumliche
Korrelationsstruktur des Restsignals: Das Korn von B1 ist weißes Rauschen, das
von B0 enthält Kratzer, Rost und JPEG-Blockstruktur. Jede Aussage zugunsten
der photographischen Bibliothek gilt gegen *diese* Modellklasse, nicht gegen
prozedurale Okkluder schlechthin. (Seite 31)

**„Welche Ihrer 14 Annahmen würde Sie am meisten sorgen?"**
Die Schaftrichtung. Sie ist an der Säulenrichtung des Clusters kalibriert,
nicht an der Rohrrichtung des Hintergrunds — in 19,1 % der Cluster kreuzt der
Schaft die Hintergrundrohre unter mehr als 60°, betroffen sind 21,6 % aller
Bilder. Weil alle Rohre eines Stapels parallel liegen, ist das dieselbe
Unstimmigkeit, gegen die die enge Streuung von ±11° *innerhalb* eines Clusters
gesetzt ist. Ein Angleich wäre billig, weil die dominante Hintergrundrichtung
gemessen vorliegt. (Abschnitt 8.2)

**„Was ist mit den 9 ‚gemischten' Parameterwerten — welche Belegart haben die?"**
Halb Messung, halb Annahme, und deshalb rechne ich sie *nicht* zu den
belegten. Beispiel Befund 2: Das Achsverhältnis der Ausschnitte 0,83–1,14 ist
gemessen, die Gleichverteilung über alle neun Ausschnitte ist Annahme. Deshalb
steht in der Bilanz „rund 67 % zurückgeführt — die neun gemischten allerdings
nur zur Hälfte". (Seite 9, Z17)

**„Warum keine Domain Randomization / warum überhaupt Realismus?"**
Die Pipeline liegt zwischen beiden Linien: breite Randomisierung, aber
ausschließlich innerhalb **physikalisch möglicher** Konfigurationen, und
Hintergründe aus der Zieldomäne selbst. Der Grund steht in der Struktur des
Divergenzterms: Er wird über die *Hypothesenklasse* gemessen. Ein Unterschied,
den ein CNN leicht erkennt — eine achsparallele Naht, ein konstanter
Farbversatz — trägt weit stärker bei als einer ohne einfache
Entscheidungsfunktion. Mehr Zufall hilft dagegen nicht. (Seite 14, Z11)

**„Warum trainieren Sie nicht auf den beiden Realbildern nach?"**
Dann gäbe es keinen unberührten Bewertungssplit mehr — und die Zahlen, die die
Arbeit berichtet, wären nicht mehr interpretierbar. `val` steuert bereits die
Modellauswahl; `test` ist der einzige Split, der Training und Modellauswahl
nie berührt hat, und genau darauf ruht die Aussagekraft von Seite 24.

**„Überträgt sich das auf andere Zäune, Rohrdurchmesser, Wetter?"**
Dazu sagen die Ergebnisse nichts. Beide Bewertungsbilder zeigen dieselbe
Objektart bei ähnlichem Wetter, und alle vier Clean Plates sind bei diffusem
Licht aufgenommen — harte Schlagschatten, Gegenlicht, Nässe und Schnee fehlen
vollständig. Die Fehleranalyse weist „Enden ohne sichtbaren hellen Kranz" als
den Fehlertyp aus, der *nicht* auf Objektgröße zurückgeht — das ist genau
diese Lücke. Gegenmaßnahme: breiteres Quellmaterial, Punkt 4 der nächsten
Schritte. (Seite 44, Z7)

---

## Vor dem Termin

```bash
make vortrag && make vortrag-pruefen && make vortrag-notizen
```

- `vortrag.pdf` — 16:9, 72 Seiten, keine überlaufende Folie, keine ungelöste
  Referenz.
- `vortrag_notizen.pdf` — doppelt breite Seiten für den Präsentationsmodus
  (zweiter Bildschirm) oder zum Ausdrucken.
- Die Fußzeile zeigt `Foliennummer / Gesamt`; im Anhang beginnt die Zählung
  neu, damit die Zusatzfolien im Hauptteil nicht mitgezählt werden.
  Abschnittstrenner tragen keine Nummer.

> Hinweis: `PRAESENTATION.md` ist der Vortragsplan einer **früheren** Fassung
> (Quellen unter `kapitel_e2/`, Okklusions*anteil* als einziges Experiment).
> Für die Fassung in `main.pdf` gilt dieses Dokument.
