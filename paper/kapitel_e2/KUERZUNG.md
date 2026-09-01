# Kürzungsplan der Kurzfassung

> **Stand 31.08 — diese Kopie liegt in `kapitel_e2/`** und ist auf die zählfreie
> Fassung nachgeführt: Die Arbeit behandelt nur noch die reine Detektion der
> Profile. Die Seitenzahlen unten sind weiterhin die der Kurzfassung vom 26.08;
> `main_e2` baut aktuell **41 Seiten Text / 62 Seiten gesamt**. Die Kopie in
> `kapitel_kurz/KUERZUNG.md` ist absichtlich unverändert geblieben, weil die
> Kurzfassung den Zählbezug noch führt.

Grundlage: `kapitel_kurz/*.tex`, Stand des Builds vom 26.08. Alle Seitenzahlen in
diesem Dokument sind **gemessen**, nicht geschätzt: Ich habe fünf Kürzungs­
varianten in einer Kopie gebaut (`latexmk`, gleiche Präambel) und die Seite
verglichen, auf der das Literaturverzeichnis beginnt.

## Ist-Zustand

| | Seiten |
|---|---|
| Titelei und Verzeichnisse | 8 (röm.) |
| **Text, Kapitel 1–8** | **45** |
| Literaturverzeichnis | 5 |
| Anhang | 10 |
| PDF gesamt | 68 |

Der Text ist damit 45 Seiten und nicht die 38–41, die `paper/README.md` nennt —
der nachträglich ergänzte Abschnitt zum Größenversatz (7.7) hat vier Seiten
gekostet. Die Kürzungstabelle im README ist entsprechend veraltet.

Inventar: 15 Abbildungen im Text (11 davon selbstgezeichnete TikZ-Schemata,
4 bildbasiert) + 4 im Anhang, 12 Tabellen im Text + 5 im Anhang,
20 abgesetzte Formeln, 72 zitierte Quellen.

## Vorab zu klären: was zählt überhaupt?

Zählt die Prüfungsordnung nur Kapitel 1–8, oder Literatur und Anhang mit? Das
entscheidet die Strategie, nicht nur die Reihenfolge:

* **nur Text** (die übliche Regel, und die, nach der das README rechnet):
  Dann ist *Verschieben in den Anhang* genauso gut wie Löschen, und die
  vollständigen Parametertabellen können bleiben, wo sie sind.
* **alles** — dann bringt Verschieben null, und es muss wirklich weg. In diesem
  Fall ist der Anhang mit 10 Seiten der erste Ort, an dem man nachsieht
  (Tabelle A.4 „Alle 36 Transformationen" allein sind 3 Seiten).

Der Plan unten nimmt den ersten Fall an.

## Die Leiter — gemessene Einsparung

| Stufe | Eingriff | Text | Δ |
|---|---|---|---|
| 0 | Ist-Zustand | 45 | – |
| 1 | Kapitel 2 entdoppeln + 5 Lehrbuchformeln + Abb. 2.1 | 43 | −2 |
| 2 | + 6 Schema-Abbildungen in Kapitel 3 und 5 | **41** | **−4** |
| 3 | + Prosastraffung (5 Blöcke), Kapitel 4 als Abschnitt von 5 | 39 | −6 |
| 4 | + nur ein Experiment (Exp. 3 in den Anhang) | **35** | **−10** |

Zwei Messergebnisse, die man wissen sollte, bevor man Arbeit investiert:

* **Tabellen kürzen bringt nichts.** Tabelle 5.1 (Überblick der Synthesestufen)
  zu streichen — die Maßnahme, die das README mit −0,7 Seiten veranschlagt —
  ergibt gemessen **0 Seiten**. Ebenso die beiden P2-Nebentabellen 7.4 und 7.5.
  Gleitobjekte werden neu gepackt; die freie Fläche verschwindet in den
  Umbrüchen. Finger weg, der Informationsverlust ist unbezahlt.
* **Prosa kürzen bringt fast nichts.** Fünf Absatzblöcke, die ich für
  entbehrlich halte (zusammen rund 700 Wörter), ergeben allein −1 Seite.
  Fließtext ist die dichteste Seite im Dokument; Kürzen lohnt dort nur, wo der
  Absatz auch sachlich weg kann.

Die Seiten liegen bei **Abbildungen und Formeln** (Stufe 1+2: −4 Seiten für
Material, das keine einzige Aussage der Arbeit trägt) und beim **Umfang der
Experimente** (Stufe 4: −4 Seiten).

## Abbildungen

Das Auswahlkriterium, das sich beim Durchgehen ergibt, ist erstaunlich scharf:

> **Abbildungen, die eine Formel illustrieren, sind entbehrlich.
> Abbildungen, die eine Messung zeigen, sind es nicht.**

Alle 4 bildbasierten Abbildungen sind Belege (reale Aufnahme, eigener Datensatz,
eigene Messung) — alle 4 bleiben. Von den 11 TikZ-Schemata illustrieren 8 eine
Gleichung, die zwei Zeilen darüber steht.

| Nr. | Titel | Empfehlung | Begründung |
|---|---|---|---|
| 1.1 | Reale Stapelaufnahme | **behalten** | Ohne sie ist der Gegenstand nicht fassbar; zeigt alle drei Schwierigkeiten auf einmal |
| 2.1 | Vollvernetzt gegen Faltung | **streichen** | Lehrbuchbild. Der Satz „409.600 Verbindungen werden zu 9" trägt dieselbe Aussage vollständig |
| 2.2 | Merkmalspyramide P2–P5 | behalten, **wenn P2 bleibt** | Einzige Stelle, an der die Schrittweiten 8/16/32 sichtbar werden; wird von Kap. 3 und 7 gebraucht. Fällt Exp. 3 weg, fällt sie mit |
| 3.1 | Boxaufblähung unter Rotation | Teil (a) **behalten**, Teil (b) streichen | (a) begründet, warum `degrees`/`shear`/`perspective` auf 0 stehen — eine echte Entwurfsentscheidung. (b) ist der Graph zu drei Zahlen, die in der Bildunterschrift stehen |
| 3.2 | Empfindlichkeit der IoU | **streichen** | Dieselbe Botschaft wie 3.1 in anderer Verpackung. Gleichung + „2 px kosten bei einem 20-px-Objekt 0,18 IoU" genügt |
| 3.3 | Effektive Objektgröße / Zuweisungsgrenze | (a) streichen, (b) behalten | (a) doppelt Tabelle 7.2. (b) und Abb. 7.1 zeigen fast dasselbe — ein Gitter mit Zellzentren im Objekt — also nur eines von beiden: bleibt Exp. 3, dann 7.1 (dort trägt es die Vorhersage); wandert Exp. 3 in den Anhang, dann 3.3(b), weil Gleichung (3.5) es braucht |
| 5.1 | Aufbau der Pipeline | **behalten** | Billig und orientiert über drei Stufen; das einzige Bild, das die Arbeitsteilung Generator/Trainingszeit sichtbar macht |
| 5.2 | Aufbau eines synth. Bildes (6 Teilbilder) | **behalten** | Die teuerste Abbildung und die wertvollste: eigenes Ergebnis, visueller Beleg, dass die Pipeline tut, was Kapitel 5 behauptet |
| 5.3 | Maßstabskopplung | **streichen** | Illustriert einen Satz: „die Stirnfläche hat den Durchmesser des Schafts, an dem sie sitzt" |
| 5.4 | Ovalisierung | **streichen** | Illustriert, dass ein geometrisches Mittel erhalten bleibt — das steht in Gleichung (5.2) selbst |
| 5.5 | Partieller Farbangleich | **streichen** | Gute Argumentation, aber die Kurven sind gezeichnet, nicht gemessen. Das Argument („voller Angleich zerstört den Albedo-Unterschied") überlebt als Satz |
| 5.6 | Layoutmodell | **streichen** | Zwei Panels für ein Modell, das Gleichung (5.5) definiert und Tabelle 5.2 härter belegt, als ein Schema es kann |
| 7.1 | Kandidatenzellen P3/P2 | behalten, **wenn P2 bleibt** | Trägt die Vorhersage, die das Experiment anschließend prüft — die stärkste Stelle der Arbeit |
| 7.2 | P2-Vorteil über der Auflösung | behalten, **wenn P2 bleibt** | Echte Messung, und die Prüfung der Vorhersage |
| 7.3 | Rote und blaue Box am selben Objekt | **behalten** | Echte Messung und Auslöser des ganzen Befunds 7.7 |
| A.1 | Beispielbilder | **behalten** | Zeigt typische Bilder; 5.2 zeigt bewusst ein untypisches |
| A.2 | T2/T3 im Detail | **streichen** | Doppelt Teilbild (d) von 5.2 |
| A.3 | T8/T9 | **streichen** | Doppelt die globale Stufe aus 5.2 |
| A.4 | Qualitativergebnis, zwei Schwellen | **behalten** | Der Vergleich 0,50 gegen 0,40 *ist* das Argument in 7.7 |

Summe: 19 → 12 Abbildungen, gemessen −3 Textseiten und −1 Anhangseite.

## Formeln

20 abgesetzte Formeln. Der Test ist nicht „ist sie richtig", sondern **„trägt
sie eine Entscheidung dieser Arbeit"**. Ein guter Indikator ist, ob später auf
sie verwiesen wird — mit einer Ausnahme in beide Richtungen.

**Streichen — 5 in Kapitel 2, eine in Kapitel 5 (unten):**

| Formel | Grund |
|---|---|
| (2.1) vollvernetzte Schicht | Existiert nur, um mit (2.2) verglichen zu werden. Der Vergleich ist ein Satz |
| Kreuzkorrelation (unnummeriert) | Wiederholt (2.2) mit anderer Notation |
| (2.3) Rekursion des rezeptiven Felds | **Kein einziger Verweis** im ganzen Dokument. Mit ihr geht der Absatz zum effektiven rezeptiven Feld (`luo2016`) weg, in dem auch der Tippfehler „für die jeweil vorangehenden Schichten" steckt |
| Max-Pooling (unnummeriert) | YOLO26 verwendet Stride-2-Faltungen; die Formel wird nie gebraucht |
| Ausgabegröße $H_\text{out}$ | Lehrbuch. Gebraucht wird daraus nur „$s=2$ halbiert", und das steht im Text |

**Behalten in Kapitel 2 (2):** die Faltung (2.2) — ein CNN-Kapitel ohne die
Faltungsgleichung sieht seltsam aus; die Messung oben hat sie mitgestrichen, das
letzte Zehntel Seite ist es wert — und die Translationsäquivarianz (2.4), die
Kapitel 3 für das Argument „Augmentierung bringt alle *anderen* Invarianzen auf
der Datenseite ein" wörtlich braucht.

**Behalten in Kapitel 3 (alle 7).** Das sind die tragenden: (3.1) augmentiertes
Risiko ist der formale Kern des Theorieteils; (3.2) IoU($\theta$) begründet die
abgeschalteten Trainingsparameter; (3.3) IoU-Versatz begründet, wie mAP zu lesen
ist; (3.4) $d_\text{eff}$ hat drei Verweise und ist die Brücke zu Experiment 1;
(3.5) $\mathbb{E}[N]=\pi d^2/(4S^2)$ ist die Vorhersage, die Experiment 3 prüft;
(3.6) Ben-David-Schranke gibt der Dreiteilung (i)/(ii)/(iii) ihr Skelett.

**Behalten in Kapitel 5, außer einer.** (5.4) Rauschschätzung mit Medianfilter
ist der einzige Kandidat: Die zugehörige Abbildung ist bereits per `\iffalse`
stillgelegt, und das Argument „Median verschmiert Kanten weniger" ist ein
Implementierungsdetail. Zwei Zeilen Text statt Formel. — Gegenbeispiel zur
Verweisregel: (5.6) Sichtbarkeit nach Porter/Duff hat **null** Verweise und muss
trotzdem bleiben, denn sie *ist* Stufe T7 und der Beleg für A1 und A5.

Summe: 20 → 14 Formeln, enthalten in den −2 Seiten von Stufe 1.

## Literatur

72 der 77 Einträge sind zitiert. Hier ist wenig zu holen, und das sollte man
wissen, bevor man Zeit investiert: **Zitate zu streichen spart praktisch keine
Seite.** Ein `\parencite` ist ein paar Wörter; das Literaturverzeichnis sind
5 Seiten, und die zählen nach der Annahme oben nicht mit. Selbst 20 Quellen
weniger wären etwa 1,3 Seiten — im Verzeichnis, nicht im Text.

Der Grund, in Kapitel 2 trotzdem aufzuräumen, ist **Ton**: 30 Zitate auf
7 Seiten Standardstoff lassen das Kapitel wie eine Literaturzusammenfassung
aussehen und nicht wie eigene Arbeit. Entbehrlich dort:

* `luo2016` — geht mit der Formel zum rezeptiven Feld
* `zhang2017rethinking` — fünf Zeilen Exkurs über memorierte Zufallslabels.
  Interessant, ändert aber keine Entscheidung dieser Arbeit
* `ren2015` / `liu2016ssd` — eine der beiden genügt für „zweistufig gegen
  einstufig"
* `zeiler2014` / `lecun2015` — eine der beiden genügt für die Merkmalshierarchie
* `micikevicius2018` (Mixed Precision), `loshchilov2017sgdr` (Kosinusabfall) —
  Implementierungsdetails in Kapitel 6, die niemand belegt braucht

Nicht antasten: die Forensik-Quellen in Tabelle 3.1 (`farid2009`, `popescu2005`,
`lukas2006`, `wallace1992jpeg`, `immerkaer1996`, `reinhard2001`) — sie sind der
Beleg für Anforderung A4 und damit für den halben Pipelineentwurf. Ebenso
`cubuk2019`/`cubuk2020`/`zoph2020` und `buslaev2020`: die begründen, was die
Arbeit *nicht* tut, und das ist die häufigste Prüfungsfrage.

**Umgekehrter Befund.** Fünf Einträge sind nirgends zitiert — biblatex lässt sie
stumm weg, also kein Fehler. Zwei davon fehlen aber inhaltlich:

* `nikolenko2021`, die Monographie *Synthetic Data for Deep Learning*, in
  Abschnitt 3.4. Ihr Fehlen fällt in einer Arbeit über rein synthetische
  Trainingsdaten mehr auf als ihre Anwesenheit.
* `trabucco2024` (diffusionsbasierte Augmentierung) im Verfahrensüberblick.
  Ein Satz dazu beantwortet die naheliegende Frage „warum keine generative
  Augmentierung?" — bei einer Arbeit von 2026 wird sie kommen.

Also: je einen Satz ergänzen, `kortylewski2020`, `mueller2021`, `simonyan2015`
aus der `.bib` löschen.

## Prosa und Struktur

Gemessen −2 Seiten in Kombination. Reihenfolge nach Verhältnis von Ertrag zu
Schaden:

1. **Kapitel 2, Unterabschnitt 2.1.2 „Aufbau eines CNNs" ganz streichen**
   (`02_theorie.tex:211–243`). Er ist redundant: ReLU wird in 2.1.2 eingeführt
   (Z. 225–230) und in 2.1.3 gleich wieder (Z. 248–250), vollvernetzte Schichten
   am Ende ebenso zweimal (Z. 239–243 und 256–259). Das ist eine kostenlose
   Kürzung — es geht keine Aussage verloren, nur eine Wiederholung.
2. **Kapitel 3, „Die Gleichung im Einzelnen" bis „…die Dokumentation des
   Datensatzes *ist*"** (`03_augmentierung_grundlagen.tex:22–72`). Die
   Bestandteilliste zu Gleichung (3.1), „Was in dieser Arbeit *ein* $t$ ist" und
   die „letzte Lesart" sind didaktisch gut, aber dreifach dieselbe Einsicht:
   $p(t)$ ist der Datensatz. Einen Absatz behalten, zwei streichen.
3. **Kapitel 3.3.3 „Mehrbildverfahren"** — steht sachlich schon in
   Abschnitt 6.2, wo Mosaic/mixup/cutmix begründet ein- bzw. abgeschaltet
   werden. Auf zwei Sätze eindampfen und dort belassen.
4. **Kapitel 4 als Abschnitt 5.1 statt eigenes Kapitel.** Drei Seiten mit einer
   Seite Kapitelanfangsfläche. Inhaltlich ist es ohnehin die Belegbasis von
   Kapitel 5, und der Anschluss „Befund 1 → Stufe 1 Matting" wird enger.
5. **Kapitel 5.6 „Nachweis der Anforderungen"** — wiederholt A1–A6 in
   Listenform, nachdem jede einzeln begründet wurde. Als Tabellenspalte „erfüllt
   durch" in Tabelle 5.1 unterbringen oder in die Diskussion ziehen, wo
   Abschnitt 8.1 dieselbe Bilanz schon differenzierter zieht.
6. **Kapitel 8, Grenze (6) zur Schaftrichtung** (`08_diskussion.tex:72–84`) —
   13 Zeilen mit vier Zahlen für einen Nebenwiderspruch. Drei Sätze genügen.

Nicht kürzen: Abschnitt 7.7 (Größenversatz) und 8.1 („Was ist belegt, was ist
nur begründet?"). Beide sind das, was die Arbeit von einer Bastelarbeit
unterscheidet — 8.1 nennt ungefragt, dass 27 von 36 Transformationen unbelegt
sind. Das schreibt niemand freiwillig hin, und genau deshalb ist es die
glaubwürdigste Seite im Dokument.

## Nur ein Experiment: Experiment 2 (Verdeckungsanteil)

Ja, das geht, und es kostet weniger, als es klingt — gemessen −4 Seiten.
Empfehlung mit Begründung:

### Behalten: Experiment 2, die Okklusionsablation

1. **Es ist das einzige Experiment zum Thema der Arbeit.** Der Titel sagt
   *Datenaugmentierung*. Experiment 1 variiert einen Inferenzparameter,
   Experiment 3 eine Architektur — beide sagen nichts über Augmentierung und
   wären auf jedem beliebigen Datensatz genauso durchführbar. Nur bei
   Experiment 2 ist die eigene Pipeline die unabhängige Variable. Auf die Frage
   „was haben Sie über Datenaugmentierung empirisch gezeigt?" gibt es sonst
   keine Antwort.
2. **Es hat die beste interne Validität von allen drei.** Gleicher Seed →
   *genau dieselben* 86.617 Instanzen an denselben Positionen in allen drei
   Armen. Der einzige Unterschied ist, wie viel davon verdeckt wird. Das ist
   eine schärfere Kontrolle, als eine Datensatz-Ablation normalerweise hat, und
   das ist ein Argument, das man in der Verteidigung gebrauchen kann.
3. **Der Befund ist kontraintuitiv und damit berichtenswert.** A2 gewinnt in
   allen sechs Metrikspalten, *obwohl* er ein Drittel weniger Labels hat. Wären
   die Labels wertvoller als die Schwierigkeit, müsste A0 gewinnen. Das ist der
   einzige Punkt der Arbeit, an dem eine Entwurfsentscheidung gegen einen
   messbaren Preis gewinnt.
4. **Es rechtfertigt die teuerste Stufe der Pipeline.** Okkluder-Bibliothek,
   Tiefenmodell und exakte Sichtbarkeitsrechnung sind der Hauptaufwand von
   Kapitel 5. Ohne Experiment 2 steht dieser Aufwand unbelegt da.
5. **Es ist das kürzeste.** 1,5 Seiten und eine Tabelle.

Ehrlich dazugesagt: Der Effekt (2,0 Prozentpunkte) liegt innerhalb des
Konfidenzintervalls von ±0,04. Tragend ist allein die Konsistenz — dieselbe
Richtung in 3 Metriken × 2 Splits × 3 Stufen. Das steht schon so im Text und
sollte so stehen bleiben.

### Experiment 1 nicht streichen, sondern zurückstufen

Der Auflösungssweep braucht keinen Experimentstatus. Er liefert den
Betriebspunkt, mit dem *alle* anderen Zahlen der Arbeit gemessen sind — also
gehört er als „Wahl des Betriebspunkts" in Abschnitt 7.1 zum
Bewertungsprotokoll, mit Tabelle 7.2 und drei Sätzen. Kostet fast nichts, und
der größte gemessene Einzeleffekt der Arbeit (47 Prozentpunkte) bleibt erhalten,
inklusive der unbequemen Bilanz in 8.1, dass er *keine*
Augmentierungsentscheidung ist. Diese Selbstrelativierung ist zu gut, um sie
mitzustreichen.

### Experiment 3 (P2) in den Anhang — aber die zwei Einsichten retten

Die vier Seiten 7.5.1–7.5.7 sind der größte Einzelposten. Was aber nicht
verloren gehen darf, sind zwei Aussagen, die ohne den Architekturapparat
funktionieren:

* **Der Betriebspunkt verschiebt sich stärker als die Detektionsgüte** (7.5.6,
  seit dem 31.08 unter diesem Titel). Das F₁-Optimum liegt bei `yolo26s-p2` bei
  c\* = 0,379 gegenüber 0,134 beim Basismodell — eine fast dreifach höhere
  Schwelle, weil der feinere Kopf 4,05-mal so viele Prädiktionspositionen hat.
  Daraus die Betriebsregel: Die Konfidenzschwelle des Basismodells darf nicht
  übernommen werden, sie ist je Modell neu zu kalibrieren. Braucht nur zwei
  Modelle mit verschiedenen Betriebspunkten, keine Herleitung der
  Merkmalspyramide — eine halbe Seite in Kapitel 8, mit Tabelle 7.6 auf zwei
  Zeilen eingekocht.
* **Die Vorhersage $\mathbb{E}[N]=\pi d^2/(4S^2)$ ist eingetroffen.** Zwei Sätze
  in der Diskussion („die Rechnung sagt den Vorteil dort vorher, wo Objekte
  unter zwei Zellen belegen; gemessen 4,6 gegen 0,4 Prozentpunkte"), Details im
  Anhang. Das ist der einzige Punkt, an dem die Arbeit aus der Theorie eine
  quantitative Vorhersage ableitet und sie prüft — als Methode wertvoller als
  das Ergebnis.

Fällt Experiment 3 weg, fällt Abbildung 2.2 (Merkmalspyramide) mit, und
Abschnitt 2.2.2 kann auf einen Absatz schrumpfen.

### Wenn die Betreuung es anders sieht

Sollte der Schwerpunkt ausdrücklich auf der Architektur liegen — die
Kurzfassung ist ja bewusst um P2 herumgebaut —, dann tauscht man: Experiment 3
bleibt in voller Länge, Experiment 2 wird auf eine halbe Seite mit Tabelle 7.3
zurückgeführt. Das ist verteidigbar, aber die Arbeit hat dann kein Experiment
zu ihrem eigenen Titelthema mehr, und die Verdeckungsstufe — der Hauptaufwand
der Pipeline — bleibt unbelegt. Ich halte das für die schwächere Position.

## Nebenbefunde beim Durchgehen

Kein Kürzungsthema, aber beim Lesen aufgefallen:

1. **`02_theorie.tex:10`** — Kapitel 2 beginnt mit „Die drei genannten Prinzipien
   sind keine Implementierungsdetails". Genannt werden sie nie: Der einleitende
   Absatz der Langfassung (`kapitel/02_theorie.tex:8–18`), der lokale rezeptive
   Felder, Weight Sharing und Subsampling einführt, ist beim Kürzen weggefallen.
   Der Leser stolpert im ersten Satz des Kapitels. Ein Satz repariert das.
2. **`02_theorie.tex:104`** — „für die jeweil vorangehenden Schichten": Tippfehler,
   entfällt mit Formel (2.3).
3. **`05_pipeline.tex:379–471`** — 93 Zeilen TikZ in `\iffalse … \fi` (Abbildung
   zur Rauschschätzung). Toter Code; zusammen mit Formel (5.4) löschen.
4. **`06_trainingszeit.tex:82–93`** — Abschnitt 6.3 „Trainingsverlauf" beschreibt
   Lernkurven, zeigt aber keine. `abbildungen/training.png` liegt aufbereitet
   vor, ist aber nicht eingebunden. Entweder Abbildung rein (dann trägt der
   Abschnitt etwas) oder die acht Zeilen in 6.1 auflösen.
5. **13 aufbereitete Abbildungen sind unbenutzt** (`training.png`,
   `ablation.png`, `imgsz_sweep.png`, `pr_kurven.png`, `modelle.png`,
   `matting_check.jpg`, `okkluder_check.png`, `label_check.jpg`, `qual_val.jpg`,
   `qual_test.jpg`, `ablation_beispiele.jpg`, `synth_002.jpg`, `synth_005.jpg`).
   Alle stammen aus der Langfassung — kein Fehler, aber `make abbildungen`
   erzeugt deutlich mehr, als die Kurzfassung braucht.
6. **Die Kurzfassung (Abstract) nennt fünf Befunde und drei Experimente**, und
   Abschnitt 1.3 kündigt „drei Experimente" an. Beides muss mit, wenn
   Experiment 3 in den Anhang wandert — sonst versprechen die ersten zwei Seiten
   etwas, das das Dokument nicht hält.

## Empfohlene Reihenfolge

Nach Ertrag pro Aufwand, und so, dass jeder Schritt einzeln übersetzbar bleibt:

1. `02_theorie.tex`: Unterabschnitt 2.1.2 löschen, 5 Formeln + Abbildung 2.1
   löschen, Einleitungssatz zu den drei Prinzipien ergänzen. **−2 Seiten**,
   eine halbe Stunde, kein Informationsverlust.
2. Abbildungen 3.2, 5.3, 5.4, 5.5, 5.6 und 3.3(a) löschen, Abbildung 3.1 auf
   Teil (a) kürzen, von 3.3(b) und 7.1 nur eine behalten. Je eine Aussage in den Fließtext
   nachtragen. **−2 Seiten**.
3. Anhangabbildungen A.2 und A.3 löschen, toten `\iffalse`-Block löschen,
   Formel (5.4) auf zwei Zeilen Text. **−1 Anhangseite**.
4. Entscheidung zu den Experimenten treffen. Wenn ja: Abschnitt 7.5 in den
   Anhang, die zwei Einsichten in Kapitel 8 retten, Abstract und 1.3 anpassen.
   **−4 Seiten**.
5. Nur wenn die 35 Seiten noch nicht reichen: Prosablöcke aus dem Abschnitt oben
   und Kapitel 4 zu Abschnitt 5.1. **−2 Seiten**.

Nach Schritt 1–4 stehen 37 Seiten, nach Schritt 5 sind es 35.
