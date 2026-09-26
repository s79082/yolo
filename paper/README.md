# LaTeX-Projekt der Bachelorarbeit

Enthält den vollständigen Text: den bereits vorhandenen Theorieteil (CNN,
YOLO26, Anwendung) und den neu ergänzten Praxisteil mit dem Schwerpunkt
**Datenaugmentierung** — allgemeine Grundlagen, die konkrete Synthese-Pipeline
mit Begründung jedes Parameters, und die Auswertung.

## Aufbau

Ein Dokument: `main.tex` bindet die Kapitel aus `kapitel/` ein. Dort liegen
neben dem Fließtext auch `zahlen.tex` und die Ergebnistabellen `tab_*.tex`,
beide erzeugt von `src/report_neu.py` bzw. `src/analyse_ablationB.py`. Jede
Zahl steht damit an genau einer Stelle.

Frühere Fassungen (Langfassung, Kurzfassung, yolo11s-Vorgänger) sind entfernt;
sie liegen im letzten Commit unter `paper/kapitel`, `paper/kapitel_kurz` und
`paper/kapitel_e2` und lassen sich mit `git show HEAD:<pfad>` wieder ansehen.

### Woher die Zahlen kommen

Jede im Fließtext genannte Messzahl steht in `kapitel/zahlen.tex` und wird
von `src/report_neu.py` erzeugt — ebenso alle Ergebnistabellen
(`kapitel/tab_*.tex`). Wird ein Lauf wiederholt, genügt

```bash
.venv/bin/python src/report_neu.py --nur-tabellen
```

im Projektwurzelverzeichnis, damit Text und Tabellen wieder zu den Daten passen.
Eine Zahl, für die noch kein Lauf vorliegt, erscheint im PDF als **??**.

### Herkunft der Zahlenwerte

Anhangabschnitt **A.4** (Tabelle A.4) nennt für jede Klasse von Zahlenwerten die
Quelle und das Programm, das sie erzeugt — von der Objektstatistik über die
Pipelineparameter und die Modellkennzahlen bis zu den Metriken. Als Prüfhilfe
beim Schreiben:

```bash
python3 tools/zahlen_audit.py
```

Das Skript zerlegt die Kapitel in Sätze und meldet jeden Satz, der eine Zahl
enthält, aber keinen Herkunftsmarker (Verweis auf Messung, Gleichung, Tabelle,
Literatur oder die Kennzeichnung als gesetzter Wert). Es ist eine
Erinnerungshilfe, kein Urteil — viele Treffer sind korrekt, weil die Herkunft im
Nachbarsatz oder in der Abschnittseinleitung steht.

## Übersetzen

```bash
make
```

Erzeugt `build/main.pdf` und kopiert das Ergebnis nach `paper/main.pdf`. Ohne
`latexmk` geht auch `make manual`.

Weitere Ziele: `make pruefen` zählt die
Warnungen (offene Referenzen, Overfull-Boxen) aus den Logdateien, `make clean`
räumt auf.

Vorgesehene Engine ist **pdfLaTeX**; die Präambel erkennt XeLaTeX und LuaLaTeX
und lädt dort `fontspec` statt `inputenc`/`fontenc` (unter XeLaTeX setzt
`fontenc` mit T1 das ß falsch). Literatur läuft über `biblatex` mit
`backend=biber` — latexmk erkennt die `.bcf`-Datei und ruft biber selbst auf.
Ohne biber im System muss man in `praeambel.tex` auf `backend=bibtex` umstellen
**und** latexmk mit `-e '$bibtex_use=2;'` aufrufen, sonst startet latexmk
trotzdem biber, findet keine `.bcf` und lässt die Literaturliste leer.

Benötigte TeX-Live-Pakete (Debian/Ubuntu):

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-lang-german texlive-bibtex-extra texlive-science latexmk
```

## Aufbau

| Datei | Inhalt |
|---|---|
| `main.tex` | Dokumentrahmen, bindet alle Kapitel ein |
| `praeambel.tex` | Pakete, Satzspiegel, Makros (`\code`, `\px`, `\PP`, `\mAPfifty` …) |
| `literatur.bib` | 94 Quellen |
| `kapitel/00_titel.tex` | Titelseite |
| `kapitel/00_kurzfassung.tex` | Kurzfassung |
| `kapitel/01_einleitung.tex` | 1 Einleitung: Aufgabe, Vorarbeiten (Verfahren der Objektdetektion), Zielsetzung, Aufbau |
| `kapitel/02_theorie.tex` | 2 Theoretische Grundlagen (CNN, YOLO26, Anwendung) — der vorhandene Theorieteil |
| `kapitel/03_augmentierung_grundlagen.tex` | 3 Datenaugmentierung: Formalisierung, Wirkmechanismen, Taxonomie, Detektionsbesonderheiten, Grenzen, Sim-to-Real; leitet die Anforderungen A1–A6 ab |
| `kapitel/04_datengrundlage.tex` | 4 Datengrundlage: alle Messungen am Quellmaterial (Belegbasis der Parameter) |
| `kapitel/05_pipeline.tex` | 5 Synthese-Pipeline: Stufen 1–3 mit T1–T9, jede Transformation mit Soll-Bereich, gemessenem Ist-Anteil und Begründung |
| `kapitel/06_trainingszeit.tex` | 6 Trainingszeit-Augmentierung und Trainingsaufbau |
| `kapitel/07_auswertung.tex` | 7 Auswertung: Metriken, Hauptergebnis, Sim-to-Real-Abstand, Experimente 1–3, Bootstrap, Fehleranalyse |
| `kapitel/08_diskussion.tex` | 8 Diskussion: Belegstärke, Grenzen, Ausblick, Fazit |
| `kapitel/09_anhang.tex` | Anhang: Reproduktion, Umgebungshinweis, Rohwerte des Erzeugungsberichts |
| `abbildungen/` | aufbereitete Abbildungen (JPEG/PNG) |
| `tools/abbildungen_vorbereiten.py` | erzeugt `abbildungen/` aus `docs/` und den Rohdaten |
| `tools/fig_p2.py` | erzeugt `abbildungen/p2_sweep.png` aus `docs/experiments/experiment3_models_imgsz.json` |
| `tools/fig_boxversatz.py` | erzeugt `abbildungen/boxversatz.png`; braucht GPU (führt das Modell aus) |

Die schematischen Abbildungen sind als TikZ direkt im Text gezeichnet und
brauchen keine Bilddatei: rezeptives Feld, Boxaufblähung unter Rotation, Aufbau
der Pipeline (Langfassung) sowie die Merkmalspyramide P2–P5 und die
Kandidatenzellen auf P3 gegen P2 (beide Varianten).

## Abbildungen neu erzeugen

```bash
make abbildungen
```

Skaliert die Diagramme und Fotos aus `docs/` auf satztaugliche Größen.
Die zugrunde liegenden Diagramme selbst entstehen mit `src/report.py`,
`src/matting.py`, `src/occluders.py`, `src/generate.py` und `src/evaluate.py`
(siehe Anhang des Dokuments und `../README.md`).

## Herkunft der Zahlen

Alle Ist-Anteile der Pipeline stammen aus
`../synth/train/generation_report.json`, alle Metriken aus
`../docs/experiments/*.json`, alle Messungen am Quellmaterial aus
`../docs/datensichtung.json`. Der Anhang des Dokuments listet die Rohwerte,
damit jede Tabelle nachprüfbar ist.

## Konventionen im Text

* Jede Parameterentscheidung ist mit einer Belegart gekennzeichnet:
  **M** = Messung an den Realdaten, **P** = physikalische Randbedingung,
  **L** = Literatur, **A** = Annahme.
* Die Signaturen **S1–S6** benennen die Copy-Paste-Artefakte, gegen die
  einzelne Pipelinestufen gerichtet sind (Langfassung Abschnitt 3.7.3,
  Kurzfassung 3.4.1 mit eigener Tabelle).
* Die Anforderungen **A1–A6** werden am Ende von Kapitel 5 gegen die Umsetzung
  geprüft.

## Seitenumfang der Kurzfassung

38 Seiten Text bei 1,5-fachem Zeilenabstand (davon 3 Seiten der nachträglich
ergänzte Abschnitt zum Größenversatz der Boxen). Rund acht dieser Seiten sind
Tabellen und Abbildungen, weshalb ein kleinerer Zeilenabstand kaum etwas bringt
(bei 1,10 sind es 34 Seiten). Die verbleibenden Kürzungsmöglichkeiten mit ihrem
jeweiligen Wert:

| Maßnahme | Ersparnis |
|---|---|
| Kapitel 2 (CNN- und YOLO26-Grundlagen, 5 S.) auf 2 S. eindampfen | −3 S. |
| Kapitel 3 (Augmentierungstheorie, 6 S.) auf 4 S. | −2 S. |
| Kapitel 4 (Datengrundlage) in Kapitel 5 integrieren | −1 S. |
| Übersichtstabelle der Synthesestufen streichen (steht vollständig im Anhang) | −0,7 S. |

Nicht empfohlen wird eine Kürzung von Abschnitt 7.5 (P2-Detektionskopf, 3,5 S.)
und von Tabelle 5.1 -- das sind die inhaltlichen Kernstücke.

## Nachtrag: Größenversatz der Boxen

Ein in der Qualitativabbildung sichtbares Muster — rote und blaue Boxen am
selben Objekt — hat sich als messbarer systematischer Effekt erwiesen und ist in
beiden Fassungen als eigener Abschnitt dokumentiert (Kurzfassung 7.7,
Langfassung 7.10):

| Split | pred/GT (Median) | GT − pred |
|---|---|---|
| `test` (real) | 0,884 | +2,53 px |
| `val` (real) | 0,961 | +1,16 px |
| `sval` (synthetisch) | **1,012** | −0,29 px |

Das Modell reproduziert die synthetische Boxkonvention auf 1 % genau; der Versatz
entsteht erst gegenüber der Handannotation. Reproduzierbar über die Skripte in
`tools/`; die Zahlen der Sensitivitätsrechnung stammen aus einer erneuten
Auswertung mit aufgeweiteten Boxen.
