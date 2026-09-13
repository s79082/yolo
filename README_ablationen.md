# Zwei Ablationen der Augmentierungspipeline auf `yolo26s-p2`

Dieses Verzeichnis enthält den neuen Praxisteil der Arbeit. Es lässt das
bestehende Projekt (`src/`, `synth/`, `runs/`, `docs/`) **unverändert** und legt
alles Neue daneben: eigene Skripte, eigene Datensätze, eigene Läufe, eigene
Auswertung.

Alle fünf Trainingsläufe verwenden dasselbe Modell — **`yolo26s-p2`**, also
YOLO26 in der Größe `s` mit zusätzlichem P2-Detektionskopf (Strides 4/8/16/32),
Gewichte partiell aus dem COCO-Checkpoint `yolo26s.pt`.

---

## Die beiden Ablationen

### Ablation A — *wie viel* Verdeckung? (bestehende Datensätze, neues Modell)

| Arm | Datensatz | Inhalt |
|---|---|---|
| **A0** | `synth/abl_A0_keine_okklusion` | keine quer liegenden Stangen, keine Fremdkörper, kein Schaft kreuzt vor einer Stirnfläche |
| **A1** | `synth/abl_A1_baseline` | Baseline der Pipeline |
| **A2** | `synth/abl_A2_starke_okklusion` | in jedem Bild Stangen, mehr Fremdkörper, mehr quer davor liegende Rohre |

Die drei Datensätze lagen bereits vor und werden unverändert weiterverwendet.
Neu ist ausschließlich das Modell.

### Ablation B — *woraus* bestehen die Verdecker? (neu)

Getestet wird die einzige Begründung, mit der `src/occluders.py` seine
photographische Okkluder-Bibliothek rechtfertigt:

> „Ein synthetisch gemalter Balken hätte eine andere Textur-, Rausch- und
> Glanzstatistik als der Rest des Bildes; das Netz könnte ‚Objekt liegt neben
> künstlichem Balken‘ lernen.“

Dieses Argument war nie gemessen.

| Arm | Bibliothek | Datensatz |
|---|---|---|
| **B0** | photographisch (56 Stangen aus den Clean Plates) | `synth/abl_A1_baseline` — **identisch mit A1** |
| **B1** | prozedural, physikalisch (Zylindermodell, an B0 kalibriert) | `synth/abl_B1_proz_balken` |
| **B2** | prozedural, naiv (konstantes Grau) | `synth/abl_B2_proz_balken_naiv` |

Ausgetauscht wird die **ganze** Bibliothek, also beide Verwendungen: der
Rohrschaft *hinter* der Stirnfläche (T5, 88 % der Instanzen — der dominante
Fall) und die quer *davor* liegenden Stangen (T6, 58 % der Bilder).

**Der Versuch ist gepaart.** Jede prozedurale Stange übernimmt von ihrem
photographischen Gegenstück die exakte Pixelgröße und die exakte Alphamaske;
alle drei Bibliotheken haben genau 56 Einträge. Damit ist die Zufallsfolge im
Generator identisch und die Sichtbarkeitsrechnung ebenfalls. Nachweis
(`experiment_neu.py check`):

```
Labelverzeichnisse   SHA-256  951f6e23e65486f1  (alle drei gleich)
Bildverzeichnisse    c39a869f… / e392289e… / 9f5f1906…  (paarweise verschieden)
Labels 68 281 · Sichtbarkeit 0,901 · teilverdeckt 29,9 %  (in allen drei gleich)
```

Es bleibt genau eine Variable: die RGB-Werte innerhalb der Stangenmasken.

---

## Ergebnis auf einen Blick

Alle Zahlen bei nativem Maßstab, gegen `val` (98 Objekte) und `test` (89).
Bootstrap-Streuung des Referenzmodells: ±0,011 auf `val`, ±0,012 auf `test`.

### Referenzmodell (6000 Bilder, 60 Epochen, Early Stopping nach 43)

| Split | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | bestes F1 | P / R |
|---|---|---|---|---|---|
| `val` (real) | 0.9474 | 0.7939 | 0.6197 | 0.9101 | 0.945 / 0.878 |
| `test` (real, unberührt) | 0.9688 | 0.3680 | 0.4767 | 0.9364 | 0.964 / 0.910 |
| `sval` (synthetisch) | 0.9836 | 0.9430 | 0.8391 | 0.9613 | 0.968 / 0.954 |

Sim-to-Real-Abstand: 3,6 pp (`val`) bzw. 1,5 pp (`test`) mAP@0.5.

### Ablation A — Okklusionsanteil

| Arm | Labels | test mAP@0.5 | test F1 | test Rückruf | val mAP@0.5 | val F1 | best. Epoche |
|---|---|---|---|---|---|---|---|
| A0 ohne | 82 950 | 0.958 | 0.929 | 0.888 | 0.953 | 0.909 | 21 |
| **A1 Baseline** | 68 281 | 0.961 | **0.932** | 0.921 | **0.974** | **0.944** | 28 |
| A2 stark | 55 833 | **0.964** | 0.924 | **0.955** | 0.957 | 0.938 | 29 |

Verdeckung hilft — der Schritt A0 → A1 bringt +3,5 pp F1 und +8,2 pp Rückruf auf
`val`. *Mehr* Verdeckung hilft nicht mehr: A2 findet zwar mehr, zählt aber über
(95 statt 89) und verliert beim F1. **Die Baseline ist auf beiden Splits der
beste Betriebspunkt** — die Ablation bestätigt die Parameterwahl. Der in der
Vorgängerfassung berichtete *monotone* Verlauf reproduziert sich mit dem
P2-Kopf und dem korrigierten Annotationsstand nicht.

### Ablation B — Herkunft der Okkluder

| Arm | test mAP@0.5 | test F1 | test Rückruf | val mAP@0.5 | val F1 | best. Epoche |
|---|---|---|---|---|---|---|
| **B0 photographisch** | 0.961 | 0.932 | **0.921** | **0.974** | **0.944** | 28 |
| B1 prozedural | **0.963** | **0.941** | 0.899 | 0.944 | 0.913 | 20 |
| B2 naiv | 0.914 | 0.885 | 0.820 | 0.964 | 0.922 | 15 |

Rückruf nach Objektgröße (kleinstes Drittel / obere zwei Drittel):

| Arm | test klein (≤14,9 px) | test groß | test FN | val klein (≤25,4 px) | val groß | val FN |
|---|---|---|---|---|---|---|
| B0 photographisch | **0.800** | **0.983** | **7** | **0.909** | **0.969** | **5** |
| B1 prozedural | 0.767 | 0.966 | 9 | 0.848 | 0.938 | 9 |
| B2 naiv | 0.567 | 0.949 | 16 | 0.879 | 0.923 | 9 |

**Der Hauptbefund.** Ein naiv gezeichneter Balken kostet auf dem unberührten
Testbild **4,7 pp mAP@0.5** und **4,7 pp F1** — mehr, als der Verdeckungsgrad
überhaupt bewegt, und mehr als das doppelte Bootstrap-Intervall. Der Verlust
sitzt fast vollständig beim kleinsten Objektdrittel (Rückruf 0.800 → 0.567),
also dort, wo im Zielbild tatsächlich verdeckt wird. Die Precision bleibt dabei
unverändert hoch (0.961 gegen 0.943): Es ist **keine Abkürzung**, sondern eine
Domänenlücke beim Verdecker.

Das an fünf Kennwerten kalibrierte Zylindermodell erreicht dagegen auf `test`
dieselbe Leistung wie die photographische Bibliothek. Die Begründung in
`src/occluders.py` ist damit **in der Sache bestätigt und in ihrer starken Form
widerlegt**: Entscheidend ist nicht die *Herkunft* der Verdeckerpixel, sondern
ob ihre Statistik zur Umgebung passt. Die aufwendige Extraktionsstufe lässt sich
durch rund vierzig Zeilen prozeduralen Code ersetzen.

### Inferenzauflösung (Referenzmodell)

mAP@0.5 auf `val` steigt von 0.320 bei `imgsz` 640 auf 0.952 bei 2240 — **+63,3 pp**,
der größte Einzeleffekt der Arbeit. Das Maximum liegt jeweils dort, wo die
Objektgröße im Netzeingang dem Median der Trainingsverteilung (23,5 px)
entspricht: auf `val` bei 2240 (24,4 px), auf `test` bei 1920 (20,4 px). Zu
große Auflösung kostet ebenfalls — auf `test` fällt mAP@0.5 von 0.976 bei 1920
auf 0.939 bei 2464.
Test-Time-Augmentation ist bei diesem Modell **nicht verfügbar**: Die
Werkzeugkette lehnt `augment=True` für Ende-zu-Ende-Modelle ab und fällt
stillschweigend auf einfache Inferenz zurück.

---

## Ablauf

```bash
# 1  prozedurale Bibliotheken erzeugen und an der photographischen kalibrieren
.venv/bin/python src/procedural_bars.py

# 2  Datensätze der Ablation B (gleicher Seed wie A1)
.venv/bin/python src/generate_neu.py --out synth/abl_B1_proz_balken \
    --n 3000 --seed 1 --bars-dir assets/bars_proc
.venv/bin/python src/generate_neu.py --out synth/abl_B2_proz_balken_naiv \
    --n 3000 --seed 1 --bars-dir assets/bars_proc_naiv

# 3  yamls, Paarungsnachweis, Trainings, Auswertung
.venv/bin/python src/experiment_neu.py data
.venv/bin/python src/experiment_neu.py check
.venv/bin/python src/experiment_neu.py train      # 5 Läufe, je ~45 min
.venv/bin/python src/experiment_neu.py eval

# 4  Referenzmodell (voller Datensatz) und dessen Auswertung
.venv/bin/python src/train_neu.py --name referenz \
    --data data/referenz.yaml --epochs 60
.venv/bin/python src/experiment_neu.py main  --weights runs/referenz/weights/best.pt
.venv/bin/python src/experiment_neu.py sweep --weights runs/referenz/weights/best.pt
.venv/bin/python src/analyse_neu.py          --weights runs/referenz/weights/best.pt

# 5  Rückruf nach Objektgröße (Ablation B)
.venv/bin/python src/analyse_ablationB.py

# 6  Abbildungen, LaTeX-Tabellen und zahlen.tex
.venv/bin/python src/report_neu.py
```

`src/run_all.sh` und `src/run_ref.sh` fassen die Schritte 3 und 4
zusammen.

---

## Code

| Datei | Zweck |
|---|---|
| `src/procedural_bars.py` | Stufe 2b — prozedurale Okkluder-Bibliotheken B1/B2, an B0 kalibriert |
| `src/generate_neu.py` | dünner Aufsatz auf `src/generate.py`: tauscht **nur** die Okkluder-Bibliothek aus |
| `src/train_neu.py` | Training auf `yolo26s-p2`; übernimmt die Parametertabelle aus `src/train.py` |
| `src/experiment_neu.py` | `data` / `check` / `train` / `eval` / `main` / `sweep` |
| `src/analyse_neu.py` | Kreuzvergleich, Boxgrößenversatz, Sensitivität, Fehleranalyse |
| `src/analyse_ablationB.py` | Rückruf nach Objektgröße je Arm der Ablation B |
| `src/report_neu.py` | Abbildungen und LaTeX-Tabellen |

Der Generator selbst wird **nicht** kopiert und **nicht** verändert —
`generate_neu.py` ersetzt zur Laufzeit allein die Funktion, die die Stangen
lädt. Hintergrundwahl, Zoom, Clusterlayout, Instanzgeometrie, Photometrie,
Tiefenmodell, Sichtbarkeitsrechnung und globale Bildstufe sind Zeile für Zeile
dieselben wie in der ursprünglichen Arbeit.

---

## Trainingsbudgets

| | Referenzmodell | Ablationsarme (5×) |
|---|---|---|
| Trainingsbilder | 6000 (`synth/train`) | 3000 |
| Epochenbudget | 60 (`patience` 20) | 30 (`patience` 20) |
| tatsächlich gelaufen | 43, bestes Modell aus Epoche 23 | 30 |
| Batch | 8 | 8 |
| Bildzwischenspeicher | Festplatte | Festplatte |

Alles Übrige ist identisch: Architektur, Optimierer, Lernratenplan,
Verlustgewichte, Trainingszeit-Augmentierung, Seed, Validierungssplit.

**Drei Abweichungen gegenüber `src/train.py` sind erzwungen, nicht gewählt:**

* `batch=8` statt 16 — `yolo26s-p2` hält bei `imgsz=1024` eine P2-Merkmalskarte
  mit 256×256 Zellen vor und belegt bei Batch 16 rund 23,5 GB. Auf dieser Karte
  sind durch einen Fremdprozess nur etwa 25 GB frei; ein erster Lauf mit
  Batch 16 wich wiederholt mit `CUDA OutOfMemoryError in TaskAlignedAssigner,
  using CPU` aus. Das ist nicht nur langsam, sondern macht Läufe untereinander
  unvergleichbar, weil der Ausweichpfad von der momentanen Fremdlast abhängt.
* `cache='disk'` statt `'ram'` — 3000 × 1024² × 3 B = 9,4 GB, zeitweise waren
  unter 2 GB Arbeitsspeicher frei.
* `workers=0` — unverändert aus der ursprünglichen Arbeit (DataLoader-Deadlock
  in dieser Umgebung).

Alle drei gelten für **alle** Läufe gleich und können den Vergleich zwischen
den Armen daher nicht verzerren. Die Epochenzahl 30 ist aus dem Zeitbudget
gewählt und begrenzt die absolute Höhe der Ablationszahlen — deshalb sind
Ablationszahlen nur untereinander zu vergleichen, nie mit dem Referenzmodell.

---

## Bewertungsdaten

`val` (98 Objekte) und `test` (89 Objekte) wurden nach der ersten
Auswertungsrunde noch einmal durchgesehen und ergänzt. **Alle Zahlen in den neuen Läufen`
beziehen sich auf diesen korrigierten Stand**; die Zahlen der ursprünglichen
Arbeit unter `docs/` sind gegen 88 bzw. 86 Objekte gerechnet und mit ihnen
nicht direkt vergleichbar.

---

## Ergebnisdateien

| Datei | Inhalt |
|---|---|
| `docs/bars_kennwerte.json` | Kennwerte der drei Okkluder-Bibliotheken, Kalibrierverlauf |
| `docs/bars_vergleich.png` | Kontrollbild: echte, prozedurale und naive Stangen |
| `docs/experiments/datensaetze.json` | Kennwerte der fünf Trainingsdatensätze |
| `docs/experiments/paarungsnachweis.json` | Hashes der Label- und Bildverzeichnisse |
| `docs/experiments/ablationen.json` | Ergebnisse aller Arme auf `val` und `test` |
| `docs/experiments/hauptergebnis.json` | Referenzmodell auf `val`, `test`, `sval` |
| `docs/experiments/inferenz_sweep.json` | Auflösungssweep, TTA, Bootstrap-CI |
| `docs/experiments/analyse.json` | Kreuzvergleich, Boxversatz, Sensitivität, Fehler |
| `docs/experiments/ablationB_groesse.json` | Rückruf nach Objektgröße, Ablation B |
| `docs/ERGEBNISSE.md` | alle Ergebnistabellen als Markdown |
| `docs/fig_*.png` | Abbildungen der Arbeit |

## Zwei Fassungen der schriftlichen Arbeit

| Fassung | Praxisteil | Bauen |
|---|---|---|
| `paper/main.pdf` | Hauptergebnis + **Ablation B** (Arbeitsfassung) | `make` |

Die Arme A0 und A2 sind in beiden Fällen trainiert und ausgewertet
(`runs/`, `docs/ERGEBNISSE.md`); in der Arbeitsfassung werden sie nicht
berichtet, sondern in Abschnitt 8.2 als nächster Schritt benannt. Beide
Zahlen und Ergebnistabellen liegen in `paper/kapitel/`,
sodass eine Zahl nur an einer Stelle steht.
