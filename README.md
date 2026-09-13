# Rohrprofil-Erkennung mit YOLO auf rein synthetischen Trainingsdaten

Bachelorarbeit — Detektion von **Stirnflächen von Rundrohrprofilen** an
gestapelten Bauzaunfeldern. Trainiert wird **ausschließlich** auf synthetischen
Bildern, die aus 9 freigestellten Objekt-Cutouts und 7 Hintergründen erzeugt
werden. Validiert und getestet wird auf echten Aufnahmen.

```
Cutouts (9)  ──▶ [1] Matting ──────────┐
                                       ├──▶ [3] Bildsynthese ──▶ 6000 Bilder
Hintergründe (7) ─▶ [2] Okkluder ──────┘                        136 384 Labels
                                                                      │
                                       [4] YOLO11s-Training ◀──────────┘
                                                │
                        [5] Test-Framework ◀────┴────▶ [6] Experimente
```

## Ergebnis auf einen Blick

Rein synthetisch trainiert, ausgewertet auf **echten** Bildern:

| Split | mAP@0.5 | mAP@0.5:0.95 | bestes F1 | Zählung |
|---|---|---|---|---|
| `val` (real, für Modellauswahl) | 0.886 | 0.597 | 0.862 | 86 / 88 |
| `test` (real, **nie im Training oder in der Auswahl**) | 0.888 | 0.437 | 0.894 | 84 / 86 |
| `sval` (synthetisch, Sim-to-Real-Referenz) | 0.982 | 0.828 | 0.964 | 7128 / 7248 |

Bei der auf `val` optimierten Inferenzauflösung (imgsz 1920) erreicht `test`
mAP@0.5 = **0.928**, mit Test-Time-Augmentation **0.934**. Der Zählfehler liegt
bei **2.3 %**.

Zwei Kernbefunde:

1. **Die Inferenzauflösung dominiert alles andere** — mAP@0.5 auf `val` steigt
   von 0.435 (imgsz 640) auf 0.904 (imgsz 1600), weil die Objekte nur 12–36 px
   groß sind.
2. **Okklusion in den Trainingsdaten hilft monoton** — über drei Ablationsarme,
   zwei Splits und drei Metriken, obwohl der stärkste Arm 33 % weniger Labels
   hat. *(Dieser Befund reproduziert sich mit `yolo26s-p2` und dem korrigierten
   Annotationsstand **nicht** in dieser Form — siehe [`README_ablationen.md`](README_ablationen.md).)*
3. **Die Modellfamilie war die schwächste Entscheidung** — `yolo26s-p2`
   erreicht bei identischem Budget `test` mAP@0.5 = 0.941 statt 0.888
   (Experiment 3). `yolo11s` bleibt die dokumentierte Baseline, für den
   produktiven Einsatz wäre `yolo26s-p2` zu wählen.

Details und Einschränkungen: [`docs/DISKUSSION.md`](docs/DISKUSSION.md).

## Eigenes CNN als Gegenentwurf

Im Unterverzeichnis [`cnn/`](cnn/) liegt **PipeNet** — eine eigene, auf dieses
Problem zugeschnittene Architektur (Zentrums-Heatmap auf Stride 4,
Peak-Extraktion statt NMS, 3.07 M Parameter). Trainiert auf demselben
Datensatz, ausgewertet mit denselben Metrikfunktionen. Bei den
Lokalisierungsmetriken auf `test` liegt sie vor allen YOLO-Varianten, bei den
Detektionsmetriken hinter `yolo26s-p2`. Siehe [`cnn/README.md`](cnn/README.md),
die Architekturgrafik in [`cnn/docs/fig_architektur.png`](cnn/docs/fig_architektur.png)
und die ausführliche Erklärung in [`cnn/einfach.md`](cnn/einfach.md)
(Zielgruppe Informatikstudierende, ohne CV-Vorkenntnisse).

## Neuer Stand: zwei Ablationen der Augmentierung auf `yolo26s-p2`

Der aktuelle Praxisteil ist in [`README_ablationen.md`](README_ablationen.md) beschrieben. Alle Modelle
dort sind **`yolo26s-p2`** (YOLO26s mit zusätzlichem P2-Kopf, Stride 4), und es
werden zwei Ablationen der Augmentierungspipeline gerechnet:

| Ablation | Frage | Arme |
|---|---|---|
| **A** | *wie viel* Verdeckung? | A0 keine / A1 Baseline / A2 stark — die bereits vorhandenen Datensätze `synth/abl_*` |
| **B** | *woraus* bestehen die Verdecker? | B0 photographisch (= A1) / B1 prozedural, physikalisch / B2 prozedural, naiv |

Ablation B prüft die einzige Begründung, mit der `src/occluders.py` seine
photographische Okkluder-Bibliothek rechtfertigt — dass ein *gemalter* Balken
eine andere Textur-, Rausch- und Glanzstatistik hätte und damit selbst zur
Abkürzung würde. Sie ist **gepaart**: Die drei Datensätze haben byte-identische
Labeldateien, es unterscheiden sich nur die RGB-Werte innerhalb der
Stangenmasken. Details in [`README_ablationen.md`](README_ablationen.md).

Achtung beim Vergleich mit den älteren Zahlen unter `docs/`: `val` und `test`
wurden nachträglich ergänzt (98 statt 88 bzw. 89 statt 86 Objekte); alle Zahlen
in den neuen Skripten beziehen sich auf den korrigierten Stand.

**Hauptbefund von Ablation B:** Ein naiv gezeichneter Balken kostet auf dem
unberührten Testbild 4,7 Prozentpunkte mAP@0.5 — mehr, als der Verdeckungsgrad
überhaupt bewegt. Ein an fünf Kennwerten kalibriertes prozedurales
Zylindermodell erreicht dagegen dieselbe Leistung wie die photographische
Bibliothek. Entscheidend ist also nicht die *Herkunft* der Verdeckerpixel,
sondern ob ihre Statistik zur Umgebung passt. Zahlen in
[`docs/ERGEBNISSE.md`](docs/ERGEBNISSE.md).

## Ausarbeitung (LaTeX)

Das Verzeichnis [`paper/`](paper/) enthält die schriftliche Arbeit als
LaTeX-Projekt in zwei Varianten: Theorieteil (CNN, YOLO26) plus Praxisteil mit
dem Schwerpunkt **Datenaugmentierung** — allgemeine Grundlagen, die
Synthese-Pipeline mit Begründung jedes Parameters und die Auswertung.

| PDF | Kapitel | Besonderheit |
|---|---|---|
| [`paper/main.pdf`](paper/main.pdf) | `kapitel/` | **Arbeitsfassung** — Hauptergebnis + Ablation B (Herkunft der Okkluder) |
| [`paper/main.pdf`](paper/main.pdf) | `kapitel/` | Langfassung, Modellvergleich über drei Varianten |

Übersetzen mit `make` in `paper/`.
Details in [`paper/README.md`](paper/README.md).

## Dokumentation

| Datei | Inhalt |
|---|---|
| [`docs/DATENSICHTUNG.md`](docs/DATENSICHTUNG.md) | Sichtung und Bewertung von Cutouts, Hintergründen, val/test — mit allen Messwerten |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | **Jede Transformation der Augmentierungs-Pipeline mit Anteil und Begründung** |
| [`docs/ERGEBNISSE.md`](docs/ERGEBNISSE.md) | Metriken, Diagramme, beide Experimente (automatisch erzeugt) |
| [`docs/DISKUSSION.md`](docs/DISKUSSION.md) | **Interpretation, Fehleranalyse, Einschränkungen** |
| [`docs/INFERENZ.md`](docs/INFERENZ.md) | **Jeder Inferenzparameter mit Wert und Begründung** |
| `runs/baseline/parameter_begruendung.md` | **Jeder Trainingsparameter mit Wert und Begründung** (automatisch aus `src/train.py` erzeugt) |
| `docs/experiments/*.json` | Rohdaten aller Sweeps |

## Code

| Datei | Stufe |
|---|---|
| `src/analyze_data.py` | 0 — Sichtung, alle Messwerte der Dokumentation |
| `src/matting.py` | 1 — Ellipsen-Fit und weiche Alphamasken für die Cutouts |
| `src/occluders.py` | 2 — Extraktion echter Rohrstangen als Okkluder |
| `src/generate.py` | 3 — Bildsynthese mit exakter Sichtbarkeitsrechnung |
| `src/train.py` | 4 — Training, Parameter + Begründungen in einer Tabelle |
| `src/evaluate.py` | 5 — eigenständiges Test-Framework (COCO-Metriken, Zählfehler) |
| `src/experiment.py` | 6 — Experiment 1 (Inferenzparameter) und 2 (Okklusionsablation) |
| `src/report.py` | 7 — Diagramme und Ergebnistabellen |
| `src/predict.py` | Anwendung — eigene Bilder zählen und markieren |

| `src/procedural_bars.py` | 2b — prozedurale Okkluder-Bibliotheken (Ablation B) |
| `src/generate_neu.py` | Aufsatz auf `src/generate.py`: tauscht nur die Okkluder-Bibliothek aus |
| `src/train_neu.py` | Training auf `yolo26s-p2` |
| `src/experiment_neu.py` | beide Ablationen: `data` / `check` / `train` / `eval` / `main` / `sweep` |
| `src/analyse_neu.py` | Kreuzvergleich, Boxversatz, Sensitivität, Fehleranalyse |
| `src/report_neu.py` | Abbildungen, LaTeX-Tabellen und `zahlen.tex` |

Trainierte Modelle: `runs/baseline` (yolo11s), `runs/yolo26s`,
`runs/yolo26s_p2`, `runs/abl_A0|A1|A2` (Okklusionsablation auf yolo11s) sowie
`runs/{A0,A1,A2,B1,B2,referenz}` (alle yolo26s-p2).

## Ablauf reproduzieren

```bash
# 0  Sichtung (erzeugt alle Messwerte der Dokumentation)
.venv/bin/python src/analyze_data.py

# 1  Cutouts freistellen  -> assets/masks/, docs/matting_check.png
.venv/bin/python src/matting.py

# 2  Okkluder-Bibliothek  -> assets/bars/, docs/bars_check.png
.venv/bin/python src/occluders.py

# 3  Trainingsdaten       -> synth/train/ (6000 Bilder) + generation_report.json
.venv/bin/python src/generate.py --out synth/train --n 6000 --seed 1 --preview 8
.venv/bin/python src/generate.py --out synth/sval  --n  400 --seed 99

# 4  Training auf GPU 0
.venv/bin/python src/train.py --name baseline

# 5  Auswertung (nativer Maßstab, eigene Metriken + Ultralytics-Kreuzvergleich)
.venv/bin/python src/evaluate.py --weights runs/baseline/weights/best.pt \
    --splits val,test --crosscheck --vis docs/qual_SPLIT_NAME.jpg \
    --out docs/experiments/final_eval.json

# 6  Experimente
.venv/bin/python src/experiment.py infer --weights runs/baseline/weights/best.pt
.venv/bin/python src/experiment.py ablation-data
.venv/bin/python src/experiment.py ablation-train
.venv/bin/python src/experiment.py ablation-eval

# 6c Modellfamilien-Vergleich (Experiment 3)
.venv/bin/python src/train.py --name yolo26s --model yolo26s.pt
.venv/bin/python src/train.py --name yolo26s_p2 --model yolo26s-p2.yaml \
    --pretrained yolo26s.pt
.venv/bin/python src/experiment.py models

# 7  Diagramme und Tabellen
.venv/bin/python src/report.py
```

## Eigene Bilder testen

```bash
.venv/bin/python src/predict.py MEIN_BILD.jpg
```

Schreibt nach `predictions/` ein markiertes Bild und eine YOLO-Labeldatei
mit Konfidenzwerten. Mit `--tta` etwa 2 Punkte mAP@0.5 mehr bei doppelter
Rechenzeit. **Nicht** `--imgsz 640` verwenden — die Objekte sind zu klein
(siehe Experiment 1a).

## Datenaufteilung

| Split | Quelle | Zweck |
|---|---|---|
| `synth/train` | synthetisch, 6000 Bilder | Training |
| `data/val` | **real**, 1 Bild, 88 Objekte | Modellauswahl (`best.pt`), Early Stopping |
| `data/test` | **real**, 1 Bild, 86 Objekte | **nur** Endbewertung, berührt das Training nie |
| `data/sval` | synthetisch, 400 Bilder | misst den Sim-to-Real-Abstand |

## Umgebungshinweis

In dieser Umgebung (opencv-python 5.0.0.93 + torch 2.13) verklemmt sich der
fork-basierte Ultralytics-DataLoader reproduzierbar beim ersten Batch, sobald
`workers > 0` ist. Getestet und jeweils erfolglos: `cv2.setNumThreads(0)`,
deaktiviertes OpenCL, `OMP/MKL_NUM_THREADS=1`, Startmethode `spawn`.
Das Training läuft deshalb mit `workers=0` und `cache='ram'`; siehe die
Begründung in `runs/baseline/parameter_begruendung.md`.
