# PipeNet — eigenes CNN für die Rohrprofil-Detektion

Gegenentwurf zum YOLO-Ansatz des Hauptprojekts. **Kein Fork von YOLO**, sondern
eine eigenständige Architektur, die aus den gemessenen Eigenschaften des
Problems abgeleitet ist.

Trainiert auf **exakt demselben** synthetischen Datensatz (`../synth/train`,
6000 Bilder, 136 384 Labels) mit demselben Protokoll (imgsz 1024, batch 16,
epochs 60, patience 20, AdamW, Kosinus-Abfall, Modellauswahl über die Fitness
auf dem realen `val`-Bild). Ausgewertet mit **denselben Metrikfunktionen**
(`../src/evaluate.py`). Der Vergleich ist damit ein Architekturvergleich und
keine Datenfrage.

## Die Idee in drei Sätzen

Das Problem hat **eine** Klasse, eine Objektgrößenspanne von nur Faktor 3.7 und
dicht gepackte, sich berührende Objekte, deren **Zentren** aber immer getrennt
sind. Also: **eine** Detektionsebene bei Stride 4 statt einer vierstufigen
Merkmalspyramide, eine **Zentrums-Heatmap** mit Peak-Extraktion statt
Box-Regression mit NMS, und ein Encoder ohne `layer4` — **3.07 M Parameter**
gegen 9.4–10.0 M bei den YOLO-Varianten.

![PipeNet Architektur](docs/fig_architektur.png)

Vollständige Begründung jeder Entscheidung: [`docs/ARCHITEKTUR.md`](docs/ARCHITEKTUR.md)
· ausführlich erklärt für Informatikstudierende: [`einfach.md`](einfach.md)

## Ergebnis

Auf dem unberührten Testbild, Betriebseinstellung je Modell **nur auf `val`**
bestimmt:

| Modell | Par. | test mAP@0.5 | test mAP@0.75 | test mAP@0.5:0.95 | test F1 | IoU (TP) |
|---|---|---|---|---|---|---|
| `yolo11s` | 9.43 M | 0.9279 | 0.3398 | 0.4482 | 0.9167 | 0.745 |
| `yolo26s` | 9.95 M | 0.9051 | 0.3326 | 0.4270 | 0.8876 | 0.729 |
| `yolo26s-p2` | 9.66 M | **0.9471** | 0.3257 | 0.4492 | **0.9341** | 0.739 |
| **PipeNet** | **3.07 M** | 0.9335 | 0.3548 | 0.4531 | 0.9167 | 0.742 |
| **PipeNet +TTA** | **3.07 M** | 0.9220 | **0.3775** | **0.4648** | 0.9102 | **0.756** |

**Bei den Lokalisierungsmetriken liegt PipeNet vorn** (mAP@0.75, mAP@0.5:0.95,
mittlere IoU — jeweils beste Werte aller fünf Modelle), bei den
Detektionsmetriken `yolo26s-p2`. PipeNet erreicht `yolo11s` exakt (F1 0.9167,
Zählung 82/86) mit einem Drittel der Parameter und einem Drittel der
Trainingszeit (0.37 h gegen 1.13 h).

Interpretation, Fehleranalyse und Einschränkungen:
[`docs/DISKUSSION.md`](docs/DISKUSSION.md)

## Dokumentation

| Datei | Inhalt |
|---|---|
| [`docs/ARCHITEKTUR.md`](docs/ARCHITEKTUR.md) | **Jede Entwurfsentscheidung mit Messwert und Begründung** |
| [`einfach.md`](einfach.md) | **Ausführliche Erklärung für Informatikstudierende** — von der Faltung bis zur Verlustfunktion, ohne CV-Vorkenntnisse |
| [`docs/einfach_kinderfassung.md`](docs/einfach_kinderfassung.md) | stark vereinfachte Fassung (Zielgruppe ~10 Jahre) |
| [`docs/ERGEBNISSE.md`](docs/ERGEBNISSE.md) | Metriken, Vergleich mit YOLO, Diagramme (automatisch erzeugt) |
| [`docs/DISKUSSION.md`](docs/DISKUSSION.md) | **Interpretation: welche Entwurfshypothese sich bestätigt hat und welche nicht** |
| `runs/pipenet/parameter_begruendung.md` | **Jeder Trainingsparameter mit Wert und Begründung** (automatisch aus `src/train.py`) |
| `docs/experiments/*.json` | Rohdaten |

## Code

| Datei | Inhalt |
|---|---|
| `src/model.py` | PipeNet: ResNet18-Encoder bis layer3, Top-Down-Dekoder auf Stride 4, drei Köpfe |
| `src/dataset.py` | Zielerzeugung (Gauss-Heatmap, log-Größe, Subpixel-Versatz) + Augmentierung |
| `src/losses.py` | Gaussian Focal Loss + maskierter L1 |
| `src/decode.py` | Peak-Extraktion, D4-Rotations-TTA, Äquivarianz-Selbsttest |
| `src/loader.py` | Thread-basierter Batch-Lader (fork-freier Ersatz für DataLoader-Worker) |
| `src/train.py` | Training, Parameter + Begründungen in einer Tabelle |
| `src/eval_pipenet.py` | Auswertung mit den Metrikfunktionen des Hauptprojekts (Name eindeutig, weil `src/evaluate.py` im Hauptprojekt existiert) |
| `src/compare.py` | Gegenüberstellung PipeNet ↔ YOLO |
| `src/predict.py` | Anwendung auf eigene Bilder, optional mit Heatmap-Overlay |
| `src/fig_architektur.py` | erzeugt beide Architekturgrafiken als SVG und PNG |

## Ablauf

```bash
# Architektur pruefen (Parameterzahl, Ausgabe-Stride)
.venv/bin/python cnn/src/model.py

# Zielerzeugung pruefen (sigma-Verteilung fuer die realen Objektgroessen)
.venv/bin/python cnn/src/dataset.py

# Datenlader-Benchmark (Threads gegen seriell)
.venv/bin/python cnn/src/loader.py

# Training auf GPU 0
.venv/bin/python cnn/src/train.py --name pipenet

# Auswertung inkl. Skalierungs-Sweep und Aequivarianz-Selbsttest
.venv/bin/python cnn/src/eval_pipenet.py --weights cnn/runs/pipenet/weights/best.pt \
    --splits val,test --sweep-scale --out cnn/docs/experiments/pipenet_eval.json

# Vergleich mit den YOLO-Modellen
.venv/bin/python cnn/src/compare.py

# Architekturgrafiken (SVG + PNG)
.venv/bin/python cnn/src/fig_architektur.py
```

## Eigene Bilder

```bash
.venv/bin/python cnn/src/predict.py MEIN_BILD.jpg --heatmap
```

Schreibt nach `cnn/predictions/` das markierte Bild, eine YOLO-Labeldatei mit
Konfidenzwerten und — mit `--heatmap` — das Heatmap-Overlay. Letzteres ist die
anschaulichste Darstellung des Verfahrens: pro Rohrende ein kompaktes,
klar getrenntes Maximum.

Es gibt **keinen** `imgsz`-Parameter (das Netz ist vollständig konvolutional und
arbeitet auf der nativen Bildgröße) und **keine** NMS-IoU-Schwelle (die
Peak-Extraktion braucht keine).

## Umgebungshinweis

Der fork-basierte DataLoader verklemmt sich in dieser Umgebung reproduzierbar
(opencv-python 5.0.0.93 + torch 2.13), auch mit diesem Datensatz. Statt auf
`workers=0` zurückzufallen wie im YOLO-Teil, lädt hier ein eigener Thread-Pool
(`src/loader.py`): die teuren OpenCV-Stufen geben die GIL frei und
parallelisieren echt. Gemessen **31.9 → 2.5 ms pro Sample**, Epochendauer
**3.19 → 0.25 min**.
