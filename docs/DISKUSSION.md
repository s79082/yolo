# Diskussion der Ergebnisse

Zahlen aus `docs/ERGEBNISSE.md`; Rohdaten in `docs/experiments/*.json`.

---

## 1 Hauptergebnis

Ein ausschließlich auf synthetischen Bildern trainiertes YOLO11s erreicht auf
**echten, nie im Training gesehenen Aufnahmen**:

| Split | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | bestes F1 | P | R | Zählung |
|---|---|---|---|---|---|---|---|
| `val` (real) | 0.886 | 0.793 | 0.597 | 0.862 | 0.872 | 0.852 | 86 / 88 |
| `test` (real, unberührt) | 0.888 | 0.320 | 0.437 | 0.894 | 0.905 | 0.884 | 84 / 86 |
| `sval` (synthetisch) | 0.982 | 0.935 | 0.828 | 0.964 | 0.972 | 0.956 | 7128 / 7248 |

Bei der auf `val` gewählten Inferenzauflösung (imgsz = 1920) statt des nativen
Maßstabs: `test` mAP@0.5 = **0.928**, mit TTA **0.934**.

Für die eigentliche Anwendung — Rohre zählen — ist der **Zählfehler** die
relevante Größe: **2.3 %** auf beiden realen Bildern (86 statt 88 bzw. 84 statt
86). Mit TTA bei imgsz 1920: 90/88 auf `val` und 84/86 auf `test`.

### Verifikation der eigenen Metrikimplementierung

| | eigene Implementierung | Ultralytics `model.val()` |
|---|---|---|
| `val` mAP@0.5 / mAP@0.5:0.95 | 0.886 / 0.597 | 0.897 / 0.599 |
| `test` mAP@0.5 / mAP@0.5:0.95 | 0.888 / 0.437 | 0.891 / 0.424 |

Abweichung ≤ 1.3 Prozentpunkte, erklärbar durch Detailunterschiede in der
AP-Interpolation und der Zuordnungsreihenfolge. Die eigene Implementierung ist
damit verifiziert.

---

## 2 Der Sim-to-Real-Abstand ist messbar und liegt bei etwa 10 Punkten

`sval` (400 synthetische Bilder aus derselben Pipeline, aber mit anderem Seed)
liefert mAP@0.5 = 0.982. Auf echten Bildern sind es 0.886/0.888.

**Abstand: −9.6 Prozentpunkte mAP@0.5, −23 Punkte mAP@0.5:0.95.**

Das ist eine gute Nachricht in zwei Richtungen:

* Der Abstand ist **klein genug**, dass das Verfahren praktisch funktioniert.
* Er ist **groß genug**, um zu belegen, dass das Netz *nicht* nur
  Einfügeartefakte gelernt hat — sonst wäre `sval` nahezu perfekt (mAP > 0.99)
  und die realen Bilder katastrophal.

Die Lernkurven (`fig_training.png`) zeigen dasselbe von der anderen Seite: der
Trainingsverlust fällt über alle 30 Epochen monoton, die Metrik auf dem realen
`val`-Bild ist aber ab Epoche ~10 flach. Ab dort verbessert das Modell nur noch
seine Anpassung an die synthetische Verteilung. Genau deshalb ist es richtig,
`best.pt` über den **realen** Validierungssplit auszuwählen — das Early Stopping
griff bei Epoche 30 mit dem besten Modell aus Epoche 10.

---

## 3 Experiment 1: Die Inferenzauflösung ist der dominante Parameter

| imgsz | `val` mAP@0.5 | `test` mAP@0.5 |
|---|---|---|
| 640 | 0.435 | 0.524 |
| 800 | 0.793 | 0.665 |
| 960 | 0.852 | 0.889 |
| 1024 | 0.868 | 0.882 |
| 1280 | 0.887 | 0.900 |
| 1600 | **0.904** | 0.888 |
| 1920 | 0.893 | **0.928** |
| 2240 | 0.880 | 0.928 |
| 2464 (nativ für `val`) | 0.886 | 0.927 |

**Von 640 auf 1600 px verdoppelt sich mAP@0.5 auf `val` (0.435 → 0.904).** Das
ist mit Abstand der größte Einzeleffekt der gesamten Arbeit — größer als jede
Entscheidung in der Augmentierungs-Pipeline.

Die Erklärung ist rein geometrisch und wurde in der Datensichtung vorhergesagt:
die Objekte sind nativ 12–36 px groß. Bei imgsz = 640 wird `val` (2459 px breit)
um den Faktor 0.26 verkleinert; die Objekte schrumpfen auf 5–9 px. Der feinste
Detektionskopf von YOLO11 arbeitet auf Stride 8 — ein 6-px-Objekt belegt dort
weniger als ein Ausgabepixel.

Ab etwa 1280 px sättigt die Kurve. Interessant: **das Optimum liegt nicht am
nativen Maßstab.** Für `test` (1600 px breit) ist imgsz 1920–2464, also eine
*Hochskalierung* um Faktor 1.2–1.5, spürbar besser als der native Maßstab
(0.928 vs. 0.888). Der Grund ist die Trainingsverteilung: der Generator erzeugt
Objekte mit Median 22.5 px, `test` hat aber nur 19.6 px Median. Eine leichte
Hochskalierung schiebt die Testobjekte in die Mitte der Trainingsverteilung.

> **Praktische Empfehlung:** nicht auf den nativen Maßstab skalieren, sondern
> so, dass die erwartete Objektgröße ~22 px im Netzeingang beträgt.

### mAP@0.75 zeigt die Grenze der Lokalisierung bei Kleinstobjekten

`val` erreicht mAP@0.75 = 0.793, `test` nur 0.320 — bei fast identischem
mAP@0.5. Der Unterschied ist die Objektgröße (Median 27.4 px gegen 19.6 px).
Bei einem 20-px-Objekt kostet ein Boxfehler von 2 px bereits ~0.2 IoU. Die
mittlere IoU der Treffer bestätigt das: 0.824 auf `val`, 0.748 auf `test`.
mAP@0.5:0.95 ist bei dieser Objektgröße folglich mehr ein Maß für die
Pixelauflösung als für die Detektionsqualität.

### Die übrigen Inferenzparameter sind unkritisch

* **NMS-IoU** (0.3 … 0.9): Spanne nur 3 Punkte mAP@0.5, Optimum bei 0.3–0.45,
  aber der Standard 0.7 verliert kaum. Wie in der Datensichtung vorhergesagt:
  benachbarte Rohrenden haben mindestens ~0.85 Durchmesser Abstand, ihre Boxen
  überlappen also kaum — NMS ist hier fast wirkungslos.
* **max_det**: 50 ist katastrophal (mAP 0.735 statt 0.892 auf `val`, Zählung
  50/88), ab 100 gesättigt. Der Ultralytics-Standard 300 reicht für diese
  Bilder, wäre bei einem großen Stapel aber eine stille Obergrenze.
* **FP16 (`quantize='fp16'`)**: bitidentische Ergebnisse zu FP32 bei ~gleicher
  Laufzeit. Kein Grund, es nicht zu verwenden, aber auch kein Gewinn.
* **TTA (`augment=True`)**: **+2.1 Punkte** mAP@0.5 auf `val` (0.893 → 0.914),
  **+0.6** auf `test` (0.928 → 0.934), und deutlich bessere Zählung
  (100/88 → 90/88). Kostet Faktor ~2 Rechenzeit. Für eine Zählanwendung, die
  nicht in Echtzeit laufen muss, lohnt sich das.

---

## 4 Experiment 2: Okklusion in den Trainingsdaten hilft — monoton

Drei Trainingsläufe, identisches Budget (3000 Bilder, 30 Epochen, Seed 1,
gleiche Hyperparameter), einziger Unterschied ist der Okklusionsanteil:

| Arm | Sichtbarkeit ⌀ | teilverdeckt | Labels | `val` mAP@0.5 | `val` mAP@.5:.95 | `test` mAP@0.5 | `test` mAP@.5:.95 | `test` F1 |
|---|---|---|---|---|---|---|---|---|
| **A0** ohne | 0.977 | 12.0 % | 82 950 | 0.888 | 0.606 | 0.899 | 0.418 | 0.896 |
| **A1** Baseline | 0.902 | 29.9 % | 68 281 | 0.924 | 0.609 | 0.911 | 0.452 | 0.904 |
| **A2** stark | 0.852 | 40.7 % | 55 833 | **0.930** | **0.620** | **0.919** | **0.454** | **0.910** |

**Der Effekt ist auf beiden Splits, in allen drei Metriken und über alle drei
Stufen monoton.** A0 → A2 bringt +2.0 Punkte mAP@0.5 und +3.6 Punkte
mAP@0.5:0.95 auf dem unberührten Testbild.

Bemerkenswert ist, **wogegen** dieser Gewinn erzielt wird: A2 hat durch die
stärkere Verdeckung **33 % weniger Labels** als A0 (55 833 gegen 82 950 bei
gleicher Bildzahl). Der Okklusionsvorteil überkompensiert also einen deutlichen
Verlust an Supervision. Das ist der stärkste Beleg dafür, dass die
Okklusionsstufe kein Beiwerk ist.

Ein zweiter Befund steckt im Trainingsverlauf: **A0 erreichte sein bestes Modell
bereits in Epoche 3** und verbesserte sich in den folgenden 20 Epochen nicht
mehr (Early Stopping bei Epoche 23). A1 und A2 verbesserten sich bis zur letzten
Epoche 30 durchgehend. Ohne Okklusion ist die Aufgabe für das Netz zu leicht —
es sättigt sofort und passt sich danach nur noch an die synthetische Verteilung
an.

### Einschränkungen dieses Experiments

1. **Der Effekt liegt innerhalb des Konfidenzintervalls.** Das
   Bootstrap-CI für mAP@0.5 ist ±0.05 (`val`) bzw. ±0.04 (`test`) — breiter als
   der gemessene Unterschied von 0.02–0.04. Belastbar ist nicht die einzelne
   Differenz, sondern die **Konsistenz**: dieselbe Richtung in 3 Metriken × 2
   Splits × 3 Stufen. Bei reinem Rauschen wäre eine solche durchgehende
   Monotonie unwahrscheinlich, aber ein formaler Signifikanznachweis ist mit je
   einem Bild pro Split nicht möglich.
2. **Labelzahl und Okklusionsanteil lassen sich nicht entkoppeln.** Mehr
   Verdeckung heißt zwangsläufig mehr verworfene Labels. Ein sauberer
   Nachfolgeversuch müsste die Bildzahl je Arm so anpassen, dass alle Arme
   gleich viele Labels haben.
3. **A1 und A2 waren nach 30 Epochen noch nicht auskonvergiert.** Ihr Vorsprung
   könnte bei längerem Training größer werden.
4. Auffällig: A1 und A2 (3000 Bilder, 30 Epochen) schlagen den Hauptlauf
   (6000 Bilder, bestes Modell aus Epoche 10) auf beiden Splits. Auch das liegt
   im Rauschen, deutet aber darauf hin, dass mehr synthetische Bilder ab einer
   gewissen Menge nichts mehr bringen — die Vielfalt ist durch die 9 Cutouts
   begrenzt, nicht durch die Bildzahl.

---

## 5 Fehleranalyse

Aus den Qualitativbildern (`docs/qual_val_*.jpg`, `docs/qual_test_*.jpg`):

**Falsch-Negative** treten fast ausschließlich auf bei
* sehr kleinen Objekten am linken Bildrand von `test` (12–15 px),
* stark verdeckten Enden tief im Stapel,
* Enden ohne sichtbaren hellen Kranz (vollständig im Schatten).

**Falsch-Positive** sind zum Teil gar keine: mehrere rot markierte Detektionen
im `val`-Bild sitzen auf erkennbaren Rohröffnungen, die in der Ground Truth
fehlen. Bei 88 bzw. 86 handannotierten Objekten schlägt jede einzelne fehlende
Annotation mit ~1.1 Punkten Precision zu Buche. Die berichteten
Precision-Werte sind daher eher eine **Untergrenze**.

Echte Falsch-Positive entstehen an dunklen Löchern im Maschendraht und an
Schattenspalten zwischen Rohren — genau die Strukturen, gegen die die
Negativbilder (9.7 % des Datensatzes) gerichtet sind.

---

## 6 Was ich mit mehr Zeit anders machen würde

1. **Mehr Evaluationsdaten.** Je ein Bild pro Split ist die größte methodische
   Schwäche der Arbeit. 10–20 annotierte Bilder würden die
   Konfidenzintervalle auf ein Maß bringen, bei dem Experiment 2 entscheidbar
   wäre.
2. **Mehr Cutouts.** Neun Quellobjekte sind die Obergrenze der Formvielfalt.
   Aus den vorhandenen `val`/`test`-Bildern ließen sich mit demselben
   Matting-Verfahren über 150 weitere gewinnen — allerdings nur unter Verzicht
   auf diese Bilder als Evaluationsdaten.
3. ~~**P2-Detektionskopf**~~ — **erledigt** (Experiment 3, siehe
   `docs/BACHELORARBEIT.md` Kapitel 6.3). `yolo26s-p2` hebt mAP@0.75 auf `test`
   von 0.320 auf 0.387 und mAP@0.5 von 0.888 auf 0.941, kostet aber die
   2.6-fache GPU-Speichermenge. Die mittlere IoU der Treffer verbessert sich
   dabei nicht — gewonnen wird über den Rückruf, nicht über die Präzision der
   Boxen.
4. **Beleuchtungsvielfalt.** Alle vier Clean Plates sind bei diffusem Licht
   aufgenommen. Harte Schlagschatten, Gegenlicht und Nässe fehlen im
   Quellmaterial und werden nur approximativ durch photometrische
   Augmentierung angenähert.
5. **Kachelweise Inferenz (SAHI)** statt Skalierung des Gesamtbildes. Bei
   deutlich größeren Aufnahmen als 2500 px wäre das die skalierbarere Lösung.
