# PipeNet — ein auf das Problem zugeschnittenes CNN

Eigenentwurf als Gegenstück zum YOLO-Ansatz. Trainiert auf **exakt demselben**
synthetischen Datensatz (`../synth/train`, 6000 Bilder, 136 384 Labels) mit
demselben Protokoll, damit der Vergleich ein Architekturvergleich bleibt und
keine Datenfrage.

---

## 1 Was am Problem besonders ist

Alle Zahlen gemessen, siehe `../../docs/DATENSICHTUNG.md`:

| Eigenschaft | Messwert |
|---|---|
| Klassen | **1** |
| zugrunde liegende Objektinstanzen | **9** Cutouts |
| Objektdurchmesser | 12–44 px (Median real 19.6–27.4 px) |
| Größenspanne | Faktor 3.7 |
| Abstand zum nächsten Nachbarn | p10 0.84, Median 1.09–1.24, p90 1.78–2.51 Durchmesser |
| Objekte, die ihren Nachbarn berühren | 46–61 % |
| Objekte je Bild | 86–88 (real), bis 74 je synthetischem Bild |
| Objektform | Kreisring, in Schrägansicht Ellipse — nahezu rotationssymmetrisch |
| eigentliches Ziel | **zählen** |

Ein universeller Detektor wie YOLO ist für 80 Klassen, Objektgrößen über zwei
Größenordnungen und beliebige Seitenverhältnisse gebaut. Nichts davon liegt
hier vor. Die folgenden fünf Entscheidungen nutzen das aus.

---

## 2 Entscheidung 1 — eine Ausgabeebene bei Stride 4

Die Kernrechnung. Ein Objekt von 22 px im Netzeingang belegt auf einer
Merkmalskarte mit Stride *s* genau 22/*s* Zellen, und der nächste Nachbar sitzt
(bei Median-Abstand ≈ 1.15 Durchmessern) etwa ebenso weit entfernt:

| Ebene | Stride | Gitter bei 1920 px | Objekt in Zellen | Nachbarzentrum in Zellen |
|---|---|---|---|---|
| **P2** | 4 | 480×480 | **5.5** | **5.5** |
| P3 | 8 | 240×240 | 2.8 | 2.8 |
| P4 | 16 | 120×120 | 1.4 | 1.4 |
| P5 | 32 | 60×60 | 0.7 | 0.7 |

P4 und P5 können ein 22-px-Objekt nicht mehr auflösen — es belegt dort
weniger als zwei bzw. weniger als eine Zelle. YOLO11/YOLO26 detektieren
standardmäßig auf P3/P4/P5, verwalten also drei Ebenen, von denen zwei für
diese Aufgabe nutzlos sind. Genau deshalb war der Zugewinn durch
`yolo26s-p2` (Experiment 3 der Hauptarbeit) so deutlich.

**PipeNet zieht die Konsequenz und hat nur P2.** Bei einer Größenspanne von
Faktor 3.7 braucht es keine Pyramide; die Größenvariation deckt der
Regressionskopf ab.

Der Encoder liefert weiterhin Merkmale bis Stride 16 — nur *detektiert* wird
ausschließlich auf Stride 4. Kontext und Detektionsauflösung sind zwei
verschiedene Dinge.

---

## 3 Entscheidung 2 — Encoder nach layer3 abgeschnitten

ResNet18 hat 11.69 M Parameter, davon **8.39 M allein in layer4**
(Stride 32). Der rezeptive Bereich am Ende von layer3 ist bereits deutlich
größer als jedes Objekt samt Umgebung.

| Variante | Encoder-Parameter |
|---|---|
| ResNet18 vollständig (ohne fc) | 11.18 M |
| **ResNet18 bis layer3** | **2.78 M** |

Das ist hier kein Sparzwang, sondern erwünscht. Der Datensatz enthält **9**
zugrunde liegende Objektinstanzen; jede erscheint rund 15 000-mal.
Überschüssige Kapazität bedeutet in dieser Lage: Auswendiglernen der neun
synthetischen Texturen. Dieselbe Überlegung stand schon hinter der Wahl von
`yolo11s` statt `yolo11m`.

**Gesamtmodell: 3.07 M Parameter** gegen 9.43 M (`yolo11s`), 9.95 M
(`yolo26s`), 9.66 M (`yolo26s-p2`).

---

## 4 Entscheidung 3 — Zentrums-Heatmap statt Box-Regression mit NMS

Der entscheidende Punkt und der Grund, warum eine Eigenentwicklung hier
überhaupt lohnt.

**Beobachtung:** Die Boxen benachbarter Objekte überlappen deutlich (10.
Perzentil des Abstands: 0.84 Durchmesser), ihre **Zentren** sind aber immer
getrennt (Minimum 0.83 Durchmesser ≈ 16 px ≈ 4 Zellen bei Stride 4).

Daraus folgt:

| | Box-NMS (YOLO) | Peak-Extraktion (PipeNet) |
|---|---|---|
| Entscheidungsgrundlage | IoU zweier Boxen | lokales Maximum der Heatmap |
| freier Parameter | IoU-Schwelle | **keiner** |
| Fehlermodus | Schwelle zu tief → Nachbar verloren; zu hoch → Dublette | zwei Zentren < 2 Zellen → verschmelzen |
| tritt in diesen Daten auf? | ja, unvermeidbar | nein (Minimum 4 Zellen) |

Ein 3×3-MaxPool markiert genau die Zellen, die Maximum ihrer
8er-Nachbarschaft sind — **ein Maximum = ein Rohr**. Der gesamte
NMS-IoU-Sweep aus Experiment 1b der Hauptarbeit entfällt, weil der Parameter
nicht existiert.

Belegt wird das durch die vorhergesagte Heatmap selbst
(`predict.py --heatmap`): kompakte, klar getrennte Maxima auch bei sich
berührenden Rohren.

---

## 5 Entscheidung 4 — drei Köpfe: Heatmap, log-Größe, Subpixel-Versatz

Ausgabe je Zelle des Stride-4-Gitters:

| Kopf | Kanäle | Inhalt | Verlust |
|---|---|---|---|
| `hm` | 1 | Zentrumswahrscheinlichkeit (Logit) | Gaussian Focal Loss (α=2, β=4) |
| `wh` | 2 | log(Breite in px), log(Höhe in px) | L1, nur an echten Zentren |
| `off` | 2 | Subpixel-Versatz in Zelleinheiten [0,1) | L1, nur an echten Zentren |

**Warum der Versatzkopf zwingend ist.** Ohne ihn wäre das Zentrum auf das
Stride-4-Gitter quantisiert, also systematisch bis zu 2 px falsch. Bei einem
20-px-Objekt kostet ein Boxfehler von 2 px bereits rund 0.2 IoU — genau die
Größenordnung, in der `mAP@0.75` auf `test` zusammenbricht (im YOLO-Teil
0.320). Der Kopf holt diese Genauigkeit zurück.

**Warum die Größe logarithmisch.** Bei einer Spanne von 12–44 px bestraft ein
L1-Verlust auf log-Größen den *relativen* Fehler. In Pixeleinheiten wären
2 px Fehler bei einem 44-px-Objekt belanglos und bei einem 12-px-Objekt
gravierend — der Verlust würde beides gleich behandeln.

**Warum der Focal Loss.** In einem 256×256-Gitter mit ~23 Objekten sind
65 513 von 65 536 Zellen negativ (99.96 %). Reines BCE wird vollständig von
den Negativen dominiert. Die beiden Exponenten wirken dagegen:
α dämpft leichte Beispiele, **β dämpft Negative in der Nähe eines Zentrums**,
wo die gauss-verschmierte Zielheatmap ohnehin nahe 1 liegt. Ohne β würde das
Netz dafür bestraft, den Nachbarpixel eines Zentrums ebenfalls hoch zu
bewerten — genau die Toleranz, die man bei 5.5 Zellen großen Objekten
braucht.

**Warum die Bias-Initialisierung −4.595.** Das entspricht logit(0.01): zu
Beginn sagt das Netz überall „kein Objekt" voraus. Ohne diesen Prior
dominieren die 65 513 negativen Zellen die ersten Iterationen vollständig.

### Breite der Gauss-Glocke

Nicht frei gewählt, sondern über die CenterNet-Radiusformel (CornerNet, Gl. 6)
aus einer geometrischen Toleranz abgeleitet: `min_overlap = 0.7` bedeutet, eine
um *r* verschobene Box hätte noch 0.7 IoU mit der Originalbox. Ergebnis für die
vorliegenden Größen:

| Objektgröße | Zellen (Stride 4) | Radius | σ |
|---|---|---|---|
| 12 px | 3.0 | 0.82 | 0.60 |
| 20 px | 5.0 | 1.37 | 0.62 |
| 27 px | 6.8 | 1.84 | 0.78 |
| 36 px | 9.0 | 2.46 | 0.99 |
| 44 px | 11.0 | 3.01 | 1.17 |

σ liegt also bei 0.60–1.17 Zellen, der Nachbarabstand bei 4.4–5.5 Zellen.
**Benachbarte Glocken überlappen damit nicht** — die Voraussetzung dafür, dass
die Peak-Extraktion sie trennen kann. Eingezeichnet wird per `np.maximum`, nicht
additiv: bei sich berührenden Objekten würde eine Summe Werte > 1 zwischen
zwei echten Zentren erzeugen und damit ein künstliches drittes Zentrum.

---

## 6 Entscheidung 5 — Rotations-TTA über die Diedergruppe D4

Rohrenden sind nahezu rotationssymmetrisch. Drehungen um 90° und Spiegelungen
sind auf einem Pixelgitter **exakt** — reine Indexpermutationen ohne
Interpolation, im Unterschied zur üblichen Mehrskalen-TTA. Gemittelt wird über
alle 8 Elemente von D4 (4 Drehungen × 2 Spiegelungen).

Details:

* Die Heatmap wird gemittelt.
* Die Größenkanäle werden gemittelt, wobei bei 90°/270° **Breite und Höhe
  getauscht** werden müssen.
* Der Versatzkopf wird **nur aus dem untransformierten Durchlauf** gelesen.
  Der Versatz ist ein Vektor in Zelleinheiten; seine korrekte
  Rücktransformation führte zusätzliche Vorzeichenfälle ein. Der Nutzen der
  TTA liegt im Finden (Heatmap), nicht im Nachjustieren um Bruchteile einer
  Zelle. Diese Einschränkung ist bewusst und dokumentiert.

**Selbsttest.** `decode.selftest_equivariance` dreht ein Bild, lässt es durch
das Netz laufen, dreht die Heatmap zurück und korreliert sie mit dem
untransformierten Durchlauf. Ein Vorzeichen- oder Achsenfehler im Code würde
eine Korrelation um 0 ergeben. Gemessen steigt sie mit dem Training
(untrainiert 0.13–0.27, nach wenigen Epochen 0.39–0.59) — das Netz lernt
zunehmend rotationsinvariante Merkmale, und die Rücktransformation ist korrekt.
Zweiter, unabhängiger Beleg: die TTA verbessert die Metrik konsistent; eine
falsche Transformation würde sie verschlechtern.

---

## 7 Ergänzend — Skip von der Stride-2-Ebene

Das diskriminative Merkmal ist der **1–3 px breite helle Rohrkranz**. Die
Stride-2-Merkmale des ResNet-Stems haben ihn noch, `layer1` (nach MaxPool) nur
abgeschwächt. Ein 3×3-Faltung mit Stride 2 bringt den Stem-Ausgang auf Stride 4
und wird dort addiert. Kosten: 37 k Parameter.

---

## 7b Koordinatenkonvention — und ein Darstellungsfehler, der daraus entstand

Bei einem Ausgabe-Stride von 4 muss festgelegt werden, **welchem Eingangspixel
eine Ausgabezelle entspricht**. Zwei Konventionen sind gängig:

| Konvention | Zelle *j* entspricht | verwendet in |
|---|---|---|
| Blockanfang | Pixel `4j` | Zielerzeugung und Dekodierung von PipeNet |
| Blockmitte | Pixel `4j + 2` | üblich bei der Darstellung |

PipeNet verwendet durchgängig den **Blockanfang**: die Zielerzeugung rechnet
`ix = int(cx/4)`, `off = cx/4 − ix`, die Dekodierung `cx = (ix + off)·4`. Beide
Seiten sind konsistent, der Versatzkopf lernt genau die fehlenden 0…4 px. Belegt
wird das durch den gemessenen systematischen Zentrumsversatz der Vorhersagen
gegenüber der Ground Truth:

| Split | scale | Δx | Δy | relativ zum Durchmesser |
|---|---|---|---|---|
| `val` | 1.0 | −0.74 ± 1.14 px | −0.98 ± 0.95 px | −2.8 % / −3.4 % |
| `val` | 1.8 | −0.39 ± 1.06 px | −0.73 ± 0.96 px | −1.5 % / −2.6 % |
| `test` | 1.0 | −0.55 ± 0.83 px | +0.01 ± 0.82 px | −2.8 % / −0.2 % |
| `test` | 1.8 | −0.49 ± 0.85 px | +0.06 ± 0.81 px | −2.5 % / +0.2 % |

Unter einem Pixel systematischer Versatz — die **Detektionen sind korrekt
zentriert**.

### Der Fehler saß in der Heatmap-Darstellung

Das Overlay von `predict.py --heatmap` war ursprünglich ein
`cv2.resize(hm, (W, H))`. Das ist aus **zwei** unabhängigen Gründen falsch:

1. **Das Padding wird mitgestreckt.** Die Vorhersage läuft auf dem auf ein
   Vielfaches von 32 gepaddeten, skalierten Bild; die Heatmap ist also größer
   als der Bildinhalt. Für `test5.jpg` bei `scale` 1.8: 1600×900 → skaliert
   2880×1620 → gepaddet 2880×1632, Heatmap 720×408 Zellen, davon nur 405 Zeilen
   Inhalt. Streckt man alle 408 auf 900 px, entsteht ein linear zum Bildunterrand
   wachsender Versatz von bis zu **6.6 px nach oben**.
2. **Der Zellbezugspunkt.** Wird Zelle *j* am Blockanfang statt in der
   Blockmitte gezeichnet, erscheint das Maximum um `2/scale` px nach oben-links
   verschoben — hier **1.1 px**.

Summe am unteren Bildrand: **bis zu 7.7 px nach oben**, bei einem 20-px-Objekt
also **39 % des Durchmessers**.

Wie groß der Fehler ausfällt und **in welche Richtung**, hängt vom
Seitenverhältnis ab — je nachdem, welche Achse mehr Padding braucht:

| Bild | Größe | Padding (h×b) | Versatz y | Versatz x |
|---|---|---|---|---|
| `test5.jpg` | 1600×900 | 12×0 px | **6.6 px** | 0.0 px |
| `PXL_20251204_073853552.jpg` | 2459×1174 | 31×22 px | **17.0 px** | **12.2 px** |
| `test4` | 2689×2879 | 2×24 px | 1.1 px | **13.3 px** |
| `PXL_20251204_073837682` | 4080×3072 | 6×16 px | 3.3 px | **8.9 px** |
| `test` | 1756×959 | 2×7 px | 1.1 px | 3.9 px |

(jeweils bei `scale` 1.8, zuzüglich 1.1 px aus dem Zellbezugspunkt)

Das erklärt beide beobachteten Erscheinungsformen: bei `test5.jpg` sitzt das
Maximum **über** dem Rohrende, bei `test4` **daneben**, beim val-Bild beides
gleichzeitig und mit über 20 px am stärksten — dort mehr als eine ganze
Objektbreite.

![Heatmap-Ausrichtung](fig_heatmap_ausrichtung.png)

*Weiße Kreuze = dekodierte Zentren. Links das ursprüngliche `cv2.resize`, rechts
die korrigierte Abbildung.*

**Korrektur** (`decode.heatmap_overlay`): die Blockmitte einer Zelle liegt in
Originalkoordinaten bei `p = (4c + 2)/scale`, umgestellt `c = p·scale/4 − 0.5`.
Das ist eine affine Abbildung; `cv2.warpAffine` mit `WARP_INVERSE_MAP` wendet
sie in genau dieser Richtung an und tastet das Padding nie an.

**Funktional war nichts betroffen.** Die Metriken sind vor und nach der
Korrektur bitidentisch, weil die Boxen aus `decode` stammen und nicht aus dem
Overlay. Zusätzlich wurde geprüft, ob der gepaddete Rand Scheinmaxima erzeugt:
bei `val` (31×22 px Padding) und `test` (12×0 px) **null** Maxima über 0.3.
Die Padding-Maske in `decode.peaks` ist deshalb reine Absicherung für andere
Bildformate.

---

## 8 Aufbau im Überblick

![PipeNet Architektur](fig_architektur.png)

Erzeugt von `cnn/src/fig_architektur.py`; alle Parameterzahlen werden dabei aus
dem Modell ausgelesen, nicht abgeschrieben (`--print` zeigt sie einzeln).
Vektorfassung fuer Satz und Druck: `fig_architektur.svg`.

Als Textdiagramm:

```
Eingang  H x W x 3   (native Bildgroesse, nur auf /32 gepaddet)
   │
   ├─ stem  conv7x7 s2 ─────────────────────┐  Stride 2,  64 Kanaele
   │                                        │
   ├─ maxpool s2 → layer1 ── Stride 4,  64 ─┤
   ├─ layer2 ──────────────── Stride 8, 128 ─┤   Top-Down-Dekoder
   └─ layer3 ──────────────── Stride 16,256 ─┘   (FPN-artig, 64 Kanaele)
                                            │
                          Stride 4, 64 Kanaele
                                            │
                    ┌───────────────┬───────┴───────┐
                   hm (1)         wh (2)         off (2)
              Heatmap        log-Groesse     Subpixel-Versatz
                    │
            3x3-MaxPool-Peaks  →  eine Detektion je lokalem Maximum
```

**layer4 (Stride 32, 8.39 M Parameter) wird nicht verwendet.**

---

## 9 Gegenüberstellung der Entwurfsentscheidungen

| Problem-Eigenschaft (gemessen) | YOLO | PipeNet | Warum |
|---|---|---|---|
| 1 Klasse, 9 Quellobjekte | Klassifikationskopf, 9.4–10.0 M Par. | kein Klassifikationskopf, 3.07 M Par. | weniger Kapazität = weniger Auswendiglernen |
| Größenspanne nur Faktor 3.7 | 3–4 Pyramidenebenen | **eine** Ebene, Stride 4 | P4/P5 lösen 22-px-Objekte nicht auf |
| Nachbarabstand 1.09–1.24 Ø | Stride 8 → 2.8 Zellen | Stride 4 → 5.5 Zellen | Nachbarn bleiben trennbar |
| dicht und berührend, Zentren getrennt | Box-NMS mit IoU-Schwelle | Peak-Extraktion | keine Schwelle nötig |
| nahezu rotationssymmetrisch | Box-Regression, Flip/Skalen-TTA | Zentrum + 2 Radien, exakte D4-TTA | passt zur Objektsymmetrie |
| Ziel ist Zählen | conf-Schwelle auf Boxen | Schwelle auf einem 1-Kanal-Feld | direkter, interpretierbar |
| hochauflösende Eingangsbilder | fixes `imgsz` + Letterbox | vollständig konvolutional, native Größe | ein Hyperparameter weniger |

---

## 10 Bekannte Grenzen des Entwurfs

1. **Zwei Zentren unter 2 Zellen (8 px) Abstand verschmelzen** zu einem
   Maximum. In den vorliegenden Daten tritt das nicht auf (Minimum 4 Zellen),
   bei Aufnahmen aus größerer Entfernung aber sehr wohl. Gegenmittel wäre
   Stride 2 statt 4 — vierfacher Speicherbedarf der Ausgabekarten.
2. **Kein Vortraining für Dekoder und Köpfe.** Nur der Encoder startet mit
   ImageNet-Gewichten; 0.29 M Parameter sind zufällig initialisiert. YOLO
   startet mit vollständig COCO-vortrainierten Detektionsköpfen. Das ist ein
   Nachteil, der im Vergleich mitzudenken ist.
3. **Nur achsparallele Boxen.** Für ovalisierte Profile in Schrägansicht wäre
   eine Ellipsenparametrisierung (Zentrum, 2 Halbachsen, Winkel) die
   natürlichere Ausgabe. Nicht umgesetzt, weil die Ground Truth in val/test
   achsparallele Boxen sind und ein Winkel damit nicht überwachbar ist.
4. **Der Versatzkopf wird bei der TTA nicht mitgemittelt** (siehe Abschnitt 6).
5. **Ein Datenpunkt für die Bewertung.** Wie im Hauptteil: je ein reales Bild
   für Validierung und Test. Alle Unterschiede sind gegen das
   Bootstrap-Konfidenzintervall von ±0.04–0.05 zu lesen.
