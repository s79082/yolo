# PipeNet — Diskussion

Zahlen aus [`ERGEBNISSE.md`](ERGEBNISSE.md), Rohdaten in
`experiments/*.json`. Entwurfsbegründung in
[`ARCHITEKTUR.md`](ARCHITEKTUR.md).

---

## 1 Das Ergebnis

Gleicher synthetischer Datensatz, gleiches Trainingsbudget, gleiche
Metrikimplementierung. Betriebseinstellung je Modell **ausschließlich auf
`val`** bestimmt und unverändert auf `test` angewandt.

| Modell | Par. | Einstellung | test mAP@0.5 | test mAP@0.75 | test mAP@0.5:0.95 | test F1 | test IoU (TP) | Zählung |
|---|---|---|---|---|---|---|---|---|
| `yolo11s` | 9.43 M | imgsz 1920 | 0.9279 | 0.3398 | 0.4482 | 0.9167 | 0.745 | 82/86 |
| `yolo26s` | 9.95 M | imgsz 1600 | 0.9051 | 0.3326 | 0.4270 | 0.8876 | 0.729 | 83/86 |
| `yolo26s-p2` | 9.66 M | imgsz 1920 | **0.9471** | 0.3257 | 0.4492 | **0.9341** | 0.739 | 81/86 |
| **PipeNet** | **3.07 M** | scale 1.8 | 0.9335 | 0.3548 | 0.4531 | 0.9167 | 0.742 | 82/86 |
| **PipeNet +TTA** | **3.07 M** | scale 1.8 +TTA | 0.9220 | **0.3775** | **0.4648** | 0.9102 | **0.756** | 81/86 |

Zusammengefasst:

* **Bei den Lokalisierungsmetriken liegt PipeNet vorn** — mAP@0.75,
  mAP@0.5:0.95 und mittlere IoU der Treffer sind mit TTA die besten Werte aller
  fünf Modelle, mit einem Drittel der Parameter.
* **Bei den Detektionsmetriken liegt `yolo26s-p2` vorn** — mAP@0.5 und F1.
* **PipeNet erreicht `yolo11s` exakt** (F1 0.9167, Zählung 82/86) und schlägt
  `yolo26s` deutlich, bei 3.07 M gegen 9.43 bzw. 9.95 M Parametern.
* Trainingszeit: **0.37 h** gegen 1.13 h (`yolo11s`) und 2.02 h
  (`yolo26s-p2`).

---

## 2 Die Entwurfshypothesen — was sich bestätigt hat und was nicht

### Bestätigt: Stride 4 plus Subpixel-Kopf verbessert die Lokalisierung

Das war die konkrete Vorhersage aus [`ARCHITEKTUR.md`](ARCHITEKTUR.md)
Abschnitt 5: ohne Versatzkopf wäre das Zentrum auf das Stride-4-Gitter
quantisiert, also systematisch bis 2 px falsch, was bei einem 20-px-Objekt
rund 0.2 IoU kostet.

Gemessen auf `test`: mAP@0.75 **0.3548** (ohne TTA) bzw. **0.3775** (mit)
gegen 0.3257–0.3398 bei allen YOLO-Varianten. Mittlere IoU der Treffer
**0.756** gegen 0.729–0.745. Beides die besten Werte im Vergleich.

Bemerkenswert dabei: `yolo26s-p2` hat ebenfalls eine P2-Ebene, erreicht aber
das **niedrigste** mAP@0.75 aller fünf (0.3257). Der Unterschied ist genau der
Versatzkopf — YOLO26 hat `reg_max = 1`, also keine Distribution-Focal-Loss über
Boxbins und damit keinen vergleichbaren Subpixel-Mechanismus. Die feine
Auflösung allein genügt nicht; man braucht auch die Subpixel-Korrektur.

### Nachtrag: die Heatmap-Darstellung war fehlerhaft, die Detektionen nicht

Bei der Sichtung der Heatmap-Overlays fiel auf, dass die Maxima teils leicht
über oder neben den Rohrenden lagen. Die Ursache war **nicht** das Modell,
sondern die Darstellung: `cv2.resize(hm, (W, H))` streckt das /32-Padding mit
und zeichnet Zellen am Blockanfang statt in der Blockmitte — zusammen bis zu
7.7 px nach oben am unteren Bildrand, also 39 % eines 20-px-Objekts.

Der gemessene systematische Zentrumsversatz der **Vorhersagen** gegenüber der
Ground Truth liegt dagegen bei nur −0.4 bis −1.0 px (2–3 % des Durchmessers).
Details, Herleitung der korrekten Abbildung und ein Vorher-Nachher-Bild in
[`ARCHITEKTUR.md`](ARCHITEKTUR.md) Abschnitt 7b. Die Metriken sind vor und nach
der Korrektur bitidentisch.

Der Fall ist als Methodenhinweis wertvoll: bei einem Detektor mit Ausgabe-Stride
> 1 ist die Koordinatenkonvention zwischen Zielerzeugung, Dekodierung und
Darstellung eine echte Fehlerquelle, und ein Darstellungsfehler sieht einem
Modellfehler zum Verwechseln ähnlich. Getrennt werden konnten beide erst durch
die Messung des Versatzes gegen die Ground Truth.

### Bestätigt: Peak-Extraktion funktioniert ohne Schwelle

Die Heatmap-Ausgabe (`predict.py --heatmap`) zeigt kompakte, klar getrennte
Maxima auch bei sich berührenden Rohren. Der gesamte NMS-IoU-Sweep aus
Experiment 1b der Hauptarbeit entfällt, weil der Parameter nicht existiert. Es
gab in keinem Bild einen Fall verschmolzener Zentren — konsistent mit der
Vorhersage (Minimum 0.83 Durchmesser ≈ 4 Zellen, Verschmelzungsgrenze 2 Zellen).

### Bestätigt: weniger Kapazität ist hier kein Nachteil

3.07 M Parameter gegen 9.4–10.0 M, und bei drei von sechs Metrikspalten auf
`test` der beste Wert. Die Überlegung aus der Datensichtung — 9 Quellobjekte,
eine Klasse, überschüssige Kapazität bedeutet Auswendiglernen — trägt.

### Nicht bestätigt: „Objektgröße auf den Trainingsmedian schieben"

Im YOLO-Teil hatte ich als Regel formuliert: so skalieren, dass die
Objektgröße im Netzeingang etwa dem Trainingsmedian (22.5 px) entspricht.
Für PipeNet ist das **falsch**. Das Optimum liegt bei `scale` 1.5–1.8, also
bei 41–49 px auf `val` — weit über dem Median:

| scale | Objekt val (px) | val mAP@0.5 | test mAP@0.5 |
|---|---|---|---|
| 0.5 | 13.7 | 0.867 | 0.728 |
| 0.7 | 19.2 | 0.895 | 0.831 |
| 0.85 | 23.3 | 0.904 | 0.852 |
| 1.0 | 27.4 | 0.870 | 0.772 |
| 1.2 | 32.9 | 0.898 | 0.867 |
| **1.5** | **41.1** | **0.919** | **0.917** |
| 1.8 | 49.3 | 0.901 | 0.898 |

Die bessere Erklärung: entscheidend ist nicht die Nähe zum Trainingsmedian,
sondern **wie viele Ausgabezellen ein Objekt belegt**. Bei Stride 4 sind das
bei 22 px 5.5 Zellen, bei 41 px aber 10.3 — und je mehr Zellen, desto
genauer lassen sich Peak und Größe bestimmen. Nach oben begrenzt wird das
durch die augmentierte Trainingsspanne: die Skalierung 0.6–1.4 dehnt die
Generatorspanne von 12–44 px auf effektiv 7–62 px, und 49 px liegt noch
darin, 60 px kaum mehr. Die Kurve fällt bei 1.8 entsprechend wieder ab.

Diese Korrektur gilt auch rückblickend für den YOLO-Teil: dort war das Optimum
ebenfalls **oberhalb** des nativen Maßstabs (imgsz 1920–2464 statt 1600 für
`test`), was ich damals über die Nähe zum Trainingsmedian erklärt habe. Die
Zellzahl-Erklärung passt auf beide Beobachtungen und ist die tragfähigere.

### Teilweise: die Rotations-TTA

Über die Diedergruppe D4 gemittelt, auf `test`:

| | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | F1 | IoU (TP) |
|---|---|---|---|---|---|
| ohne TTA | **0.9335** | 0.3548 | 0.4531 | **0.9167** | 0.742 |
| mit TTA | 0.9220 | **0.3775** | **0.4648** | 0.9102 | **0.756** |

Die TTA verbessert die **Lokalisierung** (mAP@0.75 +2.3, mAP@0.5:0.95 +1.2,
IoU +1.4 Punkte) und verschlechtert die **Detektion** (mAP@0.5 −1.2, F1 −0.7).

Mechanistisch plausibel: gemittelt werden acht Antworten, die nicht exakt
deckungsgleich sind (Faltungen sind nicht rotationsequivariant — der
Selbsttest zeigt Korrelationen von nur 0.35–0.47). Die Mittelung glättet die
Größen- und Positionsschätzung, verbreitert aber die Heatmap-Maxima; schwache
Peaks fallen dadurch unter die Schwelle. Wer zählen will, lässt die TTA aus;
wer Boxgeometrie braucht, schaltet sie ein. Der Standard in
`docs/betriebspunkt.json` ist deshalb **ohne** TTA.

---

## 3 Warum `yolo26s-p2` bei mAP@0.5 trotzdem vorn liegt

Zwei Gründe, beide benennbar:

1. **Vortraining.** Bei PipeNet sind nur die 2.78 M Encoder-Parameter
   ImageNet-vortrainiert; Dekoder und Köpfe (0.29 M) starten zufällig. YOLO
   startet mit **vollständig COCO-vortrainierten Detektionsköpfen**, also mit
   Gewichten, die schon gelernt haben, Objekte von Hintergrund zu trennen. Das
   ist der klarste verbleibende Hebel für PipeNet und kein Architekturvorteil
   von YOLO.
2. **Die Pyramide hilft an den Größenrändern.** PipeNet hat eine
   Detektionsebene; sehr kleine (12 px) und sehr große (60 px) Objekte
   erreichen dort 3 bzw. 15 Zellen. YOLO verteilt diese Fälle auf mehrere
   Ebenen mit jeweils passender Auflösung. Das erklärt den Recall-Unterschied
   (0.907 gegen 0.895).

---

## 4 Ein Befund über beide Architekturen hinweg

**Beide PipeNet-Läufe erreichen ihr bestes Modell in Epoche 6** — der Lauf mit
60-Epochen-Kosinusrampe (Early Stopping bei 26) genauso wie der mit
30-Epochen-Rampe (Early Stopping bei 26, Fitness 0.601 gegen 0.592). Die
abgeschlossene Lernratenrampe ändert praktisch nichts.

Dasselbe Muster bei YOLO: bestes Modell in Epoche 10 (`yolo11s`), 5
(`yolo26s`), 22 (`yolo26s-p2`) — immer weit vor dem Ende des Budgets.

Das ist ein **architekturunabhängiger** Befund: die Grenze ist nicht die
Optimierung, sondern der Sim-to-Real-Abstand. Nach wenigen Epochen hat das
Netz alles gelernt, was die synthetischen Daten über die reale Domäne
aussagen; danach verbessert es nur noch seine Anpassung an die synthetische
Verteilung. Für die Arbeit folgt daraus: **mehr Trainingsaufwand ist der
falsche Hebel; besseres Quellmaterial ist der richtige.**

---

## 5 Was ich empfehlen würde

| Anwendungsfall | Modell | Warum |
|---|---|---|
| Rohre zählen (F1, Recall) | `yolo26s-p2` | höchstes F1 (0.9341) und Recall |
| Boxgeometrie wird weiterverarbeitet | **PipeNet +TTA** | beste mAP@0.75, mAP@0.5:0.95, IoU |
| beschränkte Hardware, Edge | **PipeNet** | 3.07 M Parameter, 0.37 h Training, keine NMS |
| geringster Wartungsaufwand | `yolo26s-p2` | Standardwerkzeug, kein eigener Code |

---

## 6 Grenzen dieser Gegenüberstellung

1. **Ein Bild pro Split.** Das Bootstrap-Konfidenzintervall für mAP@0.5 liegt
   bei ±0.04–0.05 (siehe Hauptarbeit Kapitel 5.5). **Alle** Unterschiede in
   Abschnitt 1 liegen darin. Interpretierbar ist nicht die einzelne Differenz,
   sondern das durchgängige Muster: PipeNet vorn bei allen drei
   Lokalisierungsmetriken, `yolo26s-p2` vorn bei beiden Detektionsmetriken.
   Dass sich diese Trennung so klar an den Entwurfsentscheidungen festmachen
   lässt, macht sie plausibler als Rauschen — beweist sie aber nicht.
2. **Vortraining ist nicht kontrolliert.** YOLO bringt COCO-vortrainierte
   Detektionsköpfe mit, PipeNet nicht. Ein fairer Test wäre PipeNet mit einem
   auf COCO vortrainierten Kopf — dafür gibt es keine fertigen Gewichte.
3. **Die Betriebseinstellung wurde je Modell auf `val` optimiert.** Das ist
   methodisch korrekt (keine Selektion auf `test`), nutzt aber je Modell einen
   Freiheitsgrad, dessen Wahl auf 88 Objekten verrauscht ist.
4. **PipeNet ist nicht auskonvergiert im Sinne des Budgets**, sondern durch
   Early Stopping bei Epoche 26 beendet — wie die YOLO-Läufe. Siehe
   Abschnitt 4.
5. **Kein Ellipsenkopf.** Für die ovalisierten Profile wäre eine
   Ellipsenparametrisierung (Zentrum, zwei Halbachsen, Winkel) natürlicher.
   Nicht umgesetzt, weil die Ground Truth achsparallele Boxen sind und ein
   Winkel damit nicht überwachbar ist.
