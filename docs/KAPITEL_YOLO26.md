# Wahl der Detektorarchitektur: YOLO11, YOLO26 und die P2-Erweiterung

*Ausformulierter Kapitelteil zur Übernahme in die Bachelorarbeit. Alle
Zahlenwerte sind gemessen; die erzeugenden Skripte sind jeweils genannt.*

---

## 1 Ausgangslage und Anlass für den Vergleich

Als Detektor wurde zunächst **YOLO11 in der Größenvariante `s`** gewählt. Diese
Entscheidung setzt sich aus zwei unabhängigen Teilentscheidungen zusammen, die
unterschiedlich gut begründet sind und deshalb getrennt betrachtet werden
müssen.

Die **Größenvariante** folgt unmittelbar aus der Datenlage. Der Datensatz
umfasst genau eine Objektklasse und geht auf lediglich neun freigestellte
Objektinstanzen zurück; die Formvielfalt ist entsprechend gering und die
erforderliche Modellkapazität damit ebenfalls. Die Variante `n` verliert bei
Objektgrößen von 12 bis 40 Pixeln messbar Rückruf, während die Varianten `m`
und `l` bei nur neun Quellobjekten Gefahr laufen, deren konkrete Texturmuster
zu memorieren. Die Wahl von `s` mit 9,4 Millionen Parametern ist damit sachlich
belegt. Ebenso begründet ist die Verwendung COCO-vortrainierter Gewichte: die
vortrainierten Filter der frühen Schichten reagieren auf Kanten, Ringe und
Texturen und sind damit domänenunabhängig. Sie reduzieren gerade jenen Anteil
des Modells, der andernfalls auf die Artefakte der synthetischen Bilderzeugung
überangepasst würde.

Die **Modellfamilie** hingegen wurde konservativ gewählt, aber nicht belegt.
Die Argumente dafür waren die Reife der Familie, stabile und gut dokumentierte
Gewichte sowie eine bekannte Kopfstruktur, auf der die gesamte
Kleinobjekt-Argumentation dieser Arbeit aufbaut. Eine Überprüfung der
eingesetzten Bibliothek zeigt jedoch, dass diese Wahl nicht dem Stand der
Bibliothek entspricht: In Ultralytics 8.4.117 ist **YOLO26 die
Standardfamilie** — die interne AMP-Prüfung lädt `yolo26n.pt`, und die Funktion
`check_suffix()` verwendet denselben Vorgabewert. Konfigurationen für YOLO12
und YOLO26 liegen der Bibliothek bei, die zugehörigen Gewichte sind verfügbar.

Da die Modellfamilie damit die schwächst begründete Einzelentscheidung der
Arbeit darstellt, wird sie im Folgenden experimentell überprüft.

---

## 2 Architektonische Unterschiede

### 2.1 Merkmalspyramide und die Bedeutung von „P2“

Ein Faltungsnetz verkleinert das Eingangsbild stufenweise und gewinnt dabei an
semantischem Gehalt, während es an räumlicher Auflösung verliert. Detektoren
sagen deshalb nicht auf einer, sondern auf mehreren dieser Stufen gleichzeitig
vorher: grobe Stufen für große Objekte, feine für kleine. Die Stufen werden
nach der Konvention des Feature Pyramid Network mit **P***ᵢ* bezeichnet, wobei
der Index den Zweierexponenten der Verkleinerung angibt:

| Stufe | Schrittweite (Stride) | Auflösung relativ zum Eingang |
|---|---|---|
| P2 | 4 | 1/4 |
| P3 | 8 | 1/8 |
| P4 | 16 | 1/16 |
| P5 | 32 | 1/32 |

Die Standardkonfigurationen von YOLO11 und YOLO26 detektieren auf **P3, P4 und
P5**; die feinste Stufe hat also die Schrittweite 8. Die Bezeichnung `-p2`
kennzeichnet eine Variante, bei der eine **zusätzliche Stufe mit der
Schrittweite 4** eingeführt wird. Dazu genügt nicht das Anfügen eines weiteren
Detektionskopfes: das Neck muss um eine weitere Aufwärtsabtastung erweitert und
mit der entsprechenden Backbone-Stufe fusioniert werden.

### 2.2 Warum diese Stufe für die vorliegende Aufgabe relevant ist

Die Relevanz ergibt sich unmittelbar aus der Objektgröße. Die annotierten
Stirnflächen messen im Validierungsbild median 27,4 Pixel und im Testbild
median 19,6 Pixel. Bei der eingesetzten Netzeingangsgröße von 1920 Pixeln
entspricht das etwa 22 Pixeln im Netzeingang. Daraus ergeben sich die folgenden
Gitterauflösungen (gemessen an den trainierten Modellen):

| Stufe | Gitter | Vorhersagepositionen | belegte Zellen eines 22-px-Objekts |
|---|---|---|---|
| **P2** (nur in `-p2`) | 480 × 480 | 230 400 | **5,5** |
| P3 | 240 × 240 | 57 600 | 2,8 |
| P4 | 120 × 120 | 14 400 | 1,4 |
| P5 | 60 × 60 | 3 600 | 0,7 |

Ohne die P2-Stufe muss ein Rohrende auf einem Gitter erkannt und lokalisiert
werden, auf dem es weniger als drei Zellen überdeckt; die Stufen P4 und P5
sind bei dieser Objektgröße praktisch wirkungslos. Mit der P2-Stufe stehen
5,5 Zellen zur Verfügung.

Es handelt sich damit um denselben geometrischen Zusammenhang, der bereits die
starke Abhängigkeit von der Inferenzauflösung erklärt, lediglich von der
anderen Seite betrachtet: Anstatt das Bild zu vergrößern, wird das
Vorhersagegitter verfeinert.

Der Preis ist erheblich. Die P2-Stufe allein umfasst mit 230 400 Positionen
etwa das Dreifache der Stufen P3, P4 und P5 zusammen (75 600); die Gesamtzahl
der Vorhersagepositionen steigt auf 306 000. Da dieser Aufwand nicht in den
Gewichten, sondern in den Aktivierungen anfällt, bleibt die Parameterzahl
nahezu unverändert, während Speicherbedarf und Rechenzeit deutlich steigen.

### 2.3 Unterschiede zwischen YOLO11 und YOLO26

| Modell | Parameter | Strides | `reg_max` | Nachbearbeitung |
|---|---|---|---|---|
| `yolo11s` | 9,43 M | 8 / 16 / 32 | 16 | Box-NMS |
| `yolo26s` | 9,95 M | 8 / 16 / 32 | 1 | **keine** (end-to-end) |
| `yolo26s-p2` | 9,66 M | **4** / 8 / 16 / 32 | 1 | **keine** (end-to-end) |

Zwei Eigenschaften von YOLO26 sind für diese Arbeit von Bedeutung.

**YOLO26 arbeitet end-to-end und benötigt keine Non-Maximum Suppression.** Der
Kopf gibt eine bereits eindeutige Zuordnung aus. Nachgewiesen wurde dies durch
Variation der NMS-Schwelle über den gesamten sinnvollen Bereich: bei den Werten
0,10, 0,70 und 0,99 ergibt sich auf dem Testbild jeweils exakt
mAP@0,5 = 0,947066 bei identischer Detektionszahl (323). Der Parameter ist
somit nachweislich wirkungslos. Die im Rahmen der Inferenzparameter-Untersuchung
durchgeführte Variation der NMS-Schwelle ist für YOLO26 daher nicht anwendbar.
Für dicht gepackte Objekte ist der Entfall dieser Stufe grundsätzlich
vorteilhaft, da eine Fehlerquelle und ein zu wählender Parameter entfallen.

**YOLO26 verwendet `reg_max = 1` statt 16.** Damit entfällt die
Distribution-Focal-Loss-Verteilung über diskrete Box-Stützstellen; die
Boxkanten werden direkt regressiert. Bei welcher Objektgröße das vorteilhaft
oder nachteilig ist, war vorab offen und wird in Abschnitt 5.3 diskutiert.

Bemerkenswert ist, dass **alle drei Modelle P3 als feinste Standardstufe
besitzen**. Die Aussagen zur Auflösungsabhängigkeit gelten damit
modellunabhängig; der zentrale methodische Befund dieser Arbeit wird durch die
Modellwahl nicht berührt.

---

## 3 Versuchsaufbau

Verglichen werden drei Modelle bei **identischem Budget**: derselbe
Trainingsdatensatz (6 000 synthetische Bilder), `imgsz = 1024`, `batch = 16`,
maximal 60 Epochen, `patience = 20`, `seed = 0` sowie identische Optimierungs-
und Augmentierungsparameter. Die Auswertung erfolgt mit demselben
Evaluationsframework und denselben Metrikfunktionen.

Die Variante `yolo26s-p2` verwendet die der Bibliothek beiliegende
Konfiguration `yolo26-p2.yaml` in der Größe `s`. Die formkompatiblen Schichten
wurden aus `yolo26s.pt` übernommen (696 von 708 Tensoren); der zusätzliche
P2-Kopf beginnt zufällig initialisiert. Dieser Umstand ist bei der
Interpretation zu berücksichtigen: das Modell startet mit einem geringeren
Anteil vortrainierter Gewichte als die beiden Vergleichsmodelle.

Die Inferenzauflösung wurde für jedes Modell **ausschließlich auf dem
Validierungsbild** bestimmt und anschließend unverändert auf das Testbild
angewandt. Eine modellspezifische Optimierung auf dem Testbild hätte dessen
Aussagekraft aufgehoben.

| Modell | Epochen gelaufen | bestes Modell | Trainingszeit | GPU-Speicher |
|---|---|---|---|---|
| `yolo11s` | 30 (Abbruch durch Early Stopping) | Epoche 10 | 1,13 h | 11,6 GB |
| `yolo26s` | 25 (Abbruch durch Early Stopping) | Epoche 5 | 0,96 h | 14,5 GB |
| `yolo26s-p2` | 42 (Abbruch durch Early Stopping) | Epoche 22 | 2,02 h | 29,9 GB |

---

## 4 Ergebnisse

### 4.1 Nativer Maßstab

| Modell | Split | mAP@0,5 | mAP@0,75 | mAP@0,5:0,95 | bestes F1 | Präzision | Rückruf | mittl. IoU | Zählung |
|---|---|---|---|---|---|---|---|---|---|
| `yolo11s` | val | 0,8864 | 0,7928 | 0,5973 | 0,8621 | 0,872 | 0,852 | 0,824 | 86 / 88 |
| `yolo11s` | test | 0,8875 | 0,3196 | 0,4372 | 0,8941 | 0,905 | 0,884 | 0,748 | 84 / 86 |
| `yolo26s` | val | 0,9311 | 0,7954 | 0,6244 | 0,8913 | 0,854 | 0,932 | 0,824 | 96 / 88 |
| `yolo26s` | test | 0,9051 | 0,3326 | 0,4270 | 0,8876 | 0,904 | 0,872 | 0,729 | 83 / 86 |
| `yolo26s-p2` | val | 0,9359 | 0,7502 | 0,6065 | 0,8973 | 0,856 | 0,943 | 0,813 | 97 / 88 |
| `yolo26s-p2` | test | 0,9415 | 0,3873 | 0,4500 | 0,9333 | 0,975 | 0,895 | 0,728 | 79 / 86 |

### 4.2 Bei der jeweils auf `val` bestimmten Inferenzauflösung

| Modell | imgsz | val mAP@0,5 | val mAP@0,5:0,95 | test mAP@0,5 | test mAP@0,5:0,95 | test F1 | test Zählung |
|---|---|---|---|---|---|---|---|
| `yolo11s` | 1920 | 0,8925 | 0,6129 | 0,9279 | 0,4482 | 0,9167 | 82 / 86 |
| `yolo26s` | 1600 | 0,9391 | 0,6377 | 0,9051 | 0,4270 | 0,8876 | 83 / 86 |
| `yolo26s-p2` | 1920 | 0,9400 | 0,6417 | 0,9471 | 0,4492 | 0,9341 | 81 / 86 |

---

## 5 Diskussion

### 5.1 Die ursprüngliche Modellwahl war suboptimal

`yolo26s-p2` erreicht auf dem unberührten Testbild mAP@0,5 = 0,9415 im nativen
Maßstab beziehungsweise 0,9471 bei der auf `val` bestimmten Auflösung, gegenüber
0,8875 und 0,9279 für `yolo11s`. Das entspricht einer Verbesserung um 5,3
beziehungsweise 1,9 Prozentpunkte. Beim F1-Maß beträgt der Unterschied im
nativen Maßstab 0,9333 gegenüber 0,8941.

Bemerkenswert ist, dass die Zwischenstufe `yolo26s` auf dem Testbild die
schwächste der drei Varianten ist (mAP@0,5 = 0,9051). Der Gewinn geht demnach
nicht auf die Familie YOLO26 als solche zurück, sondern auf die
**P2-Erweiterung**. Der Familienwechsel allein hätte in dieser Anwendung keinen
Vorteil gebracht.

### 5.2 Die P2-Hypothese bestätigt sich nur zur Hälfte

Die Erwartung an den P2-Kopf lautete, er verbessere die
Lokalisierungspräzision und damit insbesondere mAP@0,75 auf dem Testbild, das
mit 0,3196 auffällig niedrig lag. Gemessen steigt dieser Wert auf 0,3873, also um
6,8 Prozentpunkte — die Vorhersage trifft in dieser Metrik zu.

Die **mittlere IoU der korrekt zugeordneten Detektionen verbessert sich jedoch
nicht**: sie liegt bei 0,728 gegenüber 0,748 für `yolo11s`, also sogar
geringfügig niedriger. Der Zugewinn in mAP@0,75 entsteht demnach überwiegend
daraus, dass **mehr Objekte überhaupt gefunden werden** (Rückruf 0,895
gegenüber 0,884 bei gleichzeitig um 5,3 Prozentpunkte höherem mAP@0,5), nicht
daraus, dass die gefundenen Objekte präziser umschlossen würden. Die Hypothese
ist damit in ihrer Metrik bestätigt, in ihrer Begründung jedoch nicht: Das
feinere Vorhersagegitter erhöht die Trefferwahrscheinlichkeit, verbessert aber
nicht die Kantengenauigkeit der Boxen.

### 5.3 Der Verzicht auf die Distribution Focal Loss ist bei den kleinsten Objekten nachteilig

`yolo26s` liegt auf dem Validierungsbild bei mAP@0,5:0,95 vorn (0,6244 gegenüber
0,5973), auf dem Testbild jedoch zurück (0,4270 gegenüber 0,4372). Die
Objektgrößen unterscheiden sich zwischen beiden Bildern deutlich (Median 27,4
gegenüber 19,6 Pixel). Das Muster ist mit der Erwartung vereinbar, dass der
Entfall der Verteilungsrepräsentation der Boxkanten gerade bei sehr kleinen
Objekten Lokalisierungsgenauigkeit kostet. Mit je einem Bild pro Split lässt
sich dieser Zusammenhang jedoch nicht belastbar von anderen Unterschieden
zwischen den beiden Aufnahmen trennen; die Aussage bleibt eine plausible
Deutung, kein Nachweis.

### 5.4 Test-Time-Augmentation steht bei YOLO26 nicht zur Verfügung

Für `yolo26s-p2` liefert `augment = True` **bitidentische** Ergebnisse zu
`augment = False` (Validierung mAP@0,5 = 0,9400, Test 0,9471 in beiden Fällen);
der end-to-end-Kopf verarbeitet das Flag nicht. Der für `yolo11s` gemessene
TTA-Gewinn von 2,1 Prozentpunkten lässt sich damit nicht auf YOLO26 übertragen.
Für einen faireren Vergleich ist daher `yolo11s` **mit** TTA gegen
`yolo26s-p2` **ohne** TTA zu stellen: 0,9341 gegenüber 0,9471 mAP@0,5 auf dem
Testbild. Der Vorsprung bleibt bestehen, fällt aber deutlich geringer aus als
die unmittelbaren Zahlen nahelegen.

### 5.5 Rechenkosten

`yolo26s-p2` benötigt 29,9 GB GPU-Speicher gegenüber 11,6 GB und 2,02 Stunden
Trainingszeit gegenüber 1,13 Stunden. Der Speicherbedarf ist die eigentliche
Einschränkung, da er die mögliche Batchgröße begrenzt. Für die vorliegende
Anwendung ist der Aufwand angesichts des Zugewinns vertretbar; auf
Hardware mit weniger Speicher wäre `yolo26s` mit 14,5 GB der geeignetere
Kompromiss, wobei dieser gegenüber `yolo11s` auf dem Testbild keinen Vorteil
bietet.

### 5.6 Einschränkung der Aussagekraft

Das über räumliche Kachel-Resampling bestimmte 95-%-Konfidenzintervall für
mAP@0,5 beträgt ±0,04 bis ±0,05. Der Unterschied zwischen `yolo11s` und
`yolo26s-p2` liegt im nativen Maßstab mit 5,3 Prozentpunkten an dessen Grenze
und bei der auf `val` bestimmten Auflösung mit 1,9 Prozentpunkten deutlich
darunter. Belastbar ist somit nicht die exakte Differenz, sondern die
**Richtung des Effekts**: `yolo26s-p2` liegt auf dem Testbild in fünf von sechs
Metrikspalten vorn. Ein formaler Signifikanznachweis ist bei einem
Evaluationsbild je Split nicht möglich.

Hinzu kommt der in Abschnitt 3 genannte Umstand, dass `yolo26s-p2` mit einem
geringeren Anteil vortrainierter Gewichte startet. Dass dieses Modell dennoch
am besten abschneidet, stützt den Befund; ein sauberer Vergleich würde jedoch
eine für die P2-Architektur vortrainierte Ausgangsgewichtung voraussetzen.

---

## 6 Konsequenz für die Arbeit

`yolo11s` bleibt die **dokumentierte Referenzkonfiguration**. Auf ihr wurden
die Untersuchung der Inferenzparameter und die Ablation des Okklusionsanteils
gerechnet; deren Aussagen sind modellunabhängig, weil alle drei verglichenen
Modelle dieselbe P3-Stufe als feinste Standardauflösung besitzen.

Für den **produktiven Einsatz** ist `yolo26s-p2` bei einer Inferenzauflösung
von 1920 Pixeln und einer Konfidenzschwelle von 0,44 zu wählen. Der
Betriebspunkt wurde über das F1-Maximum auf dem Validierungsbild bestimmt und
ist damit auf ein neues Bild übertragbar.

Methodisch bleibt festzuhalten: Die Modellfamilie war die einzige
Entscheidung dieser Arbeit, die zunächst allein aus Konvention und nicht aus
einer Messung heraus getroffen wurde — und genau diese Entscheidung erwies
sich in der Überprüfung als verbesserungsfähig.

---

## Quellenverweise im Projekt

| Aussage | Beleg |
|---|---|
| YOLO26 ist Standardfamilie in Ultralytics 8.4.117 | `ultralytics/utils/checks.py`, Zeilen 707, 761, 991, 1030 |
| Strides, `reg_max`, end-to-end-Eigenschaft | direkt aus den trainierten Modellen ausgelesen |
| NMS-Schwelle wirkungslos | Variation 0,10 / 0,70 / 0,99, bitidentisches Ergebnis |
| alle Metriken | `src/evaluate.py`, Rohdaten `docs/experiments/experiment3_models*.json` |
| Trainingsbudget und Laufzeiten | `runs/*/results.csv`, `runs/*/parameter_begruendung.md` |
| Abbildung Modellvergleich | `docs/fig_models.png` (erzeugt von `src/report.py`) |
| Reproduktion | `src/train.py --name yolo26s_p2 --model yolo26s-p2.yaml --pretrained yolo26s.pt`, danach `src/experiment.py models` |
