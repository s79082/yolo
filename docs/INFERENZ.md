# Inferenzparameter

Alle Werte, mit denen `src/evaluate.py` und `src/experiment.py` arbeiten. Die empfohlenen Werte sind durch die Sweeps in Experiment 1 belegt (siehe `docs/ERGEBNISSE.md`).

| Parameter | empfohlener Wert | Begruendung |
|---|---|---|
| `imgsz` | `val: 1920, test: 1920` | Netzeingangsgroesse. ENTSCHEIDEND fuer diese Aufgabe: die Objekte sind im Original 12-36 px gross. Ultralytics skaliert das Bild auf imgsz (laengste Kante). Bei val (2459x1174) bedeutet imgsz=640 einen Faktor 0.26 -> die Objekte schrumpfen auf 5-9 px und liegen unter der Aufloesung des feinsten Detektionskopfes (Stride 8). Der Standardwert ist deshalb 'native': imgsz = laengste Bildkante, auf ein Vielfaches von 32 aufgerundet. Dann ist der Massstab 1:1 und entspricht exakt dem, wofuer die Trainingsdaten kalibriert wurden. |
| `conf` | `0.001 fuer mAP; Betriebspunkt siehe conf* in ERGEBNISSE.md` | Konfidenzschwelle. Fuer die mAP-Berechnung 0.001 (praktisch alle Kandidaten, sonst wird der Recall-Ast der PR-Kurve abgeschnitten). Fuer den Betriebspunkt wird die F1-optimale Schwelle aus der Kurve bestimmt und separat berichtet. |
| `iou` | `0.7` | IoU-Schwelle der NMS. 0.7 (Ultralytics-Standard). Unkritisch, weil benachbarte Rohrenden real mindestens einen Durchmesser Abstand haben (gemessener 10-%-Quantil-Abstand: 1.0 d) und ihre Boxen damit IoU < 0.3 haben. Ein zu niedriger Wert wuerde dicht stehende, tatsaechlich verschiedene Rohre gegenseitig ausloeschen. |
| `max_det` | `1000` | Maximale Detektionen pro Bild. Der Ultralytics-Standard 300 reicht fuer val (88 GT) und test (86 GT), waere aber bei einem grossen Stapel eine stille Obergrenze. Auf 1000 gesetzt, damit die Metrik nie durch eine Kappungsgrenze verfaelscht wird. |
| `augment` | `False` | Test-Time-Augmentation (Mehrfachskalen + Spiegelung). Standard aus; wird im Sweep geprueft. Kostet ~3x Rechenzeit. |
| `quantize` | `None (FP32)` | Rechengenauigkeit der Inferenz (None = FP32, 'fp16' = halbe Genauigkeit; gueltig sind 8/16/32, 'int8', 'fp16', 'fp32'). Standard None, damit die Metrik nicht von der Rechengenauigkeit abhaengt; im Sweep geprueft. Hinweis: der aeltere Parameter 'half' ist in Ultralytics 8.4.117 deprecated und wird ignoriert - gemessen liefert half=True bitidentische Ergebnisse zu half=False. Verwendet wird deshalb 'quantize'. |
| `rect` | `True (implizit)` | Rechteckige Inferenz (nur auf Vielfache von 32 padden statt auf ein Quadrat). Bei Ultralytics-predict implizit aktiv. |
| `device` | `0` | GPU 0, wie gefordert. |
| `batch` | `1` | Einzelbildinferenz; die Bilder haben unterschiedliche Seitenverhaeltnisse, ein groesserer Batch wuerde zusaetzliches Padding erzwingen. |

## Nachbearbeitung

Nach der NMS wird nichts weiter gefiltert - keine Mindestgroesse, keine Randunterdrueckung, keine Cluster-Nachbearbeitung. Das ist Absicht: jede solche Regel waere ein an val angepasster Freiheitsgrad und wuerde die Aussagekraft der test-Zahlen schwaechen.

