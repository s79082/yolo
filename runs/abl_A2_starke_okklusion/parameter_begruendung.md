# Trainingsparameter - Werte und Begruendung

Automatisch aus `src/train.py` erzeugt.

| Parameter | Wert | Begruendung |
|---|---|---|
| `model` | `yolo11s.pt` | YOLO11 in der 's'-Variante (9.4 M Parameter). Begruendung fuer die Groesse: der Datensatz hat genau EINE Klasse und nur 9 zugrunde liegende Objektinstanzen - die Formvielfalt ist klein, die noetige Modellkapazitaet damit ebenfalls. 'n' verliert bei 12-40 px grossen Objekten messbar Rueckruf, 'm'/'l' wuerden die synthetischen Eigenheiten (immer dieselben 9 Cutouts) auswendig lernen. COCO-vortrainierte Gewichte statt Training from scratch: die vortrainierten Low-Level-Filter (Kanten, Ringe, Texturen) sind domaenenunabhaengig und reduzieren genau den Anteil des Modells, der sonst auf synthetische Artefakte ueberangepasst wuerde. |
| `data` | `data/abl_A2_starke_okklusion.yaml` | Training auf synthetischen Daten, Validierung auf dem REALEN val-Bild. Bewusste Entscheidung: die Modellauswahl (best.pt) muss auf der Zieldomaene erfolgen, sonst waehlt man das Modell aus, das am besten zu den eigenen Renderartefakten passt. /test bleibt unberuehrt und dient ausschliesslich der Endbewertung. |
| `epochs` | `30` | 6000 Bilder / Batch 16 = 375 Iterationen pro Epoche, also ~22.500 Iterationen. Bei einem Ein-Klassen-Problem mit vortrainiertem Backbone ist die Konvergenz typischerweise nach 30-50 Epochen erreicht; 60 gibt der Kosinus-Rampe Raum zum Ausschwingen. Gegen Ueberanpassung wirkt zusaetzlich 'patience'. |
| `patience` | `20` | Early Stopping nach 20 Epochen ohne Verbesserung der Fitness auf dem realen val-Bild. Bewusst grosszuegig: die Fitness schwankt auf nur 88 Ground-Truth-Objekten stark, ein knappes Fenster wuerde zu frueh abbrechen. |
| `imgsz` | `1024` | Identisch zur Leinwandgroesse des Generators. Damit durchlaufen die synthetischen Bilder KEINE Skalierung und die Objekte erreichen das Netz mit exakt der Pixelgroesse (12-44 px), fuer die die Pipeline ausgelegt wurde. Jede Abweichung wuerde die sorgfaeltig kalibrierte Groessenverteilung wieder zerstoeren. Bei 640 px waeren die Objekte nur noch 7-27 px gross und laegen teils unter der Aufloesung des P3-Kopfes (Stride 8). |
| `device` | `0` | GPU 0 (RTX PRO 6000 Blackwell), wie gefordert. |
| `batch` | `16` | Bei imgsz=1024 und yolo11s belegt Batch 16 rund 14 GB und passt sicher in den freien Speicher (die Karte ist bereits durch einen Fremdprozess zu ~56 GB belegt). Feste Zahl statt batch=-1, weil Autobatch je nach Fremdlast unterschiedliche Werte waehlt und der Lauf dann nicht reproduzierbar ist. |
| `workers` | `0` | ERZWUNGEN, nicht gewaehlt: in dieser Umgebung (opencv-python 5.0.0.93, torch 2.13, fork-basierte Worker) verklemmt sich der DataLoader reproduzierbar beim ersten Batch, sobald workers > 0 ist - unabhaengig von cv2.setNumThreads(0), deaktiviertem OpenCL, gesetzten OMP/MKL-Thread-Limits oder Start-Methode 'spawn'. Getestet und jeweils bestaetigt. Kompensiert wird der Verlust durch cache='ram' (siehe dort); gemessener Durchsatz 3.1 it/s bei Batch 16, also ~50 Bilder/s - die GPU ist damit zu rund 60 % ausgelastet. |
| `cache` | `ram` | Alle Bilder einmalig dekodiert im RAM halten: 6000 x 1024 x 1024 x 3 B = 18 GB bei 94 GB frei. Weil workers=0 erzwungen ist, laeuft die gesamte Datenaufbereitung im Hauptprozess; das Wegfallen der JPEG-Dekodierung ist dort der groesste einzelne Zeitgewinn (Faktor ~1.6 gemessen). Der Cache wird von einem Thread-Pool gefuellt, nicht von Prozessen, und ist vom obigen Deadlock nicht betroffen. |
| `amp` | `True` | Mixed Precision. Halbiert Speicher und Rechenzeit; Ultralytics haelt Verlustberechnung und Normierung in FP32, der Genauigkeitsverlust ist bei Detektion nicht messbar. |
| `seed` | `0` | Fester Seed. |
| `deterministic` | `True` | Deterministische cuDNN-Kernel. Kostet etwas Geschwindigkeit, ist fuer eine Abschlussarbeit aber die Voraussetzung dafuer, dass die Ablationen (Experiment 2) vergleichbar sind. |
| `optimizer` | `AdamW` | Explizit gesetzt statt 'auto'. AdamW konvergiert bei kleinen Datensaetzen deutlich schneller als SGD und ist gegenueber der Lernratenwahl robust - wichtig, weil hier kein Budget fuer eine Lernraten-Suche vorhanden ist. Die entkoppelte Gewichtsdaempfung verhindert, dass der Weight Decay mit der adaptiven Schrittweite interagiert. |
| `lr0` | `0.001` | Standard-Startlernrate fuer AdamW-Finetuning. Hoehere Werte (3e-3) zerstoeren die vortrainierten Backbone-Gewichte in den ersten Epochen, niedrigere (3e-4) verlangsamen ohne Gewinn. |
| `lrf` | `0.01` | Endlernrate = 1 % der Startlernrate. Die letzten Epochen dienen dann nur noch der Feinjustierung der Box-Regression, was gerade bei 20-30 px grossen Objekten den Lokalisierungsfehler senkt. |
| `cos_lr` | `True` | Kosinus-Abfall statt linear: laengeres Verweilen auf hoher Lernrate (bessere Exploration), dann weiches Auslaufen. |
| `warmup_epochs` | `3.0` | 3 Aufwaermepochen. Ohne Warmup erzeugt der frisch initialisierte Detektionskopf in den ersten Schritten sehr grosse Gradienten, die in den vortrainierten Backbone zurueckschlagen. |
| `weight_decay` | `0.0005` | Ultralytics-Standard. Bei nur 9 Quellobjekten ist eine gewisse Regularisierung erwuenscht. |
| `momentum` | `0.937` | Entspricht beta1 bei AdamW (Ultralytics-Konvention). |
| `box` | `7.5` | Standardgewicht der CIoU/DFL-Boxverluste. Eine Erhoehung waere bei kleinen Objekten denkbar, wurde hier aber NICHT vorgenommen: die Zielmetrik ist ueberwiegend mAP50 und Zaehlgenauigkeit, nicht Pixelpraezision, und ohne separaten Tuning-Split waere jede Abweichung vom Standard nicht belegbar. |
| `cls` | `0.5` | Bei einer einzigen Klasse ist der Klassifikationsverlust faktisch eine Objektheits-Entscheidung. Der Standardwert bleibt, weil das Verhaeltnis box:cls die Balance zwischen Finden und Lokalisieren steuert und ohne Tuning-Split nicht seriös verschoben werden kann. |
| `dfl` | `1.5` | Standardgewicht der Distribution-Focal-Loss. |
| `mosaic` | `1.0` | Immer aktiv. Setzt vier Bilder zu einem zusammen; das erhoeht die Objektzahl pro Iteration, bringt Objekte an Kachelraender (realistisch, in test sind 5.8 % der Labels angeschnitten) und kombiniert verschiedene Hintergruende in einem Bild. Bei dichten Kleinobjekten ist Mosaic die wirksamste Einzelaugmentierung. |
| `close_mosaic` | `10` | Mosaic in den letzten 10 Epochen abschalten. Mosaic erzeugt eine Statistik, die es bei der Inferenz nie gibt (Vierfachkacheln). Das Abschalten am Ende laesst das Modell auf der echten Bildstatistik ausklingen - ein in der YOLO-Literatur gut belegter Gewinn. |
| `scale` | `0.4` | Zufaellige Skalierung 0.6-1.4. Zusammen mit der Generatorspanne 12-44 px ergibt das effektiv 7-62 px und deckt damit alle Inferenzaufloesungen des Experiments 1 ab. Der Ultralytics-Standard 0.5 wurde reduziert, weil er die kleinsten Objekte auf 6 px schrumpfen laesst - unterhalb dessen, was Stride 8 aufloest, und damit reines Labelrauschen. |
| `translate` | `0.1` | Standard. Verschiebt das Mosaic-Zentrum, verhindert eine Positionsverzerrung zur Bildmitte. |
| `degrees` | `0.0` | BEWUSST AUS. Eine Bildrotation dreht achsparallele Boxen mit und blaeht sie auf (10 Grad -> +17 % Kantenlaenge bei quadratischer Box), erzeugt also Labelrauschen. Die noetige Rotationsinvarianz liefert der Generator: dort wird JEDE Instanz um 0-360 Grad gedreht und die Box danach aus der gedrehten Alphamaske exakt neu bestimmt. |
| `shear` | `0.0` | Aus, gleiche Begruendung wie degrees. |
| `perspective` | `0.0` | Aus, gleiche Begruendung. Perspektive wird im Generator pro Instanz label-exakt angewandt (30 % der Instanzen). |
| `fliplr` | `0.5` | Horizontale Spiegelung ist label-exakt und physikalisch korrekt: ein Stapel kann von beiden Enden fotografiert werden. |
| `flipud` | `0.5` | Vertikale Spiegelung ebenfalls label-exakt. Ungewoehnlich fuer Naturbilder, hier aber richtig: die Kamera wird beim Fotografieren eines Stapels in beliebiger Rollage gehalten, und ein Rohrende hat kein 'oben'. |
| `hsv_h` | `0.015` | Sehr kleiner Farbtonversatz - verzinkter Stahl ist neutral, grosse Farbtonspruenge waeren unphysikalisch. |
| `hsv_s` | `0.7` | Kraeftige Saettigungsvariation deckt die Spanne von neutral grau bis rostbraun/moosgruen ab. |
| `hsv_v` | `0.4` | Helligkeitsvariation fuer Sonne/Schatten/Bewoelkung. |
| `mixup` | `0.0` | AUS. Mixup ueberlagert zwei Bilder halbtransparent und erzeugt damit Geisterobjekte mit vollem Label. Bei 20 px grossen, dicht gepackten Objekten ist das reines Labelrauschen. |
| `cutmix` | `0.0` | AUS. Gleiche Begruendung; Okklusion wird im Generator physikalisch korrekt erzeugt (echte Rohrstangen), nicht durch eingesetzte Rechtecke aus fremden Bildern. |
| `erasing` | `0.0` | AUS. Wirkt in Ultralytics ohnehin nur bei Klassifikation; die entsprechende Funktion uebernimmt hier die Okklusionsstufe des Generators. |
| `copy_paste` | `0.0` | AUS - benoetigt Segmentierungsmasken und ist fuer die Detektionsaufgabe nicht verfuegbar. Copy-Paste ist genau das, was der Generator offline und mit korrekter Sichtbarkeitsrechnung leistet. |
| `val` | `True` | Nach jeder Epoche gegen das reale val-Bild. |
| `plots` | `True` | Erzeugt Lernkurven, PR-Kurve und Konfusionsmatrix. |
| `save_period` | `-1` | Nur last.pt und best.pt speichern. |
| `project` | `/home/mso/source/yolo-gen/runs` | Absoluter Pfad. Ultralytics stellt einem relativen project-Pfad sonst SETTINGS['runs_dir'] voran, und das zeigt auf dieser Maschine in ein fremdes Projektverzeichnis. |
