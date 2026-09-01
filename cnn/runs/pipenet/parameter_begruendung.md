# PipeNet - Trainingsparameter, Werte und Begruendung

Automatisch aus `cnn/src/train.py` erzeugt.

Lauf `pipenet`: 3.07 M Parameter, Ausgabe-Stride 4, 26 Epochen gelaufen, bestes Modell aus Epoche 6 (Fitness 0.5921), 0.36 h.

| Parameter | Wert | Begruendung |
|---|---|---|
| `train_dir` | `synth/train/images` | Exakt derselbe synthetische Datensatz wie beim YOLO-Lauf (6000 Bilder, 136 384 Labels). Nur so ist der Vergleich ein Architekturvergleich und keine Datenfrage. |
| `val_dir` | `data/val/images` | Das REALE val-Bild. Die Modellauswahl muss auf der Zieldomaene erfolgen, sonst waehlt man das Modell, das am besten zu den eigenen Renderartefakten passt. data/test bleibt unberuehrt. |
| `imgsz` | `1024` | Identisch zur Leinwandgroesse des Generators und zur YOLO-Einstellung. Die synthetischen Bilder durchlaufen damit keine Skalierung, die Objekte erreichen das Netz mit exakt der Pixelgroesse (12-44 px), fuer die die Pipeline kalibriert ist. |
| `epochs` | `60` | Wie beim YOLO-Lauf. Gegen Ueberanpassung wirkt patience. |
| `patience` | `20` | Early Stopping nach 20 Epochen ohne Fitness-Verbesserung. Grosszuegig, weil die Fitness auf nur 88 Ground-Truth-Objekten stark schwankt. |
| `batch` | `16` | Wie beim YOLO-Lauf, damit die effektive Lernrate je Beispiel vergleichbar bleibt. |
| `pretrained` | `True` | ResNet18-Encoder mit ImageNet-Gewichten. Dieselbe Begruendung wie bei YOLO: die vortrainierten Low-Level-Filter (Kanten, Ringe, Texturen) sind domaenenunabhaengig und reduzieren genau den Anteil des Modells, der sonst auf synthetische Artefakte ueberangepasst wuerde. Ohne Vortraining ist der Dekoder ohnehin zufaellig initialisiert - der Encoder ist der Teil, bei dem es etwas zu uebernehmen gibt. |
| `optimizer` | `AdamW` | Wie beim YOLO-Lauf. Konvergiert bei kleinen Datensaetzen schneller als SGD und ist robust gegenueber der Lernratenwahl. |
| `lr0` | `0.001` | Standard-Startlernrate fuer AdamW-Finetuning. |
| `lr_encoder_mult` | `0.1` | Der Encoder ist vortrainiert, Dekoder und Koepfe sind zufaellig initialisiert. Eine einheitliche Lernrate wuerde die vortrainierten Gewichte in den ersten Epochen zerstoeren, waehrend die Koepfe noch keinen sinnvollen Gradienten liefern. Faktor 0.1 ist die uebliche Wahl fuer diskriminatives Finetuning. |
| `lrf` | `0.01` | Endlernrate = 1 % der Startlernrate, Kosinus-Abfall. |
| `warmup_epochs` | `3.0` | Ohne Warmup erzeugen die frisch initialisierten Koepfe in den ersten Schritten grosse Gradienten, die in den vortrainierten Encoder zurueckschlagen. |
| `weight_decay` | `0.0005` | Wie beim YOLO-Lauf. Nur auf Faltungsgewichte, nicht auf BatchNorm-Parameter und Biases (siehe PipeNet.param_groups). |
| `clip_grad` | `10.0` | Gradienten-Norm begrenzen. Der Gaussian-Focal-Loss wird durch die Anzahl Zentren normiert; ein Batch mit sehr wenigen Objekten (die 9.7 % Negativbilder haben null) kann sonst einen einzelnen grossen Schritt erzeugen. |
| `amp` | `True` | Mixed Precision, wie beim YOLO-Lauf. |
| `threads` | `12` | Der Standard-DataLoader mit num_workers > 0 verklemmt sich in dieser Umgebung reproduzierbar (fork + OpenCV, siehe loader.py). Statt auf workers=0 zurueckzufallen wie beim YOLO-Lauf, laedt hier ein eigener Thread-Pool: die teuren cv2-Stufen geben die GIL frei und parallelisieren echt. Gemessen 31.9 -> 2.5 ms/Sample, Epochendauer 3.19 -> 0.25 min. 12 statt 16 Threads, weil der Zugewinn danach unter 10 % liegt und Kerne fuer die GPU-Zufuhr frei bleiben sollen. |
| `prefetch` | `3` | Vorlauf in Batches; ueberlappt Datenaufbereitung und GPU-Berechnung. |
| `cache` | `True` | Alle 6000 Bilder einmalig dekodiert im RAM (~18 GB bei 94 GB frei). Alle Threads teilen denselben Cache - bei Prozessen haette jeder eine eigene Kopie gebraucht. |
| `p_mosaic` | `1.0` | Wie mosaic=1.0 bei YOLO. Vier Bilder in einer Leinwand: mehr Objekte je Iteration, Objekte an Kachelraendern (in test sind 5.8 % der Labels angeschnitten) und verschiedene Hintergruende in einem Bild. |
| `scale` | `(0.6, 1.4)` | Wie scale=0.4 bei YOLO (0.6-1.4). Zusammen mit der Generatorspanne 12-44 px ergibt das effektiv 7-62 px. |
| `flip` | `0.5` | Horizontal und vertikal je 0.5, wie fliplr/flipud bei YOLO. Beide sind label-exakt und physikalisch korrekt: ein Stapel wird von beiden Enden fotografiert, und ein Rohrende hat kein 'oben'. |
| `rotation` | `0.0` | AUS - wie bei YOLO. Eine Bildrotation dreht achsparallele Boxen mit und blaeht sie auf. Die Rotationsinvarianz liefert der Generator (jede Instanz 0-360 Grad, Box danach exakt aus der gedrehten Alphamaske neu bestimmt). |
| `min_overlap` | `0.7` | Parameter der CenterNet-Radiusformel. Bestimmt, wie weit eine Detektion vom Zentrum abweichen darf und trotzdem noch min_overlap IoU erreicht - daraus folgt die Breite der Gauss-Glocke. Gemessene Folge fuer diese Daten: sigma 0.60-1.17 Zellen. Bei einem Nachbarabstand von 4.4-5.5 Zellen verschmelzen benachbarte Glocken damit nicht. |
| `sigma_clip` | `(0.6, 2.5)` | Untergrenze 0.6: darunter faellt die Glocke innerhalb einer Zelle ab und der beta=4-Term des Focal Loss hat keine Wirkung mehr. Obergrenze 2.5: verhindert, dass ein sehr grosses Objekt Nachbarzentren ueberdeckt. |
| `w_wh` | `1.0` | Gewicht des log-Groessen-Verlusts, siehe losses.py. |
| `w_off` | `1.0` | Gewicht des Versatz-Verlusts, siehe losses.py. |
| `val_thresh` | `0.01` | Peak-Schwelle bei der Validierung. Niedrig, damit die PR-Kurve nicht am Recall-Ast abgeschnitten wird. |
| `val_topk` | `2000` | Obergrenze der Detektionen je Bild. Weit ueber den 88 Ground-Truth-Objekten, damit die Metrik nie durch eine Kappungsgrenze verfaelscht wird. |
| `seed` | `0` | Fester Seed. |
