# PipeNet - Ergebnisse

Automatisch aus `cnn/docs/experiments/*.json` erzeugt (`python cnn/src/report.py`). Entwurfsbegruendung in [ARCHITEKTUR.md](ARCHITEKTUR.md).

Ausgewerteter Lauf: `pipenet_e30` (der von zwei Laeufen mit der besseren Fitness auf val).

## Trainingsverlauf

![](fig_training.png)

| Kenngroesse | Wert |
|---|---|
| Parameter | 3.07 M |
| Epochen gelaufen | 26 |
| bestes Modell | Epoche 6 |
| Laufzeit | 0.37 h |
| val mAP@0.5 (bestes) | 0.8792 |
| val mAP@0.5:0.95 (bestes) | 0.5701 |

## Eingangsskalierung

PipeNet ist vollstaendig konvolutional - es gibt keine feste Netzeingangsgroesse. Skaliert wird nur, um die Objektgroesse in die Mitte der Trainingsverteilung (Median 22.5 px) zu verschieben.

![](fig_scale_sweep.png)

| Split | scale | Objekt (px) | mAP50 | mAP75 | mAP50-95 | bestes F1 | R | Zaehlung |
|---|---|---|---|---|---|---|---|---|
| val | 0.5 | 13.7 | 0.8843 | 0.5834 | 0.5363 | 0.8671 | 0.852 | 85/88 |
| val | 0.7 | 19.2 | 0.8997 | 0.6733 | 0.5712 | 0.8852 | 0.920 | 95/88 |
| val | 0.85 | 23.3 | 0.9049 | 0.6943 | 0.5787 | 0.8925 | 0.943 | 98/88 |
| val | 1.0 | 27.4 | 0.8792 | 0.6554 | 0.5701 | 0.8743 | 0.909 | 95/88 |
| val | 1.2 | 32.9 | 0.9012 | 0.6676 | 0.5806 | 0.9011 | 0.932 | 94/88 |
| val | 1.5 | 41.1 | 0.9178 | 0.7315 | 0.6092 | 0.9011 | 0.932 | 94/88 |
| val | 1.8 | 49.3 | 0.9060 | 0.7333 | 0.6140 | 0.8778 | 0.898 | 92/88 |
| val | 2.1 | 57.5 | 0.8460 | 0.5995 | 0.5251 | 0.8571 | 0.852 | 87/88 |
| test | 0.5 | 9.8 | 0.6999 | 0.4312 | 0.3961 | 0.7170 | 0.663 | 73/86 |
| test | 0.7 | 13.7 | 0.8795 | 0.5121 | 0.4743 | 0.8462 | 0.767 | 70/86 |
| test | 0.85 | 16.7 | 0.9042 | 0.4241 | 0.4750 | 0.8645 | 0.779 | 69/86 |
| test | 1.0 | 19.6 | 0.9207 | 0.3287 | 0.4238 | 0.8743 | 0.849 | 81/86 |
| test | 1.2 | 23.5 | 0.9228 | 0.4734 | 0.4853 | 0.8834 | 0.837 | 77/86 |
| test | 1.5 | 29.4 | 0.9341 | 0.4400 | 0.5031 | 0.9136 | 0.860 | 76/86 |
| test | 1.8 | 35.3 | 0.9335 | 0.3548 | 0.4531 | 0.9167 | 0.895 | 82/86 |
| test | 2.1 | 41.2 | 0.9143 | 0.4244 | 0.4684 | 0.8916 | 0.860 | 80/86 |

### Endauswertung (mit und ohne Rotations-TTA)

| Split | TTA | mAP50 | mAP75 | mAP50-95 | bestes F1 | conf* | P | R | mittl. IoU | Zaehlung | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | False | 0.9060 | 0.7333 | 0.6140 | 0.8778 | 0.413 | 0.859 | 0.898 | 0.822 | 92/88 | 0.49 |
| val | True | 0.9066 | 0.8181 | 0.6270 | 0.8913 | 0.350 | 0.854 | 0.932 | 0.828 | 96/88 | 0.26 |
| test | False | 0.9335 | 0.3548 | 0.4531 | 0.9167 | 0.317 | 0.939 | 0.895 | 0.742 | 82/86 | 0.05 |
| test | True | 0.9220 | 0.3775 | 0.4648 | 0.9102 | 0.287 | 0.938 | 0.884 | 0.756 | 81/86 | 0.12 |

### Aequivarianz-Selbsttest der Rotations-TTA

Bild drehen, durch das Netz schicken, Heatmap zurueckdrehen, mit dem untransformierten Durchlauf korrelieren. Ein Vorzeichen- oder Achsenfehler im Code ergaebe eine Korrelation um 0.

| Transformation | Korrelation |
|---|---|
| rot90 | +0.3489 |
| rot180 | +0.3797 |
| rot270 | +0.3902 |
| fliplr | +0.4669 |

## Vergleich mit den YOLO-Modellen

Gleicher synthetischer Datensatz, gleiches Trainingsbudget, gleiche Metrikimplementierung. Die Betriebseinstellung ist je Modell **ausschliesslich auf val** bestimmt und dann unveraendert auf test angewandt.

![](fig_vergleich.png)

| Modell | Familie | Par. (M) | Einstellung | Filterung | Objekt val (px) | Objekt test (px) |
|---|---|---|---|---|---|---|
| yolo11s | YOLO | 9.43 | imgsz=1920 | Box-NMS (iou 0.7) | 21.4 | 23.5 |
| yolo26s | YOLO | 9.95 | imgsz=1600 | keine (end2end, NMS-frei) | 17.8 | 19.6 |
| yolo26s-p2 | YOLO | 9.66 | imgsz=1920 | keine (end2end, NMS-frei) | 21.4 | 23.5 |
| PipeNet | PipeNet | 3.07 | scale=1.8 | Peak-Extraktion (3x3, keine Schwelle) | 49.3 | 35.3 |
| PipeNet +TTA | PipeNet | 3.07 | scale=1.8 +TTA | Peak-Extraktion (3x3, keine Schwelle) | 49.3 | 35.3 |

| Modell | Split | mAP50 | mAP75 | mAP50-95 | bestes F1 | P | R | mittl. IoU | Zaehlung |
|---|---|---|---|---|---|---|---|---|---|
| yolo11s | val | 0.8925 | 0.7390 | 0.6129 | 0.8511 | 0.800 | 0.909 | 0.830 | 100/88 |
| yolo11s | test | 0.9279 | 0.3398 | 0.4482 | 0.9167 | 0.939 | 0.895 | 0.745 | 82/86 |
| yolo26s | val | 0.9391 | 0.8384 | 0.6377 | 0.9111 | 0.891 | 0.932 | 0.824 | 92/88 |
| yolo26s | test | 0.9051 | 0.3326 | 0.4270 | 0.8876 | 0.904 | 0.872 | 0.729 | 83/86 |
| yolo26s-p2 | val | 0.9400 | 0.7795 | 0.6417 | 0.9091 | 0.859 | 0.966 | 0.829 | 99/88 |
| yolo26s-p2 | test | 0.9471 | 0.3257 | 0.4492 | 0.9341 | 0.963 | 0.907 | 0.739 | 81/86 |
| PipeNet | val | 0.9060 | 0.7333 | 0.6140 | 0.8778 | 0.859 | 0.898 | 0.822 | 92/88 |
| PipeNet | test | 0.9335 | 0.3548 | 0.4531 | 0.9167 | 0.939 | 0.895 | 0.742 | 82/86 |
| PipeNet +TTA | val | 0.9066 | 0.8181 | 0.6270 | 0.8913 | 0.854 | 0.932 | 0.828 | 96/88 |
| PipeNet +TTA | test | 0.9220 | 0.3775 | 0.4648 | 0.9102 | 0.938 | 0.884 | 0.756 | 81/86 |

