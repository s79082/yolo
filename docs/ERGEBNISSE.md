# Ergebnisse — zwei Ablationen der Augmentierung auf `yolo26s-p2`

Automatisch erzeugt von `src/report_neu.py`. Alle Zahlen bei
nativem Maßstab, gegen den korrigierten Annotationsstand
(`val` 98, `test` 89 Objekte).

## Referenzmodell (6000 Bilder, 60 Epochen)

| Split | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | bestes F1 | P / R |
|---|---|---|---|---|---|
| `val` | 0.9163 | 0.7526 | 0.5803 | 0.8643 | 0.851 / 0.878 |
| `test` | 0.9550 | 0.3820 | 0.4783 | 0.9249 | 0.952 / 0.899 |
| `sval` | 0.9842 | 0.9520 | 0.8302 | 0.9619 | 0.967 / 0.957 |

## Ablation A — Okklusionsanteil

| Arm | Labels | Sichtb. | teilverd. | val mAP@0.5 | val mAP@0.5:0.95 | test mAP@0.5 | test mAP@0.5:0.95 | test F1 | best. Epoche |
|---|---|---|---|---|---|---|---|---|---|
| A0 ohne | 82950 | 0.9768 | 12.0 % | 0.9527 | 0.6384 | 0.9578 | 0.5057 | 0.9294 | 21 |
| A1 Baseline | 68281 | 0.9015 | 29.9 % | 0.9739 | 0.6471 | 0.9611 | 0.4814 | 0.9318 | 28 |
| A2 stark | 55833 | 0.8522 | 40.7 % | 0.9572 | 0.6170 | 0.9638 | 0.4567 | 0.9239 | 29 |

## Ablation B — Herkunft der Okkluder

| Arm | Labels | Sichtb. | teilverd. | val mAP@0.5 | val mAP@0.5:0.95 | test mAP@0.5 | test mAP@0.5:0.95 | test F1 | best. Epoche |
|---|---|---|---|---|---|---|---|---|---|
| B0 photogr. | 68281 | 0.9015 | 29.9 % | 0.9739 | 0.6471 | 0.9611 | 0.4814 | 0.9318 | 28 |
| B1 prozedural | 68281 | 0.9015 | 29.9 % | 0.9442 | 0.5853 | 0.9630 | 0.4806 | 0.9412 | 20 |
| B2 naiv | 68281 | 0.9015 | 29.9 % | 0.9642 | 0.6537 | 0.9140 | 0.4697 | 0.8848 | 15 |

## Boxgrößenversatz

| Split | Paare | Objekt-Ø (Median) | pred/GT | GT − pred | Zentrumsversatz |
|---|---|---|---|---|---|
| `val` | 95 | 26.8 px | 0.929 | +1.78 px | 0.062 Ø |
| `test` | 88 | 17.9 px | 0.893 | +2.04 px | 0.048 Ø |
| `sval` | 8892 | 24.5 px | 1.006 | -0.14 px | 0.022 Ø |

