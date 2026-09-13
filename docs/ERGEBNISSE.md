# Ergebnisse — zwei Ablationen der Augmentierung auf `yolo26s-p2`

Automatisch erzeugt von `src/report_neu.py`. Alle Zahlen bei
nativem Maßstab, gegen den korrigierten Annotationsstand
(`val` 98, `test` 89 Objekte).

## Referenzmodell (6000 Bilder, 60 Epochen)

| Split | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | bestes F1 | P / R |
|---|---|---|---|---|---|
| `val` | 0.9474 | 0.7939 | 0.6197 | 0.9101 | 0.945 / 0.878 |
| `test` | 0.9688 | 0.3680 | 0.4767 | 0.9364 | 0.964 / 0.910 |
| `sval` | 0.9836 | 0.9485 | 0.8391 | 0.9613 | 0.968 / 0.954 |

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
| `val` | 97 | 26.8 px | 0.944 | +1.58 px | 0.056 Ø |
| `test` | 88 | 17.9 px | 0.897 | +2.00 px | 0.051 Ø |
| `sval` | 7224 | 24.0 px | 1.003 | -0.08 px | 0.021 Ø |

