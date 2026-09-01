#!/usr/bin/env python3
"""Qualitativbilder bei unterschiedlicher Zuordnungsschwelle.

Rendert dasselbe Bild mit derselben Detektionsmenge zweimal und variiert
ausschliesslich die IoU-Schwelle der Zuordnung.  Damit wird sichtbar, welche
der markierten Fehler echte Fehler sind und welche nur Zuordnungsartefakte:
Eine Detektion, die auf einem annotierten Objekt sitzt, ihre IoU-Schwelle aber
knapp verfehlt, erscheint bei 0,50 gleichzeitig als Falsch-Positiv (rot) und
als Falsch-Negativ (blau).

Farbkonvention identisch zu src/evaluate.py: gruen = Treffer, rot =
Falsch-Positiv, blau = Falsch-Negativ.

Aufruf (aus dem Projektwurzelverzeichnis):
    .venv/bin/python paper/tools/fig_qual_iou.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from evaluate import match, load_gt                       # noqa: E402
from ultralytics import YOLO                              # noqa: E402

WEIGHTS = "runs/baseline/weights/best.pt"
ZIEL = os.path.join(os.path.dirname(__file__), "..", "abbildungen")

# Split, Bild, Labelverzeichnis, native imgsz, Betriebspunkt aus
# docs/experiments/final_eval.json
FAELLE = [
    dict(name="test", bild="data/test/images/test5.jpg",
         labels="data/test/labels", imgsz=1600,
         conf=0.26276811957359314),
]
SCHWELLEN = (0.50,)
BREITE = 1600          # Zielbreite der aufbereiteten Abbildung
GRUEN = (0, 220, 0)
ROT = (0, 0, 255)
BLAU = (255, 90, 0)


def rendern(im, pk, ck, gt, thr, conf_thr):
    tp, _, gidx, _ = match(pk, ck, gt, thr)
    hit = set(gidx[tp].tolist())
    out = im.copy()
    for j, b in enumerate(gt):                        # FN blau
        if j not in hit:
            cv2.rectangle(out, (int(b[0]), int(b[1])),
                          (int(b[2]), int(b[3])), BLAU, 2)
    for i, b in enumerate(pk):                        # TP gruen / FP rot
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      GRUEN if tp[i] else ROT, 2)
    kopf = (f"Zuordnung ab IoU {thr:.2f}   "
            f"gruen TP={int(tp.sum())}   rot FP={int((~tp).sum())}   "
            f"blau FN={len(gt) - len(hit)}   conf>={conf_thr:.3f}")
    cv2.rectangle(out, (0, 0), (out.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(out, kopf, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out, int(tp.sum()), int((~tp).sum()), len(gt) - len(hit)


def main():
    os.makedirs(ZIEL, exist_ok=True)
    model = YOLO(WEIGHTS)
    for fall in FAELLE:
        gt, _ = load_gt(fall["bild"], fall["labels"])
        r = model.predict(fall["bild"], verbose=False, imgsz=fall["imgsz"],
                          conf=0.001, iou=0.7, max_det=1000, device=0)[0]
        p = r.boxes.xyxy.cpu().numpy().astype(np.float64)
        c = r.boxes.conf.cpu().numpy().astype(np.float64)
        keep = c >= fall["conf"]
        pk, ck = p[keep], c[keep]
        im = cv2.imread(fall["bild"])
        print(f"{fall['name']}: {len(gt)} GT, {len(pk)} Detektionen "
              f"ab conf {fall['conf']:.3f}")
        for thr in SCHWELLEN:
            bild, tp, fp, fn = rendern(im, pk, ck, gt, thr, fall["conf"])
            s = BREITE / bild.shape[1]
            if s < 1.0:
                bild = cv2.resize(bild, None, fx=s, fy=s,
                                  interpolation=cv2.INTER_AREA)
            datei = os.path.join(
                ZIEL, f"qual_{fall['name']}_iou{int(round(thr*100)):03d}.jpg")
            cv2.imwrite(os.path.normpath(datei), bild,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  IoU {thr:.2f}: TP {tp}  FP {fp}  FN {fn}"
                  f"   -> {os.path.basename(datei)}")


if __name__ == "__main__":
    main()
