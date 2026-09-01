#!/usr/bin/env python3
"""Abbildung zum systematischen Groessenversatz zwischen Handannotation und
Generatorlabel.

Zeigt drei Ausschnitte aus dem Testbild, bei denen die Detektion (rot) auf einem
annotierten Objekt (blau) sitzt, die IoU aber knapp unter der Zuordnungsschwelle
0,5 liegt.  Solche Faelle zaehlen gleichzeitig als Falsch-Positiv und als
Falsch-Negativ, obwohl das Objekt gefunden wurde.

Aufruf (aus dem Projektwurzelverzeichnis):
    .venv/bin/python paper/tools/fig_boxversatz.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from evaluate import iou_matrix, match, load_gt          # noqa: E402
from ultralytics import YOLO                             # noqa: E402

BILD = "data/test/images/test5.jpg"
LABELS = "data/test/labels"
WEIGHTS = "runs/baseline/weights/best.pt"
IMGSZ = 1600
CONF_BETRIEBSPUNKT = 0.26343148946762085     # aus docs/experiments/final_eval.json
ZIEL = os.path.join(os.path.dirname(__file__), "..", "abbildungen",
                    "boxversatz.png")

ZOOM = 9          # Vergroesserungsfaktor der Ausschnitte
RAND = 1.9        # Ausschnittkante in GT-Durchmessern
N_FAELLE = 3

BLAU = (255, 120, 0)      # BGR - wie in den Qualitativbildern
ROT = (0, 0, 255)


def main():
    model = YOLO(WEIGHTS)
    gt, _ = load_gt(BILD, LABELS)
    r = model.predict(BILD, verbose=False, imgsz=IMGSZ, conf=0.001, iou=0.7,
                      max_det=1000, device=0)[0]
    p = r.boxes.xyxy.cpu().numpy().astype(np.float64)
    c = r.boxes.conf.cpu().numpy().astype(np.float64)
    keep = c >= CONF_BETRIEBSPUNKT
    pk, ck = p[keep], c[keep]
    tp, _, _, _ = match(pk, ck, gt, 0.5)

    # Falsch-Positive, die auf einer annotierten Box sitzen, nach IoU sortiert
    M = iou_matrix(pk, gt)
    kandidaten = []
    for i in np.where(~tp)[0]:
        j = int(M[i].argmax())
        if M[i, j] >= 0.35:
            kandidaten.append((float(M[i, j]), i, j))
    kandidaten.sort(reverse=True)
    kandidaten = kandidaten[:N_FAELLE]

    im = cv2.imread(BILD)
    kacheln = []
    for iou, i, j in kandidaten:
        gx0, gy0, gx1, gy1 = gt[j]
        d = np.sqrt((gx1 - gx0) * (gy1 - gy0))
        cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
        half = RAND * d
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        x1, y1 = int(round(cx + half)), int(round(cy + half))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(im.shape[1], x1), min(im.shape[0], y1)
        crop = im[y0:y1, x0:x1]
        big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM,
                         interpolation=cv2.INTER_LANCZOS4)

        def sk(bx):     # Boxkoordinaten in den vergroesserten Ausschnitt
            return [int(round((bx[0] - x0) * ZOOM)), int(round((bx[1] - y0) * ZOOM)),
                    int(round((bx[2] - x0) * ZOOM)), int(round((bx[3] - y0) * ZOOM))]

        g = sk(gt[j])
        q = sk(pk[i])
        cv2.rectangle(big, (g[0], g[1]), (g[2], g[3]), BLAU, 3)
        cv2.rectangle(big, (q[0], q[1]), (q[2], q[3]), ROT, 3)

        dp = np.sqrt((pk[i, 2] - pk[i, 0]) * (pk[i, 3] - pk[i, 1]))
        txt = f"IoU {iou:.3f}   Objekt {d:.0f} px   Box {dp/d*100:.0f} %"
        band = np.full((44, big.shape[1], 3), 255, np.uint8)
        cv2.putText(band, txt, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (30, 30, 30), 2, cv2.LINE_AA)
        kacheln.append(np.vstack([big, band]))

    hoehe = min(k.shape[0] for k in kacheln)
    kacheln = [cv2.resize(k, (int(k.shape[1] * hoehe / k.shape[0]), hoehe),
                          interpolation=cv2.INTER_AREA) for k in kacheln]
    luecke = np.full((hoehe, 14, 3), 255, np.uint8)
    out = kacheln[0]
    for k in kacheln[1:]:
        out = np.hstack([out, luecke, k])

    os.makedirs(os.path.dirname(ZIEL), exist_ok=True)
    cv2.imwrite(os.path.normpath(ZIEL), out,
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print("geschrieben:", os.path.normpath(ZIEL), out.shape)
    for iou, i, j in kandidaten:
        d = np.sqrt((gt[j, 2] - gt[j, 0]) * (gt[j, 3] - gt[j, 1]))
        dp = np.sqrt((pk[i, 2] - pk[i, 0]) * (pk[i, 3] - pk[i, 1]))
        print(f"  IoU {iou:.3f}  conf {ck[i]:.3f}  GT {d:.1f} px  "
              f"pred {dp:.1f} px  ({dp/d*100:.0f} %)")


if __name__ == "__main__":
    main()
