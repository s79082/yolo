#!/usr/bin/env python3
"""Teilt das Quellmaterial Pixel mit den beiden Bewertungsbildern?

Objektausschnitte (cutouts/) und Clean Plates (bg/) wurden aus Realaufnahmen
geschnitten. Ein unveraenderter Ausschnitt eines Bewertungsbildes findet sich
per Vorlagenabgleich (normierte Kreuzkorrelation, cv2.TM_CCOEFF_NORMED) dort
mit einer Korrelation nahe 1 wieder. Gesucht wird ueber mehrere Skalen; als
Treffer gilt eine Korrelation >= 0,99.

Gegenprobe: Fuer die Dateien unter cutouts/_aus_val/ und
bg/_aus_bewertungsbildern/ muss der Abgleich anschlagen. Je Treffer werden
die ueberlappten annotierten Boxen und der Flaechenanteil festgehalten.

Aufruf (aus dem Projektwurzelverzeichnis):
    python src/ueberlappung.py   -> docs/experiments/ueberlappung.json
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np

from occluders import CLEAN_PLATES

BEWERTUNG = {
    "val": ("data/val/images/PXL_20251204_073853552.jpg",
            "data/val/labels/PXL_20251204_073853552.txt"),
    "test": ("data/test/images/test5.jpg", "data/test/labels/test5.txt"),
}
SCHWELLE = 0.99
OUT = "docs/experiments/ueberlappung.json"


def boxen(pfad, w, h):
    lab = np.loadtxt(pfad, ndmin=2)
    cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def bester_abgleich(bild, vorlage, skalen):
    """Beste Korrelation ueber alle Skalen -> (score, skala, x0, y0, x1, y1)."""
    H, W = bild.shape
    best = (-1.0, 1.0, 0, 0, 0, 0)
    for s in skalen:
        v = vorlage if s == 1.0 else cv2.resize(
            vorlage, None, fx=s, fy=s,
            interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        h, w = v.shape
        if h >= H or w >= W or min(h, w) < 8:
            continue
        r = cv2.matchTemplate(bild, v, cv2.TM_CCOEFF_NORMED)
        _, mx, _, (x, y) = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (float(mx), float(s), x, y, x + w, y + h)
    return best


def ueberlappung(box, gt):
    """Groesste IoU und Zahl der ueberlappenden GT-Boxen fuer ein Rechteck."""
    x0, y0, x1, y1 = box
    ix = np.clip(np.minimum(gt[:, 2], x1) - np.maximum(gt[:, 0], x0), 0, None)
    iy = np.clip(np.minimum(gt[:, 3], y1) - np.maximum(gt[:, 1], y0), 0, None)
    inter = ix * iy
    flaeche = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    iou = inter / (flaeche + (x1 - x0) * (y1 - y0) - inter)
    return float(iou.max()), int((inter > 0).sum())


def pruefe(pfade, bilder, skalen):
    out = {}
    for path in pfade:
        v = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        eintrag = {}
        for name, (b, gt) in bilder.items():
            sc, s, x0, y0, x1, y1 = bester_abgleich(b, v, skalen)
            e = dict(score=round(sc, 4), skala=round(s, 3))
            if sc >= SCHWELLE:
                iou, n_gt = ueberlappung((x0, y0, x1, y1), gt)
                e.update(box=[x0, y0, x1, y1],
                         flaechenanteil=round((x1 - x0) * (y1 - y0) / b.size, 4),
                         iou_bester_gt=round(iou, 3), gt_ueberlappt=n_gt)
            eintrag[name] = e
        out[path] = eintrag
    return out


def treffer(gruppe):
    return {p: {n: v for n, v in e.items() if v["score"] >= SCHWELLE}
            for p, e in gruppe.items()
            if any(v["score"] >= SCHWELLE for v in e.values())}


def main():
    bilder = {}
    for name, (bpfad, lpfad) in BEWERTUNG.items():
        b = cv2.imread(bpfad, cv2.IMREAD_GRAYSCALE)
        bilder[name] = (b, boxen(lpfad, b.shape[1], b.shape[0]))

    # Clean Plates sind grosse Ausschnitte, grobe Skalenstufen genuegen;
    # Objektausschnitte sind ~30 px gross und brauchen feinere Stufen.
    s_platte = np.linspace(0.3, 1.2, 37)
    s_objekt = np.linspace(0.5, 2.0, 31)
    rep = {
        "schwelle": SCHWELLE,
        "aktuell": {
            "clean_plates": pruefe(CLEAN_PLATES, bilder, s_platte),
            "ausschnitte": pruefe(sorted(glob.glob("cutouts/*.png")),
                                  bilder, s_objekt),
        },
        "aussortiert": {
            "clean_plates": pruefe(sorted(glob.glob("bg/_aus_bewertungsbildern/*")),
                                   bilder, s_platte),
            "ausschnitte": pruefe(sorted(glob.glob("cutouts/_aus_val/*.png")),
                                  bilder, s_objekt),
        },
    }
    rep["zusammenfassung"] = {
        teil: {art: treffer(g) for art, g in rep[teil].items()}
        for teil in ("aktuell", "aussortiert")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(json.dumps(rep["zusammenfassung"], indent=2, ensure_ascii=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
