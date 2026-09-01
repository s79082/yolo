#!/usr/bin/env python3
"""Bereitet die Abbildungen des LaTeX-Projekts aus den Rohdaten des Repositorys
auf: skaliert auf eine fuer den Satz sinnvolle Kantenlaenge herunter und legt
sie unter paper/abbildungen/ ab.

Aufruf (aus dem Projektwurzelverzeichnis):
    .venv/bin/python paper/tools/abbildungen_vorbereiten.py
"""
import os
import cv2

ZIEL = os.path.join(os.path.dirname(__file__), "..", "abbildungen")
os.makedirs(ZIEL, exist_ok=True)

# (Quelle, Zielname, maximale Kantenlaenge, JPEG-Qualitaet oder None fuer PNG)
JOBS = [
    ("PXL_20251204_073837682.jpg",                  "real_stapel.jpg",        2000, 88),
    ("docs/matting_check.png",                      "matting_check.jpg",      2200, 92),
    ("docs/bars_check.png",                         "okkluder_check.png",      800, None),
    ("docs/label_check.png",                        "label_check.jpg",        1800, 92),
    ("docs/previews/preview_train_000.jpg",         "synth_000.jpg",          1024, 90),
    ("docs/previews/preview_train_001.jpg",         "synth_001.jpg",          1024, 90),
    ("docs/previews/preview_train_002.jpg",         "synth_002.jpg",          1024, 90),
    ("docs/previews/preview_train_003.jpg",         "synth_003.jpg",          1024, 90),
    ("docs/previews/preview_train_004.jpg",         "synth_004.jpg",          1024, 90),
    ("docs/previews/preview_train_005.jpg",         "synth_005.jpg",          1024, 90),
    ("docs/fig_ablation_beispiele.png",             "ablation_beispiele.jpg", 2040, 92),
    ("docs/fig_ablation.png",                       "ablation.png",           1950, None),
    ("docs/fig_imgsz_sweep.png",                    "imgsz_sweep.png",        1950, None),
    ("docs/fig_training.png",                       "training.png",           1950, None),
    ("docs/fig_pr_curves.png",                      "pr_kurven.png",          1430, None),
    ("docs/fig_models.png",                         "modelle.png",            1950, None),
    ("docs/qual_val_PXL_20251204_073853552.jpg",    "qual_val.jpg",           2000, 90),
    ("docs/qual_test_test5.jpg",                    "qual_test.jpg",          1600, 90),
]


def main():
    for src, dst, maxside, q in JOBS:
        if not os.path.exists(src):
            print(f"  fehlt: {src}")
            continue
        im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"  unlesbar: {src}")
            continue
        h, w = im.shape[:2]
        s = maxside / float(max(h, w))
        if s < 1.0:
            im = cv2.resize(im, (int(round(w * s)), int(round(h * s))),
                            interpolation=cv2.INTER_AREA)
        out = os.path.join(ZIEL, dst)
        if q is None:
            cv2.imwrite(out, im)
        else:
            cv2.imwrite(out, im, [cv2.IMWRITE_JPEG_QUALITY, q])
        print(f"  {src} -> {dst}  {im.shape[1]}x{im.shape[0]}")


if __name__ == "__main__":
    main()
