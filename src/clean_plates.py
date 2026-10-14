#!/usr/bin/env python3
"""Clean Plates aus IMG_3884.JPG und IMG_3886.JPG.

Beide Fotos zeigen einen Stapel frontal von der Laengsseite. Der Ausschnitt
liegt im Inneren der Stapelflaeche, damit die Rohrenden links und rechts
draussen bleiben, und reicht nach unten bis zum Boden.

Ein Rohr ist in den Fotos rund 40 px dick. Jeder Ausschnitt wird flaechentreu
(INTER_AREA) so verkleinert, dass die Achsensuche aus src/occluders.py einen
Rohrdurchmesser im Bereich der uebrigen Clean Plates misst (16-21 px).
Gespeichert wird als PNG, damit keine weitere JPEG-Generation entsteht.

Aufruf (aus dem Projektwurzelverzeichnis):
    python src/clean_plates.py
"""
import cv2
import numpy as np

from occluders import dominant_angle, find_axes, rotate_keep

# (Quelle, Ausschnitt x0, y0, x1, y1 im Original, Skalierung, Ziel, Soll-pipe_d)
PLATES = [
    ("IMG_3884.JPG", (1000, 600, 4100, 4284), 0.40, "bg/img3884_cutted.png", 18),
    ("IMG_3886.JPG", (950, 600, 4000, 4284), 0.50, "bg/img3886_cutted.png", 20),
]


def gemessener_durchmesser(img):
    """Rohrdurchmesser wie in occluders.build: Median der Achsabstaende."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ang = (dominant_angle(gray) + 45.0) % 90.0 - 45.0
    rot = cv2.cvtColor(rotate_keep(img, ang), cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = rot.shape
    m = int(0.06 * min(H, W)) + 2
    axes = find_axes(rot[m:H - m, m:W - m])
    return int(np.median(np.diff(axes))), ang


def main():
    for src, (x0, y0, x1, y1), s, out, soll in PLATES:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        crop = img[y0:y1, x0:x1]
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        d, ang = gemessener_durchmesser(crop)
        cv2.imwrite(out, crop)
        print(f"{out}: {crop.shape[1]}x{crop.shape[0]} px, Winkel {ang:+.1f} Grad, "
              f"Rohrdurchmesser {d} px (Soll {soll})")
        if d != soll:
            raise SystemExit(f"Rohrdurchmesser {d} != {soll}: Skalierung pruefen")


if __name__ == "__main__":
    main()
