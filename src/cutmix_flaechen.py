#!/usr/bin/env python3
"""Wie oft faende CutMix in den erzeugten Bildern ueberhaupt eine Flaeche?

Die CutMix-Implementierung der Werkzeugkette (ultralytics/data/augment.py,
Klasse CutMix) zieht num_areas Kandidatenrechtecke und behaelt nur eines, das
KEINE Box des Primaerbildes ueberlappt; findet sich keines, wird die
Augmentierung uebersprungen. Ob der Parameter in diesem Datensatz ueberhaupt
wirksam waere, haengt damit an der Objektdichte und ist messbar.

Nachgebildet sind _rand_bbox und die Bedingung aus get_params:
    lam ~ Beta(beta, beta);  cut_ratio = sqrt(1 - lam)
    cx, cy gleichverteilt;   Rechteck auf das Bild geklemmt
    brauchbar, wenn die Summe der Ueberlappungsflaechen null ist

Aufruf (aus dem Projektwurzelverzeichnis):
    .venv/bin/python src/cutmix_flaechen.py [--labels <Datensatz>/labels]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from laeufe import REFERENZ


def rand_bbox(rng, w, h, beta):
    lam = rng.beta(beta, beta)
    cr = np.sqrt(1.0 - lam)
    cw, ch = int(w * cr), int(h * cr)
    cx, cy = rng.integers(w), rng.integers(h)
    return (np.clip(cx - cw // 2, 0, w), np.clip(cy - ch // 2, 0, h),
            np.clip(cx + cw // 2, 0, w), np.clip(cy + ch // 2, 0, h))


def boxen(pfad, w, h):
    out = []
    for zeile in open(pfad):
        p = zeile.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.asarray(out, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",
                    default=os.path.join(REFERENZ["daten"], "labels"))
    ap.add_argument("--canvas", type=int, default=1024)
    ap.add_argument("--num-areas", type=int, default=3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    w = h = a.canvas
    dateien = sorted(glob.glob(os.path.join(a.labels, "*.txt")))
    if not dateien:
        raise SystemExit("keine Labeldateien unter " + a.labels)

    n = leer = feuert = 0
    frei_je_bild = []
    for f in dateien:
        b = boxen(f, w, h)
        n += 1
        if len(b) == 0:                      # Negativbild: jede Flaeche ist frei
            leer += 1
            feuert += 1
            continue
        frei = 0
        for _ in range(a.num_areas):
            x1, y1, x2, y2 = rand_bbox(rng, w, h, a.beta)
            ix = np.clip(np.minimum(b[:, 2], x2) - np.maximum(b[:, 0], x1), 0, None)
            iy = np.clip(np.minimum(b[:, 3], y2) - np.maximum(b[:, 1], y1), 0, None)
            if (ix * iy).sum() <= 0:
                frei += 1
        frei_je_bild.append(frei)
        if frei:
            feuert += 1

    pos = n - leer
    print(f"Labelverzeichnis        {a.labels}  ({n} Bilder, davon {leer} ohne Objekt)")
    print(f"Kandidatenflaechen      {a.num_areas} je Bild, Beta({a.beta}, {a.beta})")
    print(f"mindestens eine frei    {feuert} Bilder ({100 * feuert / n:.1f} %)")
    print(f"nur Positivbilder       {feuert - leer} von {pos} ({100 * (feuert - leer) / pos:.1f} %)")
    print(f"freie Flaechen je Bild  {np.mean(frei_je_bild):.2f} von {a.num_areas} (Positivbilder)")


if __name__ == "__main__":
    main()
