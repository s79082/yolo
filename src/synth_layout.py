"""
Layoutstatistik der erzeugten Labels (synthetische Spalte von Tabelle 4.3).

Misst an einem erzeugten Datensatz dieselben Kenngroessen, die
src/analyze_data.py an den realen Labels misst: Abstand zum naechsten Nachbarn
in Objektdurchmessern, Anteil beruehrender Paare (< 1,2 Durchmesser),
Nachbarn innerhalb von 5 Durchmessern, Groessenverhaeltnis zum naechsten
Nachbarn, Durchmesser als sqrt(Flaeche) und Seitenverhaeltnis w/h.

Konvention, an der die Zahlen der Arbeit haengen:
  - ausgewertet werden die ersten 1500 Labeldateien in sortierter Reihenfolge;
  - Nachbarn zaehlen nur innerhalb desselben Bildes;
  - gezaehlt werden nur Objekte aus Bildern mit mindestens zwei Labels, auch
    fuer Durchmesser und Seitenverhaeltnis. Bei synth/train sind das 34027 der
    34072 Labels; mit allen sinkt der Median-Durchmesser von 23,5 auf 23,4 px.

Aufruf:  python src/synth_layout.py
Ausgabe: docs/experiments/synth_layout.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

DATASETS = ("synth/train", "synth/abl_A1_baseline")
N_IMAGES = 1500
CANVAS = 1024                       # Leinwand des Generators in px
OUT = "docs/experiments/synth_layout.json"


def layout_stats(path, n_images=N_IMAGES, size=CANVAS):
    files = sorted(glob.glob(os.path.join(path, "labels", "*.txt")))[:n_images]
    nn, neigh, ratio, diam, aspect = [], [], [], [], []
    for f in files:
        rows = [q.split() for q in open(f)]
        rows = [q for q in rows if len(q) >= 5]
        if len(rows) < 2:
            continue
        a = np.array([[float(x) for x in q[1:5]] for q in rows]) * size
        C, B = a[:, :2], a[:, 2:4]
        d = np.sqrt(B[:, 0] * B[:, 1])
        D = np.linalg.norm(C[:, None] - C[None], axis=2) + np.eye(len(C)) * 1e9
        j = D.argmin(1)
        nn += list(D.min(1) / d)
        neigh += [(D[i] < 5 * d[i]).sum() for i in range(len(C))]
        ratio += list(np.maximum(d, d[j]) / np.minimum(d, d[j]))
        diam += list(d)
        aspect += list(B[:, 0] / B[:, 1])
    nn = np.array(nn)
    return dict(
        n=len(diam),
        nn=[float(np.percentile(nn, q)) for q in (10, 50, 90)],
        touch=float((nn < 1.2).mean()),
        neigh_med=float(np.median(neigh)),
        neigh_max=float(np.max(neigh)),
        ratio_med=float(np.median(ratio)),
        d=[float(np.percentile(diam, q)) for q in (5, 50, 95)],
        ar=[float(np.percentile(aspect, q)) for q in (10, 50, 90)],
    )


if __name__ == "__main__":
    out = {}
    for p in DATASETS:
        if not os.path.isdir(os.path.join(p, "labels")):
            print(f"  {p}: keine Labels, uebersprungen")
            continue
        out[p] = layout_stats(p)
        r = out[p]
        print(f"  {p}: n={r['n']}  NN p10/med/p90 = "
              f"{r['nn'][0]:.2f} / {r['nn'][1]:.2f} / {r['nn'][2]:.2f}  "
              f"beruehrend {r['touch']:.1%}  Durchmesser p05/med/p95 = "
              f"{r['d'][0]:.1f} / {r['d'][1]:.1f} / {r['d'][2]:.1f} px")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("->", OUT)
