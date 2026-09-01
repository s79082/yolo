#!/usr/bin/env python3
"""Abbildung fuer den P2-Abschnitt der Kurzfassung.

Links:  mAP@0.5 auf dem realen val-Bild ueber der Inferenzaufloesung fuer
        yolo26s (P3-P5) und yolo26s-p2 (P2-P5).
Rechts: Differenz beider Kurven, aufgetragen ueber der effektiven
        Objektgroesse im Netzeingang.  Der schattierte Bereich markiert die
        Zone, in der ein Objekt auf der P3-Ebene (Stride 8) weniger als
        1,5 Gitterzellen belegt.

Quelle: docs/experiments/experiment3_models_imgsz.json
Aufruf (aus dem Projektwurzelverzeichnis):
    .venv/bin/python paper/tools/fig_p2.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

QUELLE = "docs/experiments/experiment3_models_imgsz.json"
ZIEL = os.path.join(os.path.dirname(__file__), "..", "abbildungen", "p2_sweep.png")

# Realmessung val: Median-Objektdurchmesser und Bildbreite (src/analyze_data.py)
D_VAL = 27.4
W_VAL = 2459.0
STRIDE_P3 = 8.0

C_S = "#1f77b4"
C_P2 = "#d62728"


def main():
    rep = json.load(open(QUELLE))
    s = {int(x[0]): x[1] for x in rep["yolo26s"]["sweep"]}
    p2 = {int(x[0]): x[1] for x in rep["yolo26s-p2"]["sweep"]}
    xs = sorted(set(s) & set(p2))

    ys_s = [s[x] for x in xs]
    ys_p2 = [p2[x] for x in xs]
    d_eff = [D_VAL * x / W_VAL for x in xs]          # Objektgroesse im Netzeingang
    delta = [p2[x] - s[x] for x in xs]

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.1))

    # ---------------- links: beide Sweeps ---------------------------------
    a = ax[0]
    a.plot(xs, ys_s, "o-", color=C_S, label="yolo26s (P3\u2013P5)")
    a.plot(xs, ys_p2, "s-", color=C_P2, label="yolo26s-p2 (P2\u2013P5)")
    a.set_xlabel("Inferenzauflösung imgsz (px)")
    a.set_ylabel("mAP@0,5 auf val")
    a.set_xticks(xs)
    a.set_xticklabels([str(x) for x in xs])
    a.grid(alpha=0.3)
    a.legend(loc="lower right", fontsize=9)
    a.set_title("Auflösungssweep auf dem realen val-Bild", fontsize=10)

    at = a.twiny()
    at.set_xlim(a.get_xlim())
    at.set_xticks(xs)
    at.set_xticklabels([f"{v:.0f}" for v in d_eff], fontsize=8)
    at.set_xlabel("effektive Objektgröße im Netzeingang (px)", fontsize=9)

    # ---------------- rechts: Differenz -----------------------------------
    b = ax[1]
    grenze = 1.5 * STRIDE_P3          # 12 px: unter 1,5 Zellen auf P3
    b.axvspan(min(d_eff) - 1.0, grenze, color="0.85", zorder=0)
    b.axhline(0.0, color="0.4", lw=0.8)
    b.bar(d_eff, delta, width=1.1, color=C_P2, zorder=3)
    for x, y in zip(d_eff, delta):
        b.annotate(f"{y*100:+.1f}", (x, y), textcoords="offset points",
                   xytext=(0, 4 if y >= 0 else -12), ha="center", fontsize=8)
    b.set_xlabel("effektive Objektgröße im Netzeingang (px)")
    b.set_ylabel("Δ mAP@0,5   (p2 − s)")
    b.grid(alpha=0.3, axis="y")
    b.set_title("Vorsprung des P2-Kopfes über der Objektgröße", fontsize=10)
    b.set_ylim(-0.005, max(delta) * 1.35)
    b.annotate("Objekt belegt < 1,5 Zellen auf P3 (Stride 8)",
               (grenze + 0.8, max(delta) * 1.12), ha="left", fontsize=8,
               color="0.25")

    bt = b.twiny()
    bt.set_xlim(b.get_xlim())
    ticks = np.arange(4, 32, 4)
    bt.set_xticks(ticks)
    bt.set_xticklabels([f"{t/STRIDE_P3:.1f}" for t in ticks], fontsize=8)
    bt.set_xlabel("Gitterzellen je Objekt auf P3", fontsize=9)

    fig.tight_layout()
    fig.savefig(ZIEL, dpi=150)
    print("geschrieben:", os.path.normpath(ZIEL))
    for x, ds, dp, dd, de in zip(xs, ys_s, ys_p2, delta, d_eff):
        print(f"  imgsz {x:5d}  d_eff {de:5.1f} px  ({de/STRIDE_P3:4.2f} Zellen P3)"
              f"   s {ds:.4f}   p2 {dp:.4f}   delta {dd:+.4f}")


if __name__ == "__main__":
    main()
