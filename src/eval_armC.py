#!/usr/bin/env python3
"""Auswertung der Arme C0/C1 (Einbettung) -- schreibt eine eigene Ergebnisdatei.

Bewusst getrennt von experiment_neu.py eval: jener Befehl wertet die fuenf
bestehenden Arme neu aus und ueberschreibt ablationen.json, aus dem die Zahlen
der Arbeit stammen. Hier wird nur ergaenzt.

Aufruf (aus dem Projektwurzelverzeichnis):
    .venv/bin/python src/eval_armC.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evaluate import evaluate_split, summary_line          # noqa: E402

OUT = os.path.join(ROOT, "docs", "experiments")
RUNS = os.path.join(ROOT, "runs")

ARME = [
    ("C0_kontrolle", "synth/C0_kontrolle",
     "Kontrolle: vollstaendige Pipeline, mit demselben Code erzeugt wie C1"),
    ("C1_naive_paste", "synth/C1_naive_paste",
     "naives Einkleben: Gegenmassnahmen zu den Signaturen S1-S5 abgeschaltet"),
]


def main():
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    infer = dict(imgsz="native", conf=0.001, iou=0.7, max_det=1000,
                 augment=False, quantize=None, device="0")
    rows = []
    for name, ds, desc in ARME:
        w = os.path.join(RUNS, name, "weights", "best.pt")
        if not os.path.exists(w):
            print("fehlt:", w)
            continue
        model = YOLO(w)
        npar = sum(p.numel() for p in model.model.parameters())
        for s in ("val", "test"):
            r = evaluate_split(model, s, infer)
            r.pop("_pr", None)
            r.update(arm=name, datensatz=ds, beschreibung=desc,
                     weights=os.path.relpath(w, ROOT),
                     params_M=round(npar / 1e6, 2))
            rows.append(r)
            print(f"  {name:16s}", summary_line(r), flush=True)
    p = os.path.join(OUT, "ablationC.json")
    with open(p, "w") as f:
        json.dump(rows, f, indent=2)
    print("->", p)


if __name__ == "__main__":
    main()
