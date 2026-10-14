#!/usr/bin/env python3
"""Auswertung der Arme C0/C1 (Einbettung) -- schreibt eigene Ergebnisdateien.

Bewusst getrennt von experiment_neu.py eval: jener Befehl wertet die Arme der
Ablationen A und B aus und ueberschreibt ablationen.json.

  ablationC.json       beide Arme auf val und test
  ablationC_sval.json  jeder Arm auf seinem eigenen synthetischen
                       Validierungssatz (Split "sval_eigen"); zeigt, ob der Arm
                       seine Trainingsdaten gelernt hat

Laeufe und Datensaetze stehen in src/laeufe.py.

Aufruf (aus dem Projektwurzelverzeichnis):
    .venv/bin/python src/eval_armC.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from evaluate import evaluate_split, summary_line          # noqa: E402
from laeufe import ARME_C, gewichte                        # noqa: E402

OUT = os.path.join(ROOT, "docs", "experiments")


def main():
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    infer = dict(imgsz="native", conf=0.001, iou=0.7, max_det=1000,
                 augment=False, quantize=None, device="0")
    rows, sval_rows = [], []
    for name, arm in ARME_C.items():
        w = gewichte(arm["lauf"])
        if not os.path.exists(w):
            print("fehlt:", w)
            continue
        model = YOLO(w)
        npar = sum(p.numel() for p in model.model.parameters())
        meta = dict(arm=name, datensatz=arm["daten"],
                    beschreibung=arm["beschreibung"], weights=w,
                    params_M=round(npar / 1e6, 2))
        for s in ("val", "test"):
            r = evaluate_split(model, s, infer)
            r.pop("_pr", None)
            r.update(meta)
            rows.append(r)
            print(f"  {name:16s}", summary_line(r), flush=True)
        r = evaluate_split(model, "sval_eigen", infer,
                           img_dir=os.path.join(arm["sval"], "images"),
                           gt_dir=os.path.join(arm["sval"], "labels"))
        r.pop("_pr", None)
        r.update(meta, quelle=arm["sval"])
        sval_rows.append(r)
        print(f"  {name:16s}", summary_line(r), flush=True)
    for datei, inhalt in (("ablationC.json", rows),
                          ("ablationC_sval.json", sval_rows)):
        p = os.path.join(OUT, datei)
        with open(p, "w") as f:
            json.dump(inhalt, f, indent=2)
        print("->", p)


if __name__ == "__main__":
    main()
