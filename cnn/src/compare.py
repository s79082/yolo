"""
Vergleich PipeNet <-> YOLO auf demselben Testrahmen.

Beide Zweige werden mit DENSELBEN Metrikfunktionen ausgewertet
(../../src/evaluate.py). Der einzige Unterschied in der Ansteuerung:

    YOLO     skaliert intern auf eine feste Netzeingangsgroesse `imgsz`
             (Letterbox) und filtert mit NMS.
    PipeNet  ist vollstaendig konvolutional; skaliert wird das Bild selbst
             (`scale`), gefiltert wird per Peak-Extraktion.

Damit die Skalierung vergleichbar ist, wird sie fuer beide Zweige in derselben
Groesse angegeben: der resultierenden Objektgroesse im Netzeingang.
Referenz ist die Trainingsverteilung des Generators (Median 22.5 px).

Die Betriebseinstellung wird fuer JEDES Modell ausschliesslich auf `val`
bestimmt und dann unveraendert auf `test` angewandt.

Aufruf:  python cnn/src/compare.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2
cv2.setNumThreads(0)

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Achtung: es gibt zwei Module namens "evaluate" (Haupt-src und cnn/src).
# Das CNN-Modul heisst deshalb eval_pipenet, damit die Aufloesung eindeutig ist.
from evaluate import evaluate_split as yolo_eval      # ../../src/evaluate.py
from eval_pipenet import load_model, run_split        # cnn/src/eval_pipenet.py

OUT_JSON = os.path.join(ROOT, "cnn/docs/experiments/vergleich.json")

# Reale Objektgroessen (Median, gemessen) je Split
OBJ_MED = {"val": 27.4, "test": 19.6}
IMG_LONG = {"val": 2459, "test": 1600}

YOLO_MODELS = [
    ("yolo11s", f"{ROOT}/runs/baseline/weights/best.pt"),
    ("yolo26s", f"{ROOT}/runs/yolo26s/weights/best.pt"),
    ("yolo26s-p2", f"{ROOT}/runs/yolo26s_p2/weights/best.pt"),
]
# Zwei PipeNet-Laeufe: 60- und 30-Epochen-Kosinusrampe. Ausgewaehlt wird der
# mit der besseren Fitness auf VAL - dieselbe Auswahlregel wie fuer den
# Checkpoint innerhalb eines Laufs.
def _best_pipe():
    cands = []
    for n in ("pipenet", "pipenet_e30"):
        p = f"{ROOT}/cnn/runs/{n}/summary.json"
        if os.path.exists(p):
            cands.append((json.load(open(p))["best_fitness"], n))
    if not cands:
        return ("PipeNet", f"{ROOT}/cnn/runs/pipenet/weights/best.pt")
    fit, n = max(cands)
    return ("PipeNet", f"{ROOT}/cnn/runs/{n}/weights/best.pt")


PIPE = _best_pipe()


def obj_px_yolo(split, imgsz):
    """Objektgroesse im Netzeingang bei gegebener imgsz."""
    return OBJ_MED[split] * imgsz / IMG_LONG[split]


def obj_px_pipe(split, scale):
    return OBJ_MED[split] * scale


def eval_yolo(name, weights, device="0"):
    from ultralytics import YOLO
    m = YOLO(weights)
    npar = sum(p.numel() for p in m.model.parameters())
    base = dict(conf=0.001, iou=0.7, max_det=1000, augment=False,
                quantize=None, device=device)
    grid = [960, 1280, 1600, 1920, 2240, 2464]
    cand = []
    for sz in grid:
        r = yolo_eval(m, "val", {**base, "imgsz": sz})
        r.pop("_pr", None)
        cand.append((sz, r))
    best_sz, rv = max(cand, key=lambda t: t[1]["mAP50_95"])
    rt = yolo_eval(m, "test", {**base, "imgsz": best_sz})
    rt.pop("_pr", None)
    return dict(model=name, family="YOLO", params_M=round(npar / 1e6, 2),
                setting=f"imgsz={best_sz}", nms="Box-NMS (iou 0.7)",
                obj_px_val=round(obj_px_yolo("val", best_sz), 1),
                obj_px_test=round(obj_px_yolo("test", best_sz), 1),
                val=rv, test=rt)


def eval_pipe(name, weights, device="cuda:0", tta=True):
    dev = torch.device(device)
    m, ck = load_model(weights, dev)
    cand = []
    for s in (0.85, 1.0, 1.2, 1.5, 1.8, 2.1):
        r = run_split(m, "val", dev, scale=s, thresh=0.01, tta=tta)
        r.pop("_pr", None)
        cand.append((s, r))
    best_s, rv = max(cand, key=lambda t: t[1]["mAP50_95"])
    rt = run_split(m, "test", dev, scale=best_s, thresh=0.01, tta=tta)
    rt.pop("_pr", None)
    return dict(model=name, family="PipeNet",
                params_M=round(ck["n_params"] / 1e6, 2),
                setting=f"scale={best_s}" + (" +TTA" if tta else ""),
                nms="Peak-Extraktion (3x3, keine Schwelle)",
                obj_px_val=round(obj_px_pipe("val", best_s), 1),
                obj_px_test=round(obj_px_pipe("test", best_s), 1),
                val=rv, test=rt, sweep=[(s, r["mAP50"], r["mAP50_95"])
                                        for s, r in cand])


def row(d, split):
    r = d[split]
    return (d["model"], split, d["params_M"], d["setting"],
            f"{r['mAP50']:.4f}", f"{r['mAP75']:.4f}", f"{r['mAP50_95']:.4f}",
            f"{r['best_f1']:.4f}", f"{r['precision_at_best_f1']:.3f}",
            f"{r['recall_at_best_f1']:.3f}", f"{r['mean_iou_of_tp']:.3f}",
            f"{r['count_pred']}/{r['count_gt']}")


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    res = []
    for name, w in YOLO_MODELS:
        if not os.path.exists(w):
            print("fehlt:", w)
            continue
        d = eval_yolo(name, w)
        res.append(d)
        print(f"  {name:11s} {d['setting']:14s} "
              f"val mAP50 {d['val']['mAP50']:.4f} | "
              f"test mAP50 {d['test']['mAP50']:.4f} "
              f"F1 {d['test']['best_f1']:.4f}")
    for tta in (False, True):
        if not os.path.exists(PIPE[1]):
            print("fehlt:", PIPE[1])
            break
        d = eval_pipe(PIPE[0] + (" +TTA" if tta else ""), PIPE[1], tta=tta)
        res.append(d)
        print(f"  {d['model']:11s} {d['setting']:14s} "
              f"val mAP50 {d['val']['mAP50']:.4f} | "
              f"test mAP50 {d['test']['mAP50']:.4f} "
              f"F1 {d['test']['best_f1']:.4f}")
    json.dump(res, open(OUT_JSON, "w"), indent=2)
    print("->", OUT_JSON)


if __name__ == "__main__":
    main()
