"""
Auswertung von PipeNet.

Verwendet BEWUSST dieselben Metrikfunktionen wie der YOLO-Teil
(../../src/evaluate.py: match, pr_curve, average_precision). Damit sind die
Zahlen in cnn/docs/ERGEBNISSE.md direkt mit denen aus docs/ERGEBNISSE.md
vergleichbar - es gibt keine zweite Metrikimplementierung, die abweichen
koennte.

INFERENZPARAMETER von PipeNet
-----------------------------
  scale     Eingangsskalierung. Ersetzt YOLOs 'imgsz'. PipeNet ist vollstaendig
            konvolutional und arbeitet auf der nativen Bildgroesse; es gibt
            kein Letterboxing und keine feste Netzeingangsgroesse. Skaliert
            wird nur, um die Objektgroesse in die Mitte der Trainingsverteilung
            (Median 22.5 px) zu schieben.
  thresh    Peak-Schwelle auf der Heatmap. Fuer mAP niedrig (0.01), fuer den
            Betriebspunkt aus der F1-Kurve bestimmt.
  topk      Obergrenze der Detektionen je Bild.
  pool      Fenstergroesse der Peak-Extraktion (3 = 8er-Nachbarschaft).
            ERSETZT die NMS-IoU-Schwelle vollstaendig.
  tta       Mittelung ueber die Diedergruppe D4 (4 Drehungen x 2 Spiegelungen),
            auf einem Pixelgitter exakt.

Der Modulname ist bewusst NICHT "evaluate": das Hauptprojekt hat unter
../../src/evaluate.py ein gleichnamiges Modul, dessen Metrikfunktionen hier
importiert werden. Zwei Module gleichen Namens im Suchpfad haetten sich
gegenseitig verdeckt.

Aufruf:
    python cnn/src/eval_pipenet.py --weights cnn/runs/pipenet/weights/best.pt
    python cnn/src/eval_pipenet.py --weights ... --sweep-scale
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
cv2.setNumThreads(0)

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dataset import normalize_gpu                       # noqa: E402
from decode import predict_image, selftest_equivariance  # noqa: E402
from model import PipeNet, OUT_STRIDE                   # noqa: E402
from evaluate import (average_precision, load_gt, match,  # noqa: E402
                      pr_curve)

SPLITS = {
    "val": (f"{ROOT}/data/val/images", f"{ROOT}/data/val/labels"),
    "test": (f"{ROOT}/data/test/images", f"{ROOT}/data/test/labels"),
    "sval": (f"{ROOT}/data/sval/images", f"{ROOT}/data/sval/labels"),
}


def load_model(weights, device):
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    m = PipeNet(pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


@torch.no_grad()
def run_split(model, split, device, scale=1.0, thresh=0.01, topk=2000,
              tta=False, limit=None, save_vis=None):
    img_dir, lbl_dir = SPLITS[split]
    files = sorted(sum([glob.glob(os.path.join(img_dir, e))
                        for e in ("*.jpg", "*.png")], []))
    if limit:
        files = files[:limit]
    P, C, G, n_gt = [], [], [], 0
    t0 = time.time()
    for f in files:
        gt, (W, H) = load_gt(f, lbl_dir)
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        if scale != 1.0:
            im = cv2.resize(im, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_LINEAR if scale > 1
                            else cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(
            im[:, :, ::-1].transpose(2, 0, 1)))[None]
        box, sc = predict_image(model, normalize_gpu(x, device), OUT_STRIDE,
                                thresh, topk, tta=tta)
        if scale != 1.0 and len(box):
            box = box / scale
        P.append(box.astype(np.float64))
        C.append(sc.astype(np.float64))
        G.append(gt)
        n_gt += len(gt)
    wall = time.time() - t0

    aps = {}
    for t in np.arange(0.5, 0.96, 0.05):
        tps, cfs = [], []
        for p, c, g in zip(P, C, G):
            tp, _, _, _ = match(p, c, g, t)
            tps.append(tp)
            cfs.append(c)
        tp = np.concatenate(tps) if tps else np.zeros(0, bool)
        cf = np.concatenate(cfs) if cfs else np.zeros(0)
        pre, rec, _ = pr_curve(tp, cf, n_gt)
        aps[round(float(t), 2)] = average_precision(pre, rec)

    tps, cfs, ious = [], [], []
    for p, c, g in zip(P, C, G):
        tp, iou, _, _ = match(p, c, g, 0.5)
        tps.append(tp)
        cfs.append(c)
        ious.append(iou[tp])
    tp = np.concatenate(tps) if tps else np.zeros(0, bool)
    cf = np.concatenate(cfs) if cfs else np.zeros(0)
    pre, rec, cs = pr_curve(tp, cf, n_gt)
    f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-12)
    k = int(np.argmax(f1)) if len(f1) else 0
    best_conf = float(cs[k]) if len(cs) else 0.0
    cnt = np.array([[int((c >= best_conf).sum()), len(g)]
                    for c, g in zip(C, G)], float)
    err = cnt[:, 0] - cnt[:, 1]

    res = dict(
        split=split, n_images=len(files), n_gt=int(n_gt),
        infer=dict(scale=scale, thresh=thresh, topk=topk, tta=tta,
                   pool=3, stride=OUT_STRIDE),
        mAP50=aps[0.5], mAP75=aps[0.75],
        mAP50_95=float(np.mean(list(aps.values()))),
        ap_per_iou={str(a): round(b, 4) for a, b in aps.items()},
        best_f1=float(f1[k]) if len(f1) else 0.0, best_f1_conf=best_conf,
        precision_at_best_f1=float(pre[k]) if len(pre) else 0.0,
        recall_at_best_f1=float(rec[k]) if len(rec) else 0.0,
        mean_iou_of_tp=float(np.mean(np.concatenate(ious)))
        if any(len(i) for i in ious) else 0.0,
        count_pred=int(cnt[:, 0].sum()), count_gt=int(cnt[:, 1].sum()),
        count_mae=float(np.mean(np.abs(err))),
        count_rel_err=float(np.sum(np.abs(err)) / max(1, n_gt)),
        wall_s=round(wall, 3),
        _pr=dict(precision=pre.tolist(), recall=rec.tolist(),
                 conf=cs.tolist(), f1=f1.tolist()),
    )

    if save_vis:
        for f, p, c, g in zip(files[:6], P, C, G):
            keep = c >= best_conf
            pk = p[keep]
            tpk, _, gidx, _ = match(pk, c[keep], g, 0.5)
            im = cv2.imread(f)
            hit = set(gidx[tpk].tolist())
            for j, b in enumerate(g):
                if j not in hit:
                    cv2.rectangle(im, (int(b[0]), int(b[1])),
                                  (int(b[2]), int(b[3])), (255, 90, 0), 2)
            for i2, b in enumerate(pk):
                col = (0, 220, 0) if tpk[i2] else (0, 0, 255)
                cv2.rectangle(im, (int(b[0]), int(b[1])),
                              (int(b[2]), int(b[3])), col, 2)
            cv2.putText(im, f"PipeNet  gruen TP={int(tpk.sum())}  "
                            f"rot FP={int((~tpk).sum())}  "
                            f"blau FN={len(g) - len(hit)}  "
                            f"conf>={best_conf:.3f}  scale={scale}",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2)
            out = save_vis.replace("SPLIT", split).replace(
                "NAME", os.path.splitext(os.path.basename(f))[0])
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            cv2.imwrite(out, im)
    return res


def line(r):
    return (f"{r['split']:5s} scale={r['infer']['scale']:<4} "
            f"tta={str(r['infer']['tta']):5s} "
            f"mAP50={r['mAP50']:.4f} mAP75={r['mAP75']:.4f} "
            f"mAP50-95={r['mAP50_95']:.4f} F1={r['best_f1']:.4f}"
            f"@{r['best_f1_conf']:.3f} P={r['precision_at_best_f1']:.3f} "
            f"R={r['recall_at_best_f1']:.3f} IoU={r['mean_iou_of_tp']:.3f} "
            f"Zaehlung {r['count_pred']}/{r['count_gt']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="cnn/runs/pipenet/weights/best.pt")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--thresh", type=float, default=0.01)
    ap.add_argument("--topk", type=int, default=2000)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--sweep-scale", action="store_true")
    ap.add_argument("--sval-limit", type=int, default=100)
    ap.add_argument("--vis", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    dev = torch.device(a.device)
    model, ck = load_model(a.weights, dev)
    print(f"Modell {a.weights}: {ck.get('n_params', 0) / 1e6:.2f} M Parameter, "
          f"Epoche {ck.get('epoch')}, Stride {ck.get('out_stride')}")

    out = {"weights": a.weights, "n_params": ck.get("n_params"),
           "epoch": ck.get("epoch"), "results": [], "sweep_scale": []}

    if a.sweep_scale:
        print("\n--- Eingangsskalierung (ersetzt YOLOs imgsz) ---")
        for s in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8, 2.1):
            for sp in a.splits.split(","):
                r = run_split(model, sp, dev, scale=s, thresh=a.thresh,
                              topk=a.topk,
                              limit=a.sval_limit if sp == "sval" else None)
                r.pop("_pr", None)
                out["sweep_scale"].append(r)
                print("   ", line(r))

    print("\n--- Endauswertung ---")
    for sp in a.splits.split(","):
        for tta in ((False, True) if not a.sweep_scale else (a.tta,)):
            r = run_split(model, sp, dev, scale=a.scale, thresh=a.thresh,
                          topk=a.topk, tta=tta, save_vis=a.vis,
                          limit=a.sval_limit if sp == "sval" else None)
            out["results"].append(r)
            print("   ", line(r), f"[{r['wall_s']:.2f}s]")

    print("\n--- Aequivarianz-Selbsttest der Rotations-TTA ---")
    st = selftest_equivariance(model, str(dev))
    out["equivariance"] = st
    for k, v in st.items():
        print(f"    {k:8s} Korrelation {v:+.4f}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print("\n->", a.out)


if __name__ == "__main__":
    main()
