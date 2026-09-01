"""
Stufe 5: Eigenstaendiges Test- und Auswertungsframework.

Warum nicht einfach `model.val()`?
  1. Nachvollziehbarkeit: fuer eine Abschlussarbeit muss die Metrik selbst
     implementiert und erklaerbar sein, nicht Blackbox.
  2. Ultralytics liefert keine anwendungsnahen Groessen. Fuer das eigentliche
     Ziel - Rohre zaehlen - ist der Zaehlfehler wichtiger als mAP50-95.
  3. Es wird eine freie Variation ALLER Inferenzparameter gebraucht
     (imgsz, conf, iou, max_det, TTA, half) inklusive Sweeps.
Die Ultralytics-Zahlen werden zusaetzlich als Kreuzvergleich mitgefuehrt;
stimmen beide ueberein, ist die eigene Implementierung verifiziert.

Metrikdefinitionen (COCO-konform):
  - Zuordnung greedy nach absteigender Konfidenz, jede GT nur einmal.
  - AP ueber 101-Punkt-Interpolation des Precision-Recall-Verlaufs.
  - mAP50-95 = Mittel der AP ueber IoU-Schwellen 0.50, 0.55, ..., 0.95.
  - F1-optimaler Betriebspunkt = conf, das F1 maximiert (relevant fuer den
    praktischen Einsatz, wo eine feste Schwelle gewaehlt werden muss).

Aufruf:
    python src/evaluate.py --weights runs/baseline/weights/best.pt
    python src/evaluate.py --weights ... --sweep imgsz
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

SPLITS = {
    "val": ("data/val/images", "data/val/labels"),
    "test": ("data/test/images", "data/test/labels"),
    "sval": ("data/sval/images", "data/sval/labels"),
}

# ---------------------------------------------------------------------------
# Inferenzparameter - Werte und Begruendung
# ---------------------------------------------------------------------------
INFER_SPEC = {
    "imgsz": "Netzeingangsgroesse. ENTSCHEIDEND fuer diese Aufgabe: die Objekte "
             "sind im Original 12-36 px gross. Ultralytics skaliert das Bild auf "
             "imgsz (laengste Kante). Bei val (2459x1174) bedeutet imgsz=640 einen "
             "Faktor 0.26 -> die Objekte schrumpfen auf 5-9 px und liegen unter der "
             "Aufloesung des feinsten Detektionskopfes (Stride 8). Der Standardwert "
             "ist deshalb 'native': imgsz = laengste Bildkante, auf ein Vielfaches "
             "von 32 aufgerundet. Dann ist der Massstab 1:1 und entspricht exakt "
             "dem, wofuer die Trainingsdaten kalibriert wurden.",
    "conf": "Konfidenzschwelle. Fuer die mAP-Berechnung 0.001 (praktisch alle "
            "Kandidaten, sonst wird der Recall-Ast der PR-Kurve abgeschnitten). "
            "Fuer den Betriebspunkt wird die F1-optimale Schwelle aus der Kurve "
            "bestimmt und separat berichtet.",
    "iou": "IoU-Schwelle der NMS. 0.7 (Ultralytics-Standard). Unkritisch, weil "
           "benachbarte Rohrenden real mindestens einen Durchmesser Abstand haben "
           "(gemessener 10-%-Quantil-Abstand: 1.0 d) und ihre Boxen damit "
           "IoU < 0.3 haben. Ein zu niedriger Wert wuerde dicht stehende, "
           "tatsaechlich verschiedene Rohre gegenseitig ausloeschen.",
    "max_det": "Maximale Detektionen pro Bild. Der Ultralytics-Standard 300 reicht "
               "fuer val (88 GT) und test (86 GT), waere aber bei einem grossen "
               "Stapel eine stille Obergrenze. Auf 1000 gesetzt, damit die "
               "Metrik nie durch eine Kappungsgrenze verfaelscht wird.",
    "augment": "Test-Time-Augmentation (Mehrfachskalen + Spiegelung). Standard aus; "
               "wird im Sweep geprueft. Kostet ~3x Rechenzeit.",
    "quantize": "Rechengenauigkeit der Inferenz (None = FP32, 'fp16' = halbe "
                "Genauigkeit; gueltig sind 8/16/32, 'int8', 'fp16', 'fp32'). "
                "Standard None, damit die Metrik nicht von der Rechengenauigkeit "
                "abhaengt; im Sweep geprueft. Hinweis: der aeltere Parameter "
                "'half' ist in Ultralytics 8.4.117 deprecated und wird ignoriert "
                "- gemessen liefert half=True bitidentische Ergebnisse zu "
                "half=False. Verwendet wird deshalb 'quantize'.",
    "rect": "Rechteckige Inferenz (nur auf Vielfache von 32 padden statt auf ein "
            "Quadrat). Bei Ultralytics-predict implizit aktiv.",
}


# ---------------------------------------------------------------------------
# Metriken
# ---------------------------------------------------------------------------
def load_gt(img_path, lbl_dir):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    p = os.path.join(lbl_dir, stem + ".txt")
    h, w = cv2.imread(img_path).shape[:2]
    out = []
    if os.path.exists(p):
        for line in open(p):
            f = line.split()
            if len(f) < 5:
                continue
            cx, cy, bw, bh = map(float, f[1:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, dtype=np.float64).reshape(-1, 4), (w, h)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def match(pred, conf, gt, thr):
    """Greedy-Zuordnung nach absteigender Konfidenz.

    Rueckgabe: tp (bool je Prediction), iou der Treffer, Index der getroffenen GT.
    """
    order = np.argsort(-conf)
    tp = np.zeros(len(pred), bool)
    ious = np.zeros(len(pred))
    used = np.zeros(len(gt), bool)
    gidx = -np.ones(len(pred), int)
    if len(gt) and len(pred):
        M = iou_matrix(pred, gt)
        for i in order:
            j = -1
            best = thr
            for k in np.argsort(-M[i]):
                if M[i, k] < best:
                    break
                if not used[k]:
                    j, best = k, M[i, k]
                    break
            if j >= 0:
                tp[i] = True
                ious[i] = M[i, j]
                used[j] = True
                gidx[i] = j
    return tp, ious, gidx, order


def pr_curve(tp, conf, n_gt):
    """Precision-Recall-Verlauf ueber absteigende Konfidenz."""
    o = np.argsort(-conf)
    t = tp[o].astype(np.float64)
    ctp = np.cumsum(t)
    cfp = np.cumsum(1 - t)
    rec = ctp / max(1, n_gt)
    pre = ctp / np.maximum(ctp + cfp, 1e-12)
    return pre, rec, conf[o]


def average_precision(pre, rec):
    """101-Punkt-Interpolation (COCO)."""
    if len(rec) == 0:
        return 0.0
    mpre = np.concatenate(([1.0], pre, [0.0]))
    mrec = np.concatenate(([0.0], rec, [1.0]))
    for i in range(len(mpre) - 2, -1, -1):        # monoton fallende Praezision
        mpre[i] = max(mpre[i], mpre[i + 1])
    q = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(q, mrec, mpre)))


def evaluate_split(model, split, infer, save_vis=None, gt_dir=None, img_dir=None):
    img_dir = img_dir or SPLITS[split][0]
    gt_dir = gt_dir or SPLITS[split][1]
    files = sorted(sum([glob.glob(os.path.join(img_dir, e))
                        for e in ("*.jpg", "*.png", "*.jpeg")], []))
    all_pred, all_conf, all_gt, n_gt = [], [], [], 0
    per_image = []

    for f in files:
        gt, (w, h) = load_gt(f, gt_dir)
        kw = dict(infer)
        kw.pop("half", None)         # deprecated und wirkungslos, s. INFER_SPEC
        if kw.get("quantize") in (None, False, ""):
            kw.pop("quantize", None)
        if kw.get("imgsz") == "native":
            kw["imgsz"] = int(np.ceil(max(w, h) / 32) * 32)
        r = model.predict(f, verbose=False, **kw)[0]
        p = r.boxes.xyxy.cpu().numpy().astype(np.float64)
        c = r.boxes.conf.cpu().numpy().astype(np.float64)
        all_pred.append(p)
        all_conf.append(c)
        all_gt.append(gt)
        n_gt += len(gt)
        per_image.append(dict(file=os.path.basename(f), n_gt=len(gt),
                              n_pred=int(len(p)), imgsz=kw["imgsz"]))

    # ---- mAP ueber alle Bilder eines Splits -----------------------------
    aps = {}
    for thr in np.arange(0.5, 0.96, 0.05):
        tps, cfs = [], []
        for p, c, g in zip(all_pred, all_conf, all_gt):
            tp, _, _, _ = match(p, c, g, thr)
            tps.append(tp)
            cfs.append(c)
        tp = np.concatenate(tps) if tps else np.zeros(0, bool)
        cf = np.concatenate(cfs) if cfs else np.zeros(0)
        pre, rec, _ = pr_curve(tp, cf, n_gt)
        aps[round(float(thr), 2)] = average_precision(pre, rec)

    # ---- Betriebspunkt bei IoU 0.5 --------------------------------------
    tps, cfs, ious = [], [], []
    for p, c, g in zip(all_pred, all_conf, all_gt):
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

    # ---- Zaehlfehler am F1-optimalen Betriebspunkt -----------------------
    cnt = []
    for p, c, g in zip(all_pred, all_conf, all_gt):
        cnt.append((int((c >= best_conf).sum()), len(g)))
    cnt = np.array(cnt, float)
    count_err = cnt[:, 0] - cnt[:, 1]

    res = dict(
        split=split, n_images=len(files), n_gt=int(n_gt),
        infer={k: (v if not isinstance(v, np.integer) else int(v))
               for k, v in infer.items()},
        mAP50=aps[0.5], mAP75=aps[0.75],
        mAP50_95=float(np.mean(list(aps.values()))),
        ap_per_iou={str(k): round(v, 4) for k, v in aps.items()},
        best_f1=float(f1[k]) if len(f1) else 0.0,
        best_f1_conf=best_conf,
        precision_at_best_f1=float(pre[k]) if len(pre) else 0.0,
        recall_at_best_f1=float(rec[k]) if len(rec) else 0.0,
        mean_iou_of_tp=float(np.mean(np.concatenate(ious))) if any(
            len(i) for i in ious) else 0.0,
        count_pred=int(cnt[:, 0].sum()), count_gt=int(cnt[:, 1].sum()),
        count_mae=float(np.mean(np.abs(count_err))),
        count_bias=float(np.mean(count_err)),
        count_rel_err=float(np.sum(np.abs(count_err)) / max(1, n_gt)),
        per_image=per_image,
    )

    if save_vis:
        os.makedirs(os.path.dirname(save_vis) or ".", exist_ok=True)
        # hoechstens 6 Bilder je Split rendern - bei sval waeren es sonst 400
        for f, p, c, g in list(zip(files, all_pred, all_conf, all_gt))[:6]:
            keep = c >= best_conf
            pk = p[keep]
            tp, _, gidx, _ = match(pk, c[keep], g, 0.5)
            im = cv2.imread(f)
            hit = set(gidx[tp].tolist())
            for j, b in enumerate(g):                 # FN blau
                if j not in hit:
                    cv2.rectangle(im, (int(b[0]), int(b[1])),
                                  (int(b[2]), int(b[3])), (255, 90, 0), 2)
            for i, b in enumerate(pk):                # TP gruen / FP rot
                col = (0, 220, 0) if tp[i] else (0, 0, 255)
                cv2.rectangle(im, (int(b[0]), int(b[1])),
                              (int(b[2]), int(b[3])), col, 2)
            cv2.putText(im, f"gruen TP={int(tp.sum())}  rot FP={int((~tp).sum())}"
                            f"  blau FN={len(g) - len(hit)}  conf>={best_conf:.3f}",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            out = save_vis.replace("SPLIT", split).replace(
                "NAME", os.path.splitext(os.path.basename(f))[0])
            cv2.imwrite(out, im)

    # PR-Daten fuer Plots
    res["_pr"] = dict(precision=pre.tolist(), recall=rec.tolist(),
                      conf=cs.tolist(), f1=f1.tolist())
    return res


def summary_line(r):
    return (f"{r['split']:5s} imgsz={str(r['infer'].get('imgsz')):>5s} "
            f"conf={r['infer'].get('conf'):<6} "
            f"mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50_95']:.4f}  "
            f"F1={r['best_f1']:.4f}@{r['best_f1_conf']:.3f}  "
            f"P={r['precision_at_best_f1']:.3f} R={r['recall_at_best_f1']:.3f}  "
            f"IoU_tp={r['mean_iou_of_tp']:.3f}  "
            f"Zaehlung {r['count_pred']}/{r['count_gt']} "
            f"(MAE {r['count_mae']:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--imgsz", default="native")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--quantize", default=None,
                    choices=[None, "fp16", "int8", "fp32"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--crosscheck", action="store_true",
                    help="zusaetzlich ultralytics model.val() zum Vergleich")
    ap.add_argument("--vis", default=None,
                    help="Muster, z.B. docs/qual_SPLIT_NAME.jpg")
    a = ap.parse_args()

    model = YOLO(a.weights)
    imgsz = a.imgsz if a.imgsz == "native" else int(a.imgsz)
    infer = dict(imgsz=imgsz, conf=a.conf, iou=a.iou, max_det=a.max_det,
                 augment=a.augment, quantize=a.quantize, device=a.device)

    if a.crosscheck:
        print("\n--- Kreuzvergleich mit ultralytics model.val() ---")
        for s in a.splits.split(","):
            if s not in ("val", "test"):
                continue          # sval steht nicht in data/pipes.yaml
            m = model.val(data="data/pipes.yaml", split=s, imgsz=(
                2464 if s == "val" else 1600), conf=0.001, iou=a.iou,
                max_det=a.max_det, device=a.device, verbose=False, plots=False,
                project="/home/mso/source/yolo-gen/runs", name=f"xcheck_{s}",
                exist_ok=True)
            print(f"    ultralytics {s}: mAP50={m.box.map50:.4f} "
                  f"mAP50-95={m.box.map:.4f} P={m.box.mp:.4f} R={m.box.mr:.4f}")
        print("--- eigene Implementierung ---")

    results = []
    for s in a.splits.split(","):
        r = evaluate_split(model, s, infer, save_vis=a.vis)
        results.append(r)
        print(summary_line(r))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(dict(weights=a.weights, infer_spec=INFER_SPEC,
                           results=results), f, indent=2)
        print("->", a.out)


if __name__ == "__main__":
    main()
