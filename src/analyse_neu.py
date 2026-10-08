"""
Fehleranalyse und Kreuzvergleich fuer das Referenzmodell yolo26s-p2.

Erzeugt docs/experiments/analyse.json mit

  crosscheck   eigene Metrik gegen ultralytics model.val() (Verifikation des
               eigenen Bewertungsrahmens)
  boxversatz   Vergleich der vorhergesagten mit der annotierten Boxgroesse je
               Split, plus Zentrumsversatz
  sensitivitaet  dieselbe Auswertung, nachdem alle Vorhersageboxen um den
               gemessenen medianen Fehlbetrag aufgeweitet wurden
  fehler       Falsch-Positive und Falsch-Negative am F1-optimalen Betriebspunkt

sowie die Qualitativbilder docs/qual_<split>.jpg.

Aufruf:  python src/analyse_neu.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
cv2.setNumThreads(0)
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from evaluate import (SPLITS, evaluate_split, iou_matrix, load_gt,   # noqa: E402
                      match, pr_curve, average_precision)
from ultralytics import YOLO                                        # noqa: E402

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--weights", default="runs/yolo26s_p2/weights/best.pt")
_ap.add_argument("--tag", default="")
ARGS, _ = _ap.parse_known_args()
WEIGHTS = ARGS.weights
OUT = "docs/experiments"
DOC = "docs"
BASE = dict(conf=0.001, iou=0.7, max_det=1000, augment=False,
            quantize=None, device="0")


def _native(f):
    h, w = cv2.imread(f).shape[:2]
    return int(np.ceil(max(w, h) / 32) * 32)


def predict_split(model, split, imgsz="native"):
    img_dir, gt_dir = SPLITS[split]
    files = sorted(sum([glob.glob(os.path.join(img_dir, e))
                        for e in ("*.jpg", "*.png")], []))
    out = []
    for f in files:
        gt, (w, h) = load_gt(f, gt_dir)
        kw = dict(BASE)
        kw["imgsz"] = _native(f) if imgsz == "native" else imgsz
        r = model.predict(f, verbose=False, **kw)[0]
        out.append((f, r.boxes.xyxy.cpu().numpy().astype(np.float64),
                    r.boxes.conf.cpu().numpy().astype(np.float64), gt))
    return out


def _metrics(preds, grow=0.0):
    """mAP-Kennzahlen ueber eine Liste (file, pred, conf, gt); 'grow' weitet
    jede Vorhersagebox um grow Pixel je Seite auf."""
    P = []
    for f, p, c, g in preds:
        q = p.copy()
        if grow:
            q[:, 0] -= grow; q[:, 1] -= grow
            q[:, 2] += grow; q[:, 3] += grow
        P.append((f, q, c, g))
    n_gt = sum(len(g) for _, _, _, g in P)
    aps = {}
    for thr in np.arange(0.5, 0.96, 0.05):
        tps, cfs = [], []
        for _, p, c, g in P:
            tp, _, _, _ = match(p, c, g, thr)
            tps.append(tp); cfs.append(c)
        tp = np.concatenate(tps); cf = np.concatenate(cfs)
        pre, rec, _ = pr_curve(tp, cf, n_gt)
        aps[round(float(thr), 2)] = average_precision(pre, rec)
    tps, cfs = [], []
    for _, p, c, g in P:
        tp, _, _, _ = match(p, c, g, 0.5)
        tps.append(tp); cfs.append(c)
    tp = np.concatenate(tps); cf = np.concatenate(cfs)
    pre, rec, cs = pr_curve(tp, cf, n_gt)
    f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-12)
    k = int(np.argmax(f1))
    return dict(mAP50=aps[0.5], mAP75=aps[0.75],
                mAP50_95=float(np.mean(list(aps.values()))),
                best_f1=float(f1[k]), best_f1_conf=float(cs[k]),
                precision=float(pre[k]), recall=float(rec[k]))


def boxversatz(preds, split):
    """Vergleich der Boxgroessen. Nur Paare mit IoU > 0.3, damit die Zuordnung
    eindeutig ist und der Versatz nicht durch Fehlzuordnungen entsteht."""
    rp, rg, dd, ctr, dia = [], [], [], [], []
    for _, p, c, g in preds:
        if len(p) == 0 or len(g) == 0:
            continue
        M = iou_matrix(p, g)
        used = set()
        for i in np.argsort(-c):
            j = int(np.argmax(M[i]))
            if M[i, j] <= 0.3 or j in used:
                continue
            used.add(j)
            dp = np.sqrt((p[i, 2] - p[i, 0]) * (p[i, 3] - p[i, 1]))
            dg = np.sqrt((g[j, 2] - g[j, 0]) * (g[j, 3] - g[j, 1]))
            rp.append(dp); rg.append(dg); dd.append(dg - dp)
            cp = np.array([(p[i, 0] + p[i, 2]) / 2, (p[i, 1] + p[i, 3]) / 2])
            cg = np.array([(g[j, 0] + g[j, 2]) / 2, (g[j, 1] + g[j, 3]) / 2])
            ctr.append(float(np.linalg.norm(cp - cg) / dg))
            dia.append(dg)
    rp, rg, dd, dia = map(np.array, (rp, rg, dd, dia))
    res = dict(split=split, paare=int(len(rp)),
               gt_durchmesser_median=float(np.median(dia)),
               pred_durch_gt=float(np.median(rp / rg)),
               fehlbetrag_px=float(np.median(dd)),
               zentrumsversatz_in_d=float(np.median(ctr)))
    if len(dia) > 10:
        klein, gross = dia < 18, dia > 26
        if klein.sum() > 3:
            res["pred_durch_gt_unter18px"] = float(np.median((rp / rg)[klein]))
        if gross.sum() > 3:
            res["pred_durch_gt_ueber26px"] = float(np.median((rp / rg)[gross]))
    return res


def fehleranalyse(preds, split, conf):
    fp, fn, gesamt_pred = 0, 0, 0
    fp_ohne_overlap = 0
    fp_doppel = 0
    fn_groessen = []
    for _, p, c, g in preds:
        keep = c >= conf
        pk, ck = p[keep], c[keep]
        gesamt_pred += len(pk)
        tp, _, gidx, _ = match(pk, ck, g, 0.5)
        fp += int((~tp).sum())
        hit = set(gidx[tp].tolist())
        # Zweitbox auf einer bereits getroffenen Stirnflaeche. Sie zaehlt nach
        # der COCO-Regel als FP, sitzt aber auf einem echten Rohr. Bei diesem
        # Modell kann sie nicht wegfallen: der Kopf ist Ende-zu-Ende, es laeuft
        # keine NMS, der Parameter iou bleibt wirkungslos.
        if len(g):
            Mg = iou_matrix(pk, g)
            for i in np.where(~tp)[0]:
                j = int(np.argmax(Mg[i]))
                if Mg[i, j] > 0 and j in hit:
                    fp_doppel += 1
        fehl = [j for j in range(len(g)) if j not in hit]
        fn += len(fehl)
        for j in fehl:
            fn_groessen.append(float(np.sqrt((g[j, 2] - g[j, 0]) *
                                             (g[j, 3] - g[j, 1]))))
        if len(g) and (~tp).sum():
            M = iou_matrix(pk[~tp], g)
            fp_ohne_overlap += int((M.max(axis=1) <= 0.0).sum())
    return dict(split=split, conf=round(conf, 4), n_pred=gesamt_pred,
                falsch_positiv=fp, davon_ohne_ueberlappung=fp_ohne_overlap,
                davon_doppeldetektion=fp_doppel,
                falsch_negativ=fn,
                fn_durchmesser=[round(v, 1) for v in sorted(fn_groessen)])


def qualbild(preds, split, conf):
    """Qualitativbild am F1-optimalen Betriebspunkt.

    Die Falsch-Positiven werden nach Ursache eingefaerbt. Grund: die Mehrzahl
    von ihnen sitzt auf einer echten, bereits getroffenen Stirnflaeche und ist
    keine Fehldetektion im Wortsinn. Der YOLO26-Kopf arbeitet Ende-zu-Ende
    (end2end=True), es laeuft also keine NMS - der Inferenzparameter 'iou'
    bleibt wirkungslos. Zweitboxen auf demselben Rohr ueberleben deshalb und
    zaehlen nach der COCO-Regel (jede GT nur einmal) als FP. Eine einfarbige
    Darstellung legt dem Betrachter nahe, das Modell habe dort etwas erfunden.
    """
    for f, p, c, g in preds[:1]:
        keep = c >= conf
        pk = p[keep]
        tp, _, gidx, _ = match(pk, c[keep], g, 0.5)
        M = iou_matrix(pk, g)
        hit = set(gidx[tp].tolist())
        im = cv2.imread(f)
        H, W = im.shape[:2]

        for j, b in enumerate(g):                          # FN blau
            if j not in hit:
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (255, 120, 0), 2)
        n_dop = 0
        for i, b in enumerate(pk):
            if tp[i]:
                col = (0, 220, 0)                          # Treffer gruen
            else:
                j = int(np.argmax(M[i])) if len(g) else -1
                if j >= 0 and M[i, j] > 0 and j in hit:
                    col = (255, 0, 255)                    # Doppeldetektion magenta
                    n_dop += 1
                else:
                    col = (0, 0, 255)                      # uebrige FP rot
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)

        n_fp = int((~tp).sum())
        txt = (f"gruen TP={int(tp.sum())}   magenta Doppeldetektion={n_dop}   "
               f"rot FP={n_fp - n_dop}   blau FN={len(g) - len(hit)}   "
               f"conf>={conf:.3f}")
        # Schrift an die Bildbreite anpassen - test ist 1600 px breit, val 2459
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        sc = min(1.4, (W - 24) / max(tw, 1))
        cv2.putText(im, txt, (12, int(34 * max(sc, 0.6))),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255),
                    max(1, int(round(2 * sc))))
        out = os.path.join(DOC, f"qual_{split}.jpg")
        cv2.imwrite(out, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("->", out, f"(TP {int(tp.sum())}, Doppel {n_dop}, FP {n_fp-n_dop}, "
                         f"FN {len(g)-len(hit)})")


def nms_gegenprobe(preds, split, conf, iou_thr=0.7):
    """Belegt, dass die Zweitboxen redundant sind.

    YOLO26 hat einen Ende-zu-Ende-Kopf; Ultralytics ueberspringt die NMS dann
    vollstaendig, der Inferenzparameter 'iou' bleibt wirkungslos. Diese Funktion
    wendet die NMS nachtraeglich auf die bereits geschwellten Vorhersagen an.
    Entfernt sie nur Kopien, muessen Treffer und Falsch-Negative unveraendert
    bleiben - das ist der eigentliche Nachweis, nicht der F1-Gewinn.
    """
    from torchvision.ops import nms as _nms
    import torch as _torch

    def _stat(p, c, g):
        tp, _, _, _ = match(p, c, g, 0.5)
        TP, FP = int(tp.sum()), int((~tp).sum())
        pr = TP / max(TP + FP, 1)
        rc = TP / max(len(g), 1)
        return dict(tp=TP, fp=FP, fn=len(g) - TP, n=len(p),
                    f1=2 * pr * rc / max(pr + rc, 1e-9))

    res = dict(split=split, conf=round(conf, 4), iou=iou_thr)
    for _, p, c, g in preds[:1]:
        keep = c >= conf
        pk, ck = p[keep], c[keep]
        idx = _nms(_torch.tensor(pk, dtype=_torch.float32),
                   _torch.tensor(ck, dtype=_torch.float32), iou_thr).numpy()
        res["ohne"] = _stat(pk, ck, g)
        res["mit"] = _stat(pk[idx], ck[idx], g)
        res["entfernt"] = int(len(pk) - len(idx))
        # Konfidenzabstand und Deckung der entfernten Paare
        PP = iou_matrix(pk, pk)
        weg = sorted(set(range(len(pk))) - set(idx.tolist()))
        paare = []
        for i in weg:
            j = int(np.argmax([PP[i, q] if q != i else -1 for q in range(len(pk))]))
            paare.append(dict(conf_verlierer=round(float(ck[i]), 4),
                              conf_gewinner=round(float(ck[j]), 4),
                              abstand=round(float(ck[j] - ck[i]), 4),
                              iou_paar=round(float(PP[i, j]), 4)))
        res["paare"] = paare
    return res


def crosscheck(model):
    res = {}
    for s, imgsz in (("val", 2464), ("test", 1600)):
        m = model.val(data="data/pipes.yaml", split=s, imgsz=imgsz, conf=0.001,
                      iou=0.7, max_det=1000, device="0", verbose=False,
                      plots=False, project=os.path.join(ROOT, "runs"),
                      name=f"xcheck_{s}", exist_ok=True)
        res[s] = dict(ultralytics_mAP50=float(m.box.map50),
                      ultralytics_mAP50_95=float(m.box.map))
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    model = YOLO(WEIGHTS)
    rep = {"weights": WEIGHTS}

    rep["crosscheck"] = crosscheck(model)
    for s in ("val", "test"):
        r = evaluate_split(model, s, {**BASE, "imgsz": "native"})
        rep["crosscheck"][s]["eigen_mAP50"] = r["mAP50"]
        rep["crosscheck"][s]["eigen_mAP50_95"] = r["mAP50_95"]
    print(json.dumps(rep["crosscheck"], indent=2))

    rep["boxversatz"], rep["sensitivitaet"], rep["fehler"] = [], {}, []
    for s in ("val", "test", "sval"):
        preds = predict_split(model, s)
        bv = boxversatz(preds, s)
        rep["boxversatz"].append(bv)
        print(f"  {s}: pred/GT {bv['pred_durch_gt']:.3f}  "
              f"Fehlbetrag {bv['fehlbetrag_px']:+.2f} px  "
              f"Zentrum {bv['zentrumsversatz_in_d']:.3f} d  "
              f"({bv['paare']} Paare)")
        if s in ("val", "test"):
            grow = bv["fehlbetrag_px"] / 2.0
            rep["sensitivitaet"][s] = dict(
                aufweitung_px_je_seite=round(grow, 3),
                original=_metrics(preds),
                aufgeweitet=_metrics(preds, grow))
            conf = _metrics(preds)["best_f1_conf"]
            rep["fehler"].append(fehleranalyse(preds, s, conf))
            rep.setdefault("nms", []).append(nms_gegenprobe(preds, s, conf))
            qualbild(preds, s, conf)
    with open(os.path.join(OUT, f"analyse{ARGS.tag}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: rep[k] for k in ("sensitivitaet", "fehler")}, indent=2))
    print("->", os.path.join(OUT, f"analyse{ARGS.tag}.json"))
