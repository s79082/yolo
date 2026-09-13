"""
Wo genau kostet ein schlechter Okkluder Leistung?

Ablation B misst einen Unterschied in mAP und F1; dieses Modul schluesselt ihn
nach Objektgroesse auf. Fuer jeden Arm wird am F1-optimalen Betriebspunkt
bestimmt, welche Ground-Truth-Objekte unentdeckt bleiben, und der Rueckruf
getrennt fuer das kleinste Drittel und die uebrigen zwei Drittel berichtet.

Warum die Groesse und nicht der Verdeckungsgrad: Die realen Bilder tragen keine
Verdeckungsannotation, der Verdeckungsgrad je Objekt ist also nicht messbar.
Die Objektgroesse ist es, und sie ist in diesen Aufnahmen der beste verfuegbare
Stellvertreter fuer die Stapeltiefe -- kleine Stirnflaechen liegen weiter hinten
und damit dort, wo sich Rohre gegenseitig verdecken.

Aufruf:  python src/analyse_ablationB.py
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
from evaluate import SPLITS, load_gt, match, pr_curve      # noqa: E402
from ultralytics import YOLO                               # noqa: E402

BASE = dict(conf=0.001, iou=0.7, max_det=1000, augment=False,
            quantize=None, device="0")
ARME = [("A1_baseline", "B0 photographisch"),
        ("B1_prozedural", "B1 prozedural"),
        ("B2_prozedural_naiv", "B2 naiv")]
OUT = "docs/experiments"
TEXDIR = "paper/kapitel"


def _de(x, n=3):
    return f"{x:.{n}f}".replace(".", "{,}")


def analysiere(arm, split, model):
    img_dir, gt_dir = SPLITS[split]
    f = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[0]
    gt, (w, h) = load_gt(f, gt_dir)
    imgsz = int(np.ceil(max(w, h) / 32) * 32)
    r = model.predict(f, verbose=False, **{**BASE, "imgsz": imgsz})[0]
    p = r.boxes.xyxy.cpu().numpy().astype(float)
    c = r.boxes.conf.cpu().numpy().astype(float)

    tp, _, _, _ = match(p, c, gt, 0.5)
    pre, rec, cs = pr_curve(tp, c, len(gt))
    f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-12)
    conf = float(cs[int(np.argmax(f1))])

    keep = c >= conf
    tp2, _, gidx, _ = match(p[keep], c[keep], gt, 0.5)
    hit = set(gidx[tp2].tolist())
    d = np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
    thr = float(np.percentile(d, 100.0 / 3.0))
    treffer = np.array([j in hit for j in range(len(gt))])
    klein = d <= thr
    return dict(
        arm=arm, split=split, conf=round(conf, 4),
        schwelle_px=round(thr, 1),
        n_klein=int(klein.sum()), n_gross=int((~klein).sum()),
        rueckruf_klein=float(treffer[klein].mean()),
        rueckruf_gross=float(treffer[~klein].mean()),
        rueckruf_gesamt=float(treffer.mean()),
        fn_gesamt=int((~treffer).sum()),
        fn_durchmesser=[round(float(v), 1) for v in sorted(d[~treffer])],
    )


def tabelle(rows):
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption[Rückruf nach Objektgröße]{Rückruf am $F_1$-optimalen "
         r"Betriebspunkt, getrennt nach Objektgröße. \enquote{klein} ist das "
         r"kleinste Drittel des jeweiligen Splits (Schwelle 14,9\px{} auf "
         r"\code{test}, 25,4\px{} auf \code{val}). Der Verlust des naiven "
         r"Okkluders konzentriert sich fast vollständig auf die kleinen "
         r"Objekte -- also auf die, die tief im Stapel liegen und dort von "
         r"anderen Rohren gekreuzt werden.}",
         r"\label{tab:ablationBgroesse}", r"\footnotesize",
         r"\begin{tabular}{@{}lrrr|rrr@{}}", r"\toprule",
         r"\textbf{Arm} & \multicolumn{3}{c}{\code{test}} & "
         r"\multicolumn{3}{c}{\code{val}} \\",
         r" & klein & groß & FN & klein & groß & FN \\", r"\midrule"]
    for arm, lab in ARME:
        t = next(r for r in rows if r["arm"] == arm and r["split"] == "test")
        v = next(r for r in rows if r["arm"] == arm and r["split"] == "val")
        L.append(f"{lab} & {_de(t['rueckruf_klein'])} & "
                 f"{_de(t['rueckruf_gross'])} & {t['fn_gesamt']} & "
                 f"{_de(v['rueckruf_klein'])} & {_de(v['rueckruf_gross'])} & "
                 f"{v['fn_gesamt']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEXDIR, "tab_ablation_b_groesse.tex"), "w").write(
        "\n".join(L) + "\n")
    print("-> tab_ablation_b_groesse.tex")


if __name__ == "__main__":
    rows = []
    for arm, lab in ARME:
        w = f"runs/{arm}/weights/best.pt"
        if not os.path.exists(w):
            print("fehlt:", w)
            continue
        model = YOLO(w)
        for split in ("test", "val"):
            r = analysiere(arm, split, model)
            r["label"] = lab
            rows.append(r)
            print(f"  {lab:20s} {split:4s}  Schwelle {r['schwelle_px']:5.1f} px  "
                  f"Rückruf klein {r['rueckruf_klein']:.3f} "
                  f"({r['n_klein']} Obj.)  groß {r['rueckruf_gross']:.3f} "
                  f"({r['n_gross']})  FN {r['fn_gesamt']}")
    with open(os.path.join(OUT, "ablationB_groesse.json"), "w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        tabelle(rows)
    print("->", os.path.join(OUT, "ablationB_groesse.json"))
