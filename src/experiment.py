"""
Stufe 6: Experimente.

EXPERIMENT 1 - Inferenzparameter (kein erneutes Training noetig)
    1a  imgsz-Sweep          <- der dominante Parameter dieser Aufgabe
    1b  NMS-IoU-Sweep
    1c  max_det-Sweep
    1d  TTA (augment) und FP16 (quantize)
    Zusaetzlich: Bootstrap-Konfidenzintervall ueber raeumliche Kacheln, weil
    je Split nur EIN Bild vorliegt und eine Punktschaetzung sonst nicht
    einzuordnen ist.

EXPERIMENT 3 - Modellfamilie (yolo11s / yolo26s / yolo26s-p2)
    Gleiches Budget, gleicher Datensatz, gleiche Hyperparameter.
    Motivation: Ultralytics 8.4.117 hat YOLO26 als Default (der AMP-Check
    laedt yolo26n.pt). Die urspruengliche Wahl yolo11s war konservativ und
    unbelegt; dieses Experiment holt den Beleg nach.

EXPERIMENT 2 - Okklusionsanteil in den Trainingsdaten (Ablation, 3 Trainings)
    A0  ohne Stangen-/Fremdkoerper-Okklusion
    A1  Baseline
    A2  maximale Okklusion
    Alle drei mit identischem Budget (Bildzahl, Epochen, Seed, Hyperparameter),
    damit der Unterschied ausschliesslich vom Okklusionsanteil kommt.

Aufruf:
    python src/experiment.py infer   --weights runs/baseline/weights/best.pt
    python src/experiment.py ablation-data
    python src/experiment.py ablation-eval
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import cv2                       # noqa: E402
cv2.setNumThreads(0)

import numpy as np               # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import (SPLITS, evaluate_split, iou_matrix, load_gt,   # noqa: E402
                      match, pr_curve, average_precision, summary_line)
from ultralytics import YOLO     # noqa: E402

OUT = "docs/experiments"


# ---------------------------------------------------------------------------
# Bootstrap-Konfidenzintervall ueber raeumliche Kacheln
# ---------------------------------------------------------------------------
def bootstrap_map50(model, split, infer, grid=6, B=400, seed=0):
    """CI fuer mAP50 durch Resampling raeumlicher Kacheln.

    Warum Kacheln und nicht einzelne Objekte: Detektionen und Ground Truth sind
    raeumlich stark korreliert (Cluster). Einzelobjekt-Bootstrap unterschaetzt
    die Varianz deutlich. Kacheln erhalten die lokale Korrelationsstruktur.
    Warum ueberhaupt: je Split existiert nur EIN Bild - ohne Streuungsmass ist
    eine mAP-Zahl nicht interpretierbar.
    """
    import glob
    img_dir, gt_dir = SPLITS[split]
    files = sorted(sum([glob.glob(os.path.join(img_dir, e))
                        for e in ("*.jpg", "*.png")], []))
    cells = []                                  # (pred, conf, gt) je Kachel
    for f in files:
        gt, (W, H) = load_gt(f, gt_dir)
        kw = dict(infer)
        if kw.get("imgsz") == "native":
            kw["imgsz"] = int(np.ceil(max(W, H) / 32) * 32)
        r = model.predict(f, verbose=False, **kw)[0]
        p = r.boxes.xyxy.cpu().numpy().astype(np.float64)
        c = r.boxes.conf.cpu().numpy().astype(np.float64)
        gx = (np.clip((gt[:, 0] + gt[:, 2]) / 2 / W, 0, .999) * grid).astype(int)
        gy = (np.clip((gt[:, 1] + gt[:, 3]) / 2 / H, 0, .999) * grid).astype(int)
        px = (np.clip((p[:, 0] + p[:, 2]) / 2 / W, 0, .999) * grid).astype(int)
        py = (np.clip((p[:, 1] + p[:, 3]) / 2 / H, 0, .999) * grid).astype(int)
        for iy in range(grid):
            for ix in range(grid):
                mg = (gx == ix) & (gy == iy)
                mp = (px == ix) & (py == iy)
                if mg.sum() or mp.sum():
                    cells.append((p[mp], c[mp], gt[mg]))
    rng = np.random.default_rng(seed)
    vals = []
    n = len(cells)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        tps, cfs, ngt = [], [], 0
        for i in idx:
            pp, cc, gg = cells[i]
            tp, _, _, _ = match(pp, cc, gg, 0.5)
            tps.append(tp)
            cfs.append(cc)
            ngt += len(gg)
        if ngt == 0:
            continue
        tp = np.concatenate(tps) if tps else np.zeros(0, bool)
        cf = np.concatenate(cfs) if cfs else np.zeros(0)
        pre, rec, _ = pr_curve(tp, cf, ngt)
        vals.append(average_precision(pre, rec))
    v = np.array(vals)
    return dict(n_cells=n, mean=float(v.mean()), std=float(v.std()),
                ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------
def exp_inference(weights, splits=("val", "test")):
    os.makedirs(OUT, exist_ok=True)
    model = YOLO(weights)
    base = dict(imgsz="native", conf=0.001, iou=0.7, max_det=1000,
                augment=False, quantize=None, device="0")
    report = {"weights": weights, "sweeps": {}}

    def run(tag, overrides, splits=splits):
        rows = []
        for s in splits:
            inf = {**base, **overrides}
            t0 = time.time()
            r = evaluate_split(model, s, inf)
            r["wall_s"] = round(time.time() - t0, 3)
            r.pop("_pr", None)
            rows.append(r)
            print("   ", summary_line(r), f" [{r['wall_s']:.2f}s]")
        report["sweeps"].setdefault(tag, []).extend(rows)

    print("\n=== 1a  imgsz-Sweep " + "=" * 50)
    for v in [640, 800, 960, 1024, 1280, 1600, 1920, 2240, 2464]:
        run("imgsz", {"imgsz": v})
    print("   -- nativ (imgsz = laengste Bildkante, auf /32 gerundet) --")
    run("imgsz", {"imgsz": "native"})

    # Beste imgsz AUSSCHLIESSLICH auf val bestimmen und auf beide Splits
    # anwenden. Eine Wahl je Split waere eine Hyperparameterselektion auf dem
    # Testsatz und wuerde dessen Zahlen wertlos machen.
    cand = [r for r in report["sweeps"]["imgsz"]
            if r["split"] == "val" and isinstance(r["infer"]["imgsz"], int)]
    best_val = max(cand, key=lambda r: r["mAP50_95"])["infer"]["imgsz"]
    bi = {s: best_val for s in splits}
    print(f"\n   beste imgsz nach mAP50-95 auf VAL: {best_val} "
          f"(wird auf alle Splits angewandt)")
    report["best_imgsz"] = bi
    report["best_imgsz_selected_on"] = "val"

    print("\n=== 1b  NMS-IoU-Sweep " + "=" * 48)
    for v in [0.3, 0.45, 0.6, 0.7, 0.8, 0.9]:
        for s in splits:
            inf = {**base, "imgsz": bi[s], "iou": v}
            r = evaluate_split(model, s, inf)
            r.pop("_pr", None)
            report["sweeps"].setdefault("iou", []).append(r)
            print("   ", summary_line(r))

    print("\n=== 1c  max_det-Sweep " + "=" * 48)
    for v in [50, 100, 200, 300, 1000]:
        for s in splits:
            inf = {**base, "imgsz": bi[s], "max_det": v}
            r = evaluate_split(model, s, inf)
            r.pop("_pr", None)
            report["sweeps"].setdefault("max_det", []).append(r)
            print("   ", summary_line(r))

    print("\n=== 1d  TTA und FP16 " + "=" * 49)
    for tag, ov in [("plain", {}), ("tta", {"augment": True}),
                    ("fp16", {"quantize": "fp16"}),
                    ("tta+fp16", {"augment": True, "quantize": "fp16"})]:
        for s in splits:
            inf = {**base, "imgsz": bi[s], **ov}
            t0 = time.time()
            r = evaluate_split(model, s, inf)
            r["wall_s"] = round(time.time() - t0, 3)
            r["tag"] = tag
            r.pop("_pr", None)
            report["sweeps"].setdefault("tta_half", []).append(r)
            print(f"    {tag:9s}", summary_line(r), f" [{r['wall_s']:.2f}s]")

    print("\n=== Bootstrap-CI bei bester imgsz " + "=" * 36)
    report["bootstrap"] = {}
    for s in splits:
        b = bootstrap_map50(model, s, {**base, "imgsz": bi[s]})
        report["bootstrap"][s] = b
        print(f"    {s}: mAP50 = {b['mean']:.4f} +- {b['std']:.4f}  "
              f"95%-CI [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]  "
              f"({b['n_cells']} Kacheln)")

    with open(os.path.join(OUT, "experiment1_inference.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n->", os.path.join(OUT, "experiment1_inference.json"))
    return report


# ---------------------------------------------------------------------------
# Experiment 2 - Ablation Okklusionsanteil
# ---------------------------------------------------------------------------
ARMS = [
    ("A0_keine_okklusion", ["--no-occlusion"]),
    ("A1_baseline",        []),
    ("A2_starke_okklusion", ["--occlusion-heavy"]),
]
ABL_N = 3000
ABL_EPOCHS = 30


def exp_ablation_data():
    """Erzeugt die drei Trainingssaetze. Gleicher Seed => gleiche
    Hintergrund-/Layoutziehungen, der einzige beabsichtigte Unterschied ist
    der Okklusionsanteil."""
    for name, flags in ARMS:
        out = f"synth/abl_{name}"
        if os.path.isdir(os.path.join(out, "images")):
            print("bereits vorhanden:", out)
            continue
        cmd = [sys.executable, "src/generate.py", "--out", out,
               "--n", str(ABL_N), "--seed", "1"] + flags
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        yml = f"data/abl_{name}.yaml"
        with open(yml, "w") as f:
            f.write(f"path: /home/mso/source/yolo-gen\n"
                    f"train: {out}/images\nval: data/val/images\n"
                    f"names:\n  0: rohrprofil\n")
    # gemessene Okklusionsanteile gegenueberstellen
    tbl = {}
    for name, _ in ARMS:
        rep = json.load(open(f"synth/abl_{name}/generation_report.json"))
        tbl[name] = dict(
            objekte=rep["boxes_total"],
            sichtbarkeit=round(rep["mean_visibility_labeled"], 4),
            teilverdeckt=rep["shares"]["inst_occluded (je gelabelter Instanz)"],
            verworfen_verdeckt=rep["dropped"]["dropped_occluded"],
            bilder_mit_stangen=rep["shares"]["img_has_bars (je Bild)"],
        )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment2_datasets.json"), "w") as f:
        json.dump(tbl, f, indent=2)
    print(json.dumps(tbl, indent=2))


def exp_ablation_train(model=None, suffix=""):
    """Trainiert die drei Arme. 'model' und 'suffix' erlauben, die Ablation mit
    einer anderen Modellfamilie zu wiederholen, ohne die bereits gerechneten
    Laeufe zu ueberschreiben (Laufname wird um 'suffix' erweitert)."""
    for name, _ in ARMS:
        run = f"abl_{name}{suffix}"
        if os.path.exists(f"runs/{run}/weights/best.pt"):
            print("bereits trainiert:", run)
            continue
        cmd = [sys.executable, "-u", "src/train.py", "--name", run,
               "--data", f"data/abl_{name}.yaml", "--epochs", str(ABL_EPOCHS)]
        if model:
            cmd += ["--model", model]
        print(" ".join(cmd), flush=True)
        with open(f"train_{run}.log", "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def exp_ablation_eval(imgsz="native", suffix="", out="experiment2_ablation.json"):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, _ in ARMS:
        w = f"runs/abl_{name}{suffix}/weights/best.pt"
        if not os.path.exists(w):
            print("fehlt:", w)
            continue
        model = YOLO(w)
        for s in ("val", "test"):
            r = evaluate_split(model, s, dict(imgsz=imgsz, conf=0.001, iou=0.7,
                                              max_det=1000, augment=False,
                                              quantize=None, device="0"))
            r.pop("_pr", None)
            r["arm"] = name
            r["weights"] = w
            rows.append(r)
            print(f"  {name:22s}", summary_line(r))
    with open(os.path.join(OUT, out), "w") as f:
        json.dump(rows, f, indent=2)
    print("->", os.path.join(OUT, out))


MODELS = [
    ("yolo11s",     "runs/baseline/weights/best.pt",
     "YOLO11s, 9.46 M Par., Strides 8/16/32, reg_max 16, mit NMS"),
    ("yolo26s",     "runs/yolo26s/weights/best.pt",
     "YOLO26s, 9.95 M Par., Strides 8/16/32, reg_max 1, NMS-frei (end2end)"),
    ("yolo26s-p2",  "runs/yolo26s_p2/weights/best.pt",
     "YOLO26s mit zusaetzlichem P2-Kopf, Strides 4/8/16/32, NMS-frei"),
]


def exp_models(imgsz="native"):
    """Experiment 3: Modellfamilien bei identischem Budget."""
    os.makedirs(OUT, exist_ok=True)
    rows = []
    base = dict(imgsz=imgsz, conf=0.001, iou=0.7, max_det=1000,
                augment=False, quantize=None, device="0")
    for name, w, desc in MODELS:
        if not os.path.exists(w):
            print(f"  fehlt (nicht trainiert): {w}")
            continue
        model = YOLO(w)
        npar = sum(p.numel() for p in model.model.parameters())
        for s in ("val", "test"):
            t0 = time.time()
            r = evaluate_split(model, s, base)
            r["wall_s"] = round(time.time() - t0, 3)
            r.pop("_pr", None)
            r["model"] = name
            r["desc"] = desc
            r["params_M"] = round(npar / 1e6, 2)
            rows.append(r)
            print(f"  {name:11s}", summary_line(r))
    with open(os.path.join(OUT, "experiment3_models.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("->", os.path.join(OUT, "experiment3_models.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["infer", "ablation-data", "ablation-train",
                                    "ablation-eval", "models"])
    ap.add_argument("--weights", default="runs/baseline/weights/best.pt")
    ap.add_argument("--imgsz", default="native")
    ap.add_argument("--model", default=None,
                    help="Basismodell fuer ablation-train, z.B. yolo26s.pt")
    ap.add_argument("--suffix", default="",
                    help="Suffix der Laufnamen, damit eine wiederholte Ablation "
                         "die vorhandenen Laeufe nicht ueberschreibt")
    ap.add_argument("--out", default="experiment2_ablation.json",
                    help="Zieldatei fuer ablation-eval")
    a = ap.parse_args()
    imgsz = a.imgsz if a.imgsz == "native" else int(a.imgsz)
    if a.cmd == "infer":
        exp_inference(a.weights)
    elif a.cmd == "models":
        exp_models(imgsz)
    elif a.cmd == "ablation-data":
        exp_ablation_data()
    elif a.cmd == "ablation-train":
        exp_ablation_train(a.model, a.suffix)
    else:
        exp_ablation_eval(imgsz, a.suffix, a.out)
