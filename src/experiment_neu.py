"""
Zwei Ablationen der Augmentierungspipeline, beide auf yolo26s-p2.

ABLATION A -- Okklusionsanteil (wie in der urspruenglichen Arbeit)
    A0  ohne Stangen- und Fremdkoerperokklusion
    A1  Baseline
    A2  maximale Okklusion
  Die drei Datensaetze liegen bereits unter synth/abl_* und werden
  unveraendert weiterverwendet; neu ist ausschliesslich das Modell.

ABLATION B -- Herkunft der Okkluderbalken (neu)
    B0  photographische Stangen aus den Clean Plates   == A1, derselbe Lauf
    B1  prozedural erzeugte Stangen, physikalisch modelliert und an B0
        kalibriert
    B2  prozedural erzeugte Stangen, naiv (konstantes Grau)
  Alle drei Datensaetze sind mit demselben Seed erzeugt und unterscheiden sich
  ausschliesslich in den RGB-Werten innerhalb der Stangenmasken -- Positionen,
  Verdeckungsgrade und Labeldateien sind byte-identisch (Nachweis:
  ablation-check). B ist damit ein gepaarter Versuch.

  Betroffen sind beide Verwendungen der Bibliothek, weil beide aus derselben
  Bibliothek ziehen: der Rohrschaft HINTER der Stirnflaeche (T5, 88 % der
  Instanzen -- der dominante Fall) und die quer DAVOR liegenden Stangen
  (T6, 58 % der Bilder).

Aufruf:
    python src/experiment_neu.py data          # yamls + Datensatzkennwerte
    python src/experiment_neu.py check         # Nachweis der Paarung
    python src/experiment_neu.py train         # 5 Trainings
    python src/experiment_neu.py eval          # Auswertung val + test
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import cv2                                              # noqa: E402
cv2.setNumThreads(0)

import numpy as np                                      # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from evaluate import evaluate_split, summary_line       # noqa: E402

OUT = os.path.join(ROOT, "docs", "experiments")
RUNS = os.path.join(ROOT, "runs")
EPOCHS = 30

# (Laufname, Datensatz, Kurzbeschreibung)
ARME = [
    ("A0_keine_okklusion",  "synth/abl_A0_keine_okklusion",
     "ohne quer liegende Stangen, ohne Fremdkoerper, kein Schaft vor einer "
     "Stirnflaeche"),
    ("A1_baseline",         "synth/abl_A1_baseline",
     "Baseline der Pipeline; zugleich Arm B0 der zweiten Ablation"),
    ("A2_starke_okklusion", "synth/abl_A2_starke_okklusion",
     "in jedem Bild Stangen, mehr Fremdkoerper, mehr quer davor liegende Rohre"),
    ("B1_prozedural",       "synth/abl_B1_proz_balken",
     "Okkluder prozedural erzeugt (Zylindermodell, an B0 kalibriert)"),
    ("B2_prozedural_naiv",  "synth/abl_B2_proz_balken_naiv",
     "Okkluder prozedural erzeugt, naiv (konstantes Grau)"),
]

ABLATION_A = ["A0_keine_okklusion", "A1_baseline", "A2_starke_okklusion"]
ABLATION_B = ["A1_baseline", "B1_prozedural", "B2_prozedural_naiv"]
B_ALIAS = {"A1_baseline": "B0_photographisch"}


# ---------------------------------------------------------------------------
def cmd_data():
    """data/*.yaml schreiben und die Kennwerte der fuenf Datensaetze sammeln."""
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    tbl = {}
    for name, ds, desc in ARME:
        yml = os.path.join(ROOT, "data", f"{name}.yaml")
        with open(yml, "w") as f:
            f.write(f"# {desc}\npath: {ROOT}\n"
                    f"train: {ds}/images\nval: data/val/images\n"
                    f"test: data/test/images\nnames:\n  0: rohrprofil\n")
        rep = json.load(open(os.path.join(ds, "generation_report.json")))
        tbl[name] = dict(
            datensatz=ds,
            beschreibung=desc,
            bilder=rep["n_images"],
            instanzen_gerendert=rep["instances_rendered"],
            labels=rep["boxes_total"],
            mittlere_sichtbarkeit=round(rep["mean_visibility_labeled"], 4),
            teilverdeckt=rep["shares"]["inst_occluded (je gelabelter Instanz)"],
            verworfen_verdeckt=rep["dropped"]["dropped_occluded"],
            bilder_mit_stangen=rep["shares"]["img_has_bars (je Bild)"],
            schaft_je_instanz=rep["shares"]["inst_stub (je Instanz)"],
            okkluder=rep.get("bars_tag", "photographisch"),
        )
    with open(os.path.join(OUT, "datensaetze.json"), "w") as f:
        json.dump(tbl, f, indent=2)
    print(json.dumps(tbl, indent=2))


def _dirhash(d):
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        h.update(os.path.basename(p).encode())
        h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def cmd_check():
    """Nachweis, dass Ablation B ein gepaarter Versuch ist.

    Geprueft wird: (a) die Labelverzeichnisse der drei B-Arme sind
    byte-identisch, (b) die Bildverzeichnisse sind es NICHT. Damit ist jeder
    Unterschied im Ergebnis auf den Bildinhalt der Okkluder zurueckfuehrbar
    und nicht auf eine andere Objektverteilung.
    """
    dsB = {B_ALIAS.get(n, n): d for n, d, _ in ARME if n in ABLATION_B}
    res = {}
    for tag, d in dsB.items():
        res[tag] = dict(labels=_dirhash(os.path.join(d, "labels")),
                        images=_dirhash(os.path.join(d, "images")),
                        n_labels=len(glob.glob(os.path.join(d, "labels", "*.txt"))))
    lab = {v["labels"] for v in res.values()}
    img = {v["images"] for v in res.values()}
    res["_urteil"] = dict(
        labels_identisch=(len(lab) == 1),
        bilder_verschieden=(len(img) == len(dsB)),
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "paarungsnachweis.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    if not res["_urteil"]["labels_identisch"]:
        print("\nWARNUNG: Labels NICHT identisch - Ablation B ist nicht gepaart.")


def cmd_train(cache="disk", epochs=EPOCHS, batch=8):
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    for name, _ds, _desc in ARME:
        w = os.path.join(RUNS, name, "weights", "best.pt")
        if os.path.exists(w):
            print("bereits trainiert:", name, flush=True)
            continue
        cmd = [sys.executable, "-u", "src/train_neu.py",
               "--name", name, "--data", f"data/{name}.yaml",
               "--epochs", str(epochs), "--cache", cache,
               "--batch", str(batch)]
        print(" ".join(cmd), flush=True)
        t0 = time.time()
        log = os.path.join(ROOT, "logs", f"train_{name}.log")
        env = dict(os.environ,
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        with open(log, "w") as fh:
            subprocess.run(cmd, check=True, stdout=fh,
                           stderr=subprocess.STDOUT, env=env)
        print(f"  -> {name} fertig in {(time.time()-t0)/60:.1f} min", flush=True)


def cmd_eval(imgsz="native"):
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    infer = dict(imgsz=imgsz, conf=0.001, iou=0.7, max_det=1000,
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
            r["arm"] = name
            r["datensatz"] = ds
            r["beschreibung"] = desc
            r["weights"] = os.path.relpath(w, ROOT)
            r["params_M"] = round(npar / 1e6, 2)
            r["best_epoch"] = _best_epoch(name)
            rows.append(r)
            print(f"  {name:20s}", summary_line(r), flush=True)
    with open(os.path.join(OUT, "ablationen.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("->", os.path.join(OUT, "ablationen.json"))


REFERENZ = os.path.join(ROOT, "runs", "yolo26s_p2", "weights", "best.pt")


def set_referenz(w):
    global REFERENZ
    if w:
        REFERENZ = os.path.join(ROOT, w) if not os.path.isabs(w) else w


def cmd_main(imgsz="native"):
    """Hauptergebnis: das Referenzmodell der vollstaendigen Pipeline.

    runs/yolo26s_p2 ist auf dem vollen Datensatz (6000 Bilder, 60 Epochen)
    trainiert und dient als Bezugspunkt, gegen den die beiden Ablationen mit
    ihrem reduzierten Budget (3000/30) einzuordnen sind. Zusaetzlich zu val und
    test wird sval ausgewertet -- der synthetische Validierungssatz, aus dem
    sich der Sim-to-Real-Abstand ergibt.
    """
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    model = YOLO(REFERENZ)
    npar = sum(p.numel() for p in model.model.parameters())
    rows = []
    for s in ("val", "test", "sval"):
        infer = dict(imgsz=imgsz, conf=0.001, iou=0.7, max_det=1000,
                     augment=False, quantize=None, device="0")
        r = evaluate_split(model, s, infer)
        r.pop("_pr", None)
        r["modell"] = "yolo26s-p2"
        r["weights"] = os.path.relpath(REFERENZ, ROOT)
        r["params_M"] = round(npar / 1e6, 2)
        rows.append(r)
        print(f"  referenz {s:5s}", summary_line(r), flush=True)
    with open(os.path.join(OUT, "hauptergebnis.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("->", os.path.join(OUT, "hauptergebnis.json"))


def cmd_sweep():
    """Sweep der Inferenzaufloesung fuer das Referenzmodell.

    Muss fuer yolo26s-p2 neu gerechnet werden: Die Argumentation in
    Abschnitt 3.3.2 haengt an der Schrittweite der feinsten Detektionsebene,
    und der P2-Kopf senkt sie von 8 auf 4. Die Zahlen der urspruenglichen
    Arbeit stammen von einem Modell ohne P2-Kopf und gelten hier nicht.
    Die beste Aufloesung wird AUSSCHLIESSLICH auf val bestimmt.
    """
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    model = YOLO(REFERENZ)
    rep = {"weights": os.path.relpath(REFERENZ, ROOT), "imgsz": [],
           "tta": [], "bootstrap": {}}
    base = dict(conf=0.001, iou=0.7, max_det=1000, augment=False,
                quantize=None, device="0")
    for v in [640, 800, 960, 1024, 1280, 1600, 1920, 2240, 2464, "native"]:
        for s in ("val", "test"):
            t0 = time.time()
            r = evaluate_split(model, s, {**base, "imgsz": v})
            r.pop("_pr", None)
            r["wall_s"] = round(time.time() - t0, 3)
            rep["imgsz"].append(r)
            print(f"  imgsz={str(v):>6s}", summary_line(r), flush=True)
    cand = [r for r in rep["imgsz"]
            if r["split"] == "val" and isinstance(r["infer"]["imgsz"], int)]
    best = max(cand, key=lambda r: r["mAP50_95"])["infer"]["imgsz"]
    rep["best_imgsz"] = best
    rep["best_imgsz_selected_on"] = "val"
    print(f"\n  beste imgsz nach mAP50-95 auf val: {best}", flush=True)
    for tag, ov in [("plain", {}), ("tta", {"augment": True})]:
        for s in ("val", "test"):
            t0 = time.time()
            r = evaluate_split(model, s, {**base, "imgsz": best, **ov})
            r.pop("_pr", None)
            r["wall_s"] = round(time.time() - t0, 3)
            r["tag"] = tag
            rep["tta"].append(r)
            print(f"  {tag:5s}", summary_line(r), flush=True)

    # Bootstrap beim nativen Massstab, nicht bei `best`: Alle berichteten
    # Zahlen stehen beim nativen Massstab, und an diesem Intervall werden sie
    # gemessen. Bei imgsz=1920 waeren die Intervalle deutlich enger
    # (Breite val 0,042 statt 0,078, test 0,044 statt 0,062).
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from experiment import bootstrap_map50               # noqa: E402
    for s in ("val", "test"):
        b = bootstrap_map50(model, s, {**base, "imgsz": "native"})
        rep["bootstrap"][s] = b
        print(f"  bootstrap {s}: mAP50 {b['mean']:.4f} +- {b['std']:.4f} "
              f"95%-CI [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]", flush=True)
    with open(os.path.join(OUT, "inferenz_sweep.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("->", os.path.join(OUT, "inferenz_sweep.json"))


def _best_epoch(name):
    """Epoche des besten Checkpoints aus results.csv (Ultralytics-Fitness).

    Ultralytics 8.4.117 gewichtet [P, R, mAP50, mAP50-95] mit [0, 0, 0, 1]
    (Metric.fitness in ultralytics/utils/metrics.py); der beste Checkpoint
    ist also der mit dem hoechsten mAP50-95 auf val.
    """
    p = os.path.join(RUNS, name, "results.csv")
    if not os.path.exists(p):
        return None
    import csv
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    key95 = next((k for k in rows[0] if "mAP50-95(B)" in k), None)
    if not key95:
        return None
    fit = [float(r[key95]) for r in rows]
    i = int(np.argmax(fit))
    return dict(epoche=int(float(rows[i]["epoch"])), n_epochen=len(rows),
                fitness=round(fit[i], 5))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "check", "train", "eval",
                                    "main", "sweep"])
    ap.add_argument("--imgsz", default="native")
    ap.add_argument("--cache", default="disk")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--weights", default=None,
                    help="Referenzmodell fuer main/sweep")
    a = ap.parse_args()
    set_referenz(a.weights)
    if a.cmd == "data":
        cmd_data()
    elif a.cmd == "check":
        cmd_check()
    elif a.cmd == "train":
        cmd_train(a.cache, a.epochs, a.batch)
    elif a.cmd == "main":
        cmd_main(a.imgsz if a.imgsz == "native" else int(a.imgsz))
    elif a.cmd == "sweep":
        cmd_sweep()
    else:
        cmd_eval(a.imgsz if a.imgsz == "native" else int(a.imgsz))
