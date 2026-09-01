"""
Training von PipeNet.

Das Protokoll ist absichtlich identisch zum YOLO-Lauf, damit der Vergleich in
docs/ERGEBNISSE.md ein Architekturvergleich bleibt:
    dieselben 6000 synthetischen Bilder, imgsz 1024, batch 16,
    epochs 60, patience 20, AdamW, Kosinus-Abfall, 3 Aufwaermepochen,
    Modellauswahl ueber die Fitness auf dem REALEN val-Bild.

Jeder Parameter ist unten mit Wert UND Begruendung hinterlegt; beim Lauf wird
daraus runs/<name>/parameter_begruendung.md geschrieben.

Aufruf:
    python cnn/src/train.py --name pipenet
    python cnn/src/train.py --name pipenet_s2 --no-pretrained --epochs 30
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
cv2.setNumThreads(0)

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # .../yolo-gen
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))          # Metriken wiederverwenden

from dataset import PipeDataset, collate, normalize_gpu  # noqa: E402
from loader import ThreadedBatchLoader                   # noqa: E402
from decode import predict_image                    # noqa: E402
from losses import PipeLoss                         # noqa: E402
from model import PipeNet, OUT_STRIDE               # noqa: E402
from evaluate import (average_precision, load_gt,   # noqa: E402
                      match, pr_curve)

# ---------------------------------------------------------------------------
SPEC = {
    "train_dir": (os.path.join(ROOT, "synth/train/images"),
                  "Exakt derselbe synthetische Datensatz wie beim YOLO-Lauf "
                  "(6000 Bilder, 136 384 Labels). Nur so ist der Vergleich "
                  "ein Architekturvergleich und keine Datenfrage."),
    "val_dir": (os.path.join(ROOT, "data/val/images"),
                "Das REALE val-Bild. Die Modellauswahl muss auf der "
                "Zieldomaene erfolgen, sonst waehlt man das Modell, das am "
                "besten zu den eigenen Renderartefakten passt. data/test "
                "bleibt unberuehrt."),
    "imgsz": (1024,
              "Identisch zur Leinwandgroesse des Generators und zur YOLO-"
              "Einstellung. Die synthetischen Bilder durchlaufen damit keine "
              "Skalierung, die Objekte erreichen das Netz mit exakt der "
              "Pixelgroesse (12-44 px), fuer die die Pipeline kalibriert ist."),
    "epochs": (60, "Wie beim YOLO-Lauf. Gegen Ueberanpassung wirkt patience."),
    "patience": (20,
                 "Early Stopping nach 20 Epochen ohne Fitness-Verbesserung. "
                 "Grosszuegig, weil die Fitness auf nur 88 Ground-Truth-"
                 "Objekten stark schwankt."),
    "batch": (16, "Wie beim YOLO-Lauf, damit die effektive Lernrate je "
                  "Beispiel vergleichbar bleibt."),
    "pretrained": (True,
                   "ResNet18-Encoder mit ImageNet-Gewichten. Dieselbe "
                   "Begruendung wie bei YOLO: die vortrainierten Low-Level-"
                   "Filter (Kanten, Ringe, Texturen) sind domaenenunabhaengig "
                   "und reduzieren genau den Anteil des Modells, der sonst auf "
                   "synthetische Artefakte ueberangepasst wuerde. Ohne "
                   "Vortraining ist der Dekoder ohnehin zufaellig "
                   "initialisiert - der Encoder ist der Teil, bei dem es "
                   "etwas zu uebernehmen gibt."),
    "optimizer": ("AdamW",
                  "Wie beim YOLO-Lauf. Konvergiert bei kleinen Datensaetzen "
                  "schneller als SGD und ist robust gegenueber der "
                  "Lernratenwahl."),
    "lr0": (1e-3,
            "Standard-Startlernrate fuer AdamW-Finetuning."),
    "lr_encoder_mult": (0.1,
                        "Der Encoder ist vortrainiert, Dekoder und Koepfe sind "
                        "zufaellig initialisiert. Eine einheitliche Lernrate "
                        "wuerde die vortrainierten Gewichte in den ersten "
                        "Epochen zerstoeren, waehrend die Koepfe noch keinen "
                        "sinnvollen Gradienten liefern. Faktor 0.1 ist die "
                        "uebliche Wahl fuer diskriminatives Finetuning."),
    "lrf": (0.01, "Endlernrate = 1 % der Startlernrate, Kosinus-Abfall."),
    "warmup_epochs": (3.0,
                      "Ohne Warmup erzeugen die frisch initialisierten Koepfe "
                      "in den ersten Schritten grosse Gradienten, die in den "
                      "vortrainierten Encoder zurueckschlagen."),
    "weight_decay": (5e-4,
                     "Wie beim YOLO-Lauf. Nur auf Faltungsgewichte, nicht auf "
                     "BatchNorm-Parameter und Biases (siehe "
                     "PipeNet.param_groups)."),
    "clip_grad": (10.0,
                  "Gradienten-Norm begrenzen. Der Gaussian-Focal-Loss wird "
                  "durch die Anzahl Zentren normiert; ein Batch mit sehr "
                  "wenigen Objekten (die 9.7 % Negativbilder haben null) "
                  "kann sonst einen einzelnen grossen Schritt erzeugen."),
    "amp": (True, "Mixed Precision, wie beim YOLO-Lauf."),
    "threads": (12,
                "Der Standard-DataLoader mit num_workers > 0 verklemmt sich in "
                "dieser Umgebung reproduzierbar (fork + OpenCV, siehe "
                "loader.py). Statt auf workers=0 zurueckzufallen wie beim "
                "YOLO-Lauf, laedt hier ein eigener Thread-Pool: die teuren "
                "cv2-Stufen geben die GIL frei und parallelisieren echt. "
                "Gemessen 31.9 -> 2.5 ms/Sample, Epochendauer 3.19 -> 0.25 min. "
                "12 statt 16 Threads, weil der Zugewinn danach unter 10 % "
                "liegt und Kerne fuer die GPU-Zufuhr frei bleiben sollen."),
    "prefetch": (3, "Vorlauf in Batches; ueberlappt Datenaufbereitung und "
                    "GPU-Berechnung."),
    "cache": (True,
              "Alle 6000 Bilder einmalig dekodiert im RAM (~18 GB bei 94 GB "
              "frei). Alle Threads teilen denselben Cache - bei Prozessen "
              "haette jeder eine eigene Kopie gebraucht."),
    "p_mosaic": (1.0,
                 "Wie mosaic=1.0 bei YOLO. Vier Bilder in einer Leinwand: "
                 "mehr Objekte je Iteration, Objekte an Kachelraendern "
                 "(in test sind 5.8 % der Labels angeschnitten) und "
                 "verschiedene Hintergruende in einem Bild."),
    "scale": ((0.6, 1.4),
              "Wie scale=0.4 bei YOLO (0.6-1.4). Zusammen mit der "
              "Generatorspanne 12-44 px ergibt das effektiv 7-62 px."),
    "flip": (0.5,
             "Horizontal und vertikal je 0.5, wie fliplr/flipud bei YOLO. "
             "Beide sind label-exakt und physikalisch korrekt: ein Stapel "
             "wird von beiden Enden fotografiert, und ein Rohrende hat kein "
             "'oben'."),
    "rotation": (0.0,
                 "AUS - wie bei YOLO. Eine Bildrotation dreht achsparallele "
                 "Boxen mit und blaeht sie auf. Die Rotationsinvarianz "
                 "liefert der Generator (jede Instanz 0-360 Grad, Box danach "
                 "exakt aus der gedrehten Alphamaske neu bestimmt)."),
    "min_overlap": (0.7,
                    "Parameter der CenterNet-Radiusformel. Bestimmt, wie weit "
                    "eine Detektion vom Zentrum abweichen darf und trotzdem "
                    "noch min_overlap IoU erreicht - daraus folgt die Breite "
                    "der Gauss-Glocke. Gemessene Folge fuer diese Daten: "
                    "sigma 0.60-1.17 Zellen. Bei einem Nachbarabstand von "
                    "4.4-5.5 Zellen verschmelzen benachbarte Glocken damit "
                    "nicht."),
    "sigma_clip": ((0.6, 2.5),
                   "Untergrenze 0.6: darunter faellt die Glocke innerhalb "
                   "einer Zelle ab und der beta=4-Term des Focal Loss hat "
                   "keine Wirkung mehr. Obergrenze 2.5: verhindert, dass ein "
                   "sehr grosses Objekt Nachbarzentren ueberdeckt."),
    "w_wh": (1.0, "Gewicht des log-Groessen-Verlusts, siehe losses.py."),
    "w_off": (1.0, "Gewicht des Versatz-Verlusts, siehe losses.py."),
    "val_thresh": (0.01,
                   "Peak-Schwelle bei der Validierung. Niedrig, damit die "
                   "PR-Kurve nicht am Recall-Ast abgeschnitten wird."),
    "val_topk": (2000,
                 "Obergrenze der Detektionen je Bild. Weit ueber den 88 "
                 "Ground-Truth-Objekten, damit die Metrik nie durch eine "
                 "Kappungsgrenze verfaelscht wird."),
    "seed": (0, "Fester Seed."),
}


def build(overrides):
    a = {k: v[0] for k, v in SPEC.items()}
    a.update(overrides)
    return a


def write_doc(path, args, extra=None):
    L = ["# PipeNet - Trainingsparameter, Werte und Begruendung", "",
         "Automatisch aus `cnn/src/train.py` erzeugt.", ""]
    if extra:
        L += [extra, ""]
    L += ["| Parameter | Wert | Begruendung |", "|---|---|---|"]
    for k, (d, why) in SPEC.items():
        v = args.get(k, d)
        if isinstance(v, str) and ROOT in v:
            v = v.replace(ROOT + "/", "")
        L.append(f"| `{k}` | `{v}` | {why} |")
    open(path, "w").write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, img_dir, lbl_dir, thresh, topk, device, scale=1.0):
    """Metriken auf dem realen val-Bild, mit den Funktionen aus
    ../src/evaluate.py - identische Metrikdefinition wie beim YOLO-Lauf."""
    import glob
    model.eval()
    files = sorted(sum([glob.glob(os.path.join(img_dir, e))
                        for e in ("*.jpg", "*.png")], []))
    all_p, all_c, all_g, n_gt = [], [], [], 0
    for f in files:
        gt, (W, H) = load_gt(f, lbl_dir)
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        if scale != 1.0:
            im = cv2.resize(im, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(np.ascontiguousarray(
            im[:, :, ::-1].transpose(2, 0, 1)))[None]
        x = normalize_gpu(x, device)
        box, sc = predict_image(model, x, OUT_STRIDE, thresh, topk)
        if scale != 1.0 and len(box):
            box = box / scale
        all_p.append(box.astype(np.float64))
        all_c.append(sc.astype(np.float64))
        all_g.append(gt)
        n_gt += len(gt)
    aps = {}
    for t in np.arange(0.5, 0.96, 0.05):
        tps, cfs = [], []
        for p, c, g in zip(all_p, all_c, all_g):
            tp, _, _, _ = match(p, c, g, t)
            tps.append(tp)
            cfs.append(c)
        tp = np.concatenate(tps) if tps else np.zeros(0, bool)
        cf = np.concatenate(cfs) if cfs else np.zeros(0)
        pre, rec, _ = pr_curve(tp, cf, n_gt)
        aps[round(float(t), 2)] = average_precision(pre, rec)
    tps, cfs = [], []
    for p, c, g in zip(all_p, all_c, all_g):
        tp, _, _, _ = match(p, c, g, 0.5)
        tps.append(tp)
        cfs.append(c)
    tp = np.concatenate(tps) if tps else np.zeros(0, bool)
    cf = np.concatenate(cfs) if cfs else np.zeros(0)
    pre, rec, cs = pr_curve(tp, cf, n_gt)
    f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-12)
    k = int(np.argmax(f1)) if len(f1) else 0
    m50, m = aps[0.5], float(np.mean(list(aps.values())))
    return dict(mAP50=m50, mAP50_95=m, fitness=0.1 * m50 + 0.9 * m,
                P=float(pre[k]) if len(pre) else 0.0,
                R=float(rec[k]) if len(rec) else 0.0,
                F1=float(f1[k]) if len(f1) else 0.0,
                conf=float(cs[k]) if len(cs) else 0.0,
                n_pred=int(k + 1), n_gt=n_gt)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pipenet")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--train-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    ov = {}
    for k in ("epochs", "batch", "imgsz", "patience", "lr0"):
        if getattr(a, k) is not None:
            ov[k] = getattr(a, k)
    if a.no_pretrained:
        ov["pretrained"] = False
    if a.train_dir:
        ov["train_dir"] = a.train_dir
    A = build(ov)

    out = os.path.join(ROOT, "cnn/runs", a.name)
    os.makedirs(os.path.join(out, "weights"), exist_ok=True)
    torch.manual_seed(A["seed"])
    np.random.seed(A["seed"])
    dev = torch.device(a.device)

    ds = PipeDataset(A["train_dir"], imgsz=A["imgsz"], stride=OUT_STRIDE,
                     train=True, cache=A["cache"], p_mosaic=A["p_mosaic"],
                     scale=A["scale"], seed=A["seed"],
                     min_overlap=A["min_overlap"],
                     sigma_clip=A["sigma_clip"])
    print(f"Trainingsbilder: {len(ds)}")
    if A["cache"]:
        print("Bilder in den RAM laden ...")
        ds.prefill_cache()
    dl = ThreadedBatchLoader(ds, A["batch"], collate, shuffle=True,
                             threads=A["threads"], prefetch=A["prefetch"],
                             drop_last=True, seed=A["seed"])

    model = PipeNet(pretrained=A["pretrained"]).to(dev)
    print(f"PipeNet: {model.n_params() / 1e6:.2f} M Parameter, "
          f"Ausgabe-Stride {OUT_STRIDE}")

    # Encoder-Parameter separat mit kleinerer Lernrate (siehe lr_encoder_mult)
    enc = {"stem", "layer1", "layer2", "layer3"}
    enc_ids = set()
    for n, p in model.named_parameters():
        if n.split(".")[0] in enc:
            enc_ids.add(id(p))
    pg = []
    for g in model.param_groups(A["weight_decay"]):
        for tag, mult in (("enc", A["lr_encoder_mult"]), ("head", 1.0)):
            ps = [p for p in g["params"]
                  if (id(p) in enc_ids) == (tag == "enc")]
            if ps:
                pg.append({"params": ps, "weight_decay": g["weight_decay"],
                           "lr": A["lr0"] * mult, "initial_lr": A["lr0"] * mult})
    opt = torch.optim.AdamW(pg, lr=A["lr0"], betas=(0.937, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=A["amp"])
    crit = PipeLoss(A["w_wh"], A["w_off"])

    nb = len(dl)
    total_it = A["epochs"] * nb
    warm_it = int(A["warmup_epochs"] * nb)

    def lr_at(it):
        if it < warm_it:
            return (it + 1) / max(1, warm_it)
        t = (it - warm_it) / max(1, total_it - warm_it)
        return A["lrf"] + (1 - A["lrf"]) * 0.5 * (1 + math.cos(math.pi * t))

    csv_path = os.path.join(out, "results.csv")
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["epoch", "time_s", "loss", "loss_hm", "loss_wh", "loss_off",
             "mAP50", "mAP50_95", "fitness", "P", "R", "F1", "lr"])

    best, best_ep, t0 = -1.0, 0, time.time()
    it = 0
    for ep in range(1, A["epochs"] + 1):
        ds.epoch = ep
        model.train()
        agg = {"total": 0.0, "hm": 0.0, "wh": 0.0, "off": 0.0}
        for b in dl:
            f = lr_at(it)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * f
            img = normalize_gpu(b["img"], dev)
            tgt = {k: b[k].to(dev, non_blocking=True)
                   for k in ("hm", "wh", "off", "mask")}
            with torch.amp.autocast("cuda", enabled=A["amp"]):
                pred = model(img)
                loss, parts = crit({k: v.float() for k, v in pred.items()}, tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), A["clip_grad"])
            scaler.step(opt)
            scaler.update()
            for k in agg:
                agg[k] += parts[k]
            it += 1
        for k in agg:
            agg[k] /= nb

        m = validate(model, A["val_dir"],
                     A["val_dir"].replace("images", "labels"),
                     A["val_thresh"], A["val_topk"], dev)
        el = time.time() - t0
        lr_now = opt.param_groups[-1]["lr"]
        print(f"  Epoche {ep:3d}/{A['epochs']}  loss {agg['total']:.4f} "
              f"(hm {agg['hm']:.4f} wh {agg['wh']:.4f} off {agg['off']:.4f})  "
              f"| val mAP50 {m['mAP50']:.4f} mAP50-95 {m['mAP50_95']:.4f} "
              f"F1 {m['F1']:.4f}  | lr {lr_now:.2e}  {el/60:.1f} min",
              flush=True)
        with open(csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [ep, round(el, 1), round(agg["total"], 5), round(agg["hm"], 5),
                 round(agg["wh"], 5), round(agg["off"], 5),
                 round(m["mAP50"], 5), round(m["mAP50_95"], 5),
                 round(m["fitness"], 5), round(m["P"], 5), round(m["R"], 5),
                 round(m["F1"], 5), f"{lr_now:.3e}"])

        ck = {"model": model.state_dict(), "epoch": ep, "metrics": m,
              "args": {k: str(v) for k, v in A.items()},
              "n_params": model.n_params(), "out_stride": OUT_STRIDE}
        torch.save(ck, os.path.join(out, "weights/last.pt"))
        if m["fitness"] > best:
            best, best_ep = m["fitness"], ep
            torch.save(ck, os.path.join(out, "weights/best.pt"))
        if ep - best_ep >= A["patience"]:
            print(f"  Early Stopping: {A['patience']} Epochen ohne "
                  f"Verbesserung. Bestes Modell aus Epoche {best_ep}.")
            break

    write_doc(os.path.join(out, "parameter_begruendung.md"), A,
              extra=f"Lauf `{a.name}`: {model.n_params()/1e6:.2f} M Parameter, "
                    f"Ausgabe-Stride {OUT_STRIDE}, {ep} Epochen gelaufen, "
                    f"bestes Modell aus Epoche {best_ep} "
                    f"(Fitness {best:.4f}), {(time.time()-t0)/3600:.2f} h.")
    json.dump({"best_epoch": best_ep, "best_fitness": best,
               "epochs_run": ep, "hours": (time.time() - t0) / 3600,
               "n_params": model.n_params()},
              open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(f"\nFertig -> {out}  (bestes Modell Epoche {best_ep}, "
          f"Fitness {best:.4f})")


if __name__ == "__main__":
    main()
