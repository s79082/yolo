"""
Anwendung von PipeNet auf eigene Bilder.

PipeNet ist vollstaendig konvolutional: das Bild geht in seiner nativen
Groesse hinein (nur auf ein Vielfaches von 32 gepaddet). Es gibt keine
Netzeingangsgroesse, kein Letterboxing und keine NMS-IoU-Schwelle.

Aufruf:
    python cnn/src/predict.py bild.jpg
    python cnn/src/predict.py ordner/ --tta --out ergebnisse/
    python cnn/src/predict.py bild.jpg --scale 1.2 --conf 0.4 --heatmap
"""
from __future__ import annotations

import argparse
import glob
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

from dataset import normalize_gpu          # noqa: E402
from decode import forward_tta, _pad_to, decode, heatmap_overlay  # noqa: E402
from model import PipeNet, OUT_STRIDE      # noqa: E402

DEFAULTS_FILE = os.path.join(ROOT, "cnn/docs/betriebspunkt.json")
FALLBACK = {"scale": 1.0, "conf": 0.35, "tta": True}


def _best_run():
    """Von mehreren Laeufen den mit der besseren Fitness auf val - dieselbe
    Auswahlregel wie in compare.py."""
    best, sel = -1.0, "pipenet"
    for n in ("pipenet", "pipenet_e30"):
        p = os.path.join(ROOT, "cnn/runs", n, "summary.json")
        if os.path.exists(p):
            try:
                f = json.load(open(p))["best_fitness"]
            except Exception:
                continue
            if f > best:
                best, sel = f, n
    return os.path.join(ROOT, "cnn/runs", sel, "weights/best.pt")


def load_defaults():
    if os.path.exists(DEFAULTS_FILE):
        try:
            return json.load(open(DEFAULTS_FILE))
        except Exception:
            pass
    return dict(FALLBACK)


def collect(path):
    if os.path.isdir(path):
        return sorted(sum([glob.glob(os.path.join(path, e))
                           for e in ("*.jpg", "*.jpeg", "*.png",
                                     "*.JPG", "*.PNG")], []))
    return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--weights", default=_best_run())
    ap.add_argument("--scale", type=float, default=None,
                    help="Eingangsskalierung; Standard aus "
                         "cnn/docs/betriebspunkt.json")
    ap.add_argument("--conf", type=float, default=None,
                    help="Peak-Schwelle; Standard aus betriebspunkt.json")
    ap.add_argument("--tta", dest="tta", action="store_true", default=None)
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--topk", type=int, default=2000)
    ap.add_argument("--heatmap", action="store_true",
                    help="Heatmap zusaetzlich als Overlay speichern")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="cnn/predictions")
    a = ap.parse_args()

    d = load_defaults()
    scale = a.scale if a.scale is not None else d["scale"]
    conf = a.conf if a.conf is not None else d["conf"]
    tta = a.tta if a.tta is not None else d["tta"]

    if not os.path.exists(a.weights):
        raise SystemExit(f"Gewichte nicht gefunden: {a.weights}")
    files = collect(a.source)
    if not files:
        raise SystemExit(f"keine Bilder unter {a.source}")
    os.makedirs(a.out, exist_ok=True)

    dev = torch.device(a.device)
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    model = PipeNet(pretrained=False)
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval()

    print(f"Modell       : {a.weights}  "
          f"({ck.get('n_params', 0) / 1e6:.2f} M Parameter, "
          f"Epoche {ck.get('epoch')})")
    print(f"Betriebspunkt: scale={scale} conf={conf} tta={tta} "
          f"stride={OUT_STRIDE}  (Peak-Extraktion, keine NMS)")
    print()

    total = 0
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  uebersprungen: {f}")
            continue
        H, W = img.shape[:2]
        im = img if scale == 1.0 else cv2.resize(
            img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(
            im[:, :, ::-1].transpose(2, 0, 1)))[None]
        x = normalize_gpu(x, dev)
        with torch.no_grad():
            if tta:
                out, _, _ = forward_tta(model, x)
            else:
                xp, _, _ = _pad_to(x, 32)
                out = model(xp)
            # valid: Zellen jenseits des Bildinhalts stammen aus dem
            # /32-Padding und werden verworfen
            import math
            box, sc = decode(out, OUT_STRIDE, conf, a.topk,
                             valid=(math.ceil(im.shape[0] / OUT_STRIDE),
                                    math.ceil(im.shape[1] / OUT_STRIDE)))
            hm = torch.sigmoid(out["hm"])[0, 0].cpu().numpy()
        box = box.cpu().numpy() / scale
        sc = sc.cpu().numpy()

        vis = img.copy()
        for x1, y1, x2, y2 in box:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 220, 0), 2)
        txt = (f"{len(box)} Rohrprofile   PipeNet {ck.get('n_params',0)/1e6:.2f}M"
               f"   conf>={conf}   scale={scale}   tta={tta}")
        cv2.rectangle(vis, (0, 0), (min(W, 900), 46), (0, 0, 0), -1)
        cv2.putText(vis, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        stem = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(a.out, stem + "_det.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if a.heatmap:
            ov = heatmap_overlay(img, hm, scale, OUT_STRIDE)
            # Dekodierte Zentren als Marker: macht sichtbar, dass die
            # Subpixel-Korrektur die Gitteraufloesung ausgleicht
            for x1, y1, x2, y2 in box:
                cv2.drawMarker(ov, (int(round((x1 + x2) / 2)),
                                    int(round((y1 + y2) / 2))),
                               (255, 255, 255), cv2.MARKER_CROSS, 7, 1)
            cv2.imwrite(os.path.join(a.out, stem + "_heatmap.jpg"), ov,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(a.out, stem + ".txt"), "w") as fh:
            for (x1, y1, x2, y2), c in zip(box, sc):
                fh.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f} {c:.4f}\n")
        total += len(box)
        print(f"  {os.path.basename(f):40s} {W}x{H}  ->  {len(box):4d} Profile")

    print(f"\nGesamt: {total} Profile in {len(files)} Bild(ern)")
    print(f"Ergebnisse in: {a.out}/")


if __name__ == "__main__":
    main()
