"""
Anwendung: Rohrprofile in eigenen Bildern zaehlen und markieren.

Verwendet die in Experiment 1 und 3 belegten Inferenzparameter (siehe
docs/INFERENZ.md und docs/BACHELORARBEIT.md Kapitel 6.3).

Wichtigster Punkt: die Objekte sind nur 12-36 px gross. Das Bild wird deshalb
NICHT auf 640 verkleinert, sondern so skaliert, dass die Objekte im
Netzeingang etwa 22 px messen - die Groesse, fuer die die Trainingsdaten
kalibriert sind.

Aufruf:
    python src/predict.py bild.jpg
    python src/predict.py ordner/ --out ergebnisse/
    python src/predict.py bild.jpg --weights runs/baseline/weights/best.pt
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
cv2.setNumThreads(0)

import numpy as np              # noqa: E402
from ultralytics import YOLO    # noqa: E402

# ---------------------------------------------------------------------------
# Betriebspunkte je Modell
# ---------------------------------------------------------------------------
# imgsz: auf dem realen val-Bild bestimmtes Optimum (mAP@0.5:0.95).
# conf : F1-optimale Schwelle AUF VAL bei genau dieser imgsz.
#
# Warum die F1-optimale und nicht die zaehloptimale Schwelle? Gemessen wurde
# beides. Legt man die Schwelle auf val fest und wendet sie auf test an:
#     Kriterium        yolo26s-p2 test-F1   test-Zaehlung
#     F1-optimal       0.929                82/86
#     Zaehl-optimal    0.912                74/86
# Die zaehloptimale Schwelle liegt auf einer Messerschneide (dort heben sich
# Falschpositive und Falschnegative gerade auf) und uebertraegt sich schlecht
# auf ein anderes Bild. Die F1-optimale liegt auf einem breiten Plateau.
PROFILES = {
    "yolo26s_p2": dict(imgsz=1920, conf=0.44,
                       note="bestes Modell (Experiment 3)"),
    "yolo26s":    dict(imgsz=1600, conf=0.26, note=""),
    "baseline":   dict(imgsz=1920, conf=0.24, note="yolo11s-Baseline"),
}
DEFAULT_WEIGHTS = "runs/yolo26s_p2/weights/best.pt"
FALLBACK = dict(imgsz=1920, conf=0.35, note="unbekanntes Modell")


def profile_for(weights):
    """Betriebspunkt anhand des Laufnamens im Gewichtspfad waehlen."""
    parts = os.path.normpath(weights).split(os.sep)
    for p in parts:
        if p in PROFILES:
            return p, PROFILES[p]
    return os.path.basename(os.path.dirname(os.path.dirname(weights))), FALLBACK


def collect(path):
    if os.path.isdir(path):
        return sorted(sum([glob.glob(os.path.join(path, e))
                           for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG",
                                     "*.PNG")], []))
    return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Bild oder Ordner")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", default=None,
                    help="Netzeingangsgroesse; 'native' = laengste Bildkante. "
                         "Standard: modellabhaengig, siehe PROFILES")
    ap.add_argument("--conf", type=float, default=None,
                    help="Standard: modellabhaengig, F1-optimal auf val")
    ap.add_argument("--iou", type=float, default=0.7,
                    help="NMS-Schwelle; bei den YOLO26-Modellen wirkungslos, "
                         "weil sie end-to-end (NMS-frei) arbeiten")
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--tta", action="store_true",
                    help="Test-Time-Augmentation. Nur bei yolo11s wirksam "
                         "(+2 Punkte mAP50); bei YOLO26 gemessen wirkungslos.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="predictions")
    a = ap.parse_args()

    if not os.path.exists(a.weights):
        raise SystemExit(f"Gewichte nicht gefunden: {a.weights}")
    name, prof = profile_for(a.weights)
    imgsz_arg = a.imgsz if a.imgsz is not None else prof["imgsz"]
    conf = a.conf if a.conf is not None else prof["conf"]

    files = collect(a.source)
    if not files:
        raise SystemExit(f"keine Bilder unter {a.source}")
    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.weights)
    end2end = bool(getattr(model.model.model[-1], "end2end", False))

    print(f"Modell      : {a.weights}  ({name}{', ' + prof['note'] if prof['note'] else ''})")
    print(f"Betriebspunkt: imgsz={imgsz_arg} conf={conf}"
          f"{'  (Vorgabe aus Experiment 1/3)' if a.conf is None else '  (manuell)'}")
    if end2end:
        print("Hinweis     : end-to-end-Modell, NMS-frei -> --iou und --tta "
              "sind wirkungslos")
    elif a.tta:
        print("Hinweis     : TTA aktiv, ca. doppelte Rechenzeit")
    print()

    total = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"  uebersprungen (nicht lesbar): {f}")
            continue
        h, w = img.shape[:2]
        imgsz = (int(np.ceil(max(w, h) / 32) * 32) if imgsz_arg == "native"
                 else int(imgsz_arg))
        r = model.predict(f, imgsz=imgsz, conf=conf, iou=a.iou,
                          max_det=a.max_det, augment=a.tta, device=a.device,
                          verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        vis = img.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 220, 0), 2)
        txt = f"{len(boxes)} Rohrprofile   conf>={conf}   imgsz={imgsz}"
        cv2.rectangle(vis, (0, 0), (min(w, 780), 46), (0, 0, 0), -1)
        cv2.putText(vis, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2)

        stem = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(a.out, stem + "_det.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(a.out, stem + ".txt"), "w") as fh:
            for (x1, y1, x2, y2), c in zip(boxes, confs):
                fh.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} {c:.4f}\n")
        total += len(boxes)
        print(f"  {os.path.basename(f):40s} {w}x{h}  ->  "
              f"{len(boxes):4d} Profile")

    print(f"\nGesamt: {total} Profile in {len(files)} Bild(ern)")
    print(f"Markierte Bilder und YOLO-Labels (+Konfidenz) in: {a.out}/")


if __name__ == "__main__":
    main()
