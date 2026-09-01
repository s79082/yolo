"""
Stufe 0: Sichtung und Vermessung des Ausgangsmaterials.

Erzeugt alle Zahlen, die in docs/DATENSICHTUNG.md stehen, sowie die
Kontaktbogen/Vergleichsbilder in docs/.

Aufruf:  python src/analyze_data.py
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np


def cutout_report():
    print("=" * 78)
    print("CUTOUTS")
    print("=" * 78)
    rows = []
    for f in sorted(glob.glob("cutouts/*.png")):
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        h, w = im.shape[:2]
        has_a = im.shape[2] == 4
        a = im[:, :, 3] if has_a else None
        gray = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)
        # Helligkeit des Kranzes = 90. Perzentil, der Bohrung = 10. Perzentil
        rows.append(dict(
            file=os.path.basename(f), w=w, h=h, rgba=has_a,
            alpha_min=int(a.min()) if has_a else None,
            alpha_max=int(a.max()) if has_a else None,
            alpha_uniform=bool(has_a and a.min() == a.max()),
            kranz_p90=float(np.percentile(gray, 90)),
            bohrung_p10=float(np.percentile(gray, 10)),
        ))
        print(f"  {rows[-1]['file']:14s} {w:3d}x{h:<3d} RGBA={has_a} "
              f"alpha[{rows[-1]['alpha_min']},{rows[-1]['alpha_max']}] "
              f"uniform={rows[-1]['alpha_uniform']}  "
              f"Kranz p90={rows[-1]['kranz_p90']:5.1f} "
              f"Bohrung p10={rows[-1]['bohrung_p10']:5.1f}")
    print(f"\n  -> ALLE Alphakanaele uniform: "
          f"{all(r['alpha_uniform'] for r in rows)}  "
          f"(=> Matting-Stufe zwingend erforderlich)")
    if os.path.exists("assets/masks/ellipses.json"):
        e = json.load(open("assets/masks/ellipses.json"))
        print("\n  Ellipsen-Fit (aus src/matting.py):")
        for k, v in e.items():
            el = v["ellipse"]
            print(f"    {k:14s} r=({el['rx']:5.2f},{el['ry']:5.2f}) "
                  f"ry/rx={el['ry'] / el['rx']:.2f}  "
                  f"Objektanteil der Kachel={v['coverage']:.2f}")
    return rows


def background_report():
    print()
    print("=" * 78)
    print("HINTERGRUENDE")
    print("=" * 78)
    for f in sorted(glob.glob("bg/*")):
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(g, cv2.CV_64F).var()
        hp = g.astype(np.float32) - cv2.medianBlur(g, 3).astype(np.float32)
        print(f"  {os.path.basename(f):20s} {w:5d}x{h:<5d} "
              f"mean={g.mean():6.1f} std={g.std():5.1f} "
              f"Schaerfe(varLap)={lap:8.1f} Rauschen={hp.std():5.2f}")


def label_report():
    print()
    print("=" * 78)
    print("REALE LABELS")
    print("=" * 78)
    out = {}
    for name, img, lbl in [("val", "data/val/images", "data/val/labels"),
                           ("test", "data/test/images", "data/test/labels")]:
        files = sorted(glob.glob(os.path.join(img, "*.jpg")))
        B, C, S = [], [], []
        for f in files:
            H, W = cv2.imread(f).shape[:2]
            p = os.path.join(lbl, os.path.splitext(os.path.basename(f))[0] + ".txt")
            for line in open(p):
                q = line.split()
                if len(q) < 5:
                    continue
                cx, cy, bw, bh = map(float, q[1:5])
                B.append([bw * W, bh * H])
                C.append([cx * W, cy * H])
                S.append((W, H))
        B, C = np.array(B), np.array(C)
        d = np.sqrt(B[:, 0] * B[:, 1])
        ar = B[:, 0] / B[:, 1]
        D = np.linalg.norm(C[:, None] - C[None], axis=2) + np.eye(len(C)) * 1e9
        nn = D.min(1) / d
        neigh = [(D[i] < 5 * d[i]).sum() for i in range(len(C))]
        W, H = S[0]
        x1, y1 = C[:, 0] - B[:, 0] / 2, C[:, 1] - B[:, 1] / 2
        x2, y2 = C[:, 0] + B[:, 0] / 2, C[:, 1] + B[:, 1] / 2
        border = ((x1 < 1) | (y1 < 1) | (x2 > W - 1) | (y2 > H - 1)).mean()
        # Richtung zu den 3 naechsten Nachbarn (0 = horizontal, 90 = vertikal)
        angs = []
        for i in range(len(C)):
            for j in np.argsort(D[i])[:3]:
                if D[i, j] > 4 * d[i]:
                    continue
                v = C[j] - C[i]
                angs.append(np.degrees(np.arctan2(v[1], v[0])) % 180)
        hist, edges = np.histogram(angs, bins=6, range=(0, 180))
        # Groessenverhaeltnis zum naechsten Nachbarn
        ratio = [max(d[i], d[np.argmin(D[i])]) / min(d[i], d[np.argmin(D[i])])
                 for i in range(len(C))]
        r = dict(
            n=len(B), image=f"{W}x{H}",
            w_px=[float(B[:, 0].min()), float(np.median(B[:, 0])), float(B[:, 0].max())],
            h_px=[float(B[:, 1].min()), float(np.median(B[:, 1])), float(B[:, 1].max())],
            diam_px=[float(d.min()), float(np.median(d)), float(d.max())],
            aspect_p10_med_p90=[float(np.percentile(ar, 10)), float(np.median(ar)),
                                float(np.percentile(ar, 90))],
            nn_dist_in_diam=[float(np.percentile(nn, 10)), float(np.median(nn)),
                             float(np.percentile(nn, 90))],
            neighbours_within_5d=[float(np.median(neigh)), float(np.max(neigh))],
            frac_touching=float((nn < 1.2).mean()),
            frac_border=float(border),
            area_frac=float((B[:, 0] * B[:, 1]).sum() / (W * H)),
            size_ratio_to_nn=[float(np.median(ratio)), float(np.percentile(ratio, 90))],
            neighbour_direction_hist={f"{int(edges[k])}-{int(edges[k+1])}":
                                      int(hist[k]) for k in range(6)},
        )
        out[name] = r
        print(f"  [{name}] {r['image']}  n={r['n']}")
        print(f"     Durchmesser px  min/med/max : "
              f"{r['diam_px'][0]:.1f} / {r['diam_px'][1]:.1f} / {r['diam_px'][2]:.1f}")
        print(f"     Seitenverhaeltnis p10/med/p90: "
              f"{r['aspect_p10_med_p90'][0]:.2f} / {r['aspect_p10_med_p90'][1]:.2f}"
              f" / {r['aspect_p10_med_p90'][2]:.2f}")
        print(f"     NN-Abstand in Durchmessern   : "
              f"{r['nn_dist_in_diam'][0]:.2f} / {r['nn_dist_in_diam'][1]:.2f}"
              f" / {r['nn_dist_in_diam'][2]:.2f}")
        print(f"     Nachbarn in 5 Durchmessern   : "
              f"Median {r['neighbours_within_5d'][0]:.0f}, max "
              f"{r['neighbours_within_5d'][1]:.0f}")
        print(f"     beruehrend (<1.2 d)          : {r['frac_touching']:.1%}")
        print(f"     randangeschnitten            : {r['frac_border']:.1%}")
        print(f"     Groessenverh. zum Nachbarn    : "
              f"Median {r['size_ratio_to_nn'][0]:.2f}, p90 "
              f"{r['size_ratio_to_nn'][1]:.2f}")
        print(f"     Objektflaechenanteil          : {r['area_frac']:.2%}")
        print(f"     Nachbarrichtung (0=horiz)     : "
              f"{r['neighbour_direction_hist']}")
        for sz in (640, 960, 1280, 1600, 2464):
            print(f"       @imgsz {sz:4d}: Objekte ~"
                  f"{r['diam_px'][1] * sz / max(W, H):5.1f} px")
    return out


def synth_report(path="synth/train"):
    if not os.path.isdir(os.path.join(path, "labels")):
        return None
    print()
    print("=" * 78)
    print(f"SYNTHETISCHE LABELS ({path})")
    print("=" * 78)
    ws, hs = [], []
    files = sorted(glob.glob(os.path.join(path, "labels", "*.txt")))
    for f in files[:2000]:
        for line in open(f):
            q = line.split()
            if len(q) == 5:
                ws.append(float(q[3]) * 1024)
                hs.append(float(q[4]) * 1024)
    ws, hs = np.array(ws), np.array(hs)
    d = np.sqrt(ws * hs)
    print(f"  n={len(ws)} aus {min(2000, len(files))} Bildern")
    print(f"  Durchmesser px p05/p25/med/p75/p95: "
          f"{np.percentile(d, 5):.1f} / {np.percentile(d, 25):.1f} / "
          f"{np.median(d):.1f} / {np.percentile(d, 75):.1f} / "
          f"{np.percentile(d, 95):.1f}")
    ar = ws / hs
    print(f"  Seitenverhaeltnis p10/med/p90: {np.percentile(ar, 10):.2f} / "
          f"{np.median(ar):.2f} / {np.percentile(ar, 90):.2f}")


if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    c = cutout_report()
    background_report()
    lbl = label_report()
    synth_report()
    with open("docs/datensichtung.json", "w") as f:
        json.dump(dict(cutouts=c, labels=lbl), f, indent=2)
    print("\n-> docs/datensichtung.json")
