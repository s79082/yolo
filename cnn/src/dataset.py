"""
Datensatz und Zielerzeugung fuer PipeNet.

Verwendet EXAKT dieselben synthetischen Bilder wie das YOLO-Training
(../synth/train, 6000 Bilder, 136 384 Labels). Damit ist der Vergleich in
docs/ERGEBNISSE.md ein reiner Architekturvergleich und keine Datenfrage.

Trainingszeit-Augmentierung bewusst nach Vorbild des YOLO-Laufs gewaehlt,
damit auch dieser Faktor nicht konfundiert:
    Mosaic 4 Bilder     wie mosaic=1.0
    Skalierung 0.6-1.4  wie scale=0.4
    Verschiebung        durch Zufallsausschnitt
    Spiegelung h/v      wie fliplr=0.5, flipud=0.5
    HSV-Jitter          wie hsv_h/s/v = 0.015/0.7/0.4
    KEINE Rotation      Rotation ist im Generator label-exakt erfolgt
"""
from __future__ import annotations

import glob
import math
import os

import cv2
cv2.setNumThreads(0)

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------------------
# Gauss-Radius nach CenterNet
# ---------------------------------------------------------------------------
def gaussian_radius(h, w, min_overlap=0.7):
    """Kleinster Radius, bei dem eine um r verschobene Box noch min_overlap
    IoU mit der Originalbox hat (CornerNet, Gl. 6). Damit ist die Breite der
    Gauss-Glocke aus einer geometrischen Toleranz abgeleitet und nicht frei
    gewaehlt."""
    a1, b1 = 1, (h + w)
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 ** 2 - 4 * a1 * c1))) / 2
    a2, b2 = 4, 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(max(0.0, b2 ** 2 - 4 * a2 * c2))) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(0.0, b3 ** 2 - 4 * a3 * c3))) / 2
    return min(r1, r2, r3)


def draw_gaussian(hm, cx, cy, sigma):
    """Additiv-maximales Einzeichnen einer 2D-Gauss-Glocke."""
    H, W = hm.shape
    rad = max(1, int(3 * sigma + 0.5))
    x0, x1 = max(0, cx - rad), min(W, cx + rad + 1)
    y0, y1 = max(0, cy - rad), min(H, cy + rad + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    # Maximum statt Summe: bei beruehrenden Objekten sollen sich die Glocken
    # nicht zu Werten > 1 aufaddieren, sonst gibt es kuenstliche "Zentren"
    # zwischen zwei echten Objekten.
    np.maximum(hm[y0:y1, x0:x1], g, out=hm[y0:y1, x0:x1])


# ---------------------------------------------------------------------------
def hsv_jitter(img, rng, h=0.015, s=0.7, v=0.4):
    """HSV-Jitter ueber Nachschlagetabellen statt Fliesskommaarithmetik.

    Gemessen 12.4 ms -> 4.9 ms bei 1024x1024. Bei workers=0 (erzwungen, siehe
    train.py) laeuft die gesamte Augmentierung im Hauptprozess und ist der
    Engpass, nicht die GPU - solche Konstanten entscheiden hier ueber die
    Trainingsdauer.
    """
    g = rng.uniform(-1, 1, 3) * np.array([h, s, v]) + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * g[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * g[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * g[2], 0, 255).astype(np.uint8)
    m = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s),
                   cv2.LUT(val, lut_v)))
    return cv2.cvtColor(m, cv2.COLOR_HSV2BGR)


def normalize_gpu(x_uint8, device, non_blocking=True):
    """uint8 NCHW (RGB) -> normalisiertes float32 NCHW, auf der GPU.

    Warum nicht im Dataset: die Normalisierung auf der CPU kostete gemessen
    24 ms je Bild (12.6 MB float32-Verkehr in fuenf Durchlaeufen) - mehr als
    jede andere Stufe. Auf der GPU ist sie kostenlos, uebertragen werden nur
    3 MB uint8 statt 12.6 MB float32.
    """
    x = x_uint8.to(device, non_blocking=non_blocking).float().div_(255.0)
    m = torch.as_tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    sd = torch.as_tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - m) / sd


class PipeDataset(Dataset):
    """imgsz    Kantenlaenge der Trainingsbilder (Standard 1024, wie YOLO)
    stride     Ausgabe-Stride des Netzes (4)
    train      False -> keine Augmentierung, ganzes Bild
    cache      Bilder dekodiert im RAM halten
    """

    MIN_BOX_PX = 5      # kleinere Boxen verwirft schon der Generator

    def __init__(self, img_dir, imgsz=1024, stride=4, train=True,
                 cache=True, p_mosaic=1.0, scale=(0.6, 1.4), seed=0,
                 min_overlap=0.7, sigma_clip=(0.6, 2.5)):
        self.files = sorted(sum([glob.glob(os.path.join(img_dir, e))
                                 for e in ("*.jpg", "*.png")], []))
        if not self.files:
            raise FileNotFoundError(img_dir)
        self.lbl = [f.replace(f"{os.sep}images{os.sep}",
                              f"{os.sep}labels{os.sep}").rsplit(".", 1)[0]
                    + ".txt" for f in self.files]
        self.imgsz, self.stride, self.train = imgsz, stride, train
        self.p_mosaic = p_mosaic if train else 0.0
        self.scale = scale
        self.seed = seed
        self.min_overlap = min_overlap
        self.sigma_clip = sigma_clip
        self.epoch = 0
        self._cache = [None] * len(self.files) if cache else None

    def __len__(self):
        return len(self.files)

    # -- Rohdaten ---------------------------------------------------------
    def _load(self, i, copy_img=True):
        """copy_img=False gibt die Cache-Referenz zurueck. Nur fuer reine
        Lesezugriffe (Zusammensetzen der Leinwand) - spart je Mosaic 4 x 3 MB
        Kopie."""
        if self._cache is not None and self._cache[i] is not None:
            img, box = self._cache[i]
            return (img.copy() if copy_img else img), box.copy()
        img = cv2.imread(self.files[i], cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        b = []
        if os.path.exists(self.lbl[i]):
            for line in open(self.lbl[i]):
                q = line.split()
                if len(q) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in q[1:5])
                b.append([cx * w, cy * h, bw * w, bh * h])
        box = np.array(b, np.float32).reshape(-1, 4)
        if self._cache is not None:
            self._cache[i] = (img, box)
            return (img.copy() if copy_img else img), box.copy()
        return img, box

    def prefill_cache(self, log=print):
        if self._cache is None:
            return
        for i in range(len(self.files)):
            self._load(i)
            if log and (i + 1) % 1000 == 0:
                log(f"    Cache {i + 1}/{len(self.files)}")

    # -- Leinwand + Ausschnitt in EINEM Schritt ---------------------------
    def _build(self, i, rng):
        """Baut direkt die benoetigte Bildregion.

        Die naive Reihenfolge waere: 2048x2048-Mosaic-Leinwand zusammensetzen
        (12.6 MB), skalieren, dann 1024x1024 ausschneiden. Hier wird stattdessen
        zuerst die Zielregion bestimmt und dann nur deren Ueberlappung mit den
        vier Kacheln kopiert. Gemessen 32.9 ms -> 9.6 ms.

        Die Boxen aller Kacheln werden unbesehen uebernommen; solche ausserhalb
        der Region verwirft _clip_boxes ohnehin.
        """
        S = self.imgsz
        mosaic = self.train and rng.random() < self.p_mosaic
        idx = ([i] + [int(rng.integers(0, len(self.files))) for _ in range(3)]
               if mosaic else [i])
        tiles = [self._load(j, copy_img=False) for j in idx]
        Hm = max(t[0].shape[0] for t in tiles)
        Wm = max(t[0].shape[1] for t in tiles)
        if mosaic:
            CH, CW = 2 * Hm, 2 * Wm
            offs = [(0, 0), (0, Wm), (Hm, 0), (Hm, Wm)]
        else:
            CH, CW = Hm, Wm
            offs = [(0, 0)]

        if not self.train:
            # Auswertung: ganzes Bild, nur auf ein Vielfaches des Strides padden
            img, box = tiles[0][0], tiles[0][1]
            h, w = img.shape[:2]
            ph, pw = (-h) % (self.stride * 8), (-w) % (self.stride * 8)
            if ph or pw:
                img = cv2.copyMakeBorder(img, 0, ph, 0, pw,
                                         cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))
            return img, box

        s = float(rng.uniform(*self.scale))
        R = max(16, int(round(S / s)))
        # Region bevorzugt aus der Leinwandmitte (dort treffen bei Mosaic vier
        # Bilder zusammen), Streuung eine halbe Regionsbreite
        x0 = int(round(CW / 2 + rng.uniform(-0.5, 0.5) * R - R / 2))
        y0 = int(round(CH / 2 + rng.uniform(-0.5, 0.5) * R - R / 2))

        reg = np.full((R, R, 3), 114, np.uint8)
        boxes = []
        for (im, bx), (oy, ox) in zip(tiles, offs):
            th, tw = im.shape[:2]
            ix0, iy0 = max(x0, ox), max(y0, oy)
            ix1, iy1 = min(x0 + R, ox + tw), min(y0 + R, oy + th)
            if ix1 > ix0 and iy1 > iy0:
                reg[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = \
                    im[iy0 - oy:iy1 - oy, ix0 - ox:ix1 - ox]
            if len(bx):
                b = bx.copy()
                b[:, 0] += ox
                b[:, 1] += oy
                boxes.append(b)

        f = S / R
        out = cv2.resize(reg, (S, S), interpolation=(cv2.INTER_AREA if R > S
                                                    else cv2.INTER_LINEAR))
        if boxes:
            box = np.concatenate(boxes, 0)
            box[:, 0] = (box[:, 0] - x0) * f
            box[:, 1] = (box[:, 1] - y0) * f
            box[:, 2:] *= f
        else:
            box = np.zeros((0, 4), np.float32)
        return out, box.astype(np.float32)

    @staticmethod
    def _clip_boxes(box, S, min_px):
        """Boxen am Rand auf das Bild beschneiden (modale Box, wie im
        Generator) und zu kleine Reste verwerfen."""
        if not len(box):
            return box
        x1 = np.clip(box[:, 0] - box[:, 2] / 2, 0, S)
        y1 = np.clip(box[:, 1] - box[:, 3] / 2, 0, S)
        x2 = np.clip(box[:, 0] + box[:, 2] / 2, 0, S)
        y2 = np.clip(box[:, 1] + box[:, 3] / 2, 0, S)
        w, h = x2 - x1, y2 - y1
        keep = (w >= min_px) & (h >= min_px)
        out = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, w, h], 1)[keep]
        return out.astype(np.float32)

    # -- Ziele ------------------------------------------------------------
    def _targets(self, box, S):
        G = S // self.stride
        hm = np.zeros((G, G), np.float32)
        wh = np.zeros((2, G, G), np.float32)
        off = np.zeros((2, G, G), np.float32)
        mask = np.zeros((G, G), np.float32)
        for cx, cy, bw, bh in box:
            ox, oy = cx / self.stride, cy / self.stride
            ix, iy = int(ox), int(oy)
            if not (0 <= ix < G and 0 <= iy < G):
                continue
            r = gaussian_radius(bh / self.stride, bw / self.stride,
                                self.min_overlap)
            sigma = float(np.clip((2 * max(1.0, r) + 1) / 6.0,
                                  *self.sigma_clip))
            draw_gaussian(hm, ix, iy, sigma)
            hm[iy, ix] = 1.0            # exaktes Zentrum bleibt hartes Positiv
            wh[0, iy, ix] = math.log(max(bw, 1e-3))
            wh[1, iy, ix] = math.log(max(bh, 1e-3))
            off[0, iy, ix] = ox - ix
            off[1, iy, ix] = oy - iy
            mask[iy, ix] = 1.0
        return hm, wh, off, mask

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003
                                    + self.epoch * 7_919 + i)
        img, box = self._build(i, rng)
        S = img.shape[0] if self.train else None

        if self.train:
            if rng.random() < 0.5:                     # horizontal
                img = img[:, ::-1]
                if len(box):
                    box[:, 0] = S - box[:, 0]
            if rng.random() < 0.5:                     # vertikal
                img = img[::-1]
                if len(box):
                    box[:, 1] = S - box[:, 1]
            img = np.ascontiguousarray(img)
            img = hsv_jitter(img, rng)
            box = self._clip_boxes(box, S, self.MIN_BOX_PX)

        # uint8 CHW in RGB. Die Normalisierung erfolgt auf der GPU
        # (normalize_gpu) - siehe Begruendung dort.
        x = torch.from_numpy(np.ascontiguousarray(
            img[:, :, ::-1].transpose(2, 0, 1)))

        if not self.train:
            return {"img": x, "box": torch.from_numpy(box),
                    "file": self.files[i]}
        hm, wh, off, mask = self._targets(box, S)
        return {"img": x,
                "hm": torch.from_numpy(hm)[None],
                "wh": torch.from_numpy(wh),
                "off": torch.from_numpy(off),
                "mask": torch.from_numpy(mask),
                "n": len(box)}


def collate(batch):
    out = {}
    for k in ("img", "hm", "wh", "off", "mask"):
        out[k] = torch.stack([b[k] for b in batch])
    out["n"] = sum(b["n"] for b in batch)
    return out


if __name__ == "__main__":
    ds = PipeDataset("../synth/train/images" if os.path.isdir(
        "../synth/train/images") else "synth/train/images",
        imgsz=1024, cache=False, seed=0)
    print(f"{len(ds)} Bilder")
    ns, sig = [], []
    for i in range(0, 40):
        s = ds[i]
        ns.append(s["n"])
        m = s["mask"].numpy() > 0
        ns[-1] = int(m.sum())
    print(f"Objekte je Bild (40 Stichproben): Median {np.median(ns):.0f}, "
          f"Spanne {min(ns)}-{max(ns)}")
    # Sigma-Verteilung fuer die realen Objektgroessen
    for d in (12, 20, 27, 36, 44):
        r = gaussian_radius(d / 4, d / 4, 0.7)
        print(f"  Objekt {d:2d} px -> {d/4:4.1f} Zellen -> Radius {r:4.2f} "
              f"-> sigma {np.clip((2*max(1.0,r)+1)/6, 0.6, 2.5):4.2f}")
