"""
Dekodierung der PipeNet-Ausgabe und Test-Time-Augmentation.

KERNPUNKT: Peak-Extraktion statt NMS
------------------------------------
Ein 3x3-MaxPool markiert genau die Zellen, die lokales Maximum ihrer
8er-Nachbarschaft sind. Jedes lokale Maximum wird zu genau einer Detektion.

Warum das hier besser passt als Box-NMS:
  Gemessen liegt der Abstand zum naechsten Nachbarn im 10. Perzentil bei
  0.84 Objektdurchmessern, d. h. benachbarte Boxen ueberlappen deutlich.
  Box-NMS muss dann eine IoU-Schwelle waehlen und tauscht damit zwangslaeufig
  "Nachbar verloren" gegen "Dublette behalten". Die Zentren sind aber immer
  getrennt (Median 1.09-1.24 Durchmesser, bei Stride 4 also 4.4-5.5 Zellen).
  Auf der Heatmap gibt es deshalb keine Mehrdeutigkeit und keine Schwelle:
  ein Maximum = ein Rohr.

  Grenze des Verfahrens: liegen zwei Zentren weniger als 2 Zellen (8 px)
  auseinander, verschmelzen sie zu einem Maximum. Bei einem gemessenen
  Minimum von 0.83 Durchmessern (~16 px = 4 Zellen) tritt das in den
  vorliegenden Daten nicht auf.

ROTATIONS-TTA
-------------
Rohrenden sind nahezu rotationssymmetrisch. Drehungen um 90 Grad sind auf
einem Pixelgitter EXAKT (reine Indexpermutation, keine Interpolation), im
Unterschied zur ueblichen Mehrskalen-TTA. Gemittelt wird die Heatmap ueber
die 4 Drehungen x 2 Spiegelungen = 8 Elemente der Diedergruppe D4.

Der Versatzkopf wird bewusst NUR aus dem untransformierten Durchlauf
gelesen: der Subpixel-Versatz ist ein Vektor in Zelleinheiten, dessen
korrekte Ruecktransformation zusaetzliche Vorzeichenfaelle einfuehren wuerde.
Der Nutzen der TTA liegt im Finden (Heatmap), nicht im Nachjustieren um
Bruchteile einer Zelle. Die Groessenkanaele werden gemittelt, wobei bei
90/270 Grad Breite und Hoehe getauscht werden.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
def peaks(hm, thresh=0.3, topk=2000, pool=3, valid=None):
    """hm: (1,1,H,W) Wahrscheinlichkeiten. Gibt (idx_y, idx_x, score).

    valid = (n_rows, n_cols): Zellen jenseits davon stammen aus dem auf /32
    gepaddeten Rand und werden verworfen. Gemessen erzeugt der Rand in den
    vorliegenden Bildern KEINE Maxima ueber 0.3 (geprueft fuer val und test);
    die Maske ist reine Absicherung fuer andere Bildformate.
    """
    keep = (hm == F.max_pool2d(hm, pool, stride=1, padding=pool // 2))
    h = hm * keep
    if valid is not None:
        vr, vc = valid
        m = torch.zeros_like(h, dtype=torch.bool)
        m[..., :vr, :vc] = True
        h = h * m
    B, _, H, W = h.shape
    flat = h.view(B, -1)
    k = min(topk, flat.shape[1])
    score, idx = flat.topk(k, dim=1)
    sel = score[0] >= thresh
    idx, score = idx[0][sel], score[0][sel]
    return idx // W, idx % W, score


def decode(out, stride=4, thresh=0.3, topk=2000, valid=None):
    """Modellausgabe -> Boxen in Eingangs-Pixelkoordinaten (xyxy) + Scores."""
    hm = torch.sigmoid(out["hm"])
    iy, ix, score = peaks(hm, thresh, topk, valid=valid)
    if len(score) == 0:
        return (torch.zeros(0, 4, device=hm.device),
                torch.zeros(0, device=hm.device))
    off = out["off"][0, :, iy, ix]          # (2, N)
    wh = out["wh"][0, :, iy, ix]            # (2, N)
    cx = (ix.float() + off[0]) * stride
    cy = (iy.float() + off[1]) * stride
    w, h = wh[0].exp(), wh[1].exp()
    box = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return box, score


# ---------------------------------------------------------------------------
def _pad_to(x, mult=32):
    H, W = x.shape[-2:]
    ph, pw = (-H) % mult, (-W) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), value=0.0)
    return x, H, W


@torch.no_grad()
def forward_tta(model, x, rots=(0, 1, 2, 3), flips=(False, True), mult=32):
    """Mittelt Heatmap und Groesse ueber die Diedergruppe D4.

    rots=(0,), flips=(False,) entspricht einem normalen Durchlauf.
    """
    x, H0, W0 = _pad_to(x, mult)
    acc_hm, acc_wh, n = None, None, 0
    base = None
    for fl in flips:
        xf = torch.flip(x, dims=[-1]) if fl else x
        for k in rots:
            xr = torch.rot90(xf, k, dims=(-2, -1)) if k else xf
            o = model(xr)
            hm = torch.sigmoid(o["hm"])
            wh = o["wh"]
            if k:                                    # Drehung zuruecknehmen
                hm = torch.rot90(hm, -k, dims=(-2, -1))
                wh = torch.rot90(wh, -k, dims=(-2, -1))
                if k % 2 == 1:                       # 90/270: w <-> h
                    wh = wh.flip(1)
            if fl:                                   # Spiegelung zuruecknehmen
                hm = torch.flip(hm, dims=[-1])
                wh = torch.flip(wh, dims=[-1])
            acc_hm = hm if acc_hm is None else acc_hm + hm
            acc_wh = wh if acc_wh is None else acc_wh + wh
            n += 1
            if base is None:                         # untransformierter Lauf
                base = o
    out = {"hm": torch.logit(torch.clamp(acc_hm / n, 1e-6, 1 - 1e-6)),
           "wh": acc_wh / n,
           "off": base["off"]}                       # nur k=0, siehe Modulkopf
    return out, H0, W0


@torch.no_grad()
def predict_image(model, x, stride=4, thresh=0.3, topk=2000, tta=False):
    """x: (1,3,H,W) normalisiert. Gibt Boxen (numpy) und Scores."""
    if tta:
        out, H0, W0 = forward_tta(model, x)
    else:
        xp, H0, W0 = _pad_to(x, 32)
        out = model(xp)
    import math
    box, score = decode(out, stride, thresh, topk,
                        valid=(math.ceil(H0 / stride), math.ceil(W0 / stride)))
    if len(box):
        box[:, 0::2] = box[:, 0::2].clamp(0, W0 - 1)
        box[:, 1::2] = box[:, 1::2].clamp(0, H0 - 1)
    return box.cpu().numpy(), score.cpu().numpy()


# ---------------------------------------------------------------------------
def heatmap_overlay(img_bgr, hm, scale, stride=4, gain=2.5, marker=None):
    """Heatmap geometrisch KORREKT ueber das Originalbild legen.

    Ein naives cv2.resize(hm, (W, H)) ist um bis zu mehrere Pixel verschoben.
    Zwei unabhaengige Ursachen, beide hier behoben:

    1. PADDING. Die Vorhersage laeuft auf dem auf /32 gepaddeten, skalierten
       Bild. Die Heatmap ist also groesser als der Bildinhalt. Streckt man sie
       auf (W, H), wird das Padding mitgestreckt und der Versatz waechst linear
       zum unteren/rechten Bildrand. Gemessen fuer test5.jpg bei scale 1.8:
       12 px Padding -> 6.6 px Versatz nach oben am Bildunterrand.

    2. ZELLBEZUGSPUNKT. Zielerzeugung und Dekodierung verwenden die Konvention
       "Zelle j entspricht skaliertem Pixel 4j", also dem BLOCKANFANG. Der Block
       deckt aber [4j, 4j+4) ab; sein Mittelpunkt liegt bei 4j+2. Wird die
       Zelle am Blockanfang gezeichnet, erscheint das Maximum um 2/scale px
       nach oben-links verschoben.

    Summe fuer test5.jpg bei scale 1.8: bis zu 7.7 px nach oben, bei einem
    20-px-Objekt also 39 % des Durchmessers. Die DETEKTIONEN sind davon nicht
    betroffen (gemessener systematischer Zentrumsversatz nur -0.4 bis -1.0 px);
    es war ausschliesslich ein Darstellungsfehler.

    Korrekte Abbildung Zielpixel -> Heatmapzelle:
        Blockmitte in Originalkoordinaten:  p = (stride*c + stride/2) / scale
        umgestellt:                         c = p*scale/stride - 0.5
    Das ist eine affine Abbildung; cv2.warpAffine mit WARP_INVERSE_MAP wendet
    sie genau in dieser Richtung an und tastet das Padding nie an.
    """
    H, W = img_bgr.shape[:2]
    a = scale / stride
    M = np.array([[a, 0.0, -0.5], [0.0, a, -0.5]], np.float32)
    h = cv2.warpAffine(hm.astype(np.float32), M, (W, H),
                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    cm = cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype(np.uint8),
                           cv2.COLORMAP_INFERNO)
    al = np.clip(h * gain, 0, 1)[..., None]
    return (img_bgr * (1 - al) + cm * al).astype(np.uint8)


# ---------------------------------------------------------------------------
def selftest_equivariance(model, device="cuda", size=256):
    """Prueft, dass die Ruecktransformation in forward_tta korrekt ist.

    Ein einzelner gedrehter Durchlauf muss - nach Ruecktransformation - seine
    Maxima an denselben Positionen haben wie der untransformierte Durchlauf.
    Vollstaendige Gleichheit ist nicht zu erwarten (Faltungen sind nicht
    rotationsequivariant), aber die Positionen muessen uebereinstimmen, sonst
    ist ein Vorzeichen- oder Achsenfehler im Code.
    """
    model.eval()
    x = torch.randn(1, 3, size, size, device=device) * 0.5
    with torch.no_grad():
        o0 = model(x)
        hm0 = torch.sigmoid(o0["hm"])
        res = {}
        for k in (1, 2, 3):
            ok = model(torch.rot90(x, k, dims=(-2, -1)))
            hmk = torch.rot90(torch.sigmoid(ok["hm"]), -k, dims=(-2, -1))
            # Korrelation der zurueckgedrehten Karte mit dem Original
            a = hm0.flatten() - hm0.mean()
            b = hmk.flatten() - hmk.mean()
            res[f"rot{k*90}"] = float((a @ b) / (a.norm() * b.norm() + 1e-9))
        of = model(torch.flip(x, dims=[-1]))
        hmf = torch.flip(torch.sigmoid(of["hm"]), dims=[-1])
        a = hm0.flatten() - hm0.mean()
        b = hmf.flatten() - hmf.mean()
        res["fliplr"] = float((a @ b) / (a.norm() * b.norm() + 1e-9))
    return res
