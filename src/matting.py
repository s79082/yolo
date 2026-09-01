"""
Stufe 1 der Pipeline: Matting der Rohrprofil-Cutouts.

PROBLEM
-------
Alle 9 Dateien in /cutouts sind RGBA, aber der Alphakanal ist durchgaengig 255
(nachgewiesen in docs/DATENSICHTUNG.md). Es sind also rechteckige Bildausschnitte,
keine freigestellten Objekte. Wuerde man sie direkt einkopieren, entstuende an jeder
Einfuegestelle eine rechteckige Naht (Helligkeits-/Textursprung entlang einer
achsparallelen Kante). Ein Detektor lernt diese Naht als Abkuerzung ("shortcut
learning") statt der Objektform -> exzellente Metriken auf synthetischen Daten,
Totalausfall auf echten Bildern.

LOESUNG
-------
Rohrprofile sind Kreisringe; in Schraegansicht projizieren sie zu Ellipsen.
Wir bestimmen pro Cutout eine Ellipse (cx, cy, rx, ry, theta), die die AUSSENKANTE
des hellen verzinkten Rohrkranzes beschreibt, und erzeugen daraus eine weiche
Alphamaske.

Vorgehen (zweistufig, reproduzierbar):
  1. Manuelle Startwerte, abgelesen aus einer 14x vergroesserten Rasteransicht
     (docs/cutouts_grid.png). Bei 9 Objekten ist das exakter als jeder Auto-Fit.
  2. Lokale numerische Verfeinerung: maximiert die radiale Kantenenergie
     -dI/dr entlang der Ellipse (innen hell = Rohrkranz, aussen dunkler =
     Hintergrund). Coordinate-Descent auf 6x upgesampeltem, geglaettetem Bild.
  3. Radiale Halbwertskorrektur: Schritt 2 konvergiert konstruktionsbedingt auf
     den GRAT des hellen Kranzes (dort ist der Kontrast innen-aussen maximal),
     nicht auf dessen Aussenkante. Ein mittleres Radialprofil bestimmt daher
     zusaetzlich, wo die Helligkeit auf den Halbwert zwischen Kranz-Maximum und
     Hintergrundniveau faellt (klassisches FWHM-Kantenkriterium). Um genau
     diesen Faktor wird die Ellipse aufgeweitet.
  4. Visuelle Endkontrolle ueber docs/matting_check.png.
"""
import json
import os
import numpy as np
import cv2

CUTOUT_DIR = "cutouts"
OUT_DIR = "assets/masks"
UP = 6  # Upsampling-Faktor fuer subpixelgenaue Kantensuche

# (cx, cy, rx, ry, theta_deg) in ORIGINAL-Pixelkoordinaten, manuell abgelesen
SEEDS = {
    "cutted1.png":  (16.0, 15.5, 16.5, 16.0, 0.0),
    "cutted_4.png": (15.5, 15.0, 14.0, 14.5, 0.0),
    "cutted_5.png": (21.0, 19.0, 20.0, 17.5, 0.0),
    "cutted_6.png": (14.0, 15.5, 13.5, 14.0, 0.0),
    "cutted_7.png": (19.5, 17.5, 19.0, 16.5, 0.0),
    "cutted_8.png": (19.0, 18.0, 18.0, 16.5, 0.0),
    "cutted_9.png": (13.0, 15.5, 13.0, 14.0, 0.0),
    "cuuted2.png":  (17.5, 16.5, 16.5, 15.5, 0.0),
    "cuuted3.png":  (17.5, 16.0, 16.5, 16.0, 0.0),
}


def _edge_score(gray_up, p, n_theta=180, band=1.2):
    """Mittlere radiale Kantenenergie entlang der Ellipse.

    Fuer jeden Winkel wird die Helligkeit knapp innerhalb und knapp ausserhalb
    der Ellipse abgetastet; die Differenz (innen - aussen) ist gross, wenn die
    Ellipse genau auf der Aussenkante des hellen Rohrkranzes liegt.
    """
    cx, cy, rx, ry, th = p
    if rx <= 2 or ry <= 2:
        return -1e9
    H, W = gray_up.shape
    t = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(th), np.sin(th)

    def sample(scale):
        ex, ey = rx * scale * np.cos(t), ry * scale * np.sin(t)
        x = (cx + ex * ct - ey * st) * UP
        y = (cy + ex * st + ey * ct) * UP
        x = np.clip(x, 0, W - 1).astype(np.float32)
        y = np.clip(y, 0, H - 1).astype(np.float32)
        return cv2.remap(gray_up, x.reshape(1, -1), y.reshape(1, -1),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()

    inner = sample(1.0 - band / max(rx, ry))
    outer = sample(1.0 + band / max(rx, ry))
    return float(np.mean(inner - outer))


def refine(gray_up, seed, iters=60):
    """Coordinate-Descent mit schrumpfender Schrittweite (kein scipy noetig)."""
    p = np.array(seed, dtype=np.float64)
    p[4] = np.deg2rad(p[4])
    best = _edge_score(gray_up, p)
    steps = np.array([0.8, 0.8, 0.8, 0.8, np.deg2rad(6)])
    # Suchraum bewusst eng: die Handstartwerte sind gut, die Optimierung soll
    # nur nachjustieren, nicht auf einen Nachbarring wegspringen.
    lo = p - np.array([3.0, 3.0, 3.0, 3.0, np.deg2rad(25)])
    hi = p + np.array([3.0, 3.0, 3.0, 3.0, np.deg2rad(25)])
    for _ in range(iters):
        improved = False
        for i in range(5):
            for s in (+steps[i], -steps[i]):
                q = p.copy()
                q[i] = np.clip(q[i] + s, lo[i], hi[i])
                sc = _edge_score(gray_up, q)
                if sc > best:
                    best, p, improved = sc, q, True
        if not improved:
            steps *= 0.5
            if steps[0] < 0.02:
                break
    return p, best


def outer_edge_scale(gray_up, p, n_theta=180, s_lo=0.75, s_hi=1.55, n_s=80,
                     level=0.25):
    """Findet den Radiusfaktor der AUSSENKANTE ueber ein mittleres Radialprofil.

    Rueckgabe: Faktor s* mit  I(s*) = I_far + level * (I_peak - I_far),
    wobei I_peak das Kranzmaximum und I_far das Hintergrundniveau
    (Mittel ueber s in [s_hi-0.15, s_hi]) ist.
    """
    cx, cy, rx, ry, th = p
    H, W = gray_up.shape
    t = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    ss = np.linspace(s_lo, s_hi, n_s)
    prof = np.empty(n_s, np.float32)
    for i, s in enumerate(ss):
        ex, ey = rx * s * np.cos(t), ry * s * np.sin(t)
        x = np.clip((cx + ex * ct - ey * st) * UP, 0, W - 1).astype(np.float32)
        y = np.clip((cy + ex * st + ey * ct) * UP, 0, H - 1).astype(np.float32)
        v = cv2.remap(gray_up, x.reshape(1, -1), y.reshape(1, -1),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
        # Median statt Mittel: robust gegen einzelne Nachbarrohre im Ausschnitt
        prof[i] = np.median(v)
    k = int(np.argmax(prof))
    i_peak = prof[k]
    i_far = float(np.mean(prof[ss >= s_hi - 0.15]))
    if i_peak - i_far < 3.0:          # kein auswertbarer Kontrast -> Fallback
        return 1.10
    half = i_far + level * (i_peak - i_far)
    for i in range(k, n_s):
        if prof[i] <= half:
            # lineare Interpolation zwischen i-1 und i
            if i == 0:
                return float(ss[0])
            a, b = prof[i - 1], prof[i]
            w = 0.0 if a == b else (a - half) / (a - b)
            return float(ss[i - 1] + w * (ss[i] - ss[i - 1]))
    return float(ss[-1])


def make_alpha(shape, p, feather=1.1, supersample=8):
    """Weiche Ellipsenmaske.

    feather: Breite des Alpha-Uebergangs in Original-Pixeln. ~1 px entspricht der
    Unschaerfe, die eine Kamera bei 25-px-Objekten ohnehin erzeugt; eine harte
    Kante wuerde beim Einkopieren als Aliasing-Artefakt sichtbar bleiben.
    """
    h, w = shape
    cx, cy, rx, ry, th = p
    ys, xs = np.mgrid[0:h * supersample, 0:w * supersample].astype(np.float32)
    xs = (xs + 0.5) / supersample
    ys = (ys + 0.5) / supersample
    dx, dy = xs - cx, ys - cy
    ct, st = np.cos(th), np.sin(th)
    u = dx * ct + dy * st
    v = -dx * st + dy * ct
    # normierter Radius (1.0 == Ellipsenrand)
    r = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
    scale = min(rx, ry)
    a = np.clip((1.0 - r) * scale / feather + 0.5, 0.0, 1.0)
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    return a


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {}
    tiles = []
    for name, seed in SEEDS.items():
        path = os.path.join(CUTOUT_DIR, name)
        bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, :3]
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_up = cv2.resize(gray, (w * UP, h * UP), interpolation=cv2.INTER_CUBIC)
        gray_up = cv2.GaussianBlur(gray_up, (0, 0), UP * 0.35)

        p, score = refine(gray_up, seed)
        # Aussenkanten-Korrektur, hart auf [1.02, 1.10] begrenzt: bei den acht
        # kontrastreichen Cutouts liegt der 25%-Punkt bei 1.02-1.05. Nur
        # cutted_8 hat einen dunklen, kontraktarmen Kranz; dort liefe das
        # Kriterium ohne Deckel auf 1.19 und die Maske wuerde Hintergrund
        # aufsaugen. Der Deckel macht die Stufe fuer alle 9 Objekte stabil.
        s_out = float(np.clip(outer_edge_scale(gray_up, p), 1.02, 1.10))
        p_out = p.copy()
        p_out[2] *= s_out
        p_out[3] *= s_out
        alpha = make_alpha((h, w), p_out)

        rgba = np.dstack([bgr, (alpha * 255).astype(np.uint8)])
        cv2.imwrite(os.path.join(OUT_DIR, name), rgba)

        # tight bbox der Maske (alpha > 0.5) -> spaetere Labelgrundlage
        m = alpha > 0.5
        ys, xs = np.where(m)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        meta[name] = {
            "ellipse": {"cx": p_out[0], "cy": p_out[1], "rx": p_out[2],
                        "ry": p_out[3], "theta_deg": float(np.rad2deg(p_out[4]))},
            "ellipse_ridge": {"rx": p[2], "ry": p[3]},
            "outer_edge_scale": s_out,
            "seed": list(seed),
            "edge_score": score,
            "bbox_xyxy": bbox,
            "size": [w, h],
            "coverage": float(alpha.mean()),
        }
        print(f"{name:14s} c=({p_out[0]:5.2f},{p_out[1]:5.2f}) "
              f"r_grat=({p[2]:5.2f},{p[3]:5.2f}) x{s_out:4.2f} -> "
              f"r=({p_out[2]:5.2f},{p_out[3]:5.2f}) th={np.rad2deg(p[4]):+6.1f} "
              f"score={score:6.2f} bbox={bbox}")
        tiles.append((name, bgr, alpha, p_out))

    with open(os.path.join(OUT_DIR, "ellipses.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- Kontrollbild ---
    S = 12
    cw = max(t[1].shape[1] for t in tiles) * S + 30
    ch = max(t[1].shape[0] for t in tiles) * S + 40
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * ch, cols * cw * 2, 3), 25, np.uint8)
    for i, (name, bgr, alpha, p) in enumerate(tiles):
        h, w = bgr.shape[:2]
        big = cv2.resize(bgr, (w * S, h * S), interpolation=cv2.INTER_NEAREST)
        ov = big.copy()
        cv2.ellipse(ov, (int(p[0] * S), int(p[1] * S)),
                    (int(p[2] * S), int(p[3] * S)), np.rad2deg(p[4]), 0, 360,
                    (0, 0, 255), 2)
        # rechte Haelfte: freigestellt auf neutralem Grau -> Randqualitaet sichtbar
        ab = cv2.resize(alpha, (w * S, h * S), interpolation=cv2.INTER_LINEAR)[..., None]
        cut = (big * ab + np.array([120, 120, 120]) * (1 - ab)).astype(np.uint8)
        r, c = divmod(i, cols)
        y0, x0 = r * ch + 10, c * cw * 2 + 10
        sheet[y0:y0 + h * S, x0:x0 + w * S] = ov
        sheet[y0:y0 + h * S, x0 + cw:x0 + cw + w * S] = cut
        cv2.putText(sheet, name, (x0, y0 + h * S + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite("docs/matting_check.png", sheet)
    print("\n-> assets/masks/*.png, assets/masks/ellipses.json, docs/matting_check.png")


if __name__ == "__main__":
    main()
