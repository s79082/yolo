"""
Stufe 3: Synthetische Trainingsdatengenerierung (Copy-Paste-Pipeline).

Erzeugt YOLO-Datensaetze aus 6 freigestellten Rohrprofilen (assets/masks),
55 echten Rohrstangen als Okkluder (assets/bars) und 7 Hintergruenden (bg/).

Die Begruendung jeder einzelnen Transformation und ihres Anteils steht in
docs/PIPELINE.md; der Code hier nennt zu jedem Block die Kurzbegruendung.

Aufruf:
    python src/generate.py --out synth/train --n 4000 --seed 1
    python src/generate.py --out synth/sval  --n  300 --seed 99
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------
CANVAS = 1024          # Kantenlaenge der erzeugten Bilder

# Hintergruende. "clean plate" = Rohrstapel OHNE sichtbare Stirnflaechen,
# also exakt der Kontext, in dem die echten Objekte vorkommen.
# pipe_d = gemessener Rohrdurchmesser in Pixeln (aus src/occluders.py),
# er verankert den physikalisch korrekten Groessenbezug Profil <-> Umgebung.
BACKGROUNDS = [
    dict(path="bg/test_cutted.png",    clean=True,  pipe_d=16, w=6.0),
    dict(path="bg/img3884_cutted.png", clean=True,  pipe_d=18, w=6.0),
    dict(path="bg/test4_cutted.png",   clean=True,  pipe_d=21, w=6.0),
    dict(path="bg/img3886_cutted.png", clean=True,  pipe_d=20, w=6.0),
    dict(path="bg/bg_3.jpg",         clean=False, pipe_d=None, w=1.0),
    dict(path="bg/bg_4.jpg",         clean=False, pipe_d=None, w=1.0),
    dict(path="bg/bg_5.jpg",         clean=False, pipe_d=None, w=1.0),
]

# Zielgroesse der Objekte im erzeugten Bild (Durchmesser in px).
# Messung an den echten Labels (native Aufloesung):
#   val  20.6 - 35.8 px (Median 27.4) | test 12.2 - 31.9 px (Median 19.6)
# Der Generator deckt 12 - 44 px ab: die reale Spanne plus ~25 % Reserve nach
# oben, damit auch Aufnahmen aus geringerem Abstand abgedeckt sind.
D_MIN, D_MAX = 12.0, 44.0

P = dict(  # zentrale Wahrscheinlichkeitstabelle (siehe docs/PIPELINE.md)
    negative_image=0.10,
    bg_clean=0.80,
    bg_fliplr=0.50, bg_flipud=0.15, bg_rot=0.35,
    inst_oval_oblate=0.45, inst_oval_prolate=0.12,
    inst_dent=0.18,
    inst_perspective=0.30,
    inst_photometric=1.00, inst_gamma=0.80, inst_hsv=0.70,
    inst_harmonize=0.85,
    inst_blur=0.60, inst_noise=0.70, inst_shadow=0.50,
    inst_stub=0.88,
    stub_behind_front=0.75,
    cluster_recess=0.55,
    img_has_bars=0.65,
    img_clutter=0.35,
    glob_bc=0.90, glob_hsv=0.70, glob_blur=0.35, glob_motion=0.10,
    glob_noise=0.80, glob_vignette=0.25, glob_jpeg=0.85,
    alpha_feather=1.00,
)

STATS = Counter()


# ----------------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------------
def _u(rng, a, b):
    return float(rng.uniform(a, b))


def load_assets():
    with open("assets/masks/ellipses.json") as f:
        meta = json.load(f)
    profiles = []
    for name, m in meta.items():
        rgba = cv2.imread(os.path.join("assets/masks", name), cv2.IMREAD_UNCHANGED)
        e = m["ellipse"]
        profiles.append(dict(name=name, rgba=rgba, e=e,
                             d_src=2.0 * np.sqrt(e["rx"] * e["ry"])))
    bars = [cv2.imread(p, cv2.IMREAD_UNCHANGED)
            for p in sorted(glob.glob("assets/bars/*.png"))]
    bgs = []
    for b in BACKGROUNDS:
        img = cv2.imread(b["path"], cv2.IMREAD_COLOR)
        bgs.append({**b, "img": img})
    return profiles, bars, bgs


def alpha_paste(canvas, patch_bgr, patch_a, x0, y0):
    """Alpha-Blending von patch an (x0, y0). Gibt das beschnittene
    Alpha-Fenster (a, x0, y0) zurueck, das fuer die Sichtbarkeits-
    buchhaltung gebraucht wird."""
    H, W = canvas.shape[:2]
    h, w = patch_a.shape
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + w), min(H, y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return None
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    a = patch_a[sy0:sy1, sx0:sx1][..., None]
    canvas[dy0:dy1, dx0:dx1] = (canvas[dy0:dy1, dx0:dx1] * (1 - a)
                                + patch_bgr[sy0:sy1, sx0:sx1] * a)
    return patch_a[sy0:sy1, sx0:sx1], dx0, dy0


# ----------------------------------------------------------------------------
# T1 - Hintergrund: Zufallsausschnitt, Zoom, Spiegelung, Kleinrotation
# ----------------------------------------------------------------------------
def make_background(rng, bgs):
    """Waehlt Hintergrund und Zoom so, dass die spaetere Objektgroesse
    physikalisch zur Rohrstaerke der Umgebung passt."""
    clean = rng.random() < P["bg_clean"]
    pool = [b for b in bgs if b["clean"] == clean] or bgs
    w = np.array([b["w"] for b in pool], float)
    bg = pool[int(rng.choice(len(pool), p=w / w.sum()))]
    img = bg["img"]
    H, W = img.shape[:2]

    if bg["pipe_d"]:
        # Zoom so, dass Objektdurchmesser = pipe_d * zoom im Zielbereich liegt.
        # Begruendung: Stirnflaechen und Rohrschaefte derselben Aufnahme werden
        # annaehernd im selben Massstab abgebildet; unabhaengig gezogen,
        # wuerden zufaellige Missverhaeltnisse dem Netz eine falsche
        # Kontext-Groessen-Relation antrainieren. Das Verhaeltnis 1:1 ist
        # gesetzt: Die seitlichen Rahmenrohre, deren Stirnflaechen detektiert
        # werden, muessen nicht denselben Durchmesser haben wie die Rohre der
        # Clean Plates.
        z_lo = max(0.55, D_MIN / bg["pipe_d"])
        z_hi = min(2.20, D_MAX / bg["pipe_d"])
        zoom = _u(rng, z_lo, max(z_lo + 1e-3, z_hi))
        d_base = bg["pipe_d"] * zoom
    else:
        zoom = _u(rng, 0.6, 1.6)
        d_base = _u(rng, D_MIN, D_MAX)

    # Ausschnittgroesse auf 1.35 x kuerzeste Bildkante begrenzen und fehlende
    # Flaeche gespiegelt ergaenzen. np.tile wuerde harte, achsparallele Naehte
    # erzeugen - genau die Art von Artefakt, die ein CNN sofort als Merkmal
    # missbraucht.
    crop = int(round(CANVAS / zoom))
    crop = min(crop, int(1.35 * min(H, W)))
    if crop > min(H, W):
        pad = int(np.ceil((crop - min(H, W)) / 2)) + 2
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT101)
        H, W = img.shape[:2]
    x0 = int(rng.integers(0, W - crop + 1))
    y0 = int(rng.integers(0, H - crop + 1))
    out = img[y0:y0 + crop, x0:x0 + crop]
    interp = cv2.INTER_AREA if crop > CANVAS else cv2.INTER_CUBIC
    out = cv2.resize(out, (CANVAS, CANVAS), interpolation=interp)

    if rng.random() < P["bg_fliplr"]:
        out = out[:, ::-1]
        STATS["bg_fliplr"] += 1
    if rng.random() < P["bg_flipud"]:
        out = out[::-1]
        STATS["bg_flipud"] += 1
    if rng.random() < P["bg_rot"]:
        # +-6 Grad: die Rohrlinien im Bild sind nie exakt achsparallel;
        # ohne diese Variation waere die Vorzugsrichtung im Datensatz
        # kuenstlich diskret.
        ang = _u(rng, -6, 6)
        M = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), ang, 1.06)
        out = cv2.warpAffine(out, M, (CANVAS, CANVAS), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT101)
        STATS["bg_rot"] += 1
    return np.ascontiguousarray(out).astype(np.float32), d_base, bg


# ----------------------------------------------------------------------------
# T2 - Instanz-Geometrie: Skalierung, Ovalisierung, Delle, Perspektive, Rotation
# ----------------------------------------------------------------------------
def apply_dent(bgr, a, rng):
    """Radiale Einbeulung ueber 1-2 Winkelsektoren.

    Modelliert mechanisch beschaedigte Rohrenden (im Bestand haeufig, da
    Bauzaunfelder geworfen und gestapelt werden). Umgesetzt als glatte
    radiale Verschiebung: Inhalt wird in Richtung Zentrum gezogen, dadurch
    wandert die Aussenkante lokal nach innen.
    """
    h, w = a.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-6
    phi = np.arctan2(dy, dx)
    R = 0.5 * min(h, w)
    disp = np.zeros_like(r)
    for _ in range(int(rng.integers(1, 3))):
        phi0 = _u(rng, -np.pi, np.pi)
        sig = _u(rng, 0.30, 0.85)                 # Sektorbreite (rad)
        amp = _u(rng, 0.10, 0.30)                 # relative Eindringtiefe
        d = np.arctan2(np.sin(phi - phi0), np.cos(phi - phi0))
        disp += amp * np.exp(-(d / sig) ** 2) * (r / R) ** 2
    scale = 1.0 + disp
    mx = (cx + dx * scale).astype(np.float32)
    my = (cy + dy * scale).astype(np.float32)
    bgr = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                  borderValue=0)
    return bgr, a


def render_instance(prof, d_target, rng):
    """Erzeugt (bgr, alpha) einer transformierten Instanz."""
    rgba = prof["rgba"]
    src = rgba[:, :, :3].astype(np.float32)
    sa = rgba[:, :, 3].astype(np.float32) / 255.0

    # -- T2a Ovalisierung -----------------------------------------------------
    # Reale Messung: Seitenverhaeltnis w/h der Labels val p10-p90 = 0.97-1.31,
    # test 0.85-1.22. Ursachen: Schraegansicht (Projektion des Kreises auf
    # eine Ellipse) und tatsaechlich gequetschte Rohre.
    aspect = 1.0
    u = rng.random()
    if u < P["inst_oval_oblate"]:
        aspect = _u(rng, 0.66, 0.97)
        STATS["inst_oval"] += 1
    elif u < P["inst_oval_oblate"] + P["inst_oval_prolate"]:
        aspect = _u(rng, 1.03, 1.30)
        STATS["inst_oval"] += 1

    sx = d_target / prof["d_src"] / np.sqrt(aspect)
    sy = d_target / prof["d_src"] * np.sqrt(aspect)
    nw = max(5, int(round(src.shape[1] * sx)))
    nh = max(5, int(round(src.shape[0] * sy)))
    # INTER_AREA beim Verkleinern: verhindert Aliasing, das sonst als
    # verraeterisches Hochfrequenzmuster im Objekt zurueckbliebe.
    interp = cv2.INTER_AREA if (sx < 1 or sy < 1) else cv2.INTER_CUBIC
    bgr = cv2.resize(src, (nw, nh), interpolation=interp)
    a = cv2.resize(sa, (nw, nh), interpolation=interp)

    # -- T2b Delle ------------------------------------------------------------
    if rng.random() < P["inst_dent"]:
        bgr, a = apply_dent(bgr, a, rng)
        STATS["inst_dent"] += 1

    # -- T2c Perspektive ------------------------------------------------------
    # Milde Homographie: der Kreisring wird bei Schraegblick nicht nur zur
    # Ellipse, der helle Kranz wird auf einer Seite auch breiter (Sichtbarkeit
    # der Rohrinnenwand). Eine reine Skalierung kann das nicht abbilden.
    if rng.random() < P["inst_perspective"]:
        h, w = a.shape
        m = 0.09
        srcp = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dstp = srcp + np.float32([[_u(rng, -m, m) * w, _u(rng, -m, m) * h]
                                  for _ in range(4)])
        M = cv2.getPerspectiveTransform(srcp, dstp)
        bgr = cv2.warpPerspective(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        a = cv2.warpPerspective(a, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        STATS["inst_perspective"] += 1

    # -- T2d Rotation (immer) -------------------------------------------------
    # Rohrenden besitzen keine kanonische Bildorientierung: Kamerarollwinkel
    # und Stapelausrichtung sind beliebig. Einzige Orientierungsinformation
    # ist die Richtung des Glanzlichts auf dem Kranz, die vom Sonnenstand
    # abhaengt. Volle Rotationsinvarianz ist daher korrekt und kostenlos.
    ang = _u(rng, 0, 360)
    STATS["inst_total"] += 1
    h, w = a.shape
    diag = int(np.ceil(np.hypot(h, w))) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[0, 2] += (diag - w) / 2
    M[1, 2] += (diag - h) / 2
    bgr = cv2.warpAffine(bgr, M, (diag, diag), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpAffine(a, M, (diag, diag), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    STATS["inst_rotation"] += 1
    return bgr, np.clip(a, 0, 1)


# ----------------------------------------------------------------------------
# T3 - Instanz-Photometrie und Einbettungsrealismus
# ----------------------------------------------------------------------------
def photometric(bgr, a, rng):
    if rng.random() < P["inst_photometric"]:
        # Helligkeit: Stirnflaechen liegen je nach Stapelposition in praller
        # Sonne oder tiefem Schatten. Spannweite aus den Cutouts selbst
        # ablesbar (mittlere Kranzhelligkeit 88-156 Grauwerte).
        bgr = bgr * _u(rng, 0.70, 1.35)
        STATS["inst_photometric"] += 1
    if rng.random() < P["inst_gamma"]:
        # Gamma statt reinem Kontrast: bildet die nichtlineare Tonwertkurve
        # unterschiedlicher Kameras/JPEG-Pipelines ab und veraendert vor allem
        # das Verhaeltnis dunkle Bohrung <-> heller Kranz - genau das Merkmal,
        # auf das der Detektor sonst zu starr konditioniert waere.
        g = _u(rng, 0.75, 1.35)
        bgr = np.clip(bgr, 0, 255)
        bgr = 255.0 * np.power(bgr / 255.0, g)
        STATS["inst_gamma"] += 1
    if rng.random() < P["inst_hsv"]:
        # Verzinkter Stahl ist nahezu neutralgrau, nimmt aber Farbstiche an
        # (Flugrost braun, Moos gruen, blauer Himmel als Reflex).
        hsv = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + _u(rng, -8, 8)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * _u(rng, 0.45, 1.60), 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        STATS["inst_hsv"] += 1
    return bgr, a


def harmonize(bgr, a, bg_patch, rng):
    """Teilweiser Abgleich an die lokale Hintergrundstatistik.

    Zweck: Cutout und Hintergrund stammen aus verschiedenen Aufnahmen mit
    verschiedenem Weissabgleich und Belichtungsniveau. Ohne Angleich bleibt
    ein konstanter Farb-/Helligkeitsversatz zurueck, den ein CNN trivial als
    "hier wurde etwas eingeklebt" lernen kann.

    Bewusst NUR teilweise (alpha <= 0.45 fuer den Mittelwert, <= 0.35 fuer die
    Streuung): eine vollstaendige Angleichung wuerde den intrinsischen
    Albedo-Unterschied Bohrung/Kranz einebnen und damit das Objekt zerstoeren.
    """
    m = a > 0.5
    if m.sum() < 12 or bg_patch.size == 0:
        return bgr
    am = _u(rng, 0.10, 0.45)
    asd = _u(rng, 0.00, 0.35)
    for c in range(3):
        mi, si = bgr[..., c][m].mean(), bgr[..., c][m].std() + 1e-3
        mb, sb = bg_patch[..., c].mean(), bg_patch[..., c].std() + 1e-3
        k = 1.0 + asd * (np.clip(sb / si, 0.6, 1.6) - 1.0)
        bgr[..., c] = (bgr[..., c] - mi) * k + mi + am * (mb - mi)
    STATS["inst_harmonize"] += 1
    return bgr


def noise_sigma(patch):
    """Schaetzt das Rauschniveau eines Bildausschnitts (Hochpass-Streuung)."""
    if patch.size < 64:
        return 2.0
    g = cv2.cvtColor(np.clip(patch, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    hp = g.astype(np.float32) - cv2.medianBlur(g, 3).astype(np.float32)
    return float(np.std(hp))


def embed_realism(bgr, a, bg_patch, rng):
    """Unschaerfe- und Rauschangleich - die beiden Sensor-Signaturen, an denen
    eingeklebte Pixel sonst am leichtesten erkennbar sind."""
    if rng.random() < P["inst_blur"]:
        # Die Cutouts wurden beim Skalieren interpoliert und sind dadurch
        # glatter oder schaerfer als das Zielbild; zusaetzlich liegen reale
        # Objekte je nach Stapeltiefe ausserhalb der Schaerfeebene.
        s = _u(rng, 0.25, 1.10)
        bgr = cv2.GaussianBlur(bgr, (0, 0), s)
        a = cv2.GaussianBlur(a, (0, 0), s * 0.7)
        STATS["inst_blur"] += 1
    if rng.random() < P["inst_noise"]:
        # Beim Verkleinern eines Cutouts sinkt dessen Rauschen; der eingeklebte
        # Bereich waere dann messbar "sauberer" als seine Umgebung.
        s = noise_sigma(bg_patch) * _u(rng, 0.5, 1.25)
        bgr = bgr + rng.normal(0, max(0.5, s), bgr.shape)
        STATS["inst_noise"] += 1
    # Alphakante leicht weichzeichnen: ein 25-px-Objekt hat in einer
    # echten Aufnahme nie eine pixelharte Silhouette. Im Arm C1 aus, weil
    # die harte Silhouette gerade Signatur S1 ist.
    if P["alpha_feather"] > 0:
        a = cv2.GaussianBlur(a, (0, 0), _u(rng, 0.4, 1.0))
    return bgr, np.clip(a, 0, 1)


def cast_shadow(canvas, a, x0, y0, rng):
    """Weicher Schlagschatten unter der Instanz.

    Ohne Kontaktschatten "schwebt" ein eingeklebtes Objekt sichtbar. Der
    Schatten ist derselbe Hinweis, den auch ein menschlicher Betrachter
    zur Erkennung einer Montage nutzt.
    """
    s = cv2.GaussianBlur(a, (0, 0), max(1.0, 0.10 * max(a.shape)))
    k = _u(rng, 0.18, 0.45)
    dx = int(round(_u(rng, -0.22, 0.22) * a.shape[1]))
    dy = int(round(_u(rng, 0.02, 0.26) * a.shape[0]))
    H, W = canvas.shape[:2]
    h, w = s.shape
    X, Y = x0 + dx, y0 + dy
    sx0, sy0 = max(0, -X), max(0, -Y)
    dx0, dy0 = max(0, X), max(0, Y)
    dx1, dy1 = min(W, X + w), min(H, Y + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sm = s[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)][..., None]
    canvas[dy0:dy1, dx0:dx1] *= (1.0 - k * sm)
    STATS["inst_shadow"] += 1


def paste_stub(canvas, bars, bar, cxy, d, psi_deg, rng):
    """Setzt den Rohrschaft HINTER eine Stirnflaeche.

    PHYSIKALISCHE RANDBEDINGUNG (aus der Sichtung der echten Bilder):
    Eine Stirnflaeche existiert nie isoliert - sie ist das Ende eines Rohres,
    dessen Schaft sichtbar von ihr wegfuehrt. In val/test gilt das ausnahmslos
    fuer jedes annotierte Objekt. Ohne diesen Schaft entstuenden dunkle Kreise
    mitten auf durchgehenden Rohren: physikalisch unmoeglich und ein falscher
    Kontextprior.

    Der Schaft hat exakt den Durchmesser der Stirnflaeche (dasselbe Rohr) und
    wird VOR dem Profil gemalt, liegt also dahinter. Nebeneffekt: Schaefte
    benachbarter Rohre kreuzen andere Stirnflaechen und erzeugen damit genau
    die gegenseitige Verdeckung, die in den echten Bildern dominiert.
    """
    bw = max(3, int(round(d * _u(rng, 0.96, 1.06))))
    # Lange Schaefte: in den echten Bildern laeuft praktisch jedes Rohr aus dem
    # Ausschnitt heraus oder verschwindet hinter anderen Rohren. Kurze Stummel
    # mit sichtbarer runder Kappe waeren ein verraeterisches Muster.
    bl = max(bw + 2, int(round(d * _u(rng, 2.0, 24.0))))
    b = cv2.resize(bar, (bl, bw), interpolation=cv2.INTER_AREA)
    bb = b[:, :, :3].astype(np.float32) * _u(rng, 0.88, 1.12)
    ba = b[:, :, 3].astype(np.float32) / 255.0

    ang = np.deg2rad(psi_deg)
    ca, sa = np.cos(-ang), np.sin(-ang)          # Bildkoordinaten: y nach unten
    piv = np.array([bw / 2.0, bw / 2.0])         # Mitte der linken Kappe
    corners = np.array([[0, 0], [bl, 0], [bl, bw], [0, bw]], float) - piv
    R = np.array([[ca, -sa], [sa, ca]])
    rc = corners @ R.T
    mn, mx = rc.min(0), rc.max(0)
    ow = int(np.ceil(mx[0] - mn[0])) + 2
    oh = int(np.ceil(mx[1] - mn[1])) + 2
    M = cv2.getRotationMatrix2D((float(piv[0]), float(piv[1])), psi_deg, 1.0)
    M[0, 2] += (-mn[0] + 1) - piv[0]
    M[1, 2] += (-mn[1] + 1) - piv[1]
    bb = cv2.warpAffine(bb, M, (ow, oh), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    ba = cv2.warpAffine(ba, M, (ow, oh), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ba = cv2.GaussianBlur(np.clip(ba, 0, 1), (0, 0), _u(rng, 0.4, 0.9))
    x0 = int(round(cxy[0] + mn[0] - 1))
    y0 = int(round(cxy[1] + mn[1] - 1))

    # Helligkeitsangleich an die Umgebung. Die Stangenbibliothek stammt aus vier
    # verschiedenen Aufnahmen mit stark unterschiedlichem Belichtungsniveau;
    # ohne Angleich klebten sichtbar zu dunkle oder zu helle Rohre im Bild.
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(CANVAS, x0 + ow), min(CANVAS, y0 + oh)
    if cx1 > cx0 and cy1 > cy0:
        loc = canvas[cy0:cy1, cx0:cx1]
        m = ba[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] > 0.5
        if m.sum() > 10:
            k = _u(rng, 0.35, 0.75)
            for c in range(3):
                mi = bb[..., c][ba > 0.5].mean()
                mb = loc[..., c].mean()
                bb[..., c] += k * (mb - mi)
    STATS["inst_stub"] += 1
    return alpha_paste(canvas, bb, ba, x0, y0)


def recess(canvas, ctr, d, rng):
    """Abgedunkelte, weiche Mulde unter einem Cluster.

    In den echten Aufnahmen liegen die Stirnflaechen am Stapelrand vor dem
    verschatteten Stapelinneren; der Hintergrund hinter ihnen ist deutlich
    dunkler als die besonnten Rohrschaefte davor. Ohne diese Abdunklung
    stuenden die dunklen Bohrungen kontrastmaessig zu isoliert da.
    """
    R = d * _u(rng, 2.5, 7.0)
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS].astype(np.float32)
    m = np.exp(-(((xx - ctr[0]) / R) ** 2 + ((yy - ctr[1]) / (R * _u(rng, .5, 1.6))) ** 2))
    canvas *= (1.0 - _u(rng, 0.10, 0.38) * m)[..., None]
    STATS["recess"] += 1


# ----------------------------------------------------------------------------
# T4 - Layout: Cluster-Modell aus den echten Labels abgeleitet
# ----------------------------------------------------------------------------
def cluster_positions(rng, d0):
    """Erzeugt Objektzentren eines Stapelendes.

    Empirie aus val+test (n=174):
      - Abstand zum naechsten Nachbarn (in Objektdurchmessern):
          val  p10 0.86 | Median 1.09 | p90 1.78
          test p10 0.83 | Median 1.24 | p90 2.51
      - Median 10 Nachbarn innerhalb von 5 Durchmessern -> sehr dichte Cluster
      - 46-61 % aller Objekte beruehren ihren naechsten Nachbarn (< 1.2 d)
      - Nachbarrichtungen quasi isotrop mit leichter Haeufung um 90-150 Grad
        (die Stirnflaechen bilden geneigte Saeulen am Stapelrand)
      - Groessenverhaeltnis benachbarter Objekte: Median 1.05, p90 1.15
        -> innerhalb eines Clusters gleiche Nennweite, nur leichte Streuung
    Modell: gescherte, verrauschte Gitterpackung.
    """
    phi = np.deg2rad(_u(rng, 60, 150) if rng.random() < 0.6 else _u(rng, 0, 180))
    psi = phi + np.deg2rad(_u(rng, 55, 125)) * (1 if rng.random() < .5 else -1)
    s1 = d0 * _u(rng, 0.98, 1.60)
    s2 = d0 * _u(rng, 0.98, 1.85)
    n1 = int(rng.integers(2, 14))
    n2 = int(rng.integers(1, 7))
    jit = 0.17 * d0
    e1 = np.array([np.cos(phi), np.sin(phi)])
    e2 = np.array([np.cos(psi), np.sin(psi)])
    pts = []
    for i in range(n1):
        for j in range(n2):
            p = i * s1 * e1 + j * s2 * e2
            p = p + rng.normal(0, jit, 2)
            pts.append(p)
    pts = np.array(pts)
    # Cluster-Zentrum darf teilweise ausserhalb der Leinwand liegen ->
    # angeschnittene Objekte am Bildrand (in test 5.8 % der Labels).
    ctr = rng.uniform(-0.12, 1.12, 2) * CANVAS
    # Schaftrichtung des Clusters: alle Rohre eines Stapels liegen parallel.
    psi = float(np.rad2deg(psi_axis(phi, rng)))
    return pts + ctr, ctr, psi


def psi_axis(phi, rng):
    """Rohrschaft-Richtung: ungefaehr senkrecht zur Saeulenrichtung des
    Clusters (die Stirnflaechen bilden eine Front, die Schaefte fuehren
    dahinter weg), mit breiter Streuung."""
    return phi + np.deg2rad(_u(rng, 55, 125)) * (1 if rng.random() < .5 else -1)


# ----------------------------------------------------------------------------
# T5 - Globale Bildstufe
# ----------------------------------------------------------------------------
def global_stage(img, rng):
    if rng.random() < P["glob_bc"]:
        img = img * _u(rng, 0.80, 1.22) + _u(rng, -18, 18)
        STATS["glob_bc"] += 1
    if rng.random() < P["glob_hsv"]:
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + _u(rng, -6, 6)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * _u(rng, 0.6, 1.4), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * _u(rng, 0.85, 1.15), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        STATS["glob_hsv"] += 1
    if rng.random() < P["glob_blur"]:
        img = cv2.GaussianBlur(img, (0, 0), _u(rng, 0.3, 1.0))
        STATS["glob_blur"] += 1
    if rng.random() < P["glob_motion"]:
        # Verwacklung aus der Hand - im Bestand haeufig, da die Aufnahmen
        # auf der Baustelle im Vorbeigehen entstehen.
        k = int(rng.integers(3, 9))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        Mr = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), _u(rng, 0, 180), 1)
        ker = cv2.warpAffine(ker, Mr, (k, k))
        ker /= ker.sum() + 1e-8
        img = cv2.filter2D(img, -1, ker)
        STATS["glob_motion"] += 1
    if rng.random() < P["glob_noise"]:
        img = img + rng.normal(0, _u(rng, 1.0, 5.0), img.shape)
        STATS["glob_noise"] += 1
    if rng.random() < P["glob_vignette"]:
        yy, xx = np.mgrid[0:CANVAS, 0:CANVAS].astype(np.float32)
        r = np.sqrt((xx / CANVAS - .5) ** 2 + (yy / CANVAS - .5) ** 2)
        img = img * (1.0 - _u(rng, 0.10, 0.35) * (r / 0.707) ** 2)[..., None]
        STATS["glob_vignette"] += 1
    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Hauptgenerator
# ----------------------------------------------------------------------------
def generate_one(idx, rng, profiles, bars, bgs, snap=None):
    # snap: optionaler Rueckruf snap(stufe, bild, boxen) fuer
    # paper/tools/fig_stufen.py. Ohne snap ist das Verhalten unveraendert.
    def _s(tag, im=None, bx=None):
        if snap is not None:
            snap(tag, canvas if im is None else im, bx)

    canvas, d_base, bg = make_background(rng, bgs)
    _s("T1")

    if rng.random() < P["negative_image"]:
        # Negativbild: Rohrstapel ohne Stirnflaechen. Die haeufigste
        # Fehlerquelle sind Rohrschaefte, Gitterknoten und dunkle Loecher im
        # Maschendraht; explizite Negative bestrafen genau diese direkt.
        STATS["negative_image"] += 1
        img = global_stage(canvas, rng)
        _s("T9", img, [])
        return img, [], jpeg_quality(rng)

    layers = []   # (alpha, x0, y0) in Malreihenfolge - Basis der Verdeckung
    inst = []     # Instanzen mit Index in layers

    # ---- Objektpositionen ----------------------------------------------
    n_cluster = int(rng.choice([1, 2, 3], p=[.45, .35, .20]))
    positions, diams, psis = [], [], []
    for _ in range(n_cluster):
        d0 = float(np.clip(d_base * _u(rng, 0.85, 1.15), D_MIN, D_MAX))
        pts, ctr, psi = cluster_positions(rng, d0)
        if rng.random() < P["cluster_recess"]:
            recess(canvas, ctr, d0, rng)
        for p in pts:
            positions.append(p)
            diams.append(d0 * _u(rng, 0.92, 1.09))   # Messung: Median 1.05
            # Schaefte eines Stapels laufen parallel; nur +-11 Grad Streuung,
            # weil die Felder im Stapel lediglich leicht verkanten. Groessere
            # Streuung erzeugt einen sternfoermigen Faecher, den es real nicht
            # gibt.
            psis.append(psi + _u(rng, -11, 11))
    for _ in range(int(rng.integers(0, 4))):         # vereinzelte Profile
        positions.append(rng.uniform(0, 1, 2) * CANVAS)
        diams.append(float(np.clip(d_base * _u(rng, 0.8, 1.2), D_MIN, D_MAX)))
        psis.append(_u(rng, 0, 360))
    _s("T4", None, [(p, d) for p, d in zip(positions, diams)])

    order = rng.permutation(len(positions))          # zufaellige Tiefenordnung

    # TIEFENMODELL
    # Die Stirnflaechen eines Stapels bilden eine Front; die zugehoerigen
    # Schaefte laufen von dieser Front WEG, also nach hinten. In der Projektion
    # liegen sie damit hinter allen Stirnflaechen. Nur Rohre, die quer vor dem
    # Stapel liegen (in val/test klar erkennbar, aber die Minderheit), kreuzen
    # vor anderen Enden.
    # Werden alle Schaefte in zufaelliger Tiefenordnung gemalt, verdecken sie
    # so viele Nachbarn, dass deren Labels wegfallen - die erzeugten Cluster
    # werden dann messbar duenner als die echten (NN-Abstand 1.55 statt
    # 1.09-1.24 Durchmesser). Deshalb: 75 % der Schaefte zuerst (ganz hinten),
    # 25 % in die Tiefenordnung eingereiht.
    behind_all = rng.random(len(positions)) < P["stub_behind_front"]
    if bars:
        for k in order:
            if not behind_all[k]:
                continue
            cxy, d = positions[k], diams[k]
            marg = 0.8 * d
            if not (-marg < cxy[0] < CANVAS + marg
                    and -marg < cxy[1] < CANVAS + marg):
                continue
            if rng.random() < P["inst_stub"]:
                r = paste_stub(canvas, bars,
                               bars[int(rng.integers(0, len(bars)))],
                               cxy, d, psis[k], rng)
                if r:
                    layers.append(r)
    _s("T5")

    for k in order:
        cxy, d = positions[k], diams[k]
        # Nur Positionen rendern, die die Leinwand tatsaechlich schneiden
        # koennen (Rand = halbe Diagonale nach Rotation).
        marg = 0.8 * d
        if not (-marg < cxy[0] < CANVAS + marg
                and -marg < cxy[1] < CANVAS + marg):
            continue
        prof = profiles[int(rng.integers(0, len(profiles)))]
        bgr, a = render_instance(prof, d, rng)
        x0 = int(round(cxy[0] - a.shape[1] / 2))
        y0 = int(round(cxy[1] - a.shape[0] / 2))

        px0, py0 = max(0, x0), max(0, y0)
        px1 = min(CANVAS, x0 + a.shape[1])
        py1 = min(CANVAS, y0 + a.shape[0])
        if px1 <= px0 or py1 <= py0:
            continue
        # --- Rohrschaft hinter das Profil (physikalische Randbedingung) ---
        if bars and not behind_all[k] and rng.random() < P["inst_stub"]:
            r = paste_stub(canvas, bars, bars[int(rng.integers(0, len(bars)))],
                           cxy, d, psis[k], rng)
            if r:
                layers.append(r)

        bg_patch = canvas[py0:py1, px0:px1].copy()

        bgr, a = photometric(bgr, a, rng)
        if rng.random() < P["inst_harmonize"]:
            bgr = harmonize(bgr, a, bg_patch, rng)
        bgr, a = embed_realism(bgr, a, bg_patch, rng)
        if rng.random() < P["inst_shadow"]:
            cast_shadow(canvas, a, x0, y0, rng)

        res = alpha_paste(canvas, np.clip(bgr, 0, 255), a, x0, y0)
        if res is None:
            continue
        layers.append(res)
        inst.append(dict(layer=len(layers) - 1, a_full=float(a.sum())))
    _s("T2T3")

    # ---- T6 Okklusion durch echte Rohrstangen ---------------------------
    if bars and rng.random() < P["img_has_bars"]:
        for _ in range(int(rng.integers(1, 7))):
            bar = bars[int(rng.integers(0, len(bars)))]
            bw = max(3, int(round(d_base * _u(rng, 0.55, 1.35))))
            bl = int(round(bw * _u(rng, 4, 26)))
            b = cv2.resize(bar, (bl, bw), interpolation=cv2.INTER_AREA)
            bb, ba = b[:, :, :3].astype(np.float32), b[:, :, 3].astype(np.float32) / 255
            ang = _u(rng, 0, 180)
            diag = int(np.hypot(bl, bw)) + 2
            M = cv2.getRotationMatrix2D((bl / 2, bw / 2), ang, 1)
            M[0, 2] += (diag - bl) / 2
            M[1, 2] += (diag - bw) / 2
            bb = cv2.warpAffine(bb, M, (diag, diag), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            ba = cv2.warpAffine(ba, M, (diag, diag), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            bb = bb * _u(rng, 0.8, 1.2)
            ba = cv2.GaussianBlur(np.clip(ba, 0, 1), (0, 0), _u(rng, 0.4, 1.0))
            # Stangen bevorzugt durch die Cluster legen, nicht ins Leere
            if inst and rng.random() < 0.8:
                j = int(rng.integers(0, len(positions)))
                c = positions[j] + rng.normal(0, 0.5 * d_base, 2)
            else:
                c = rng.uniform(0, 1, 2) * CANVAS
            x0 = int(c[0] - diag / 2)
            y0 = int(c[1] - diag / 2)
            res = alpha_paste(canvas, bb, ba, x0, y0)
            if res:
                layers.append(res)
                STATS["bar_pasted"] += 1
        STATS["img_has_bars"] += 1
    _s("T6a")

    # ---- T7 Fremdkoerper (Laub, Draht, Schnee) --------------------------
    if rng.random() < P["img_clutter"]:
        src = bgs[int(rng.integers(0, len(bgs)))]["img"]
        for _ in range(int(rng.integers(1, 5))):
            ch = int(rng.integers(int(0.4 * d_base) + 3, int(3 * d_base) + 8))
            cw = int(rng.integers(int(0.4 * d_base) + 3, int(3 * d_base) + 8))
            sy = int(rng.integers(0, max(1, src.shape[0] - ch)))
            sx = int(rng.integers(0, max(1, src.shape[1] - cw)))
            patch = src[sy:sy + ch, sx:sx + cw].astype(np.float32)
            if patch.shape[0] < 3 or patch.shape[1] < 3:
                continue
            # unregelmaessige, weiche Maske aus geglaettetem Rauschen
            n = rng.random((patch.shape[0], patch.shape[1])).astype(np.float32)
            n = cv2.GaussianBlur(n, (0, 0), max(1.0, 0.18 * max(patch.shape[:2])))
            n = (n - n.min()) / (np.ptp(n) + 1e-6)
            m = np.clip((n - _u(rng, 0.45, 0.65)) * 6, 0, 1)
            x0 = int(rng.integers(0, CANVAS))
            y0 = int(rng.integers(0, CANVAS))
            res = alpha_paste(canvas, patch, m, x0, y0)
            if res:
                layers.append(res)
        STATS["img_clutter"] += 1
    _s("T6b")

    # ---- T8 Sichtbarkeitsrechnung und Labels ----------------------------
    boxes = []
    diag = []          # nur fuer snap: (box, sichtbarkeit, Urteil) je Instanz
    for it in inst:
        li = it["layer"]
        a_i, ix, iy = layers[li]
        vis = a_i.copy()
        for lj in range(li + 1, len(layers)):
            a_j, jx, jy = layers[lj]
            ox0 = max(ix, jx)
            oy0 = max(iy, jy)
            ox1 = min(ix + a_i.shape[1], jx + a_j.shape[1])
            oy1 = min(iy + a_i.shape[0], jy + a_j.shape[0])
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            vis[oy0 - iy:oy1 - iy, ox0 - ix:ox1 - ix] *= \
                (1.0 - a_j[oy0 - jy:oy1 - jy, ox0 - jx:ox1 - jx])
        a_in = float(a_i.sum())                       # Flaeche innerhalb des Bildes
        if a_in < 1e-6 or it["a_full"] < 1e-6:
            continue
        if snap is not None:
            _ys, _xs = np.where(a_i > 0.5)
            _box = ([int(_xs.min() + ix), int(_ys.min() + iy),
                     int(_xs.max() + ix) + 1, int(_ys.max() + iy) + 1]
                    if len(_xs) else None)
        in_frame = a_in / it["a_full"]
        v = float(vis.sum()) / a_in
        STATS["_vis_sum"] += v
        STATS["_vis_n"] += 1
        if v < 0.30:
            if snap is not None:
                diag.append((_box, v, "verdeckt"))
            STATS["_visdrop"] += 1
            # Unter 30 % sichtbarer Flaeche waere das Objekt auch fuer einen
            # menschlichen Annotator nicht mehr sicher als Rohrende
            # identifizierbar; ein Label hier zwingt das Netz zum Halluzinieren.
            STATS["dropped_occluded"] += 1
            continue
        if in_frame < 0.15:
            if snap is not None:
                diag.append((_box, v, "angeschnitten"))
            STATS["dropped_offscreen"] += 1
            continue
        ys, xs = np.where(vis > 0.5)
        if len(xs) < 6:
            if snap is not None:
                diag.append((_box, v, "zu klein"))
            STATS["dropped_tiny"] += 1
            continue
        # Modale Box: der SICHTBARE Umriss, genau wie in val/test annotiert.
        bx0, bx1 = xs.min() + ix, xs.max() + ix + 1
        by0, by1 = ys.min() + iy, ys.max() + iy + 1
        if (bx1 - bx0) < 5 or (by1 - by0) < 5:
            if snap is not None:
                diag.append((_box, v, "zu klein"))
            STATS["dropped_tiny"] += 1
            continue
        if snap is not None:
            diag.append(([bx0, by0, bx1, by1], v, "gelabelt"))
        boxes.append([bx0, by0, bx1, by1])
        STATS["_viskept_sum"] += v
        STATS["_viskept_n"] += 1
        if v < 0.95:
            STATS["inst_occluded"] += 1
        STATS["kept"] += 1
    _s("T7", None, diag)

    img = global_stage(canvas, rng)
    _s("T8", img, list(boxes))
    return img, boxes, jpeg_quality(rng)


def jpeg_quality(rng):
    """JPEG-Requantisierung.

    Wichtigste "unsichtbare" Transformation: eingeklebte Pixel behalten sonst
    eine andere DCT-Blockstatistik als der Hintergrund (der bereits einmal
    JPEG-komprimiert war). Ein erneuter Kompressionsdurchlauf ueber das
    Gesamtbild vereinheitlicht diese Signatur.
    """
    return int(rng.integers(55, 96)) if rng.random() < P["glob_jpeg"] else 95


_W = {}


def _init_worker(args):
    """Assets einmal pro Worker-Prozess laden (statt pro Bild)."""
    _W["assets"] = load_assets()
    _W["args"] = args
    # Ablationsarme (Experiment 2). Der Rohrschaft selbst bleibt in ALLEN
    # Armen erhalten - er ist physikalische Randbedingung, keine Augmentierung.
    # Variiert wird nur, wie stark verdeckt wird.
    if args.no_occlusion:
        P["img_has_bars"] = 0.0        # keine querliegenden Stangen
        P["img_clutter"] = 0.0         # keine Fremdkoerper
        P["stub_behind_front"] = 1.0   # kein Schaft kreuzt vor einer Stirnflaeche
    if args.naive_paste:
        # Arm C1 -- naives Einkleben im Sinne der Copy-Paste-Literatur.
        # Abgeschaltet sind genau die Gegenmassnahmen zu den Signaturen
        # S1-S5 aus Tabelle "Betrachtete synthetische Signaturen"; Layout,
        # Instanzgeometrie, photometrische Variation, Rohrschaft und
        # Okkluder bleiben unveraendert, damit der Vergleich nur die
        # Einbettung betrifft und nicht die Szene.
        P["alpha_feather"] = 0.0    # S1 harte Silhouette
        P["inst_harmonize"] = 0.0   # S2 kein Farbangleich
        P["inst_blur"] = 0.0        # S3 kein Unschaerfeangleich
        P["inst_noise"] = 0.0       # S4 kein Rauschangleich
        P["glob_jpeg"] = 0.0        # S5 keine Requantisierung
        P["inst_shadow"] = 0.0      # S6 kein Schlagschatten
    if args.occlusion_heavy:
        P["img_has_bars"] = 1.0
        P["img_clutter"] = 0.70
        P["stub_behind_front"] = 0.40  # mehr quer davor liegende Rohre
    STATS.clear()


def _work(job):
    i, args = job
    profiles, bars, bgs = _W["assets"]
    STATS.clear()
    rng = np.random.default_rng(args.seed * 1_000_003 + i)
    img, boxes, q = generate_one(i, rng, profiles, bars, bgs)
    name = f"syn_{args.seed}_{i:06d}"
    cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, q])
    with open(os.path.join(args.out, "labels", name + ".txt"), "w") as f:
        for x0, y0, x1, y1 in boxes:
            cx = (x0 + x1) / 2 / CANVAS
            cy = (y0 + y1) / 2 / CANVAS
            w = (x1 - x0) / CANVAS
            h = (y1 - y0) / CANVAS
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    if i < args.preview:
        vis = img.copy()
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.imwrite(f"docs/preview_{os.path.basename(args.out)}_{i:03d}.jpg",
                    vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return len(boxes), dict(STATS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--procs", type=int, default=0,
                    help="Worker-Prozesse (0 = cpu_count-1)")
    ap.add_argument("--preview", type=int, default=0,
                    help="zusaetzlich N Kontrollbilder mit Boxen nach docs/")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="Ablation A0: keine Stangen-/Fremdkoerper-Okklusion")
    ap.add_argument("--naive-paste", action="store_true",
                    help="Arm C1: naives Einkleben, Gegenmassnahmen zu den "
                         "Signaturen S1-S5 abgeschaltet")
    ap.add_argument("--occlusion-heavy", action="store_true",
                    help="Ablation A2: maximale Stangen-/Fremdkoerper-Okklusion")
    args = ap.parse_args()

    profiles, bars, bgs = load_assets()
    imdir = os.path.join(args.out, "images")
    lbdir = os.path.join(args.out, "labels")
    os.makedirs(imdir, exist_ok=True)
    os.makedirs(lbdir, exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    jobs = [(i, args) for i in range(args.n)]
    chunk = max(1, args.n // (args.procs or (os.cpu_count() or 8)) // 4)
    nbox = []
    with Pool(processes=(args.procs or max(1, (os.cpu_count() or 8) - 1)),
              initializer=_init_worker, initargs=(args,)) as pool:
        for k, (nb, st) in enumerate(pool.imap_unordered(_work, jobs, chunk)):
            nbox.append(nb)
            for key, v in st.items():
                STATS[key] += v
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{args.n}  Boxen/Bild im Mittel {np.mean(nbox):.1f}",
                      flush=True)

    n = args.n
    rep = {
        "n_images": n,
        "arm": ("C1_naive_paste" if args.naive_paste else
                "A0_no_occlusion" if args.no_occlusion else
                "A2_heavy_occlusion" if args.occlusion_heavy else "A1_baseline"),
        "canvas": CANVAS,
        "boxes_total": int(np.sum(nbox)),
        "boxes_per_image_mean": float(np.mean(nbox)),
        "boxes_per_image_p05_p95": [float(np.percentile(nbox, 5)),
                                    float(np.percentile(nbox, 95))],
        "mean_visibility_all_rendered": (STATS["_vis_sum"] / max(1, STATS["_vis_n"])),
        "mean_visibility_labeled": (STATS["_viskept_sum"] / max(1, STATS["_viskept_n"])),
        "shares": {},
    }
    tot = max(1, STATS["inst_total"])
    kept = max(1, STATS["kept"])
    rep["instances_rendered"] = STATS["inst_total"]
    rep["instances_labeled"] = STATS["kept"]
    for k in ["inst_rotation", "inst_oval", "inst_dent", "inst_perspective",
              "inst_photometric", "inst_gamma", "inst_hsv", "inst_harmonize",
              "inst_blur", "inst_noise", "inst_shadow", "inst_stub"]:
        rep["shares"][k + " (je Instanz)"] = round(STATS[k] / tot, 4)
    rep["shares"]["inst_occluded (je gelabelter Instanz)"] = \
        round(STATS["inst_occluded"] / kept, 4)
    for k in ["negative_image", "img_has_bars", "img_clutter", "recess",
              "bg_fliplr",
              "bg_flipud", "bg_rot", "glob_bc", "glob_hsv", "glob_blur",
              "glob_motion", "glob_noise", "glob_vignette"]:
        rep["shares"][k + " (je Bild)"] = round(STATS[k] / n, 4)
    rep["dropped"] = {k: STATS[k] for k in
                      ["dropped_occluded", "dropped_offscreen", "dropped_tiny"]}
    with open(os.path.join(args.out, "generation_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
