#!/usr/bin/env python3
"""Eine Beispielgrafik je Stufe der Synthese-Pipeline.

Die Bilder entstehen im echten Generator: fuer die kumulativen Stufen ueber den
Rueckruf snap() in src/generate.py, fuer die Detailstufen durch direkten Aufruf
derselben Funktionen (render_instance, photometric, harmonize, embed_realism,
cast_shadow, global_stage).  Es gibt keine Nachbildung.

Aufruf (aus dem Projektwurzelverzeichnis):
    .venv/bin/python paper/tools/fig_stufen.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
import generate as G                                      # noqa: E402

ZIEL = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                     "abbildungen"))
SEED = 102         # Clean Plate, grosses Cluster, sechs Stangen
VOLL = 760         # Zielbreite der Ganzbild-Panels
GRUEN, ROT, ORANGE = (0, 200, 0), (0, 0, 235), (0, 140, 240)


def speichern(name, img, breite=None, q=95):
    """Fotografische Panels als JPEG (Dateigroesse), Strichgrafik als PNG."""
    if breite and img.shape[1] != breite:
        s = breite / img.shape[1]
        ip = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=ip)
    pfad = os.path.join(ZIEL, name)
    par = ([cv2.IMWRITE_JPEG_QUALITY, q] if name.endswith(".jpg")
           else [cv2.IMWRITE_PNG_COMPRESSION, 6])
    cv2.imwrite(pfad, np.clip(img, 0, 255).astype(np.uint8), par)
    kb = os.path.getsize(pfad) / 1024
    print(f"   {name}  {img.shape[1]}x{img.shape[0]}  {kb:.0f} kB")


def kacheln(bilder, rand=6):
    # Beschneiden ist Pflicht: float-Werte ausserhalb 0..255 wuerden beim
    # Einsetzen in ein uint8-Feld umlaufen und Farbsprenkel erzeugen.
    bilder = [np.clip(b, 0, 255).astype(np.uint8) for b in bilder]
    h = max(b.shape[0] for b in bilder)
    w = max(b.shape[1] for b in bilder)
    out = []
    for b in bilder:
        t = np.full((h, w, 3), 245, np.uint8)
        y0, x0 = (h - b.shape[0]) // 2, (w - b.shape[1]) // 2
        t[y0:y0 + b.shape[0], x0:x0 + b.shape[1]] = b
        out.append(t)
        out.append(np.full((h, rand, 3), 245, np.uint8))
    return np.hstack(out[:-1])


def auf_grund(bgr, a, grund):
    """Instanz mit Alphamaske auf einen Untergrund legen (nur Darstellung)."""
    t = grund[:bgr.shape[0], :bgr.shape[1]].astype(np.float32).copy()
    if t.shape[:2] != bgr.shape[:2]:
        t = cv2.resize(t, (bgr.shape[1], bgr.shape[0]))
    m = a[:, :, None]
    return (bgr.astype(np.float32) * m + t * (1 - m))


# ---------------------------------------------------------------------------
def kumulative_stufen(profiles, bars, bgs):
    """T1, T4, T5, T2/T3-Ganzbild, T6, T7 aus einem einzigen Durchlauf."""
    snaps = {}
    G.STATS.clear()
    rng = np.random.default_rng(SEED)
    img, boxes, _ = G.generate_one(
        0, rng, profiles, bars, bgs,
        snap=lambda tag, im, x: snaps.__setitem__(tag, (im.copy(), x)))
    st = dict(G.STATS)

    speichern("stufe_T1_hintergrund.jpg", snaps["T1"][0], VOLL)

    # T4: Layout -- Mulden liegen schon auf der Leinwand, die Positionen noch
    # nicht.  Kreise markieren die gezogenen Zentren mit ihrem Durchmesser.
    lay = snaps["T4"][0].copy()
    for (p, d) in snaps["T4"][1]:
        cv2.circle(lay, (int(p[0]), int(p[1])), max(2, int(d / 2)), ORANGE, 2)
    speichern("stufe_T4_layout.jpg", lay, VOLL)

    speichern("stufe_T5_schaft.jpg", snaps["T5"][0], VOLL)
    speichern("stufe_T2T3_instanzen.jpg", snaps["T2T3"][0], VOLL)

    # T6: was die Okklusionsstufe hinzugefuegt hat, per Differenzmaske
    vor, nach = snaps["T2T3"][0], snaps["T6b"][0]
    diff = (np.abs(nach.astype(np.int16) - vor.astype(np.int16)).sum(2) > 18)
    okk = nach.copy()
    kant = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8))
    okk[kant > 0] = ORANGE
    speichern("stufe_T6_okklusion.jpg", okk, VOLL)

    # T7: Urteil je Instanz
    sic = snaps["T2T3"][0].copy()
    n = {"gelabelt": 0, "verdeckt": 0, "angeschnitten": 0, "zu klein": 0}
    for box, v, urteil in snaps["T7"][1]:
        if box is None:
            continue
        n[urteil] += 1
        c = GRUEN if urteil == "gelabelt" else ROT
        cv2.rectangle(sic, (box[0], box[1]), (box[2], box[3]), c, 2)
    speichern("stufe_T7_sichtbarkeit.jpg", sic, VOLL)

    # T8: globale Bildstufe als Vorher/Nachher-Ausschnitt, auf den
    # Schwerpunkt der Labels gelegt
    if boxes:
        cx = int(np.mean([(b[0] + b[2]) / 2 for b in boxes]))
        cy = int(np.mean([(b[1] + b[3]) / 2 for b in boxes]))
    else:
        cx = cy = G.CANVAS // 2
    x0 = int(np.clip(cx - 240, 0, G.CANVAS - 480))
    y0 = int(np.clip(cy - 240, 0, G.CANVAS - 480))
    a = snaps["T7"][0][y0:y0 + 480, x0:x0 + 480]
    b = snaps["T8"][0][y0:y0 + 480, x0:x0 + 480]
    speichern("stufe_T8_global.jpg", kacheln([a, b]))
    return st, n, len(boxes)


def profil(profiles, name):
    return next(p for p in profiles if p["name"] == name)


# Ausgewaehlte Ziehungen fuer T2: neben der stets aktiven Rotation greift je
# nur eine Transformation, und zwar nahe am Rand ihres Bereichs.  Gefunden
# durch Mitschreiben der Zufallszahlen von render_instance ueber die Seeds
# 0..19999; zufaellige Seeds zeigen meist nur schwache Ovalisierung und fast
# nie eine Delle (P = 0,18).
T2_ZIEHUNGEN = (
    11710,   # nur Rotation, 180 Grad
    7755,    # Ovalisierung 0,66
    7131,    # Delle: ein Sektor, Breite 0,30 rad, Tiefe 0,30
    9514,    # Perspektive, Eckversatz bis 8,7 %
)


def detail_geometrie(profiles, bgs):
    """T2: dieselbe Quelle, fuenf Ziehungen der Instanzgeometrie."""
    # cutted_9 traegt ein Glanzlicht auf dem Kranz, an dem die Drehung
    # sichtbar wird; bei einem gleichmaessigen Kranz sieht man sie nicht.
    prof = profil(profiles, "cutted_9.png")
    src = prof["rgba"]
    D = 100          # Zieldurchmesser der Instanz in dieser Abbildung
    K = 190          # Kachelgroesse
    roh = auf_grund(src[:, :, :3].astype(np.float32),
                    src[:, :, 3].astype(np.float32) / 255,
                    np.full((*src.shape[:2], 3), 235, np.uint8))
    # Quellausschnitt auf denselben Objektdurchmesser bringen wie die
    # Ziehungen, sonst wirkt er groesser als sie.  Kubisch wie in
    # render_instance, sonst sieht die Quelle als einzige Kachel pixelig aus.
    sc = D / prof["d_src"]
    roh = cv2.resize(roh, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    q = np.full((K, K, 3), 235, np.float32)
    ry, rx = (K - roh.shape[0]) // 2, (K - roh.shape[1]) // 2
    q[max(0, ry):max(0, ry) + min(K, roh.shape[0]),
      max(0, rx):max(0, rx) + min(K, roh.shape[1])] = roh[:K, :K]
    tiles = [q]
    for seed in T2_ZIEHUNGEN:
        rng = np.random.default_rng(seed)
        bgr, a = G.render_instance(prof, D, rng)
        t = auf_grund(bgr, a, np.full((*a.shape, 3), 235, np.uint8))
        if max(t.shape[:2]) > K:
            sc = K / max(t.shape[:2])
            t = cv2.resize(t, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        pad = np.full((K, K, 3), 235, np.float32)
        y0, x0 = (K - t.shape[0]) // 2, (K - t.shape[1]) // 2
        pad[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        tiles.append(pad)
    speichern("stufe_T2_geometrie.png", kacheln(tiles))


def detail_photometrie(profiles, bgs):
    """T3: derselbe Instanzausschnitt in vier Bearbeitungsstufen.

    Wie bei T2 sind kraeftige Ziehungen ausgewaehlt, damit jeder Schritt
    sichtbar wird: Die Photometrie dunkelt ab (Helligkeit 0,72, Gamma 1,34),
    der Farbangleich zieht die dunkle Instanz zum hellen Clean Plate zurueck
    (alpha_mu 0,45, alpha_sigma 0,35), die Einbettung zeichnet weich
    (sigma 1,10) und rauscht (Faktor 0,90), der Schatten faellt nach unten
    (kappa 0,45).  Auf einem dunklen Hintergrund hebt der Farbangleich eine
    dunkle Instanz kaum an.
    """
    prof = profil(profiles, "cutted_7.png")
    rng = np.random.default_rng(77)
    bgr0, a0 = G.render_instance(prof, 130, rng)
    h, w = a0.shape
    ph, pw = h + 60, w + 60
    grund = next(b["img"] for b in bgs
                 if b["path"].endswith("img3884_cutted.png"))
    gy = min(620, grund.shape[0] - ph - 1)
    gx = min(420, grund.shape[1] - pw - 1)
    gp = grund[gy:gy + ph, gx:gx + pw].astype(np.float32)

    def tile(bgr, a, schatten=False):
        out = gp.copy()
        y0 = (ph - a.shape[0]) // 2
        x0 = (pw - a.shape[1]) // 2
        if schatten:
            G.cast_shadow(out, a, x0, y0, np.random.default_rng(1074))
        roi = out[y0:y0 + a.shape[0], x0:x0 + a.shape[1]]
        m = a[:, :, None]
        out[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = \
            bgr.astype(np.float32) * m + roi * (1 - m)
        return out

    t1 = tile(bgr0, a0)
    b2, a2 = G.photometric(bgr0.copy(), a0.copy(), np.random.default_rng(60))
    t2 = tile(b2, a2)
    b3 = G.harmonize(b2.copy(), a2, gp, np.random.default_rng(588))
    t3 = tile(b3, a2)
    b4, a4 = G.embed_realism(b3.copy(), a2.copy(), gp,
                             np.random.default_rng(2745))
    t4 = tile(b4, a4, schatten=True)
    speichern("stufe_T3_photometrie.png", kacheln([t1, t2, t3, t4]))


def negativbild(profiles, bars, bgs):
    """T9: Bild ohne jedes Objekt.

    Gesucht wird ein Negativbild auf einem Clean Plate: Ein Rohrstapel ohne
    Stirnflaechen ist das harte Negativ, um das es der Stufe geht -- eine
    Fremdszene ohne Rohre waere ein triviales.
    """
    orig, info = G.make_background, {}

    def wrapped(rng, bgs_):
        c, d, b = orig(rng, bgs_)
        info["clean"] = b["clean"]
        return c, d, b

    G.make_background = wrapped
    try:
        for seed in range(1, 600):
            G.STATS.clear()
            info.clear()
            got = []
            img, _, _ = G.generate_one(
                0, np.random.default_rng(seed), profiles, bars, bgs,
                snap=lambda t, i, x: got.append(t))
            if "T9" in got and info.get("clean"):
                print(f"   Negativbild auf Clean Plate bei Seed {seed}")
                speichern("stufe_T9_negativ.jpg", img, VOLL)
                return seed
    finally:
        G.make_background = orig
    raise RuntimeError("kein Negativbild auf einem Clean Plate gefunden")


def main():
    os.makedirs(ZIEL, exist_ok=True)
    profiles, bars, bgs = G.load_assets()
    print("kumulative Stufen:")
    st, n, nb = kumulative_stufen(profiles, bars, bgs)
    print("Detailstufen:")
    detail_geometrie(profiles, bgs)
    detail_photometrie(profiles, bgs)
    negativbild(profiles, bars, bgs)
    print(f"\nBeispielbild (Seed {SEED}): {st.get('inst_total', 0)} Instanzen "
          f"gerendert, {nb} gelabelt")
    print("   Urteile der Sichtbarkeitsstufe:", n)
    print(f"   Mulden {st.get('recess', 0)}, Schaefte {st.get('inst_stub', 0)}, "
          f"Stangen {st.get('bar_pasted', 0)}")


if __name__ == "__main__":
    main()
