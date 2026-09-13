"""
Stufe 2b: Prozedurale Okkluder-Bibliothek (Gegenstueck zu src/occluders.py).

WOZU
----
src/occluders.py begruendet die Entscheidung, Okkluder aus echten Aufnahmen zu
schneiden, ausschliesslich mit einem Argument:

    "Ein synthetisch gemalter Balken haette eine andere Textur-, Rausch- und
     Glanzstatistik als der Rest des Bildes; das Netz koennte 'Objekt liegt
     neben kuenstlichem Balken' lernen."

Das Argument war nie gemessen. Dieses Modul erzeugt die beiden Gegenstuecke,
mit denen es geprueft werden kann (Ablation B in src/experiment_neu.py):

    B1  "prozedural, physikalisch"  - Zylindermodell aus Lambert-Term,
        Glanzlicht, Umgebungslicht, Laengsvariation und Korn. Alle Parameter
        sind an der PHOTOGRAPHISCHEN Bibliothek assets/bars/ gemessen
        (siehe measure() unten), nicht geschaetzt.
    B2  "prozedural, naiv"          - konstantes Grau in derselben Kapselmaske.
        Kein Verlauf, kein Glanzlicht, kein Korn, keine Farbe. Das ist der
        "gezeichnete Balken", vor dem occluders.py warnt.

STRIKTE VERGLEICHBARKEIT
------------------------
Beide Bibliotheken uebernehmen von jeder der 56 echten Stangen
  * die exakte Pixelgroesse (h, w) und
  * die exakte Alphamaske (dieselbe capsule_alpha aus src/occluders.py),
sodass Geometrie, Kantenweichheit und die spaetere Skalierung im Generator
identisch sind. Der EINZIGE Unterschied zwischen den drei Bibliotheken sind
die RGB-Werte innerhalb der Maske.

Aufruf:
    python src/procedural_bars.py            # erzeugt beide Bibliotheken
    python src/procedural_bars.py --measure  # nur die Messung ausgeben
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from occluders import capsule_alpha          # noqa: E402  (identische Maske)

REAL = "assets/bars"
OUT_PHYS = "assets/bars_proc"
OUT_NAIV = "assets/bars_proc_naiv"
DOCS = "docs"


# ---------------------------------------------------------------------------
# Messung an der photographischen Bibliothek
# ---------------------------------------------------------------------------
def measure(paths):
    """Kennwerte, an denen das prozedurale Modell kalibriert wird.

    Gemessen wird nur, was das Modell auch erzeugen kann: mittlere Helligkeit,
    Querschnittsprofil (auf die mittlere Helligkeit normiert), Kornstaerke,
    Farbsaettigung und die Laengsvariation. Alles andere -- die eigentliche
    Mikrotextur des verzinkten Stahls -- ist genau das, was das prozedurale
    Modell NICHT hat; darum geht die Ablation.

    'kontrast' ist bewusst JE STANGE gemessen (p90/p10 des
    Querschnittsprofils) und dann gemittelt, nicht am gemittelten Profil: die
    Glanzlichtlage streut ueber die Bibliothek stark (argmax-Streuung 0.33 der
    Streifenhoehe), ein Mittelprofil waere dadurch kuenstlich flach.
    Perzentile statt Maximum/Minimum, weil eine einzelne dunkle Zeile am
    Streifenrand den Quotienten sonst dominiert.
    """
    prof, gray_mean, grain, sat, longvar = [], [], [], [], []
    contrast, peakpos, chan = [], [], []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        bgr = im[:, :, :3]
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        core = g[:, w // 4: 3 * w // 4]                 # Kapselenden ausblenden
        pr = np.interp(np.linspace(0, 1, 32), np.linspace(0, 1, h),
                       core.mean(axis=1))
        prof.append(pr)
        pn = pr / pr.mean()
        contrast.append(float(np.percentile(pn, 90) /
                             max(1e-3, np.percentile(pn, 10))))
        peakpos.append(float(np.argmax(pn) / (len(pn) - 1)))
        gray_mean.append(float(g.mean()))
        hp = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
        grain.append(float(hp.std()))
        sat.append(float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean()))
        cm = bgr.reshape(-1, 3).astype(np.float32).mean(axis=0)
        chan.append(cm / max(1e-6, cm.mean()))
        L = cv2.GaussianBlur(g.mean(axis=0).reshape(-1, 1), (1, 31), 0).ravel()
        longvar.append(float(L.std() / max(1.0, L.mean())))
    prof = np.array(prof)
    chan = np.array(chan)
    pn = prof / prof.mean(axis=1, keepdims=True)
    return dict(
        n=len(paths),
        gray_mean=[float(np.mean(gray_mean)), float(np.std(gray_mean))],
        gray_range=[float(np.min(gray_mean)), float(np.max(gray_mean))],
        grain_sigma=[float(np.percentile(grain, 10)), float(np.mean(grain)),
                     float(np.percentile(grain, 90))],
        saturation_mean=float(np.mean(sat)),
        longitudinal_rel_std=[float(np.mean(longvar)),
                              float(np.percentile(longvar, 90))],
        kontrast_je_stange=[float(np.percentile(contrast, 10)),
                            float(np.mean(contrast)),
                            float(np.percentile(contrast, 90))],
        kanalverhaeltnis_bgr=[[float(v) for v in np.mean(chan, axis=0)],
                              [float(v) for v in np.std(chan, axis=0)]],
        glanzlichtlage=[float(np.mean(peakpos)), float(np.std(peakpos))],
        profile_norm_mean=[round(float(v), 4) for v in pn.mean(axis=0)],
    )


# ---------------------------------------------------------------------------
# Prozedurales Zylindermodell
# ---------------------------------------------------------------------------
def cylinder_bar(h, w, rng, M, gain=None):
    """Ein Rohrschaft aus einem physikalischen Beleuchtungsmodell.

    Querkoordinate u in [-1, 1] ueber die Streifenhoehe. Der Streifen zeigt
    einen ARC des Zylindermantels, nicht den vollen Halbkreis: benachbarte
    Rohre im Stapel verdecken sich gegenseitig, bevor der Blick den
    Streifwinkel erreicht. Modelliert als

        phi(u) = phi_c + u * dphi ,   dphi in [0.85, 1.30] rad (49-75 Grad)

    mit einem Versatz phi_c, weil die Achsdetektion in src/occluders.py den
    Streifen nicht exakt auf die Rohrachse zentriert -- gemessen streut die
    Glanzlichtlage in der echten Bibliothek um 0.33 der Streifenhoehe.

    Auf diesem Arc:
      Lambert    max(0, cos(phi - phi_L))          diffuse Reflexion
      Glanzlicht exp(-((phi - phi_L/2)/sigma)^2)   Spiegelrichtung, Betrachter
                                                   frontal
      Umgebung   k_amb                             Himmelslicht im Stapel
      Randabfall (1 - k_e u^2)                     Schattenspalt zum Nachbarrohr

    'gain' skaliert Querschnittskontrast, Farbsaettigung und Korn. Die drei
    Faktoren werden in calibrate() automatisch so bestimmt, dass die erzeugte
    Bibliothek die an assets/bars/ GEMESSENEN Kennwerte trifft; sie sind
    Kalibrierung, keine freien Parameter.
    """
    gain = gain or dict(kontrast=1.0, saettigung=1.0, korn=1.0)
    u = (np.arange(h, dtype=np.float32) + 0.5) / h * 2.0 - 1.0

    dphi = float(rng.uniform(0.85, 1.30))              # halber sichtbarer Arc
    phi_c = float(rng.uniform(-0.85, 0.85))            # Versatz des Streifens
    phi = phi_c + u * dphi

    phi_L = float(rng.uniform(-1.0, 1.0))              # Lichteinfall
    k_amb = float(rng.uniform(0.20, 0.75))             # Umgebungsanteil
    k_dif = 1.0 - k_amb
    s_amp = float(rng.uniform(0.05, 0.80))             # Glanzlichtstaerke
    s_sig = float(rng.uniform(0.15, 0.45))             # Glanzlichtbreite
    k_edge = float(rng.uniform(0.05, 0.25))            # Schattenspalt zum Nachbar

    lam = np.maximum(0.0, np.cos(phi - phi_L))
    spec = np.exp(-((phi - 0.5 * phi_L) / s_sig) ** 2)
    shade = (k_amb + k_dif * lam + s_amp * spec) * (1.0 - k_edge * u ** 2)
    shade = shade / shade.mean()
    shade = np.maximum(0.02, 1.0 + gain["kontrast"] * (shade - 1.0))

    # Laengsvariation: geglaettetes 1-D-Rauschen, relative Streuung wie gemessen
    rel = float(rng.uniform(0.5, 1.5)) * M["longitudinal_rel_std"][0]
    n1 = rng.standard_normal(w).astype(np.float32)
    n1 = cv2.GaussianBlur(n1.reshape(-1, 1), (1, 0),
                          float(rng.uniform(0.03, 0.12)) * w).ravel()
    n1 = n1 / (n1.std() + 1e-6)
    lon = 1.0 + rel * n1

    field = shade[:, None] * lon[None, :]

    # Zielhelligkeit aus der gemessenen Verteilung der echten Stangen
    lo, hi = M["gray_range"]
    mu = float(np.clip(rng.normal(*M["gray_mean"]), lo, hi))
    gray = field * mu / max(1e-6, field.mean())

    # Farbstich der GANZEN Stange. Nicht frei gewuerfelt, sondern aus dem
    # gemessenen Kanalverhaeltnis der echten Bibliothek gezogen: verzinkter
    # Stahl unter bedecktem Himmel ist fast neutral mit leichtem Blaustich
    # (B/G/R = 1.026 / 1.007 / 0.967, Streuung 0.041 / 0.016 / 0.034).
    # Ein gleichverteilter Farbton haette die Stangen rosa und gruen gemacht -
    # ein Unterschied, den die Ablation nicht messen soll.
    cm, cs = M["kanalverhaeltnis_bgr"]
    tint = np.array([rng.normal(m, s) for m, s in zip(cm, cs)], np.float32)
    bgr = gray[..., None] * tint[None, None, :]

    # Korn: Rauschstaerke aus der gemessenen Spanne der echten Stangen.
    # Ueberwiegend LUMINANZ-Rauschen, also ueber alle drei Kanaele identisch.
    # Grund: Bayer-Interpolation und die 4:2:0-Chroma-Unterabtastung von JPEG
    # tiefpassfiltern den Farbanteil. Kanalunabhaengiges Rauschen erzeugte
    # dagegen eine Farbsaettigung von ~55 statt der gemessenen 28.7 - das
    # Korn allein wuerde die Stange bunter machen als jede echte.
    g10, _, g90 = M["grain_sigma"]
    sig = max(0.05, float(rng.uniform(g10, g90)) * gain["korn"])
    lum = rng.normal(0.0, sig, (h, w, 1))
    # Der Chroma-Anteil traegt praktisch die gesamte gemessene Saettigung
    # (der globale Farbstich oben liegt bei nur ~3 %); sein Pegel ist deshalb
    # die Stellgroesse, mit der calibrate() die Saettigung trifft.
    chroma = rng.normal(0.0, max(0.02, gain["saettigung"] * sig), (h, w, 3))
    chroma = cv2.GaussianBlur(chroma, (0, 0), 1.2)
    bgr = bgr + lum + chroma
    return np.clip(bgr, 0, 255).astype(np.uint8)


def flat_bar(h, w, rng, M):
    """Der naive gezeichnete Balken: konstantes Grau, sonst nichts.

    Bewusst so arm wie moeglich - er ist die Kontrollbedingung fuer die Frage,
    wie viel von der Wirkung eines Okkluders ueberhaupt an seiner Textur haengt
    und wie viel allein an seiner Geometrie (er verdeckt Flaeche).
    """
    lo, hi = M["gray_range"]
    mu = float(np.clip(rng.normal(*M["gray_mean"]), lo, hi))
    return np.full((h, w, 3), mu, np.float32).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
def _measure_arrays(imgs):
    """measure() auf bereits im Speicher liegenden BGR-Bildern."""
    contrast, grain, sat = [], [], []
    for bgr in imgs:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        pr = g[:, w // 4: 3 * w // 4].mean(axis=1)
        pn = pr / pr.mean()
        contrast.append(np.percentile(pn, 90) /
                        max(1e-3, np.percentile(pn, 10)))
        hp = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
        grain.append(hp.std())
        sat.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean())
    return (float(np.mean(contrast)), float(np.mean(grain)), float(np.mean(sat)))


def calibrate(shapes, M, seed, rounds=12):
    """Drei Skalenfaktoren so bestimmen, dass die prozedurale Bibliothek die
    an der photographischen GEMESSENEN Kennwerte trifft.

    Warum ueberhaupt: der Umweg ueber ein Beleuchtungsmodell verzerrt die
    Kennwerte systematisch (der Lambert-Term erzeugt mehr Querschnittskontrast
    als real gemessen, der Hochpass-Schaetzer unterschaetzt weisses Rauschen
    gegenueber korrelierter Sensortextur). Ohne Angleich unterschieden sich die
    Bibliotheken nicht nur in der Mikrotextur -- dem Gegenstand der Ablation --
    sondern auch in Kontrast, Korn und Farbe, und das Ergebnis waere nicht
    zuzuordnen. Fixpunktiteration ueber drei entkoppelte, monotone Groessen;
    konvergiert in wenigen Runden.
    """
    tgt = (M["kontrast_je_stange"][1], M["grain_sigma"][1], M["saturation_mean"])
    gain = dict(kontrast=1.0, korn=1.0, saettigung=1.0)
    hist = []
    for _ in range(rounds):
        rng = np.random.default_rng(seed)
        imgs = [cylinder_bar(h, w, rng, M, gain) for h, w in shapes]
        cur = _measure_arrays(imgs)
        hist.append(dict(gain=dict(gain), kontrast=cur[0], korn=cur[1],
                         saettigung=cur[2]))
        err = max(abs(c / t - 1.0) for c, t in zip(cur, tgt))
        if err < 0.02:
            break
        # gedaempfte multiplikative Korrektur (Exponent 0.6 statt 1.0), weil
        # der Kontrast nichtlinear von seinem Faktor abhaengt
        for key, c, t in zip(("kontrast", "korn", "saettigung"), cur, tgt):
            gain[key] *= float((t / max(1e-6, c)) ** 0.6)
    return gain, hist


def build(seed=7):
    paths = sorted(glob.glob(os.path.join(REAL, "*.png")))
    if not paths:
        raise SystemExit(f"keine echten Stangen in {REAL} - erst "
                         f"src/occluders.py laufen lassen")
    M = measure(paths)
    os.makedirs(OUT_PHYS, exist_ok=True)
    os.makedirs(OUT_NAIV, exist_ok=True)
    os.makedirs(DOCS, exist_ok=True)

    reals = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in paths]
    shapes = [r.shape[:2] for r in reals]
    gain, hist = calibrate(shapes, M, seed)
    print("Kalibrierfaktoren:", {k: round(v, 4) for k, v in gain.items()})

    # Getrennte Stroeme: rng_p muss exakt dem Strom aus calibrate() folgen,
    # sonst greifen die dort bestimmten Faktoren auf andere Ziehungen zu.
    rng_p = np.random.default_rng(seed)
    rng_n = np.random.default_rng(seed + 1)
    sheet = []
    for i, (real, (h, w)) in enumerate(zip(reals, shapes)):
        # Identische Alphamaske wie die echte Stange (gleiche Funktion,
        # gleiche Federung) -> Geometrie und Kante sind exakt gleich.
        a = (capsule_alpha(w, h) * 255).astype(np.uint8)

        phys = cylinder_bar(h, w, rng_p, M, gain)
        naiv = flat_bar(h, w, rng_n, M)
        cv2.imwrite(os.path.join(OUT_PHYS, f"bar_{i:03d}.png"),
                    np.dstack([phys, a]))
        cv2.imwrite(os.path.join(OUT_NAIV, f"bar_{i:03d}.png"),
                    np.dstack([naiv, a]))
        if len(sheet) < 8:
            sheet.append((real, np.dstack([phys, a]), np.dstack([naiv, a])))

    # Messung der drei Bibliotheken nebeneinander -> Beleg, dass das
    # prozedurale Modell die kalibrierten Kennwerte tatsaechlich trifft.
    rep = {
        "kalibrierfaktoren": gain,
        "kalibrierverlauf": hist,
        "photographisch": M,
        "prozedural_physikalisch": measure(sorted(glob.glob(OUT_PHYS + "/*.png"))),
        "prozedural_naiv": measure(sorted(glob.glob(OUT_NAIV + "/*.png"))),
    }
    with open(os.path.join(DOCS, "bars_kennwerte.json"), "w") as f:
        json.dump(rep, f, indent=2)

    # Kontrollbild: drei Bibliotheken untereinander, 4-fach vergroessert
    tiles = []
    for real, phys, naiv in sheet:
        for r in (real, phys, naiv):
            aa = r[:, :, 3:4] / 255.0
            comp = (r[:, :, :3] * aa + 90 * (1 - aa)).astype(np.uint8)
            tiles.append(cv2.resize(comp, None, fx=4, fy=4,
                                    interpolation=cv2.INTER_NEAREST))
    Wm = max(t.shape[1] for t in tiles)
    Hm = max(t.shape[0] for t in tiles)
    sh = np.full((len(tiles) * (Hm + 8) + 8, Wm + 16, 3), 90, np.uint8)
    for i, t in enumerate(tiles):
        y = 8 + i * (Hm + 8)
        sh[y:y + t.shape[0], 8:8 + t.shape[1]] = t
    cv2.imwrite(os.path.join(DOCS, "bars_vergleich.png"), sh)

    print(json.dumps(rep, indent=2))
    print(f"\n-> {len(paths)} Stangen je Bibliothek")
    print(f"   {OUT_PHYS}/  (B1 prozedural, physikalisch)")
    print(f"   {OUT_NAIV}/  (B2 prozedural, naiv)")
    print(f"   {DOCS}/bars_vergleich.png, {DOCS}/bars_kennwerte.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true",
                    help="nur die Messung an assets/bars/ ausgeben")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.measure:
        print(json.dumps(measure(sorted(glob.glob(REAL + "/*.png"))), indent=2))
    else:
        build(a.seed)
