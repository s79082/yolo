"""
Stufe 2: Aufbau einer Okkluder-Bibliothek aus echten Rohrstangen.

WARUM ECHTE STANGEN STATT GEZEICHNETER BALKEN
---------------------------------------------
In allen realen Aufnahmen liegen vor den Stirnflaechen weitere Rohre, die
Teile der Profile verdecken. Ein synthetisch gemalter Balken haette eine
andere Textur-, Rausch- und Glanzstatistik als der Rest des Bildes; das Netz
koennte "Objekt liegt neben kuenstlichem Balken" lernen. Die vier
Clean-Plate-Hintergruende (bg/test*_cutted.*) sind buchstaeblich Waende aus
Rohrschaeften desselben Materials, derselben Kamera und derselben Beleuchtung.
Daraus geschnittene Streifen sind photometrisch exakt konsistent.

VERFAHREN
---------
1. Dominante Rohrrichtung ueber den Struktur-Tensor bestimmen und das Bild so
   drehen, dass die Rohre horizontal liegen.
2. Zeilenmittel-Profil bilden. Jedes Rohr erzeugt eine helle Glanzlinie
   (Maximum) und dazwischen einen dunklen Schattenspalt (Minimum). Die
   Maxima liefern die Rohrachsen, ihr Abstand den Durchmesser.
3. Streifen der Hoehe = Achsabstand um jede Achse ausschneiden.
4. Streifen mit weicher Kapselmaske (abgerundete Enden, gefederter Rand)
   versehen -> RGBA-Okkluder.
"""
import os
import numpy as np
import cv2

CLEAN_PLATES = ["bg/test_cutted.png", "bg/test2_cutted.jpg",
                "bg/test4_cutted.png", "bg/test5_cutted.jpg"]
OUT = "assets/bars"


def dominant_angle(gray):
    """Struktur-Tensor -> Winkel der dominanten Bildstruktur in Grad."""
    g = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    jxx, jyy, jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    # Richtung der geringsten Gradientenenergie = Richtung der Rohrachse
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)      # Richtung max. Energie
    return np.rad2deg(theta) + 90.0


def rotate_keep(img, ang):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT101)


def find_axes(gray):
    """Rohrachsen = lokale Maxima des Zeilenmittelprofils."""
    prof = gray.mean(axis=1).astype(np.float32)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 5), 0).ravel()
    base = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 61), 0).ravel()
    d = prof - base                                    # Hochpass
    peaks = [i for i in range(2, len(d) - 2)
             if d[i] == max(d[i - 2:i + 3]) and d[i] > 0.15 * d.std()]
    return np.array(peaks)


def capsule_alpha(w, h, feather=1.6, supersample=4):
    """Weiche Kapselmaske (Rechteck mit halbrunden Enden)."""
    S = supersample
    ys, xs = np.mgrid[0:h * S, 0:w * S].astype(np.float32)
    xs = (xs + 0.5) / S
    ys = (ys + 0.5) / S
    r = h / 2.0
    cx1, cx2 = r, w - r
    dx = np.clip(xs, cx1, cx2)
    dist = np.sqrt((xs - dx) ** 2 + (ys - h / 2.0) ** 2)
    a = np.clip((r - dist) / feather + 0.5, 0, 1)
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)


def build(max_per_plate=14, min_d=9, max_d=90):
    os.makedirs(OUT, exist_ok=True)
    n = 0
    sheet_rows = []
    for path in CLEAN_PLATES:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Der Struktur-Tensor ist 180-periodisch; auf (-45, 45] falten, damit
        # immer die kleinstmoegliche Drehung zur Horizontalen gewaehlt wird.
        ang = (dominant_angle(gray) + 45.0) % 90.0 - 45.0
        rot = rotate_keep(img, ang)
        rgray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = rgray.shape
        # Raender wegen Rotationsartefakten beschneiden
        m = int(0.06 * min(H, W)) + 2
        rot, rgray = rot[m:H - m, m:W - m], rgray[m:H - m, m:W - m]
        H, W = rgray.shape
        axes = find_axes(rgray)
        if len(axes) < 3:
            print(f"  {path}: keine Achsen gefunden (ang={ang:.1f})")
            continue
        diam = int(np.median(np.diff(axes)))
        print(f"  {os.path.basename(path):20s} ang={ang:+5.1f} deg  "
              f"{len(axes)} Achsen  Durchmesser~{diam}px")
        if not (min_d <= diam <= max_d):
            print("    -> Durchmesser ausserhalb Plausibilitaetsbereich, uebersprungen")
            continue
        step = max(1, len(axes) // max_per_plate)
        for y in axes[::step][:max_per_plate]:
            y0, y1 = int(y - diam / 2), int(y + diam / 2)
            if y0 < 0 or y1 > H:
                continue
            L = min(W, int(diam * np.random.uniform(9, 22)))
            x0 = np.random.randint(0, max(1, W - L))
            strip = rot[y0:y1, x0:x0 + L]
            if strip.shape[0] < 4 or strip.shape[1] < 20:
                continue
            a = capsule_alpha(strip.shape[1], strip.shape[0])
            rgba = np.dstack([strip, (a * 255).astype(np.uint8)])
            name = f"bar_{n:03d}.png"
            cv2.imwrite(os.path.join(OUT, name), rgba)
            n += 1
            if len(sheet_rows) < 16:
                sheet_rows.append(rgba)
    # Kontrollbild
    if sheet_rows:
        Wm = max(r.shape[1] for r in sheet_rows)
        Hm = max(r.shape[0] for r in sheet_rows)
        sh = np.full((len(sheet_rows) * (Hm + 6) + 6, Wm + 12, 3), 90, np.uint8)
        for i, r in enumerate(sheet_rows):
            a = r[:, :, 3:4] / 255.0
            comp = (r[:, :, :3] * a + 90 * (1 - a)).astype(np.uint8)
            y = 6 + i * (Hm + 6)
            sh[y:y + comp.shape[0], 6:6 + comp.shape[1]] = comp
        cv2.imwrite("docs/bars_check.png", sh)
    print(f"-> {n} Okkluder-Stangen in {OUT}/, Kontrollbild docs/bars_check.png")


if __name__ == "__main__":
    np.random.seed(0)
    build()
