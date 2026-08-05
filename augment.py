"""Synthetische Trainingsdaten fuer die Rohrenden-Detektion.

Baut aus drei freigestellten Rohrprofil-Ausschnitten Trainingsbilder mit
exakten YOLO-Labels. Uebernimmt die Struktur von copy_paste_augmentation.py,
passt sie aber an die andere Objektgeometrie an:

  Riegel                          Rohrende
  ------------------------------  ------------------------------------------
  stark anisotrop (L/t ~ 33)      kompakt, elliptisch (Achsenverh. ~ 1,1)
  Rotation blaeht Box auf         Rotation laesst Box unveraendert
  Periode T = Dicke + Luecke      Periode T = Durchmesser + Luecke
  Zeilenprofil zaehlbar           nur per Kreis-/Blobdetektion zaehlbar

Konsequenz fuer die Augmentierung: Rotation ist hier GRATIS. Sie veraendert
die achsenparallele Box eines Kreises nicht (vgl. Gl. box-inflation-kreis),
variiert aber die innere Textur -- Gratkante, Rost, Lichtreflex. Deshalb wird
hier voll rotiert (0-360 Grad), waehrend die Riegel-Pipeline auf ~2 Grad
begrenzt bleiben musste.

Beobachtungen aus dem Referenzfoto, die nachgebildet werden:
  1. Rohrenden liegen in CLUSTERN (ein Zaunfeld = 2 Rahmenrohre), nicht in
     regelmaessigen Spalten. Zwischen den Clustern groessere Luecken.
  2. Starke Tiefenokklusion: viele Rohrenden sind nur teilweise sichtbar,
     manche nur als Sichel. Wird ueber Belegungskarte + z-Ordnung erzeugt.
  3. Perspektivische Verkuerzung: je nach Blickwinkel wird der Kreis zur
     flachen Ellipse gestaucht.
  4. Materialvariation: galvanisiertes Grau bis Rostbraun.

WICHTIGE EINSCHRAENKUNG
Mit nur drei Quellausschnitten ist die Texturdiversitaet minimal. Das Netz
kann die drei konkreten Muster memorieren statt "Rohrende" zu lernen. Zur
Entschaerfung wird stark photometrisch variiert, aber das ersetzt keine
echten zusaetzlichen Ausschnitte. Fuer die Arbeit gilt: Validierung
ausschliesslich auf realen, nicht aus diesen drei Ausschnitten abgeleiteten
Aufnahmen (vgl. Abschnitt Datenstrategie).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

CUTOUT_PATHS = [
    "cutted1.png",
    "cuuted2.png",
    "cuuted3.png",
]

STRIDE_P3 = 8      # kumulative Schrittweite der feinsten Vorhersageebene
NET_IN = 640       # Netzeingangsgroesse


# =====================================================================
#  Konfiguration
# =====================================================================


@dataclass
class Config:
    out_w: int = 1280
    out_h: int = 960

    # --- Clusterstruktur: ein Zaunfeld = pipes_per_field Rahmenrohre ---
    pipes_per_field: int = 2
    fields_min: int = 6
    fields_max: int = 18
    # Luecke INNERHALB eines Feldes (zwischen den 2 Rahmenrohren),
    # als Faktor des Rohrdurchmessers. 0 = beruehrend.
    intra_gap_frac: tuple[float, float] = (-0.05, 0.35)
    # Luecke ZWISCHEN Feldern, als Faktor des Rohrdurchmessers.
    inter_gap_frac: tuple[float, float] = (0.25, 1.10)
    # horizontaler Versatz je Rohr, als Faktor des Durchmessers
    x_jitter_frac: float = 0.30
    # Anzahl paralleler Spalten (im Foto liegen mehrere Stapel nebeneinander)
    columns_min: int = 2
    columns_max: int = 5

    # --- Groesse ---
    # Zieldurchmesser in Pixel. Referenzschaetzung aus dem Foto: ~40 px.
    diameter_px: tuple[float, float] = (26.0, 58.0)
    # Perspektivische Stauchung: Verhaeltnis Hoehe/Breite der Ellipse.
    # 1.0 = frontal, 0.45 = stark angewinkelt.
    squash: tuple[float, float] = (0.45, 1.0)
    # Rotation in Grad -- hier voll erlaubt, da bei Kreisen kostenlos.
    rotation_deg: tuple[float, float] = (0.0, 360.0)

    # --- Okklusion ---
    visibility_min: float = 0.30
    min_box_px: float = 6.0
    # Anteil der Rohrenden, die zusaetzlich absichtlich teilverdeckt werden
    # (dunkler Balken davor), um die reale Tiefenokklusion nachzubilden.
    extra_occluder_prob: float = 0.35

    # --- Material / Photometrik ---
    rust_prob: float = 0.30
    alpha_feather_px: float = 0.8
    color_transfer_strength: float = 0.25
    brightness_jitter: float = 0.30
    contrast_jitter: float = 0.22
    noise_sigma_max: float = 2.2
    blur_prob: float = 0.30
    blur_sigma_max: float = 0.9
    jpeg_q: tuple[int, int] = (72, 96)

    # --- Inferenz-Annahme fuer die Nyquist-Pruefung ---
    strip_height_px: int = 640

    seed: int = 4711


# =====================================================================
#  Ausschnitte laden und Alphamaske erzeugen
# =====================================================================


def build_alpha(bgr: np.ndarray, feather: float) -> np.ndarray:
    """Erzeugt eine elliptische Alphamaske fuer einen Rohrend-Ausschnitt.

    Die gelieferten PNGs haben einen vollflaechig opaken Alphakanal, also
    keine echte Objektsilhouette. Da das Rohrende den Ausschnitt fast
    vollstaendig ausfuellt, ist eine eingeschriebene Ellipse eine gute
    Approximation der tatsaechlichen Kontur.
    """
    h, w = bgr.shape[:2]
    m = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(
        m,
        center=(w // 2, h // 2),
        axes=(int(w * 0.49), int(h * 0.49)),
        angle=0, startAngle=0, endAngle=360,
        color=1.0, thickness=-1,
    )
    if feather > 0:
        k = int(2 * round(3 * feather) + 1)
        m = cv2.GaussianBlur(m, (k, k), feather)
    return m


@dataclass
class Cutout:
    cid: str
    bgr: np.ndarray
    alpha: np.ndarray


def load_cutouts(cfg: Config) -> list[Cutout]:
    outs = []
    for p in CUTOUT_PATHS:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise SystemExit(f"nicht lesbar: {p}")
        bgr = im[:, :, :3].copy()
        outs.append(Cutout(Path(p).stem, bgr, build_alpha(bgr, cfg.alpha_feather_px)))
    return outs


def apply_rust(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Verschiebt die Farbe Richtung Rostbraun.

    Im Referenzfoto treten sowohl galvanisiert-graue als auch deutlich
    verrostete Rohrenden auf; ohne diese Variation lernt das Netz nur den
    grauen Fall.
    """
    out = bgr.astype(np.float32)
    strength = float(rng.uniform(0.25, 0.7))
    # BGR: Rost = wenig Blau, mehr Rot
    tint = np.array([0.55, 0.85, 1.45], dtype=np.float32)
    out *= (1.0 - strength) + strength * tint
    return np.clip(out, 0, 255).astype(np.uint8)


# =====================================================================
#  Geometrie
# =====================================================================


def transform_cutout(
    cut: Cutout, diameter: float, squash: float, rot_deg: float,
    rng: np.random.Generator, do_rust: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Skaliert, stauscht und rotiert einen Ausschnitt samt Maske.

    Reihenfolge: Zielgroesse setzen (Durchmesser x Durchmesser*squash),
    dann rotieren. Bei einem Kreis aendert die Rotation die Boxgroesse nicht;
    bei gestauchter Ellipse schon, weshalb die Box anschliessend ohnehin aus
    der Maske neu bestimmt wird (siehe box_from_mask).
    """
    d = max(6, int(round(diameter)))

    # 1) kreisrund skalieren
    bgr = cv2.resize(cut.bgr, (d, d), interpolation=cv2.INTER_CUBIC)
    alpha = cv2.resize(cut.alpha, (d, d), interpolation=cv2.INTER_LINEAR)
    if do_rust:
        bgr = apply_rust(bgr, rng)

    # 2) rotieren, solange das Objekt noch kreisfoermig ist. Die Rotation
    #    veraendert dadurch nur die innere Textur (Gratkante, Rost, Reflex),
    #    nicht die Silhouette -- genau die Eigenschaft, die Rotation bei
    #    kreisfoermigen Objekten kostenlos macht.
    M = cv2.getRotationMatrix2D((d / 2 - 0.5, d / 2 - 0.5), rot_deg, 1.0)
    bgr = cv2.warpAffine(bgr, M, (d, d), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    alpha = cv2.warpAffine(alpha, M, (d, d), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # 3) ERST DANACH vertikal stauchen. Perspektivische Verkuerzung erfolgt
    #    entlang der Blickachse und ist unabhaengig davon, wie das Rohr um
    #    seine eigene Achse verdreht ist. Die Ellipse bleibt damit
    #    achsenparallel, so wie im Referenzfoto.
    th = max(4, int(round(d * squash)))
    if th != d:
        bgr = cv2.resize(bgr, (d, th), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (d, th), interpolation=cv2.INTER_AREA)
    return bgr, alpha


def box_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Engste achsenparallele Box um die sichtbare Maske.

    Wie in der Riegel-Pipeline wird die Box AUS DER MASKE bestimmt, nicht aus
    transformierten Eckpunkten -- hier zusaetzlich wichtig, weil nach
    Teilverdeckung nur noch ein Sichelrest sichtbar sein kann.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


# =====================================================================
#  Hintergrund
# =====================================================================


def make_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Dunkler Stahlstapel-Hintergrund mit horizontaler Streifenstruktur.

    Nachgebildet wird die Umgebung, in der Rohrenden im Foto tatsaechlich
    liegen: dunkle Schattenluecken zwischen Rohrkoerpern, mittelgraue
    Stahlflaechen, gelegentlich Maschendraht dahinter.
    """
    # Stahl ist nahezu unbunt: Luminanz einkanalig erzeugen und erst am Ende
    # auf drei Kanaele legen. Kanalweises Rauschen wuerde farbige Flecken
    # erzeugen, die im Referenzfoto nicht vorkommen.
    base = float(rng.uniform(40, 90))
    lum = np.full((h, w), base, dtype=np.float32)
    lum += rng.normal(0, 6, (h, w))

    # liegende Rohrkoerper: heller Reflexstreifen mit dunklem Schattenrand
    n = int(rng.integers(30, 70))
    for _ in range(n):
        y = int(rng.integers(-10, h))
        th = int(rng.integers(4, 20))
        y2 = min(h, max(0, y + th))
        y1 = max(0, y)
        if y2 <= y1:
            continue
        peak = float(rng.uniform(25, 95))
        prof = np.sin(np.linspace(0, np.pi, y2 - y1, dtype=np.float32)) * peak
        lum[y1:y2, :] += prof[:, None]
        sh = min(h, y2 + int(rng.integers(1, 5)))
        if sh > y2:
            lum[y2:sh, :] -= float(rng.uniform(15, 45))

    # grobe Helligkeitswolken (ebenfalls einkanalig)
    small = rng.normal(0, 14, (max(4, h // 40), max(4, w // 40)))
    lum += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    lum += np.linspace(-12, 12, h, dtype=np.float32)[:, None]

    img = np.repeat(np.clip(lum, 0, 255)[:, :, None], 3, axis=2)
    # ganz leichter, global konstanter Farbstich (Tageslicht/Verzinkung)
    tint = np.array([1.0 + rng.uniform(0.0, 0.06),
                     1.0,
                     1.0 - rng.uniform(0.0, 0.05)], dtype=np.float32)
    img *= tint
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# =====================================================================
#  Photometrik auf dem Komposit
# =====================================================================


def photometric(img: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Erst NACH der Komposition -- sonst unterscheiden sich Rausch- und
    Kompressionsstatistik von eingefuegtem Objekt und Hintergrund."""
    out = img.astype(np.float32)
    b = 1.0 + float(rng.uniform(-cfg.brightness_jitter, cfg.brightness_jitter))
    c = 1.0 + float(rng.uniform(-cfg.contrast_jitter, cfg.contrast_jitter))
    mean = out.mean()
    out = np.clip((out - mean) * c + mean * b, 0, 255)

    hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + float(rng.uniform(-0.25, 0.25))), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if rng.random() < cfg.blur_prob:
        sg = float(rng.uniform(0.3, cfg.blur_sigma_max))
        k = int(2 * round(3 * sg) + 1)
        out = cv2.GaussianBlur(out, (k, k), sg)

    out += rng.normal(0, float(rng.uniform(0, cfg.noise_sigma_max)), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)

    q = int(rng.integers(cfg.jpeg_q[0], cfg.jpeg_q[1] + 1))
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else out


# =====================================================================
#  Komposition
# =====================================================================


@dataclass
class Ann:
    x1: float
    y1: float
    x2: float
    y2: float
    vis: float
    cid: str

    def to_yolo(self, w: int, h: int) -> str:
        xc = (self.x1 + self.x2) / 2 / w
        yc = (self.y1 + self.y2) / 2 / h
        bw = (self.x2 - self.x1) / w
        bh = (self.y2 - self.y1) / h
        return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def compose(cuts: list[Cutout], cfg: Config, rng: np.random.Generator):
    W, H = cfg.out_w, cfg.out_h
    canvas = make_background(H, W, rng).astype(np.float32)
    occupied = np.zeros((H, W), dtype=bool)
    anns: list[Ann] = []
    used_cids: set[str] = set()

    n_cols = int(rng.integers(cfg.columns_min, cfg.columns_max + 1))
    n_fields_total = 0
    diam_all: list[float] = []

    for col in range(n_cols):
        # Ein Stapel: gemeinsame Rohrgroesse und Stauchung, damit die
        # Perspektive innerhalb einer Spalte konsistent bleibt.
        diameter = float(rng.uniform(*cfg.diameter_px))
        squash = float(rng.uniform(*cfg.squash))
        eff_h = diameter * squash        # tatsaechliche vertikale Ausdehnung

        col_x = W * (col + 0.5) / n_cols + float(rng.uniform(-0.06, 0.06)) * W
        n_fields = int(rng.integers(cfg.fields_min, cfg.fields_max + 1))
        y = float(rng.uniform(0.02, 0.12)) * H

        for _ in range(n_fields):
            if y > H - eff_h * 0.5:
                break
            n_fields_total += 1
            for k in range(cfg.pipes_per_field):
                cut = cuts[int(rng.integers(0, len(cuts)))]
                rot = float(rng.uniform(*cfg.rotation_deg))
                do_rust = bool(rng.random() < cfg.rust_prob)
                w_bgr, w_a = transform_cutout(
                    cut, diameter, squash, rot, rng, do_rust
                )
                ch, cw = w_a.shape[:2]

                px = int(round(col_x + float(rng.uniform(
                    -cfg.x_jitter_frac, cfg.x_jitter_frac)) * diameter - cw / 2))
                py = int(round(y - ch / 2))

                x0, y0 = max(0, px), max(0, py)
                x1, y1 = min(W, px + cw), min(H, py + ch)
                if x1 - x0 < 3 or y1 - y0 < 3:
                    y += eff_h * (1.0 + float(rng.uniform(*cfg.intra_gap_frac)))
                    continue

                sub_a = w_a[y0 - py:y1 - py, x0 - px:x1 - px]
                sub_b = w_bgr[y0 - py:y1 - py, x0 - px:x1 - px]

                full = sub_a > 0.5
                n_full = int(full.sum())
                if n_full == 0:
                    y += eff_h * (1.0 + float(rng.uniform(*cfg.intra_gap_frac)))
                    continue

                region_free = ~occupied[y0:y1, x0:x1]
                vis_mask = full & region_free
                nu = float(vis_mask.sum()) / float(n_full)

                # Farbangleichung an die lokale Umgebung
                loc = canvas[y0:y1, x0:x1].astype(np.uint8)
                s = cfg.color_transfer_strength
                if s > 0 and full.sum() > 16:
                    sb = sub_b.astype(np.float32)
                    for c3 in range(3):
                        sv = sb[:, :, c3][full]
                        dv = loc[:, :, c3].astype(np.float32)
                        adj = ((sb[:, :, c3] - sv.mean()) *
                               (dv.std() + 1e-6) / (sv.std() + 1e-6) + dv.mean())
                        sb[:, :, c3] = (1 - s) * sb[:, :, c3] + s * adj
                    sub_b = np.clip(sb, 0, 255).astype(np.uint8)

                # Einfuegen
                a3 = sub_a[..., None]
                canvas[y0:y1, x0:x1] = (
                    canvas[y0:y1, x0:x1] * (1 - a3) + sub_b.astype(np.float32) * a3
                )

                # Annotation nur bei ausreichender Sichtbarkeit
                if nu >= cfg.visibility_min:
                    bx = box_from_mask(vis_mask)
                    if bx is not None:
                        bx1, by1, bx2, by2 = bx
                        aw, ah = bx2 - bx1, by2 - by1
                        if aw >= cfg.min_box_px and ah >= cfg.min_box_px:
                            anns.append(Ann(
                                x0 + bx1, y0 + by1, x0 + bx2, y0 + by2, nu, cut.cid
                            ))
                            used_cids.add(cut.cid)
                            diam_all.append(diameter)

                occupied[y0:y1, x0:x1] |= full

                # gelegentlich zusaetzlicher Verdecker (Rohrkoerper davor),
                # bildet die reale Tiefenokklusion nach
                if rng.random() < cfg.extra_occluder_prob:
                    oy = int(y0 + rng.integers(0, max(1, (y1 - y0) - 2)))
                    oh = int(rng.integers(3, max(4, int(eff_h * 0.55))))
                    oy2 = min(H, oy + oh)
                    ox0 = max(0, x0 - int(diameter * rng.uniform(0.5, 2.0)))
                    ox1 = min(W, x1 + int(diameter * rng.uniform(0.5, 2.0)))
                    if oy2 > oy and ox1 > ox0:
                        # zylindrischer Reflexverlauf, damit der Verdecker wie
                        # ein davorliegender Rohrkoerper wirkt und nicht wie
                        # ein flaches Rechteck
                        dark = float(rng.uniform(28, 70))
                        peak = float(rng.uniform(45, 120))
                        prof = (dark + np.sin(np.linspace(0, np.pi, oy2 - oy,
                                dtype=np.float32)) * peak)
                        bar = np.repeat(prof[:, None], ox1 - ox0, axis=1)
                        canvas[oy:oy2, ox0:ox1] = np.repeat(
                            bar[:, :, None], 3, axis=2)
                        occupied[oy:oy2, ox0:ox1] = True

                y += eff_h * (1.0 + float(rng.uniform(*cfg.intra_gap_frac)))

            y += eff_h * float(rng.uniform(*cfg.inter_gap_frac))

    img = photometric(np.clip(canvas, 0, 255).astype(np.uint8), cfg, rng)

    # Periode und Nyquist-Pruefung
    d_mean = float(np.mean(diam_all)) if diam_all else 40.0
    period = median_period(anns, col_gap=d_mean * 1.6)
    sigma = NET_IN / max(cfg.strip_height_px, 1)
    period_in = period * sigma if np.isfinite(period) else float("nan")

    meta = dict(
        n_boxes=len(anns),
        n_fields=n_fields_total,
        n_columns=n_cols,
        mean_diameter_px=round(float(np.mean(diam_all)), 1) if diam_all else 0.0,
        median_period_px=round(period, 1) if np.isfinite(period) else 0.0,
        period_at_input_px=round(period_in, 1) if np.isfinite(period_in) else 0.0,
        nyquist_ok=bool(period_in > 2 * STRIDE_P3) if np.isfinite(period_in) else True,
        cutouts_used=sorted(used_cids),
        implied_field_count=len(anns) / max(cfg.pipes_per_field, 1),
    )
    return img, anns, meta


def median_period(anns: list[Ann], col_gap: float = 40.0) -> float:
    """Median des vertikalen Mittelpunktabstands benachbarter Rohrenden,
    getrennt je Spalte gemessen.

    Wichtig: Die Nyquist-Bedingung gilt entlang der Richtung, in der die
    Struktur periodisch ist -- hier vertikal INNERHALB einer Stapelspalte.
    Zwei Rohrenden in verschiedenen Spalten sind horizontal getrennt und
    muessen nicht vertikal aufgeloest werden. Eine globale Messung ueber alle
    y-Mittelpunkte mischt die Spalten und unterschaetzt die Periode um etwa
    den Faktor der Spaltenzahl.
    """
    if len(anns) < 2:
        return float("nan")

    # Spalten ueber die x-Mittelpunkte gruppieren: sortieren und dort
    # trennen, wo die Luecke groesser als col_gap ist.
    items = sorted(anns, key=lambda a: (a.x1 + a.x2) / 2)
    clusters: list[list[Ann]] = [[items[0]]]
    for a in items[1:]:
        prev_x = (clusters[-1][-1].x1 + clusters[-1][-1].x2) / 2
        cur_x = (a.x1 + a.x2) / 2
        if cur_x - prev_x > col_gap:
            clusters.append([a])
        else:
            clusters[-1].append(a)

    periods: list[float] = []
    for cl in clusters:
        if len(cl) < 2:
            continue
        yc = np.sort(np.array([(a.y1 + a.y2) / 2 for a in cl]))
        d = np.diff(yc)
        d = d[d > 0.5]
        if d.size:
            periods.append(float(np.median(d)))
    return float(np.median(periods)) if periods else float("nan")


# =====================================================================
#  Visualisierung
# =====================================================================


def draw(img: np.ndarray, anns: list[Ann]) -> np.ndarray:
    out = img.copy()
    for a in anns:
        # Farbe kodiert Sichtbarkeit: gruen = frei, gelb = teilverdeckt
        col = (0, 235, 0) if a.vis > 0.75 else (0, 215, 235)
        cv2.rectangle(out, (int(a.x1), int(a.y1)), (int(a.x2), int(a.y2)), col, 1)
    txt = f"n = {len(anns)}  ->  Felder = {len(anns)/2:.1f}"
    cv2.rectangle(out, (8, 8), (8 + 9 * len(txt), 42), (0, 0, 0), -1)
    cv2.putText(out, txt, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 235, 0), 2, cv2.LINE_AA)
    return out


# =====================================================================
#  Hauptprogramm
# =====================================================================


def main(n_images: int = 8, out_root: str = "rohr_dataset") -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    cuts = load_cutouts(cfg)

    root = Path(out_root)
    #(root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    #(root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "vis").mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(asdict(cfg), indent=2,
                                                  default=str), encoding="utf-8")

    records = []
    n_sub = 0
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as mf:
        for i in range(n_images):
            img, anns, meta = compose(cuts, cfg, rng)
            name = f"rohr_{i:04d}"
            cv2.imwrite(str(root / "images" / "train" / f"{name}.jpg"), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            (root / "labels" / "train" / f"{name}.txt").write_text(
                "\n".join(a.to_yolo(cfg.out_w, cfg.out_h) for a in anns) + "\n",
                encoding="utf-8")
            cv2.imwrite(str(root / "vis" / f"{name}_annotiert.jpg"), draw(img, anns),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            meta["image"] = f"images/train/{name}.jpg"
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            records.append(meta)
            if not meta["nyquist_ok"]:
                n_sub += 1
            print(f"{name}: {meta['n_boxes']:3d} Boxen, "
                  f"D={meta['mean_diameter_px']:5.1f}px, "
                  f"T={meta['median_period_px']:5.1f}px, "
                  f"T@input={meta['period_at_input_px']:5.1f}px, "
                  f"Nyquist={'ok' if meta['nyquist_ok'] else 'VERLETZT'}")

    tot = sum(r["n_boxes"] for r in records)
    print(f"\n{n_images} Bilder, {tot} Boxen "
          f"(Durchschnitt {tot/max(n_images,1):.1f}/Bild)")
    if n_sub:
        print(f"WARNUNG: {n_sub}/{n_images} Bilder unter der Nyquist-Schranke "
              f"T > {2*STRIDE_P3} px -- Durchmesserbereich anheben oder "
              f"Streifenhoehe senken.")
    else:
        print(f"Alle Bilder erfuellen T > {2*STRIDE_P3} px am Netzeingang.")


if __name__ == "__main__":
    main()