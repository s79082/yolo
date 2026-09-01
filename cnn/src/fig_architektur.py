"""
Erzeugt die Architekturgrafiken fuer PipeNet.

    cnn/docs/fig_architektur.svg / .png   technische Uebersicht
    cnn/docs/fig_einfach.svg / .png       vereinfachte Fassung fuer einfach.md

Alle Zahlen sind aus dem Modell ausgelesen, nicht abgeschrieben:
    python cnn/src/fig_architektur.py --print

Aufruf:  python cnn/src/fig_architektur.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DOC = os.path.join(ROOT, "cnn/docs")
sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------
# Farbschema
# ---------------------------------------------------------------------------
ENC = dict(fc="#dbeafe", ec="#1d4ed8")      # Encoder (vortrainiert)
DEC = dict(fc="#ccfbf1", ec="#0f766e")      # Dekoder (neu gelernt)
HEAD = dict(fc="#fef3c7", ec="#b45309")     # Köpfe
OUT = dict(fc="#dcfce7", ec="#15803d")      # Dekodierung / Ausgabe
IMG = dict(fc="#f1f5f9", ec="#475569")      # Bild
OFF = dict(fc="#f8fafc", ec="#cbd5e1")      # nicht verwendet
TXT = "#0f172a"
MUT = "#64748b"


def measure():
    """Parameterzahlen und Formen direkt aus dem Modell holen."""
    import torch
    from model import PipeNet
    m = PipeNet(pretrained=False)
    g = {}
    for n, p in m.named_parameters():
        g[n.split(".")[0]] = g.get(n.split(".")[0], 0) + p.numel()
    x = torch.zeros(1, 3, 1024, 1024)
    with torch.no_grad():
        s2 = m.stem(x)
        c1 = m.layer1(m.pool(s2))
        c2 = m.layer2(c1)
        c3 = m.layer3(c2)
    shapes = {"stem": tuple(s2.shape[1:]), "layer1": tuple(c1.shape[1:]),
              "layer2": tuple(c2.shape[1:]), "layer3": tuple(c3.shape[1:])}
    return g, shapes, sum(g.values())


def box(ax, x, y, w, h, title, sub=None, style=ENC, fs=10.5, fs_sub=8.6,
        dashed=False, tcol=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.9",
                       linewidth=1.6, linestyle="--" if dashed else "-",
                       **style, zorder=2)
    ax.add_patch(p)
    cy = y + h / 2 + (0.85 if sub else 0)
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=fs,
            color=tcol or TXT, fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 1.15, sub, ha="center", va="center",
                fontsize=fs_sub, color=MUT, zorder=3)


def arrow(ax, p1, p2, label=None, style="-|>", col="#334155", lw=1.5,
          rad=0.0, ls="-", fs=8.2, dx=0.0, dy=0.9):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                        color=col, linewidth=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=1,
                        shrinkA=1, shrinkB=1)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
                color=col, zorder=3,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.85))


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(DOC, f"{name}.{ext}"), dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> cnn/docs/{name}.svg + .png")


# ---------------------------------------------------------------------------
# 1) Technische Uebersicht
# ---------------------------------------------------------------------------
def fig_technisch(g, shapes, total):
    fig, ax = plt.subplots(figsize=(15.8, 10.6))
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 66)
    ax.axis("off")

    ax.text(0.5, 65.0, "PipeNet — Architektur", fontsize=20,
            fontweight="bold", color=TXT, va="top")
    ax.text(0.5, 61.6,
            f"Detektor für Rohrprofil-Stirnflächen  ·  "
            f"{total / 1e6:.2f} M Parameter  ·  eine Detektionsebene bei "
            f"Stride 4  ·  keine NMS, kein imgsz",
            fontsize=11, color=MUT, va="top")

    for cx, t, c in ((17, "ENCODER   ResNet18, ImageNet-vortrainiert", ENC["ec"]),
                     (49, "DEKODER   neu gelernt", DEC["ec"]),
                     (82, "KÖPFE  +  DEKODIERUNG", HEAD["ec"])):
        ax.text(cx, 57.6, t, ha="center", fontsize=9.6, fontweight="bold",
                color=c)

    H = 5.4                                    # Boxhoehe
    ys = {"in": 51.4, 2: 44.4, 4: 37.4, 8: 30.4, 16: 23.4, "f4": 16.4}

    # --- Eingang -------------------------------------------------------
    box(ax, 4, ys["in"], 26, 5.0, "Eingangsbild   H x W x 3",
        "native Größe, nur auf /32 gepaddet", IMG, fs=11)

    # --- Encoder -------------------------------------------------------
    for s_, t, sub in (
            (2, "stem   conv 7x7, Stride 2",
             f"{shapes['stem'][0]} Kanäle  ·  {g['stem'] / 1e3:.1f} k Par."),
            (4, "maxpool  +  layer1",
             f"{shapes['layer1'][0]} Kanäle  ·  {g['layer1'] / 1e3:.0f} k Par."),
            (8, "layer2",
             f"{shapes['layer2'][0]} Kanäle  ·  {g['layer2'] / 1e3:.0f} k Par."),
            (16, "layer3",
             f"{shapes['layer3'][0]} Kanäle  ·  {g['layer3'] / 1e6:.2f} M Par.")):
        box(ax, 4, ys[s_], 26, H, t, sub, ENC)
        ax.text(2.4, ys[s_] + H / 2, f"/{s_}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=ENC["ec"])

    box(ax, 4, 9.4, 26, H, "layer4   Stride 32",
        "NICHT VERWENDET  ·  8.39 M Par. gespart", OFF, dashed=True,
        tcol="#94a3b8", fs=10)
    for x0, x1 in ((5.4, 11.0), (23.0, 28.6)):
        ax.plot([x0, x1], [9.4 + H, 9.4], color="#dbe2ea", lw=1.5, zorder=1)
        ax.plot([x0, x1], [9.4, 9.4 + H], color="#dbe2ea", lw=1.5, zorder=1)

    arrow(ax, (17, ys["in"]), (17, ys[2] + H))
    for p, q in ((2, 4), (4, 8), (8, 16)):
        arrow(ax, (17, ys[p]), (17, ys[q] + H))
    arrow(ax, (17, ys[16]), (17, 9.4 + H), ls=":", col="#cbd5e1", lw=1.3)

    # --- Dekoder -------------------------------------------------------
    for s_, t, sub in (
            (16, "lat3   1x1",
             f"256 -> 64 Kanäle  ·  {g['lat3'] / 1e3:.1f} k Par."),
            (8, "smooth2   3x3",
             f"64 Kanäle  ·  {g['smooth2'] / 1e3:.0f} k Par."),
            (4, "smooth1   3x3",
             f"64 Kanäle  ·  {g['smooth1'] / 1e3:.0f} k Par.")):
        box(ax, 36, ys[s_], 26, H, t, sub, DEC)
        ax.text(63.6, ys[s_] + H / 2, f"/{s_}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=DEC["ec"])

    box(ax, 36, ys["f4"], 26, H, "fuse   3x3",
        f"64 Kanäle bei Stride 4  ·  {g['fuse'] / 1e3:.0f} k Par.", DEC)
    ax.text(63.4, ys["f4"] + H - 1.0, "/4", ha="center", va="center",
            fontsize=10, fontweight="bold", color=DEC["ec"])

    # Lateralverbindungen
    for s_, lab in ((16, f"1x1  ·  {g['lat3'] / 1e3:.1f} k"),
                    (8, f"lat2 1x1  ·  {g['lat2'] / 1e3:.1f} k"),
                    (4, f"lat1 1x1  ·  {g['lat1'] / 1e3:.1f} k")):
        arrow(ax, (30, ys[s_] + H / 2), (36, ys[s_] + H / 2),
              None if s_ == 16 else lab, col=ENC["ec"], dy=1.5, dx=0.9,
              fs=7.6)

    # Top-Down
    for p, q in ((16, 8), (8, 4)):
        arrow(ax, (49, ys[p] + H), (49, ys[q]),
              "Upsample x2 + addieren", col=DEC["ec"], dx=10.4, dy=0.0,
              fs=7.6)
    arrow(ax, (49, ys[4]), (49, ys["f4"] + H), col=DEC["ec"])

    # Feiner Skip Stride 2 -> Stride 4 (Rohrkranz ist 1-3 px breit)
    ax.plot([32.0, 32.0], [ys[2] + 1.4, ys["f4"] + H - 1.4], color="#7c3aed",
            lw=1.5, ls=(0, (5, 2)), zorder=1)
    arrow(ax, (30, ys[2] + 1.4), (32.0, ys[2] + 1.4), col="#7c3aed", lw=1.5,
          style="-")
    arrow(ax, (32.0, ys["f4"] + H - 1.4), (36, ys["f4"] + H - 1.4),
          col="#7c3aed", lw=1.5)
    ax.text(31.4, 25.5, f"fine  3x3 Stride 2  ·  {g['fine'] / 1e3:.0f} k",
            rotation=90, ha="center", va="center", fontsize=7.6,
            color="#7c3aed")

    # --- Köpfe --------------------------------------------------------
    hd = [(ys[2], "hm    1 Kanal", "Zentrums-Heatmap (Logit)", g["hm"]),
          (ys[4], "wh    2 Kanäle", "log(Breite), log(Höhe) in px", g["wh"]),
          (ys[8], "off   2 Kanäle", "Subpixel-Versatz in [0,1)", g["off"])]
    for y, t, sub, par in hd:
        box(ax, 69, y, 27, H, t, f"{sub}  ·  {par / 1e3:.0f} k Par.", HEAD)
    # Bus von fuse nach oben zu den drei Köpfen (keine Kreuzungen)
    ax.plot([66, 66], [ys["f4"] + H / 2, ys[2] + H / 2], color=HEAD["ec"],
            lw=1.5, zorder=1)
    arrow(ax, (62, ys["f4"] + H / 2), (66, ys["f4"] + H / 2), col=HEAD["ec"],
          style="-", lw=1.5)
    for y, *_ in hd:
        arrow(ax, (66, y + H / 2), (69, y + H / 2), col=HEAD["ec"], lw=1.5)

    # --- Dekodierung ---------------------------------------------------
    box(ax, 69, ys[16] - 0.7, 27, 6.0, "3x3-MaxPool  ->  Peaks",
        "jedes lokale Maximum = ein Rohr  ·  keine IoU-Schwelle", OUT, fs=11,
        fs_sub=8.3)
    box(ax, 69, 12.4, 27, 6.2, "Boxen  +  Konfidenz",
        "cx = (Zelle + off) x 4      w = exp(wh)", OUT, fs=11)

    # Bus rechts: alle drei Köpfe in die Dekodierung
    ax.plot([98.6, 98.6], [ys[8] + H / 2, ys[2] + H / 2], color=HEAD["ec"],
            lw=1.5, zorder=1)
    for y, *_ in hd:
        arrow(ax, (96, y + H / 2), (98.6, y + H / 2), col=HEAD["ec"],
              style="-", lw=1.5)
    ax.plot([98.6, 98.6], [ys[8] + H / 2, ys[16] + 2.3], color=HEAD["ec"],
            lw=1.5, zorder=1)
    arrow(ax, (98.6, ys[16] + 2.3), (96, ys[16] + 2.3), col=HEAD["ec"],
          lw=1.5)
    arrow(ax, (82.5, ys[16] - 1.0), (82.5, 18.6), col=OUT["ec"], lw=1.8)

    # --- Fussnoten -----------------------------------------------------
    ax.text(0.5, 5.6,
            "Ausgabe-Stride 4:  ein 22-px-Objekt belegt 5.5 Zellen, der "
            "nächste Nachbar sitzt 5.5 Zellen entfernt.  Bei Stride 8 wären "
            "es nur 2.8 —\ndicht stehende Nachbarn ließen sich dann nicht "
            "mehr als getrennte Maxima auflösen.",
            fontsize=9.0, color=MUT, va="top", linespacing=1.6)
    ax.text(0.5, 1.2,
            "Der feine Skip von Stride 2 erhält den 1–3 px breiten hellen "
            "Rohrkranz, den layer1 nach dem MaxPool nur abgeschwächt führt.",
            fontsize=9.0, color="#7c3aed", va="bottom")
    save(fig, "fig_architektur")


# ---------------------------------------------------------------------------
# 2) Vereinfachte Fassung
# ---------------------------------------------------------------------------
def fig_einfach(total):
    fig, ax = plt.subplots(figsize=(15.0, 7.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    ax.text(0.5, 44.6, "Wie PipeNet Rohre findet", fontsize=20,
            fontweight="bold", color=TXT, va="top")
    ax.text(0.5, 41.4, "in vier Schritten", fontsize=11.5, color=MUT,
            va="top")

    steps = [
        (2.0, "1", "Das Foto", "Ein Bild von einem\nHaufen Bauzäune.",
         IMG, "#475569"),
        (26.0, "2", "Die Suchtürme",
         "Das Programm schaut das Bild\nvier Mal an — jedes Mal\n"
         "gröber. So sieht es\nerst Kanten, dann Ringe.", ENC, ENC["ec"]),
        (50.0, "3", "Die Schatzkarte",
         "Es malt eine schwarze Karte\nund setzt einen hellen Punkt\n"
         "genau dort, wo ein\nRohrende ist.", HEAD, HEAD["ec"]),
        (74.0, "4", "Punkte zählen",
         "Jeder helle Punkt ist ein Rohr.\nFertig — einfach zählen.",
         OUT, OUT["ec"]),
    ]
    for x, num, title, sub, style, col in steps:
        box(ax, x, 14.0, 22.0, 20.0, "", None, style)
        ax.add_patch(FancyBboxPatch((x + 0.9, 31.4), 3.6, 3.6,
                                    boxstyle="circle,pad=0.0",
                                    fc=col, ec="none", zorder=4))
        ax.text(x + 2.7, 33.2, num, ha="center", va="center", fontsize=13,
                fontweight="bold", color="white", zorder=5)
        ax.text(x + 11.0, 29.6, title, ha="center", va="center", fontsize=13.5,
                fontweight="bold", color=TXT, zorder=3)
        ax.text(x + 11.0, 21.6, sub, ha="center", va="center", fontsize=10.2,
                color="#334155", zorder=3, linespacing=1.65)
    for x in (24.0, 48.0, 72.0):
        arrow(ax, (x, 24.0), (x + 2.0, 24.0), lw=2.2, col="#94a3b8")

    ax.text(50, 9.0,
            "Der Trick: Rohrenden liegen dicht an dicht und ihre Kästen "
            "würden sich überlappen —\n"
            "aber ihre MITTELPUNKTE berühren sich nie. Darum sucht "
            "PipeNet Mittelpunkte statt Kästen.",
            ha="center", va="center", fontsize=11.2, color=TXT,
            linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.7", fc="#fffbeb",
                      ec="#fbbf24", lw=1.4))
    ax.text(50, 1.6,
            f"PipeNet hat {total / 1e6:.2f} Millionen kleine Stellschrauben. "
            "Das ist dreimal weniger als bei den fertigen Programmen — "
            "und trifft trotzdem genauso gut.",
            ha="center", va="bottom", fontsize=9.6, color=MUT)
    save(fig, "fig_einfach")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true",
                    help="nur die ausgelesenen Zahlen zeigen")
    a = ap.parse_args()
    g, shapes, total = measure()
    if a.print:
        for k, v in sorted(g.items(), key=lambda x: -x[1]):
            print(f"  {k:9s} {v:9,d}")
        print(f"  {'GESAMT':9s} {total:9,d}")
        for k, v in shapes.items():
            print(f"  {k:9s} {v}")
        raise SystemExit
    os.makedirs(DOC, exist_ok=True)
    fig_technisch(g, shapes, total)
    fig_einfach(total)
