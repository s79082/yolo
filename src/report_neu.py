"""
Stufe 7 (neu): Abbildungen und LaTeX-Tabellen der beiden Ablationen.

Liest ausschliesslich die JSON-Ausgaben von src/experiment_neu.py und
src/procedural_bars.py und erzeugt daraus

    docs/fig_balken.png          drei Okkluder-Bibliotheken im Vergleich
    docs/fig_balken_szene.png    dieselbe synthetische Szene, drei Arme
    docs/fig_ablation_a.png      Ablation A -- Okklusionsanteil
    docs/fig_ablation_b.png      Ablation B -- Herkunft der Okkluder
    docs/fig_imgsz.png           Inferenzaufloesung, yolo26s-p2
    docs/fig_training.png        Lernkurven aller fuenf Arme
    docs/fig_training_c.png      Lernkurven der Arme C0/C1
    paper/kapitel/tab_*.tex      Tabellen zum Einbinden

Alle Abbildungen werden zusaetzlich nach paper/abbildungen/neu_* kopiert.

Aufruf:  python src/report_neu.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
EXP = "docs/experiments"
DOC = "docs"
RUNS = "runs"
FIGDIR = "paper/abbildungen"
TEXDIR = "paper/kapitel"
# Die gekuerzte Fassung teilt sich zahlen.tex und die Tabellen mit der langen;
# beim Suchen nach verwendeten Makros muessen beide Kapitelverzeichnisse
# beruecksichtigt werden.
TEXDIRS = ["paper/kapitel"]

# Arme in Darstellungsreihenfolge
A_ARME = [("A0_keine_okklusion", "A0 ohne"),
          ("A1_baseline", "A1 Baseline"),
          ("A2_starke_okklusion", "A2 stark")]
B_ARME = [("A1_baseline", "B0 photogr."),
          ("B1_prozedural", "B1 prozedural"),
          ("B2_prozedural_naiv", "B2 naiv")]
# Ablation C liegt in eigenen Ergebnisdateien (src/eval_armC.py), damit ein
# erneutes "experiment_neu.py eval" ablationen.json nicht ueberschreibt.
C_ARME = [("C0_kontrolle", "C0 Kontrolle"),
          ("C1_naive_paste", "C1 naiv eingeklebt")]
CS = {"val": "#1f77b4", "test": "#d62728"}

# Beispielstangen der Bibliotheksabbildung. Bewusst ueber Dicke (16/18/20/21 px)
# und Grauwert (50/163/70/143) gestreut, damit die vier Kacheln die Bandbreite
# der Bibliothek zeigen und nicht nur ihren Anfang. Dieselben Namen gelten fuer
# alle drei Bibliotheken -- die Stangen sind paarweise gleich gross und tragen
# dieselbe Alphamaske.
BAR_BEISPIELE = ["bar_007", "bar_019", "bar_048", "bar_037"]


def _load(name):
    p = os.path.join(EXP, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _row(rows, arm, split):
    for r in rows:
        if r.get("arm") == arm and r["split"] == split:
            return r
    return None


def _de(x, n=3):
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{n}f}".replace(".", "{,}")


# ---------------------------------------------------------------------------
# Abbildung 1: die drei Okkluder-Bibliotheken
# ---------------------------------------------------------------------------
def fig_balken():
    libs = [("assets/bars", "photographisch (B0)"),
            ("assets/bars_proc", "prozedural, physikalisch (B1)"),
            ("assets/bars_proc_naiv", "prozedural, naiv (B2)")]
    ken = _load("../bars_kennwerte.json") or json.load(
        open(os.path.join(DOC, "bars_kennwerte.json")))
    keys = ["photographisch", "prozedural_physikalisch", "prozedural_naiv"]

    fig = plt.figure(figsize=(13.0, 5.4))
    gs = fig.add_gridspec(3, 3, width_ratios=[2.3, 1.0, 1.0], hspace=0.45,
                          wspace=0.28)
    for i, ((d, titel), key) in enumerate(zip(libs, keys)):
        paths = [os.path.join(d, n + ".png") for n in BAR_BEISPIELE]
        tiles = []
        for p in paths:
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            a = im[:, :, 3:4] / 255.0
            comp = (im[:, :, :3] * a + 105 * (1 - a)).astype(np.uint8)
            comp = comp[:, :190]
            comp = cv2.resize(comp, None, fx=3, fy=3,
                              interpolation=cv2.INTER_NEAREST)
            tiles.append(comp)
        Hm = max(t.shape[0] for t in tiles)
        Wm = max(t.shape[1] for t in tiles)
        sheet = np.full((len(tiles) * (Hm + 10), Wm, 3), 105, np.uint8)
        for k, t in enumerate(tiles):
            sheet[k * (Hm + 10):k * (Hm + 10) + t.shape[0], :t.shape[1]] = t
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titel, fontsize=9, loc="left")

        # Querschnittsprofil
        ax = fig.add_subplot(gs[i, 1])
        prof = np.array(ken[key]["profile_norm_mean"])
        u = np.linspace(-1, 1, len(prof))
        ax.plot(u, prof, color="#333", lw=1.6)
        ax.fill_between(u, 1, prof, color="#4c72b0", alpha=.25)
        ax.axhline(1, color="grey", lw=.6, ls=":")
        ax.set_ylim(0.55, 1.35)
        ax.set_xlabel("Querschnitt", fontsize=8)
        ax.set_ylabel("rel. Helligkeit", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.25)

        # Kennwerte
        ax = fig.add_subplot(gs[i, 2]); ax.axis("off")
        m = ken[key]
        txt = (f"Grauwert      {m['gray_mean'][0]:6.1f}\n"
               f"Korn $\\sigma$        {m['grain_sigma'][1]:6.2f}\n"
               f"Sättigung     {m['saturation_mean']:6.1f}\n"
               f"Längsvar.     {m['longitudinal_rel_std'][0]:6.3f}\n"
               f"Kontrast      {m['kontrast_je_stange'][1]:6.2f}")
        ax.text(0, .95, txt, fontsize=8, family="monospace", va="top")
    fig.suptitle("Die drei Okkluder-Bibliotheken: gleiche Geometrie, gleiche "
                 "Alphamaske, verschiedener Inhalt", fontsize=10)
    out = os.path.join(DOC, "fig_balken.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(out, os.path.join(FIGDIR, "neu_balken.png"))
    print("->", out)


# ---------------------------------------------------------------------------
# Abbildung 2: dieselbe Szene mit den drei Bibliotheken
# ---------------------------------------------------------------------------
def fig_balken_szene(idx=15, crop=(360, 420, 260, 260)):
    ds = [("synth/abl_A1_baseline", "B0 photographisch"),
          ("synth/abl_B1_proz_balken", "B1 prozedural"),
          ("synth/abl_B2_proz_balken_naiv", "B2 naiv")]
    name = f"syn_1_{idx:06d}"
    x, y, w, h = crop
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5))
    for ax, (d, titel) in zip(axes, ds):
        im = cv2.imread(os.path.join(d, "images", name + ".jpg"))
        if im is None:
            ax.axis("off")
            continue
        sub = im[y:y + h, x:x + w]
        sub = cv2.resize(sub, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        # Labels einzeichnen (in allen drei Armen identisch)
        for line in open(os.path.join(d, "labels", name + ".txt")):
            q = line.split()
            cx, cy, bw, bh = (float(v) * 1024 for v in q[1:5])
            x0, y0 = (cx - bw / 2 - x) * 3, (cy - bh / 2 - y) * 3
            ax.add_patch(plt.Rectangle((x0, y0), bw * 3, bh * 3, fill=False,
                                       ec="#00d000", lw=1.0))
        ax.imshow(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
        ax.set_xlim(0, w * 3); ax.set_ylim(h * 3, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titel, fontsize=10)
    fig.tight_layout()
    out = os.path.join(DOC, "fig_balken_szene.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(out, os.path.join(FIGDIR, "neu_balken_szene.png"))
    print("->", out)


# ---------------------------------------------------------------------------
# Abbildungen 3/4: Ergebnisse der beiden Ablationen
# ---------------------------------------------------------------------------
def _fig_ablation(rows, arme, titel, datei, refarm=None):
    metriken = [("mAP50", "mAP@0,5"), ("mAP50_95", "mAP@0,5:0,95"),
                ("best_f1", "bestes $F_1$"), ("count_rel_err", "Zählfehler")]
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5))
    xs = np.arange(len(arme))
    for ax, (key, lab) in zip(axes, metriken):
        for k, split in enumerate(("val", "test")):
            ys = [(_row(rows, a, split) or {}).get(key, np.nan) for a, _ in arme]
            ax.bar(xs + (k - .5) * .36, ys, .34, color=CS[split], label=split,
                   alpha=.9)
            for xx, yy in zip(xs + (k - .5) * .36, ys):
                if not np.isnan(yy):
                    ax.text(xx, yy, f"{yy:.3f}".replace(".", ","), ha="center",
                            va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([l for _, l in arme], fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="y", alpha=.3)
        vals = [v for a, _ in arme for split in ("val", "test")
                for v in [(_row(rows, a, split) or {}).get(key, np.nan)]
                if not np.isnan(v)]
        if vals:
            lo, hi = min(vals), max(vals)
            pad = max(0.02, (hi - lo) * 0.55)
            ax.set_ylim(max(0, lo - pad), hi + pad * 1.5)
    axes[0].legend(fontsize=8)
    fig.suptitle(titel, fontsize=11)
    fig.tight_layout()
    out = os.path.join(DOC, datei)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(out, os.path.join(FIGDIR, "neu_" + datei.replace("fig_", "")))
    print("->", out)


# ---------------------------------------------------------------------------
# Abbildung 5: Inferenzaufloesung
# ---------------------------------------------------------------------------
def fig_imgsz():
    rep = _load("inferenz_sweep.json")
    if not rep:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for split in ("val", "test"):
        r = [x for x in rep["imgsz"] if x["split"] == split
             and isinstance(x["infer"]["imgsz"], int)]
        r.sort(key=lambda x: x["infer"]["imgsz"])
        xs = [x["infer"]["imgsz"] for x in r]
        for k, (key, lab) in enumerate([("mAP50", "mAP@0,5"),
                                        ("mAP50_95", "mAP@0,5:0,95"),
                                        ("best_f1", "bestes $F_1$")]):
            ax[k].plot(xs, [x[key] for x in r], "o-", ms=4, color=CS[split],
                       label=split)
            ax[k].set_xlabel("imgsz (px)", fontsize=9)
            ax[k].set_title(lab, fontsize=10)
            ax[k].grid(alpha=.3)
    for split, nat in (("val", 2464), ("test", 1600)):
        for a in ax:
            a.axvline(nat, color=CS[split], ls=":", alpha=.55)
    ax[0].legend(fontsize=8)
    fig.suptitle("Inferenzauflösung, yolo26s-p2 (gepunktet: nativer Maßstab)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(DOC, "fig_imgsz.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(out, os.path.join(FIGDIR, "neu_imgsz.png"))
    print("->", out)


# ---------------------------------------------------------------------------
# Abbildung 6: Lernkurven
# ---------------------------------------------------------------------------
def fig_training(arme=None, datei="fig_training.png", titel=None):
    alle = arme or (A_ARME + [b for b in B_ARME if b[0] != "A1_baseline"])
    farben = {"A0_keine_okklusion": "#9467bd", "A1_baseline": "#2ca02c",
              "A2_starke_okklusion": "#8c564b", "B1_prozedural": "#ff7f0e",
              "B2_prozedural_naiv": "#17becf", "C0_kontrolle": "#2ca02c",
              "C1_naive_paste": "#d62728"}
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for arm, lab in alle:
        p = os.path.join(RUNS, arm, "results.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        ep = [float(r["epoch"]) for r in rows]
        for k, key in enumerate(["train/box_loss", "metrics/mAP50(B)",
                                 "metrics/mAP50-95(B)"]):
            kk = next((c for c in rows[0] if c.strip() == key), None)
            if kk is None:
                continue
            ax[k].plot(ep, [float(r[kk]) for r in rows], lw=1.4,
                       color=farben.get(arm, None), label=lab)
    for k, t in enumerate(["Trainingsverlust (Box)",
                           "mAP@0,5 auf realem \\code{val}",
                           "mAP@0,5:0,95 auf realem val"]):
        ax[k].set_xlabel("Epoche", fontsize=9)
        ax[k].set_title(t.replace("\\code{", "").replace("}", ""), fontsize=10)
        ax[k].grid(alpha=.3)
    ax[2].legend(fontsize=7)
    fig.suptitle(titel or ("Lernkurven der fünf Ablationsarme (yolo26s-p2, "
                           "3000 Bilder, 30 Epochen)"), fontsize=11)
    fig.tight_layout()
    out = os.path.join(DOC, datei)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(out, os.path.join(FIGDIR, "neu_" + datei.replace("fig_", "")))
    print("->", out)


# ---------------------------------------------------------------------------
# LaTeX-Tabellen
# ---------------------------------------------------------------------------
def tab_ablation(rows, ds, arme, label, caption, datei, extra_spalte):
    """Ergebnistabelle eines Ablationsarms.

    Zweizeiliger Kopf (Split oben, Metrik unten): Eine einzeilige Beschriftung
    mit \\code{val} \\mAPfifty{} je Spalte wird ueber neun Spalten breiter als
    der Satzspiegel.
    """
    n_extra = len(extra_spalte)
    L = [r"\begin{table}[htbp]", r"\centering",
         f"\\caption{caption}", f"\\label{{{label}}}", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l" + "r" * n_extra + r"|rrr|rr@{}}",
         r"\toprule",
         r"\textbf{Arm} & \multicolumn{" + str(n_extra) +
         r"}{c|}{\textbf{Datensatz}} & \multicolumn{3}{c|}{\code{test}} & "
         r"\multicolumn{2}{c}{\code{val}} \\",
         r" & " + " & ".join(h for h, _ in extra_spalte) +
         r" & \mAPfifty{} & $F_1$ & $R$ & \mAPfifty{} & $F_1$ \\",
         r"\midrule"]
    spalten = [("test", "mAP50"), ("test", "best_f1"),
               ("test", "recall_at_best_f1"), ("val", "mAP50"),
               ("val", "best_f1")]
    beste = {}
    for sp, key in spalten:
        vals = [(_row(rows, a, sp) or {}).get(key) for a, _ in arme]
        vals = [v for v in vals if v is not None]
        beste[(sp, key)] = max(vals) if vals else None
    for arm, lab in arme:
        cells = [lab]
        for _, fn in extra_spalte:
            cells.append(fn(ds.get(arm, {})))
        for sp, key in spalten:
            r = _row(rows, arm, sp)
            if r is None:
                cells.append("--")
                continue
            t = _de(r[key], 3)
            if beste[(sp, key)] is not None and abs(r[key] - beste[(sp, key)]) < 1e-12:
                t = r"\textbf{" + t + "}"
            cells.append(t)
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    p = os.path.join(TEXDIR, datei)
    open(p, "w").write("\n".join(L) + "\n")
    print("->", p)


def tabellen():
    rows = _load("ablationen.json")
    ds = _load("datensaetze.json")
    if not rows or not ds:
        print("Ergebnisse fehlen -- Tabellen uebersprungen")
        return
    tab_ablation(
        rows, ds, A_ARME, "tab:ablationA",
        "[Ablation A: Okklusionsanteil]{Ablation~A -- Okklusionsanteil. "
        "Links die erzeugten Datensätze, rechts das Ergebnis bei nativem "
        "Maßstab. Alle Arme: yolo26s-p2, 3000 Bilder, 30 Epochen, gleicher "
        "Seed. Fett der jeweils beste Wert einer Spalte.}",
        "tab_ablation_a.tex",
        [(r"\textbf{Labels}", lambda d: f"{d.get('labels',0):,}".replace(",", r"\,")),
         (r"\textbf{Sichtb.}", lambda d: _de(d.get("mittlere_sichtbarkeit", 0), 3)),
         (r"\textbf{teilverd.}",
          lambda d: _de(100 * d.get("teilverdeckt", 0), 1) + r"\,\%")])
    tab_ablation(
        rows, ds, B_ARME, "tab:ablationB",
        "[Ablation B: Herkunft der Okkluder]{Ablation~B -- Herkunft der "
        "Okkluderbalken. Die drei Datensätze sind bis auf die RGB-Werte "
        "innerhalb der Stangenmasken identisch; Labelzahl, Sichtbarkeit und "
        "Verdeckungsanteil stimmen deshalb exakt überein. Fett der jeweils "
        "beste Wert einer Spalte.}",
        "tab_ablation_b.tex",
        [(r"\textbf{Labels}", lambda d: f"{d.get('labels',0):,}".replace(",", r"\,")),
         (r"\textbf{Sichtb.}", lambda d: _de(d.get("mittlere_sichtbarkeit", 0), 3)),
         (r"\textbf{teilverd.}",
          lambda d: _de(100 * d.get("teilverdeckt", 0), 1) + r"\,\%")])


def fig_einbettung(idx=15, crop=(360, 420, 260, 260)):
    """Je ein Bildausschnitt aus den Armen C0 und C1.

    Ausdruecklich NICHT dieselbe Szene: Anders als bei Ablation B sind die
    beiden Datensaetze nicht gepaart. Sobald eine Einbettungsstufe
    uebersprungen wird, entfallen ihre inneren Ziehungen und der Zufallsstrom
    laeuft auseinander (vgl. den Abschnitt zur Ablation C). Gezeigt wird
    deshalb der typische Bildeindruck beider Arme, nicht ein Paar.

    Labels sind bewusst nicht eingezeichnet -- die Abbildung soll die
    Einbettung zeigen, nicht die Annotation.
    """
    ds = [("synth/C0_kontrolle", "C0 -- vollständige Einbettung"),
          ("synth/C1_naive_paste", "C1 -- naiv eingeklebt")]
    name = f"syn_1_{idx:06d}"
    x, y, w, h = crop
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    for ax, (d, titel) in zip(axes, ds):
        im = cv2.imread(os.path.join(d, "images", name + ".jpg"))
        if im is None:
            ax.axis("off")
            continue
        sub = im[y:y + h, x:x + w]
        sub = cv2.resize(sub, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        ax.imshow(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titel, fontsize=10)
    fig.tight_layout()
    out = os.path.join(DOC, "fig_einbettung.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(out, os.path.join(FIGDIR, "neu_einbettung.png"))
    print("->", out)


def _c_daten():
    """Ergebnisse und Datensatzkennwerte der Arme C0/C1.

    Anders als bei A und B stehen die Kennwerte nicht in datensaetze.json --
    die Arme sind nach jenem Lauf entstanden. Gelesen wird deshalb direkt der
    Erzeugungsbericht, den der Generator neben jeden Datensatz schreibt.
    """
    rows = _load("ablationC.json")
    sval = _load("ablationC_sval.json")
    if not rows:
        return None, None, None
    ds = {}
    for arm, _ in C_ARME:
        p = os.path.join("synth", arm, "generation_report.json")
        if os.path.exists(p):
            ds[arm] = json.load(open(p))
    return rows, sval, ds


def tab_ablation_c():
    """Ergebnistabelle der Ablation C mit Differenzzeile.

    Bei nur zwei Armen ist die Differenz die eigentliche Aussage; sie steht
    deshalb als eigene Zeile und nicht nur im Fliesstext. Die sval-Spalte
    gehoert dazu: Erst sie zeigt, dass der Einbruch auf den Realbildern
    nicht von einem schlechter gelernten Trainingsdatensatz kommt.
    """
    rows, sval, ds = _c_daten()
    if not rows:
        print("Ablation C: Ergebnisse fehlen -- Tabelle uebersprungen")
        return
    def hole(arm, sp, key):
        if sp == "sval":
            r = _row(sval or [], arm, "sval_eigen")
        else:
            r = _row(rows, arm, sp)
        return None if r is None else r[key]

    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption[Ablation: Einbettung der eingefügten Objekte]{Ablation zur"
         r" Einbettung. \code{C0} ist die vollständige Pipeline, \code{C1}"
         r" dieselbe Pipeline ohne die Gegenmaßnahmen gegen die Signaturen"
         r" S1--S5 und ohne Schlagschatten; beide Datensätze ziehen aus"
         r" derselben Layoutverteilung, sind aber \emph{nicht} gepaart."
         r" \code{sval} ist der \emph{jeweils eigene} synthetische"
         r" Validierungssplit des Arms und beantwortet damit die Frage, ob der"
         r" Arm seine eigenen Trainingsdaten gelernt hat."
         r" Die Differenzzeile steht in Prozentpunkten, die Labelspalte"
         r" absolut.}",
         r"\label{tab:ablationC}", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lrr|rrr|rr@{}}", r"\toprule",
         r"\textbf{Arm} & \textbf{Labels} & \code{sval} & "
         r"\multicolumn{3}{c|}{\code{test}} & \multicolumn{2}{c}{\code{val}} \\",
         r" & & \mAPfifty{} & \mAPfifty{} & $F_1$ & $R$ & \mAPfifty{} & $F_1$ \\",
         r"\midrule"]
    reihen = [("sval", "mAP50"), ("test", "mAP50"), ("test", "best_f1"),
              ("test", "recall_at_best_f1"), ("val", "mAP50"), ("val", "best_f1")]
    for arm, lab in C_ARME:
        n = (ds.get(arm) or {}).get("instances_labeled")
        cells = [lab, "--" if n is None else f"{n:,}".replace(",", r"\,")]
        cells += [("--" if (v := hole(arm, sp, k)) is None else _de(v, 3))
                  for sp, k in reihen]
        L.append(" & ".join(cells) + r" \\")
    a, b = C_ARME[0][0], C_ARME[1][0]
    na, nb = ((ds.get(a) or {}).get("instances_labeled"),
              (ds.get(b) or {}).get("instances_labeled"))
    diff = [r"\emph{Differenz}",
            "--" if None in (na, nb) else f"${nb - na}$"]
    for sp, k in reihen:
        va, vb = hole(a, sp, k), hole(b, sp, k)
        diff.append("--" if None in (va, vb) else
                    r"$" + _de(100 * (vb - va), 1).replace("-", "-") + r"$")
    L.append(r"\midrule")
    L.append(" & ".join(diff) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    p = os.path.join(TEXDIR, "tab_ablation_c.tex")
    open(p, "w").write("\n".join(L) + "\n")
    print("->", p)


# ---------------------------------------------------------------------------
# Weitere LaTeX-Tabellen
# ---------------------------------------------------------------------------
def tab_hauptergebnis():
    rows = _load("hauptergebnis.json")
    if not rows:
        return
    nam = {"val": r"\code{val} (real)", "test": r"\code{test} (real)",
           "sval": r"\code{sval} (synth.)"}
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption[Hauptergebnis]{Ergebnis des Referenzmodells "
         r"(\code{yolo26s-p2}, 6000 Bilder, 60 Epochen) bei nativem Maßstab. "
         r"$c^*$ ist die $F_1$-optimale Konfidenzschwelle, $P$ und $R$ sind "
         r"Precision und Rückruf an diesem Betriebspunkt.}",
         r"\label{tab:hauptergebnis}", r"\footnotesize",
         r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"\textbf{Split} & \mAPfifty{} & \mAPseven{} & \mAPall{} & "
         r"\textbf{bestes $F_1$} & $c^*$ & $P$\,/\,$R$ & \textbf{mittl.\ IoU} \\",
         r"\midrule"]
    for s_ in ("val", "test", "sval"):
        r = next((x for x in rows if x["split"] == s_), None)
        if not r:
            continue
        L.append(f"{nam[s_]} & \\textbf{{{_de(r['mAP50'],3)}}} & "
                 f"{_de(r['mAP75'],3)} & {_de(r['mAP50_95'],3)} & "
                 f"{_de(r['best_f1'],3)} & {_de(r['best_f1_conf'],3)} & "
                 f"{_de(r['precision_at_best_f1'],3)}\\,/\\,"
                 f"{_de(r['recall_at_best_f1'],3)} & "
                 f"{_de(r['mean_iou_of_tp'],3)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEXDIR, "tab_hauptergebnis.tex"), "w").write(
        "\n".join(L) + "\n")
    print("-> tab_hauptergebnis.tex")


def _imgsz_rows():
    rep = _load("inferenz_sweep.json")
    if not rep:
        return None, None
    out = {}
    for r in rep["imgsz"]:
        v = r["infer"]["imgsz"]
        if not isinstance(v, int):
            continue
        out.setdefault(v, {})[r["split"]] = r
    return out, rep


def tab_imgsz():
    tab, rep = _imgsz_rows()
    if not tab:
        return
    for datei, label, cap, stufen in [
        ("tab_imgsz.tex", "tab:imgsz",
         r"[Sweep der Inferenzauflösung]{Auszug aus dem Sweep der "
         r"Inferenzauflösung für das Referenzmodell. Nativer Maßstab ist 2464 "
         r"(\code{val}) bzw.\ 1600 (\code{test}); der vollständige Sweep steht "
         r"in Tabelle~\ref{tab:imgszvoll}.}",
         [640, 960, 1280, 1600, 1920, 2464]),
        ("tab_imgsz_voll.tex", "tab:imgszvoll",
         r"[Vollständiger Auflösungssweep]{Vollständiger Sweep der "
         r"Inferenzauflösung für das Referenzmodell \code{yolo26s-p2}.}",
         sorted(tab))]:
        L = [r"\begin{table}[htbp]", r"\centering", f"\\caption{cap}",
             f"\\label{{{label}}}", r"\footnotesize",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"\code{imgsz} & \multicolumn{3}{c}{\code{val}} & "
             r"\multicolumn{3}{c}{\code{test}} \\",
             r" & \mAPfifty{} & \mAPall{} & $R$ & \mAPfifty{} & \mAPall{} & $R$ \\",
             r"\midrule"]
        best = {sp: max((tab[v][sp]["mAP50"] for v in tab if sp in tab[v]))
                for sp in ("val", "test")}
        for v in stufen:
            if v not in tab:
                continue
            c = [str(v)]
            for sp in ("val", "test"):
                r = tab[v].get(sp)
                if not r:
                    c += ["--", "--", "--"]
                    continue
                t = _de(r["mAP50"], 3)
                if abs(r["mAP50"] - best[sp]) < 1e-12:
                    t = r"\textbf{" + t + "}"
                c += [t, _de(r["mAP50_95"], 3), _de(r["recall_at_best_f1"], 3)]
            L.append(" & ".join(c) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(TEXDIR, datei), "w").write("\n".join(L) + "\n")
        print("->", datei)


def tab_sensitivitaet():
    a = _load("analyse.json")
    if not a:
        return
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption[Sensitivität gegenüber dem Größenversatz]{"
         r"Sensitivitätsrechnung: alle vorhergesagten Boxen um den medianen "
         r"Fehlbetrag je Seite aufgeweitet. Das ist \emph{kein} Ergebnis eines "
         r"neuen Trainings, sondern eine Obergrenzenschätzung für den Effekt "
         r"einer angeglichenen Labelkonvention.}",
         r"\label{tab:boxsens}", r"\footnotesize",
         r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"\textbf{Split} & & \mAPfifty{} & \mAPseven{} & \mAPall{} & "
         r"\textbf{bestes $F_1$} \\", r"\midrule"]
    for sp in ("test", "val"):
        d = a["sensitivitaet"].get(sp)
        if not d:
            continue
        for tag, key in (("unverändert", "original"),
                         (f"Boxen +{_de(d['aufweitung_px_je_seite'],2)}\\px{{}} "
                          "je Seite", "aufgeweitet")):
            m = d[key]
            L.append(f"\\code{{{sp}}} & {tag} & {_de(m['mAP50'],3)} & "
                     f"{_de(m['mAP75'],3)} & {_de(m['mAP50_95'],3)} & "
                     f"{_de(m['best_f1'],3)} \\\\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEXDIR, "tab_sensitivitaet.tex"), "w").write(
        "\n".join(L) + "\n")
    print("-> tab_sensitivitaet.tex")


def tab_boxversatz():
    a = _load("analyse.json")
    if not a:
        return
    nam = {"test": r"\code{test} (real)", "val": r"\code{val} (real)",
           "sval": r"\code{sval} (synthetisch)"}
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption[Größenversatz der Boxen]{Vergleich der vorhergesagten mit "
         r"der annotierten Boxgröße. Als Boxgröße dient "
         r"$\sqrt{\text{Fläche}}$, der Zentrumsversatz ist in "
         r"Objektdurchmessern angegeben. \code{sval} ist der synthetische "
         r"Split, dessen Labels der Generator selbst erzeugt hat.}",
         r"\label{tab:boxversatz}", r"\footnotesize",
         r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"\textbf{Split} & \textbf{Paare} & \textbf{Objekt-\Oe{} (Median)} & "
         r"\textbf{pred/GT} & \textbf{GT $-$ pred} & "
         r"\textbf{Zentrumsversatz} \\", r"\midrule"]
    for sp in ("test", "val", "sval"):
        r = next((x for x in a["boxversatz"] if x["split"] == sp), None)
        if not r:
            continue
        v = _de(r["pred_durch_gt"], 3)
        if sp == "sval":
            v = r"\textbf{" + v + "}"
        L.append(f"{nam[sp]} & {r['paare']} & "
                 f"{_de(r['gt_durchmesser_median'],1)}\\px & {v} & "
                 f"${'+' if r['fehlbetrag_px']>=0 else '-'}"
                 f"{_de(abs(r['fehlbetrag_px']),2)}$\\px & "
                 f"{_de(r['zentrumsversatz_in_d'],3)}\\,\\Oe \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEXDIR, "tab_boxversatz.tex"), "w").write(
        "\n".join(L) + "\n")
    print("-> tab_boxversatz.tex")


def qualbilder():
    for sp in ("val", "test"):
        src = os.path.join(DOC, f"qual_{sp}.jpg")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(FIGDIR, f"neu_qual_{sp}.jpg"))
            print("->", f"{FIGDIR}/neu_qual_{sp}.jpg")


# ---------------------------------------------------------------------------
# Zahlen fuer den Fliesstext als LaTeX-Makros
# ---------------------------------------------------------------------------
def zahlen():
    """Schreibt paper/kapitel/zahlen.tex.

    Jede Zahl, die im Fliesstext genannt wird, kommt aus dieser Datei. Damit
    kann der Text nicht auseinanderlaufen, wenn ein Lauf wiederholt wird -- ein
    erneuter Aufruf von report_neu.py setzt alle Werte neu.
    """
    M = {}

    def put(name, val, n=3, pp=False):
        if val is None:
            M[name] = "??"
            return
        M[name] = _de(100 * val, n) if pp else _de(val, n)

    haupt = _load("hauptergebnis.json") or []
    for r in haupt:
        sp = r["split"]
        put(f"ref{sp.capitalize()}Map", r["mAP50"])
        put(f"ref{sp.capitalize()}MapS", r["mAP75"])
        put(f"ref{sp.capitalize()}MapA", r["mAP50_95"])
        put(f"ref{sp.capitalize()}Ff", r["best_f1"])
        put(f"ref{sp.capitalize()}Conf", r["best_f1_conf"])
        put(f"ref{sp.capitalize()}Prec", r["precision_at_best_f1"])
        put(f"ref{sp.capitalize()}Rec", r["recall_at_best_f1"])
        put(f"ref{sp.capitalize()}Iou", r["mean_iou_of_tp"])
    d = {r["split"]: r for r in haupt}
    if {"val", "test", "sval"} <= set(d):
        put("simRealVal", d["sval"]["mAP50"] - d["val"]["mAP50"], 1, pp=True)
        put("simRealTest", d["sval"]["mAP50"] - d["test"]["mAP50"], 1, pp=True)
        put("simRealValA", d["sval"]["mAP50_95"] - d["val"]["mAP50_95"], 1, pp=True)
        put("simRealTestA", d["sval"]["mAP50_95"] - d["test"]["mAP50_95"], 1, pp=True)
        M["epsS"] = _de(1 - d["sval"]["mAP50"], 3)
        M["epsT"] = _de(1 - d["test"]["mAP50"], 3)
        M["epsDiff"] = _de(d["sval"]["mAP50"] - d["test"]["mAP50"], 3)

    tab, rep = _imgsz_rows()
    if tab:
        # LaTeX-Makronamen duerfen keine Ziffern enthalten -- die Stufen
        # werden deshalb benannt, nicht durchnummeriert.
        nativ = {"val": 2464, "test": 1600}
        for sp in ("val", "test"):
            S = sp.capitalize()
            for tag, v in (("Niedrig", 640), ("Nativ", nativ[sp])):
                r = tab.get(v, {}).get(sp)
                if r:
                    put(f"sw{S}{tag}", r["mAP50"])
                    put(f"sw{S}{tag}Rec", r["recall_at_best_f1"])
                    M[f"sw{S}{tag}At"] = str(v)
            cand = [(v, tab[v][sp]) for v in tab if sp in tab[v]]
            bv, br = max(cand, key=lambda t: t[1]["mAP50"])
            M[f"sw{S}BestAt"] = str(bv)
            put(f"sw{S}Best", br["mAP50"])
            put(f"sw{S}BestRec", br["recall_at_best_f1"])
            lo = tab.get(640, {}).get(sp)
            if lo:
                put(f"sw{S}Gain", br["mAP50"] - lo["mAP50"], 1, pp=True)
            # Effektive Objektgroesse beim jeweils besten imgsz: d_eff = d*M/N
            med = {"val": 26.8, "test": 17.0}[sp]
            N = {"val": 2459, "test": 1600}[sp]
            put(f"sw{S}BestDeff", med * bv / N, 1)
            hoch = max(tab)
            r_hoch = tab[hoch].get(sp)
            if r_hoch:
                put(f"sw{S}Hoch", r_hoch["mAP50"])
                M[f"sw{S}HochAt"] = str(hoch)
                put(f"sw{S}HochDeff", med * hoch / N, 1)
        M["bestImgsz"] = str(rep.get("best_imgsz", "??"))
        # Metriken bei der ausgewaehlten Auflaesung. Gebraucht werden sie,
        # weil die Arbeit ihre Zahlen beim nativen Massstab berichtet, die
        # Auswahl aber eine andere Stufe trifft -- der Unterschied gehoert
        # benannt und nicht dem Leser ueberlassen.
        try:
            _sel = int(rep.get("best_imgsz"))
        except (TypeError, ValueError):
            _sel = None
        for sp in ("val", "test"):
            r = tab.get(_sel, {}).get(sp) if _sel else None
            if r:
                put(f"sel{sp.capitalize()}Map", r["mAP50"])
                put(f"sel{sp.capitalize()}MapA", r["mAP50_95"])
                put(f"sel{sp.capitalize()}Rec", r["recall_at_best_f1"])
        for sp in ("val", "test"):
            o = next((x for x in rep["tta"] if x["split"] == sp
                      and x["tag"] == "plain"), None)
            t = next((x for x in rep["tta"] if x["split"] == sp
                      and x["tag"] == "tta"), None)
            if o and t:
                put(f"tta{sp.capitalize()}", t["mAP50"] - o["mAP50"], 1, pp=True)
        for sp, b in (rep.get("bootstrap") or {}).items():
            put(f"boot{sp.capitalize()}Mean", b["mean"])
            put(f"boot{sp.capitalize()}Std", b["std"])
            put(f"boot{sp.capitalize()}Lo", b["ci95"][0])
            put(f"boot{sp.capitalize()}Hi", b["ci95"][1])
            M[f"boot{sp.capitalize()}N"] = str(b["n_cells"])

    rows = _load("ablationen.json")
    ds = _load("datensaetze.json")
    if rows and ds:
        kurz = {"A0_keine_okklusion": "AZero", "A1_baseline": "AOne",
                "A2_starke_okklusion": "ATwo", "B1_prozedural": "BOne",
                "B2_prozedural_naiv": "BTwo"}
        for arm, tag in kurz.items():
            for sp in ("val", "test"):
                r = _row(rows, arm, sp)
                if not r:
                    continue
                put(f"{tag}{sp.capitalize()}Map", r["mAP50"])
                put(f"{tag}{sp.capitalize()}MapA", r["mAP50_95"])
                put(f"{tag}{sp.capitalize()}Ff", r["best_f1"])
                put(f"{tag}{sp.capitalize()}Rec", r["recall_at_best_f1"])
                put(f"{tag}{sp.capitalize()}Prec", r["precision_at_best_f1"])
                M[f"{tag}{sp.capitalize()}Cnt"] = str(r["count_pred"])
            e = (_row(rows, arm, "val") or {}).get("best_epoch") or {}
            M[f"{tag}Epoche"] = str(e.get("epoche", "??"))
            M[f"{tag}Labels"] = f"{ds[arm]['labels']:,}".replace(",", r"\,")
        # Spannen
        def sp_delta(a, b, sp, key):
            ra, rb = _row(rows, a, sp), _row(rows, b, sp)
            return None if not (ra and rb) else rb[key] - ra[key]
        put("ablAValGain", sp_delta("A0_keine_okklusion",
                                    "A2_starke_okklusion", "val", "mAP50"),
            1, pp=True)
        put("ablATestGain", sp_delta("A0_keine_okklusion",
                                     "A2_starke_okklusion", "test", "best_f1"),
            1, pp=True)
        la, lb = ds["A0_keine_okklusion"]["labels"], ds["A2_starke_okklusion"]["labels"]
        put("ablALabelDrop", (la - lb) / la, 1, pp=True)
        put("ablBTestGap", sp_delta("B2_prozedural_naiv", "A1_baseline",
                                    "test", "mAP50"), 1, pp=True)
        put("ablBValGap", sp_delta("B2_prozedural_naiv", "A1_baseline",
                                   "val", "mAP50"), 1, pp=True)
        put("ablBTestGapEins", sp_delta("B1_prozedural", "A1_baseline",
                                        "test", "mAP50"), 1, pp=True)
        put("ablBValGapEins", sp_delta("B1_prozedural", "A1_baseline",
                                       "val", "mAP50"), 1, pp=True)
        put("ablBTestGapFf", sp_delta("B2_prozedural_naiv", "A1_baseline",
                                      "test", "best_f1"), 1, pp=True)
        put("ablBTestGapRec", sp_delta("B2_prozedural_naiv", "A1_baseline",
                                       "test", "recall_at_best_f1"), 1, pp=True)

    # --- Ablation C: Einbettung -------------------------------------------
    crows, csval, cds = _c_daten()
    if crows:
        kurzC = {"C0_kontrolle": "CZero", "C1_naive_paste": "COne"}
        for arm, tag in kurzC.items():
            for sp in ("val", "test"):
                r = _row(crows, arm, sp)
                if not r:
                    continue
                put(f"{tag}{sp.capitalize()}Map", r["mAP50"])
                put(f"{tag}{sp.capitalize()}MapA", r["mAP50_95"])
                put(f"{tag}{sp.capitalize()}Ff", r["best_f1"])
                put(f"{tag}{sp.capitalize()}Rec", r["recall_at_best_f1"])
                put(f"{tag}{sp.capitalize()}Prec", r["precision_at_best_f1"])
                M[f"{tag}{sp.capitalize()}Cnt"] = str(r["count_pred"])
            s = _row(csval or [], arm, "sval_eigen")
            if s:
                put(f"{tag}Sval", s["mAP50"])
                put(f"{tag}SvalFf", s["best_f1"])
                # Sim-to-Real-Abstand gegen den EIGENEN sval des Arms: nur so
                # misst die Groesse die Uebertragung und nicht zusaetzlich den
                # Unterschied der beiden synthetischen Verteilungen.
                for sp in ("val", "test"):
                    r = _row(crows, arm, sp)
                    if r:
                        put(f"gap{tag}{sp.capitalize()}",
                            s["mAP50"] - r["mAP50"], 1, pp=True)
            n = (cds.get(arm) or {}).get("instances_labeled")
            if n is not None:
                M[f"{tag}Labels"] = f"{n:,}".replace(",", r"\,")

        def c_delta(sp, key):
            ra = _row(crows, "C0_kontrolle", sp)
            rb = _row(crows, "C1_naive_paste", sp)
            return None if not (ra and rb) else rb[key] - ra[key]

        put("ablCTestGap", c_delta("test", "mAP50"), 1, pp=True)
        put("ablCValGap", c_delta("val", "mAP50"), 1, pp=True)
        put("ablCTestGapFf", c_delta("test", "best_f1"), 1, pp=True)
        put("ablCValGapFf", c_delta("val", "best_f1"), 1, pp=True)
        put("ablCTestGapRec", c_delta("test", "recall_at_best_f1"), 1, pp=True)
        put("ablCValGapRec", c_delta("val", "recall_at_best_f1"), 1, pp=True)
        sa = _row(csval or [], "C0_kontrolle", "sval_eigen")
        sb = _row(csval or [], "C1_naive_paste", "sval_eigen")
        if sa and sb:
            put("ablCSvalGap", sb["mAP50"] - sa["mAP50"], 1, pp=True)
        na = (cds.get("C0_kontrolle") or {}).get("instances_labeled")
        nb = (cds.get("C1_naive_paste") or {}).get("instances_labeled")
        if na and nb:
            M["ablCLabelDiff"] = str(na - nb)
            put("ablCLabelDiffProz", (na - nb) / na, 1, pp=True)

    a = _load("analyse.json")
    if a:
        for r in a["boxversatz"]:
            sp = r["split"].capitalize()
            put(f"box{sp}Ratio", r["pred_durch_gt"])
            put(f"box{sp}Off", r["fehlbetrag_px"], 2)
            put(f"box{sp}Ctr", r["zentrumsversatz_in_d"])
            M[f"box{sp}Paare"] = str(r["paare"])
            if "pred_durch_gt_unter18px" in r:
                put(f"box{sp}Klein", r["pred_durch_gt_unter18px"])
            if "pred_durch_gt_ueber26px" in r:
                put(f"box{sp}Gross", r["pred_durch_gt_ueber26px"])
        for sp, v in a["sensitivitaet"].items():
            S = sp.capitalize()
            put(f"sens{S}Grow", v["aufweitung_px_je_seite"], 2)
            put(f"sens{S}MapSOrig", v["original"]["mAP75"])
            put(f"sens{S}MapSNeu", v["aufgeweitet"]["mAP75"])
            put(f"sens{S}MapAOrig", v["original"]["mAP50_95"])
            put(f"sens{S}MapANeu", v["aufgeweitet"]["mAP50_95"])
            put(f"sens{S}MapOrig", v["original"]["mAP50"])
            put(f"sens{S}MapNeu", v["aufgeweitet"]["mAP50"])
            put(f"sens{S}Gain", v["aufgeweitet"]["mAP75"] - v["original"]["mAP75"],
                1, pp=True)
            put(f"sens{S}FfOrig", v["original"]["best_f1"])
            put(f"sens{S}FfNeu", v["aufgeweitet"]["best_f1"])
        # Sim-to-Real-Abstand bei mAP@0,5:0,95 NACH Angleich der
        # Boxkonvention. Erst diese Groesse trennt im grossen mAPall-Abstand
        # den Anteil der Labelkonvention von dem, was Verteilungsdivergenz
        # bleibt; der rohe Abstand allein ueberzeichnet die Divergenz.
        _hs = {r["split"]: r for r in (haupt or [])}
        if "sval" in _hs:
            for sp in ("val", "test"):
                v = a["sensitivitaet"].get(sp)
                if v:
                    put(f"simReal{sp.capitalize()}AKorr",
                        _hs["sval"]["mAP50_95"] - v["aufgeweitet"]["mAP50_95"],
                        1, pp=True)
        for r in a["fehler"]:
            S = r["split"].capitalize()
            M[f"fp{S}"] = str(r["falsch_positiv"])
            M[f"fpOhne{S}"] = str(r["davon_ueberlappungsfrei"]
                                  if "davon_ueberlappungsfrei" in r
                                  else r["davon_ohne_ueberlappung"])
            if "davon_doppeldetektion" in r:
                M[f"fpDoppel{S}"] = str(r["davon_doppeldetektion"])
            M[f"fn{S}"] = str(r["falsch_negativ"])
            M[f"nPred{S}"] = str(r["n_pred"])
            if r["fn_durchmesser"]:
                M[f"fnMax{S}"] = _de(max(r["fn_durchmesser"]), 1)
                M[f"fnKlein{S}"] = str(sum(1 for v in r["fn_durchmesser"]
                                           if v < 18))
        for r in a.get("nms", []):
            S = r["split"].capitalize()
            M[f"nms{S}Entfernt"] = str(r["entfernt"])
            M[f"nms{S}BoxenOhne"] = str(r["ohne"]["n"])
            M[f"nms{S}BoxenMit"] = str(r["mit"]["n"])
            M[f"nms{S}Tp"] = str(r["ohne"]["tp"])
            M[f"nms{S}Fn"] = str(r["ohne"]["fn"])
            M[f"nms{S}FpOhne"] = str(r["ohne"]["fp"])
            M[f"nms{S}FpMit"] = str(r["mit"]["fp"])
            put(f"nms{S}FfOhne", r["ohne"]["f1"])
            put(f"nms{S}FfMit", r["mit"]["f1"])
            put(f"nms{S}Gain", r["mit"]["f1"] - r["ohne"]["f1"], 1, pp=True)
            if r["paare"]:
                put(f"nms{S}AbstandMin", min(x["abstand"] for x in r["paare"]))
                put(f"nms{S}IouMin", min(x["iou_paar"] for x in r["paare"]))
                put(f"nms{S}IouMax", max(x["iou_paar"] for x in r["paare"]))
        alleP = [x for r in a.get("nms", []) for x in r["paare"]]
        if alleP:
            M["nmsKopien"] = str(len(alleP))
            put("nmsAbstandMin", min(x["abstand"] for x in alleP))
            put("nmsIouMin", min(x["iou_paar"] for x in alleP))
            put("nmsIouMax", max(x["iou_paar"] for x in alleP))
            n_ges = sum(r["ohne"]["n"] for r in a["nms"])
            M["nmsVorhersagen"] = str(n_ges)
            put("nmsAnteil", len(alleP) / n_ges, 1, pp=True)
        for sp, v in a["crosscheck"].items():
            S = sp.capitalize()
            put(f"xc{S}Eigen", v["eigen_mAP50"])
            put(f"xc{S}Ult", v["ultralytics_mAP50"])
            put(f"xc{S}EigenA", v["eigen_mAP50_95"])
            put(f"xc{S}UltA", v["ultralytics_mAP50_95"])
            put(f"xc{S}Diff", abs(v["eigen_mAP50"] - v["ultralytics_mAP50"]),
                1, pp=True)
        put("xcMaxDiff", max(abs(v[f"eigen_{m}"] - v[f"ultralytics_{m}"])
                             for v in a["crosscheck"].values()
                             for m in ("mAP50", "mAP50_95")), 1, pp=True)
        e = a.get("best_epoch") or {}

    import csv as _csv
    _rp = os.path.join(RUNS, "referenz", "results.csv")
    if os.path.exists(_rp):
        _rows = list(_csv.DictReader(open(_rp)))
        M["refEpochen"] = str(len(_rows))
        # Fitness wie in Ultralytics 8.4.117: allein mAP50-95 (siehe
        # _best_epoch in experiment_neu.py)
        _k95 = next((k for k in _rows[0] if "mAP50-95(B)" in k), None)
        if _k95:
            _fit = [float(r[_k95]) for r in _rows]
            M["refBestEpoche"] = str(int(float(_rows[int(np.argmax(_fit))]["epoch"])))

    gr = _load("ablationB_groesse.json")
    if gr:
        kurz = {"A1_baseline": "BZero", "B1_prozedural": "BOne",
                "B2_prozedural_naiv": "BTwo"}
        for r in gr:
            t = kurz.get(r["arm"])
            if not t:
                continue
            S = r["split"].capitalize()
            put(f"{t}{S}RecKlein", r["rueckruf_klein"])
            put(f"{t}{S}RecGross", r["rueckruf_gross"])
            M[f"{t}{S}Fn"] = str(r["fn_gesamt"])
            M[f"grenze{S}"] = _de(r["schwelle_px"], 1)
            M[f"nKlein{S}"] = str(r["n_klein"])
            M[f"nGross{S}"] = str(r["n_gross"])

    # Arm A1 ist zugleich B0; seine Werte deshalb unter beiden Namen ablegen.
    for k in [k for k in list(M) if k.startswith("AOne")]:
        M["BZero" + k[4:]] = M[k]

    # Im Text verwendete, aber (noch) nicht berechenbare Makros mit '??'
    # belegen, damit die Uebersetzung nicht an einem fehlenden Lauf scheitert
    # und die Luecke im PDF sichtbar wird.
    import re as _re
    benutzt = set()
    for f in sum([glob.glob(os.path.join(d, "*.tex")) for d in TEXDIRS], []):
        if os.path.basename(f) == "zahlen.tex":
            continue
        # Die Namen in M tragen kein fuehrendes z -- das setzt erst
        # die Ausgabe davor. Beim Einlesen also abschneiden.
        benutzt |= {m[1:] for m in
                    _re.findall(r"\\(z[A-Za-z]+)", open(f).read())}
    offen = sorted(benutzt - set(M))
    for k in offen:
        M[k] = r"\textbf{??}"

    L = ["% Automatisch erzeugt von src/report_neu.py -- nicht von Hand",
         "% bearbeiten. Jede im Fliesstext genannte Messzahl steht hier.", ""]
    for k in sorted(M):
        L.append("\\newcommand{\\z%s}{%s}" % (k, M[k]))
    if offen:
        print("   ohne Messwert (im PDF als ??):", ", ".join(offen))
    open(os.path.join(TEXDIR, "zahlen.tex"), "w").write("\n".join(L) + "\n")
    print(f"-> zahlen.tex ({len(M)} Makros)")
    fehlend = [k for k, v in M.items() if v == "??"]
    if fehlend:
        print("   noch unbesetzt:", ", ".join(fehlend))


# ---------------------------------------------------------------------------
# Markdown-Zusammenfassung
# ---------------------------------------------------------------------------
def ergebnisse_md():
    rows = _load("ablationen.json")
    ds = _load("datensaetze.json")
    haupt = _load("hauptergebnis.json")
    L = ["# Ergebnisse — zwei Ablationen der Augmentierung auf `yolo26s-p2`", "",
         "Automatisch erzeugt von `src/report_neu.py`. Alle Zahlen bei",
         "nativem Maßstab, gegen den korrigierten Annotationsstand",
         "(`val` 98, `test` 89 Objekte).", ""]
    if haupt:
        L += ["## Referenzmodell (6000 Bilder, 60 Epochen)", "",
              "| Split | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | bestes F1 | P / R |",
              "|---|---|---|---|---|---|"]
        for r in haupt:
            L.append(f"| `{r['split']}` | {r['mAP50']:.4f} | {r['mAP75']:.4f} | "
                     f"{r['mAP50_95']:.4f} | {r['best_f1']:.4f} | "
                     f"{r['precision_at_best_f1']:.3f} / "
                     f"{r['recall_at_best_f1']:.3f} |")
        L.append("")
    if rows and ds:
        for titel, arme in (("Ablation A — Okklusionsanteil", A_ARME),
                            ("Ablation B — Herkunft der Okkluder", B_ARME)):
            L += [f"## {titel}", "",
                  "| Arm | Labels | Sichtb. | teilverd. | val mAP@0.5 | "
                  "val mAP@0.5:0.95 | test mAP@0.5 | test mAP@0.5:0.95 | "
                  "test F1 | best. Epoche |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for arm, lab in arme:
                d = ds.get(arm, {})
                rv, rt = _row(rows, arm, "val"), _row(rows, arm, "test")
                if not (rv and rt):
                    continue
                e = (rv.get("best_epoch") or {}).get("epoche", "?")
                L.append(f"| {lab} | {d.get('labels','?')} | "
                         f"{d.get('mittlere_sichtbarkeit','?')} | "
                         f"{100*d.get('teilverdeckt',0):.1f} % | "
                         f"{rv['mAP50']:.4f} | {rv['mAP50_95']:.4f} | "
                         f"{rt['mAP50']:.4f} | {rt['mAP50_95']:.4f} | "
                         f"{rt['best_f1']:.4f} | {e} |")
            L.append("")
    a = _load("analyse.json")
    if a:
        L += ["## Boxgrößenversatz", "",
              "| Split | Paare | Objekt-Ø (Median) | pred/GT | GT − pred | "
              "Zentrumsversatz |", "|---|---|---|---|---|---|"]
        for r in a["boxversatz"]:
            L.append(f"| `{r['split']}` | {r['paare']} | "
                     f"{r['gt_durchmesser_median']:.1f} px | "
                     f"{r['pred_durch_gt']:.3f} | {r['fehlbetrag_px']:+.2f} px | "
                     f"{r['zentrumsversatz_in_d']:.3f} Ø |")
        L.append("")
    open(os.path.join(DOC, "ERGEBNISSE.md"), "w").write("\n".join(L) + "\n")
    print("-> docs/ERGEBNISSE.md")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nur-tabellen", action="store_true",
                    help="nur tab_*.tex und zahlen.tex erzeugen; die "
                         "Abbildungen bleiben unveraendert (schnell, und "
                         "braucht die Bilddatensaetze nicht)")
    args = ap.parse_args()
    os.makedirs(DOC, exist_ok=True)
    if not args.nur_tabellen:
        fig_balken()
        fig_balken_szene()
        rows = _load("ablationen.json")
        if rows:
            _fig_ablation(rows, A_ARME,
                          "Ablation A -- Okklusionsanteil in den Trainingsdaten",
                          "fig_ablation_a.png")
            _fig_ablation(rows, B_ARME,
                          "Ablation B -- Herkunft der Okkluderbalken",
                          "fig_ablation_b.png")
        fig_einbettung()
        fig_imgsz()
        fig_training()
        fig_training(B_ARME, "fig_training_b.png",
                     "Lernkurven der drei Arme "
                     "(yolo26s-p2, 3000 Bilder, 30 Epochen)")
        fig_training(C_ARME, "fig_training_c.png",
                     "Lernkurven der beiden Arme "
                     "(yolo26s-p2, 3000 Bilder, 30 Epochen)")
    tabellen()
    tab_ablation_c()
    tab_hauptergebnis()
    tab_imgsz()
    tab_sensitivitaet()
    tab_boxversatz()
    if not args.nur_tabellen:
        qualbilder()
    zahlen()
    ergebnisse_md()
