"""
Stufe 7: Ergebnisaufbereitung - Diagramme und Markdown-Tabellen.

Erzeugt aus den JSON-Ausgaben von src/evaluate.py und src/experiment.py:
    docs/fig_imgsz_sweep.png     Metrik ueber Inferenzaufloesung
    docs/fig_pr_curves.png       PR-Kurven val/test
    docs/fig_training.png        Lernkurven
    docs/fig_ablation.png        Experiment 2
    docs/ERGEBNISSE.md           alle Tabellen

Aufruf:  python src/report.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

EXP = "docs/experiments"
DOC = "docs"
C = {"val": "#1f77b4", "test": "#d62728"}


def fig_imgsz():
    p = os.path.join(EXP, "experiment1_inference.json")
    if not os.path.exists(p):
        return None
    rep = json.load(open(p))
    rows = rep["sweeps"]["imgsz"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    tab = []
    for split in ("val", "test"):
        r = [x for x in rows if x["split"] == split
             and isinstance(x["infer"]["imgsz"], int)]
        r.sort(key=lambda x: x["infer"]["imgsz"])
        xs = [x["infer"]["imgsz"] for x in r]
        for k, (key, lab) in enumerate([("mAP50", "mAP@0.5"),
                                        ("mAP50_95", "mAP@0.5:0.95"),
                                        ("best_f1", "bestes F1")]):
            ax[k].plot(xs, [x[key] for x in r], "o-", color=C[split], label=split)
            ax[k].set_xlabel("imgsz (px)")
            ax[k].set_ylabel(lab)
            ax[k].grid(alpha=.3)
            ax[k].legend()
        for x in r:
            tab.append((split, x["infer"]["imgsz"], x["mAP50"], x["mAP50_95"],
                        x["best_f1"], x["recall_at_best_f1"],
                        x["precision_at_best_f1"], x["mean_iou_of_tp"],
                        x["count_pred"], x["count_gt"], x.get("wall_s", 0)))
    # native-Marker
    for split in ("val", "test"):
        nat = 2464 if split == "val" else 1600
        for a in ax:
            a.axvline(nat, color=C[split], ls=":", alpha=.5)
    ax[0].set_title("Aufloesungsabhaengigkeit (gepunktet: nativer Massstab)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOC, "fig_imgsz_sweep.png"), dpi=130)
    plt.close(fig)
    return tab


def fig_pr(weights_json="docs/experiments/final_eval.json"):
    if not os.path.exists(weights_json):
        return
    d = json.load(open(weights_json))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for r in d["results"]:
        pr = r.get("_pr")
        if not pr:
            continue
        s = r["split"]
        ax[0].plot(pr["recall"], pr["precision"], color=C.get(s, "k"),
                   label=f"{s}  AP50={r['mAP50']:.3f}")
        ax[1].plot(pr["conf"], pr["f1"], color=C.get(s, "k"),
                   label=f"{s}  max F1={r['best_f1']:.3f} @ {r['best_f1_conf']:.3f}")
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")
    ax[0].set_title("PR-Kurve (IoU 0.5)")
    ax[1].set_xlabel("Konfidenzschwelle")
    ax[1].set_ylabel("F1")
    ax[1].set_title("F1 ueber Konfidenz")
    for a in ax:
        a.grid(alpha=.3)
        a.legend()
        a.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(DOC, "fig_pr_curves.png"), dpi=130)
    plt.close(fig)


def fig_training(run="runs/baseline"):
    p = os.path.join(run, "results.csv")
    if not os.path.exists(p):
        return
    rows = [l.strip().split(",") for l in open(p) if l.strip()]
    hdr, data = rows[0], np.array(rows[1:], dtype=object)
    idx = {h.strip(): i for i, h in enumerate(hdr)}
    ep = data[:, idx["epoch"]].astype(float)

    def col(n):
        return data[:, idx[n]].astype(float)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for n, lab in [("train/box_loss", "box"), ("train/cls_loss", "cls"),
                   ("train/dfl_loss", "dfl")]:
        ax[0].plot(ep, col(n), label="train " + lab)
    for n, lab in [("val/box_loss", "box"), ("val/cls_loss", "cls")]:
        if n in idx:
            ax[0].plot(ep, col(n), "--", label="val " + lab)
    ax[0].set_title("Verluste")
    ax[1].plot(ep, col("metrics/mAP50(B)"), label="mAP50")
    ax[1].plot(ep, col("metrics/mAP50-95(B)"), label="mAP50-95")
    ax[1].set_title("Metriken auf dem realen val-Bild")
    ax[2].plot(ep, col("metrics/precision(B)"), label="Precision")
    ax[2].plot(ep, col("metrics/recall(B)"), label="Recall")
    ax[2].set_title("Precision / Recall")
    for a in ax:
        a.set_xlabel("Epoche")
        a.grid(alpha=.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(DOC, "fig_training.png"), dpi=130)
    plt.close(fig)


def fig_ablation():
    p = os.path.join(EXP, "experiment2_ablation.json")
    q = os.path.join(EXP, "experiment2_datasets.json")
    if not (os.path.exists(p) and os.path.exists(q)):
        return None
    rows = json.load(open(p))
    ds = json.load(open(q))
    arms = [a for a in ds]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(arms))
    for k, (key, lab) in enumerate([("mAP50", "mAP@0.5"),
                                    ("mAP50_95", "mAP@0.5:0.95"),
                                    ("recall_at_best_f1", "Recall @ bestes F1")]):
        for j, split in enumerate(("val", "test")):
            y = [next((r[key] for r in rows
                       if r["arm"] == a and r["split"] == split), np.nan)
                 for a in arms]
            b = ax[k].bar(x + (j - .5) * .35, y, .33, color=C[split], label=split)
            # Zahlenwerte an die Balken: die Achse beginnt bewusst bei 0
            # (keine abgeschnittene Skala), dadurch sind die Unterschiede
            # optisch klein - die Beschriftung macht sie trotzdem ablesbar.
            ax[k].bar_label(b, fmt="%.3f", fontsize=8, padding=2)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels([f"{a.split('_')[0]}\nSicht "
                               f"{ds[a]['sichtbarkeit']:.2f}" for a in arms])
        ax[k].set_ylabel(lab)
        ax[k].grid(alpha=.3, axis="y")
        ax[k].set_ylim(0, 1.05 * max(1e-6, np.nanmax(
            [r[key] for r in rows if key in r])) * 1.12)
        ax[k].legend(loc="lower right", fontsize=8)
    ax[0].set_title("Experiment 2: Okklusionsanteil in den Trainingsdaten")
    fig.tight_layout()
    fig.savefig(os.path.join(DOC, "fig_ablation.png"), dpi=130)
    plt.close(fig)
    return rows, ds


def fig_ablation_examples():
    """Dasselbe Bild (gleicher Seed) aus allen drei Ablationsarmen."""
    import cv2
    arms = [("A0_keine_okklusion", "A0  ohne Okklusions-Augmentierung"),
            ("A1_baseline", "A1  Baseline"),
            ("A2_starke_okklusion", "A2  starke Okklusion")]
    if not all(os.path.isdir(f"synth/abl_{a}/labels") for a, _ in arms):
        return
    idx = 3
    for i in range(80):
        try:
            n = [sum(1 for l in open(f"synth/abl_{a}/labels/syn_1_{i:06d}.txt")
                     if l.strip()) for a, _ in arms]
        except FileNotFoundError:
            continue
        if min(n) > 25:
            idx = i
            break
    tiles = []
    for a, title in arms:
        im = cv2.imread(f"synth/abl_{a}/images/syn_1_{idx:06d}.jpg")
        n = 0
        for l in open(f"synth/abl_{a}/labels/syn_1_{idx:06d}.txt"):
            q = l.split()
            if len(q) != 5:
                continue
            cx, cy, w, h = [float(v) * 1024 for v in q[1:]]
            cv2.rectangle(im, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)), (0, 0, 255), 2)
            n += 1
        cv2.rectangle(im, (0, 0), (1024, 46), (0, 0, 0), -1)
        cv2.putText(im, f"{title}   {n} Labels", (14, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (680, 680)))
    cv2.imwrite(os.path.join(DOC, "fig_ablation_beispiele.png"),
                np.hstack(tiles))


def fig_models():
    """Experiment 3: Modellfamilien-Vergleich."""
    p = os.path.join(EXP, "experiment3_models.json")
    q = os.path.join(EXP, "experiment3_models_imgsz.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    names = []
    for r in rows:
        if r["model"] not in names:
            names.append(r["model"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(names))
    for k, (key, lab) in enumerate([("mAP50", "mAP@0.5"),
                                    ("mAP75", "mAP@0.75"),
                                    ("mAP50_95", "mAP@0.5:0.95")]):
        for j, split in enumerate(("val", "test")):
            y = [next((r[key] for r in rows
                       if r["model"] == n and r["split"] == split), np.nan)
                 for n in names]
            b = ax[k].bar(x + (j - .5) * .35, y, .33, color=C[split], label=split)
            ax[k].bar_label(b, fmt="%.3f", fontsize=8, padding=2)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(names, fontsize=9)
        ax[k].set_ylabel(lab)
        ax[k].grid(alpha=.3, axis="y")
        ax[k].set_ylim(0, 1.15 * max(
            [r[key] for r in rows if key in r]))
        ax[k].legend(loc="lower right", fontsize=8)
    ax[0].set_title("Experiment 3: Modellfamilie (nativer Massstab, "
                    "identisches Budget)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOC, "fig_models.png"), dpi=130)
    plt.close(fig)
    return rows, (json.load(open(q)) if os.path.exists(q) else None)


def write_inference_doc():
    """docs/INFERENZ.md aus evaluate.INFER_SPEC + gemessenen Sweeps."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate import INFER_SPEC

    p = os.path.join(EXP, "experiment1_inference.json")
    rep = json.load(open(p)) if os.path.exists(p) else {}
    best = rep.get("best_imgsz", {})

    empf = {
        "imgsz": ("nativ (val 2464, test 1600)"
                  if not best else
                  ", ".join(f"{k}: {v}" for k, v in best.items())),
        "conf": "0.001 fuer mAP; Betriebspunkt siehe conf* in ERGEBNISSE.md",
        "iou": "0.7",
        "max_det": "1000",
        "augment": "False",
        "quantize": "None (FP32)",
        "rect": "True (implizit)",
        "device": "0",
        "batch": "1 (Einzelbildinferenz)",
    }
    lines = ["# Inferenzparameter", "",
             "Alle Werte, mit denen `src/evaluate.py` und `src/experiment.py` "
             "arbeiten. Die empfohlenen Werte sind durch die Sweeps in "
             "Experiment 1 belegt (siehe `docs/ERGEBNISSE.md`).", "",
             "| Parameter | empfohlener Wert | Begruendung |", "|---|---|---|"]
    for k, why in INFER_SPEC.items():
        lines.append(f"| `{k}` | `{empf.get(k, '-')}` | {why} |")
    lines += ["| `device` | `0` | GPU 0, wie gefordert. |",
              "| `batch` | `1` | Einzelbildinferenz; die Bilder haben "
              "unterschiedliche Seitenverhaeltnisse, ein groesserer Batch "
              "wuerde zusaetzliches Padding erzwingen. |", ""]
    lines += ["## Nachbearbeitung", "",
              "Nach der NMS wird nichts weiter gefiltert - keine "
              "Mindestgroesse, keine Randunterdrueckung, keine "
              "Cluster-Nachbearbeitung. Das ist Absicht: jede solche Regel "
              "waere ein an val angepasster Freiheitsgrad und wuerde die "
              "Aussagekraft der test-Zahlen schwaechen.", ""]
    with open(os.path.join(DOC, "INFERENZ.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("-> docs/INFERENZ.md")


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main():
    os.makedirs(DOC, exist_ok=True)
    write_inference_doc()
    parts = ["# Ergebnisse", "",
             "Automatisch aus `docs/experiments/*.json` erzeugt "
             "(`python src/report.py`). Die Interpretation, Fehleranalyse und "
             "die Einschraenkungen stehen in [DISKUSSION.md](DISKUSSION.md); "
             "die Inferenzparameter in [INFERENZ.md](INFERENZ.md).", ""]

    fig_training()
    if os.path.exists("runs/baseline/results.csv"):
        parts += ["## Trainingsverlauf", "", "![](fig_training.png)", ""]

    fp = "docs/experiments/final_eval.json"
    if os.path.exists(fp):
        d = json.load(open(fp))
        rows = []
        for r in d["results"]:
            rows.append((r["split"], r["infer"]["imgsz"],
                         f"{r['mAP50']:.4f}", f"{r['mAP75']:.4f}",
                         f"{r['mAP50_95']:.4f}", f"{r['best_f1']:.4f}",
                         f"{r['best_f1_conf']:.3f}",
                         f"{r['precision_at_best_f1']:.3f}",
                         f"{r['recall_at_best_f1']:.3f}",
                         f"{r['mean_iou_of_tp']:.3f}",
                         f"{r['count_pred']}/{r['count_gt']}"))
        parts += ["## Endergebnis (bestes Modell, nativer Massstab)", "",
                  md_table(["Split", "imgsz", "mAP50", "mAP75", "mAP50-95",
                            "bestes F1", "conf*", "P", "R", "mittl. IoU (TP)",
                            "Zaehlung"], rows), "",
                  "![](fig_pr_curves.png)", "",
                  "### Qualitativ (gruen = TP, rot = FP, blau = FN)", "",
                  "![](qual_val_PXL_20251204_073853552.jpg)", "",
                  "![](qual_test_test5.jpg)", ""]
        fig_pr(fp)

    t = fig_imgsz()
    if t:
        parts += ["## Experiment 1a - Inferenzaufloesung", "",
                  "![](fig_imgsz_sweep.png)", "",
                  md_table(["Split", "imgsz", "mAP50", "mAP50-95", "bestes F1",
                            "R", "P", "mittl. IoU", "pred", "gt", "s"],
                           [(a, b, f"{c:.4f}", f"{d_:.4f}", f"{e:.4f}",
                             f"{f:.3f}", f"{g:.3f}", f"{h:.3f}", i, j,
                             f"{k:.2f}")
                            for a, b, c, d_, e, f, g, h, i, j, k in t]), ""]
        rep = json.load(open(os.path.join(EXP, "experiment1_inference.json")))
        for tag, title in [("iou", "Experiment 1b - NMS-IoU"),
                           ("max_det", "Experiment 1c - max_det"),
                           ("tta_half", "Experiment 1d - TTA und FP16")]:
            if tag not in rep["sweeps"]:
                continue
            rows = [(r.get("tag", ""), r["split"], r["infer"].get("imgsz"),
                     r["infer"].get(tag if tag in ("iou", "max_det") else "augment"),
                     f"{r['mAP50']:.4f}", f"{r['mAP50_95']:.4f}",
                     f"{r['best_f1']:.4f}", f"{r.get('wall_s', 0):.2f}")
                    for r in rep["sweeps"][tag]]
            parts += [f"## {title}", "",
                      md_table(["Variante", "Split", "imgsz", "Wert", "mAP50",
                                "mAP50-95", "bestes F1", "s"], rows), ""]
        if "bootstrap" in rep:
            rows = [(s, f"{b['mean']:.4f}", f"{b['std']:.4f}",
                     f"[{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]", b["n_cells"])
                    for s, b in rep["bootstrap"].items()]
            parts += ["## Bootstrap-Konfidenzintervall (mAP50)", "",
                      "Resampling raeumlicher Kacheln, 400 Wiederholungen. "
                      "Notwendig, weil je Split nur ein Bild vorliegt.", "",
                      md_table(["Split", "Mittel", "Std", "95 %-CI",
                                "Kacheln"], rows), ""]

    mo = fig_models()
    if mo:
        rows, byimg = mo
        parts += ["## Experiment 3 - Modellfamilie", "",
                  "![](fig_models.png)", "",
                  "### Nativer Massstab", "",
                  md_table(["Modell", "Split", "Par. (M)", "mAP50", "mAP75",
                            "mAP50-95", "bestes F1", "P", "R", "Zaehlung"],
                           [(r["model"], r["split"], r["params_M"],
                             f"{r['mAP50']:.4f}", f"{r['mAP75']:.4f}",
                             f"{r['mAP50_95']:.4f}", f"{r['best_f1']:.4f}",
                             f"{r['precision_at_best_f1']:.3f}",
                             f"{r['recall_at_best_f1']:.3f}",
                             f"{r['count_pred']}/{r['count_gt']}")
                            for r in rows]), ""]
        if byimg:
            parts += ["### Bei der je Modell auf **val** gewaehlten imgsz", "",
                      md_table(["Modell", "imgsz", "val mAP50", "val mAP50-95",
                                "test mAP50", "test mAP50-95", "test F1",
                                "test Zaehlung"],
                               [(n, v["best_imgsz"],
                                 f"{v['val']['mAP50']:.4f}",
                                 f"{v['val']['mAP50_95']:.4f}",
                                 f"{v['test']['mAP50']:.4f}",
                                 f"{v['test']['mAP50_95']:.4f}",
                                 f"{v['test']['best_f1']:.4f}",
                                 f"{v['test']['count_pred']}/"
                                 f"{v['test']['count_gt']}")
                                for n, v in byimg.items()]), ""]

    ab = fig_ablation()
    if ab:
        rows, ds = ab
        fig_ablation_examples()
        parts += ["## Experiment 2 - Okklusionsanteil", "",
                  "Dasselbe Bild (identischer Seed) aus allen drei Armen:", "",
                  "![](fig_ablation_beispiele.png)", "",
                  "![](fig_ablation.png)", "",
                  "### Erzeugte Datensaetze", "",
                  md_table(["Arm", "Objekte", "mittl. Sichtbarkeit",
                            "teilverdeckt", "Labels wegen Verdeckung verworfen",
                            "Bilder mit Stangen"],
                           [(a, v["objekte"], v["sichtbarkeit"],
                             v["teilverdeckt"], v["verworfen_verdeckt"],
                             v["bilder_mit_stangen"]) for a, v in ds.items()]),
                  "", "### Ergebnis", "",
                  md_table(["Arm", "Split", "mAP50", "mAP50-95", "bestes F1",
                            "P", "R", "Zaehlung"],
                           [(r["arm"], r["split"], f"{r['mAP50']:.4f}",
                             f"{r['mAP50_95']:.4f}", f"{r['best_f1']:.4f}",
                             f"{r['precision_at_best_f1']:.3f}",
                             f"{r['recall_at_best_f1']:.3f}",
                             f"{r['count_pred']}/{r['count_gt']}")
                            for r in rows]), ""]

    with open(os.path.join(DOC, "ERGEBNISSE.md"), "w") as f:
        f.write("\n".join(parts) + "\n")
    print("-> docs/ERGEBNISSE.md")


if __name__ == "__main__":
    main()
