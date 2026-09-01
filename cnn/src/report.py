"""
Ergebnisaufbereitung fuer PipeNet: Diagramme und cnn/docs/ERGEBNISSE.md.

Aufruf:  python cnn/src/report.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DOC = os.path.join(ROOT, "cnn/docs")
EXP = os.path.join(DOC, "experiments")
# Der Lauf mit der besseren Fitness auf VAL (siehe compare.py)
def _run_dir():
    best, sel = -1, "pipenet"
    for n in ("pipenet", "pipenet_e30"):
        p = os.path.join(ROOT, "cnn/runs", n, "summary.json")
        if os.path.exists(p):
            f = json.load(open(p))["best_fitness"]
            if f > best:
                best, sel = f, n
    return os.path.join(ROOT, "cnn/runs", sel), sel


RUN, RUN_NAME = _run_dir()
C = {"val": "#1f77b4", "test": "#d62728"}


def md(head, rows):
    out = ["| " + " | ".join(map(str, head)) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)


def fig_training():
    p = os.path.join(RUN, "results.csv")
    if not os.path.exists(p):
        return None
    import csv
    rows = list(csv.DictReader(open(p)))
    ep = np.array([float(r["epoch"]) for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k in ("loss", "loss_hm", "loss_wh", "loss_off"):
        ax[0].plot(ep, [float(r[k]) for r in rows], label=k)
    ax[0].set_yscale("log")
    ax[0].set_title("Verluste (log)")
    ax[1].plot(ep, [float(r["mAP50"]) for r in rows], label="mAP50")
    ax[1].plot(ep, [float(r["mAP50_95"]) for r in rows], label="mAP50-95")
    ax[1].set_title("Metriken auf dem realen val-Bild")
    ax[2].plot(ep, [float(r["P"]) for r in rows], label="Precision")
    ax[2].plot(ep, [float(r["R"]) for r in rows], label="Recall")
    ax[2].plot(ep, [float(r["F1"]) for r in rows], label="F1")
    ax[2].set_title("Betriebspunkt (F1-optimal)")
    for a in ax:
        a.set_xlabel("Epoche")
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(DOC, "fig_training.png"), dpi=130)
    plt.close(fig)
    return rows


def fig_scale():
    p = os.path.join(EXP, "pipenet_eval.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    sw = d.get("sweep_scale") or []
    if not sw:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    tab = []
    for sp in ("val", "test"):
        r = sorted([x for x in sw if x["split"] == sp],
                   key=lambda x: x["infer"]["scale"])
        xs = [x["infer"]["scale"] for x in r]
        for k, (key, lab) in enumerate([("mAP50", "mAP@0.5"),
                                        ("mAP50_95", "mAP@0.5:0.95"),
                                        ("best_f1", "bestes F1")]):
            ax[k].plot(xs, [x[key] for x in r], "o-", color=C[sp], label=sp)
            ax[k].set_xlabel("Eingangsskalierung")
            ax[k].set_ylabel(lab)
            ax[k].grid(alpha=.3)
            ax[k].legend()
        med = 27.4 if sp == "val" else 19.6
        for x in r:
            tab.append((sp, x["infer"]["scale"],
                        round(med * x["infer"]["scale"], 1),
                        f"{x['mAP50']:.4f}", f"{x['mAP75']:.4f}",
                        f"{x['mAP50_95']:.4f}", f"{x['best_f1']:.4f}",
                        f"{x['recall_at_best_f1']:.3f}",
                        f"{x['count_pred']}/{x['count_gt']}"))
    ax[0].axvline(22.5 / 27.4, color=C["val"], ls=":", alpha=.6)
    ax[0].axvline(22.5 / 19.6, color=C["test"], ls=":", alpha=.6)
    ax[0].set_title("Eingangsskalierung (gepunktet: Objekte auf 22.5 px, "
                    "den Trainingsmedian)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOC, "fig_scale_sweep.png"), dpi=130)
    plt.close(fig)
    return tab, d


def fig_compare():
    p = os.path.join(EXP, "vergleich.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    names = [x["model"] for x in d]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    x = np.arange(len(names))
    keys = [("mAP50", "mAP@0.5"), ("mAP75", "mAP@0.75"),
            ("mAP50_95", "mAP@0.5:0.95"), ("best_f1", "bestes F1")]
    for k, (key, lab) in enumerate(keys):
        for j, sp in enumerate(("val", "test")):
            y = [m[sp][key] for m in d]
            b = ax[k].bar(x + (j - .5) * .35, y, .33, color=C[sp], label=sp)
            ax[k].bar_label(b, fmt="%.3f", fontsize=7, padding=2)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels([f"{n}\n{m['params_M']} M"
                               for n, m in zip(names, d)], fontsize=8)
        ax[k].set_ylabel(lab)
        ax[k].grid(alpha=.3, axis="y")
        ax[k].set_ylim(0, 1.18 * max(m[sp][key] for m in d for sp in
                                     ("val", "test")))
        ax[k].legend(loc="lower right", fontsize=7)
    ax[0].set_title("PipeNet gegen YOLO - gleicher Datensatz, gleiches Budget, "
                    "gleiche Metrik")
    fig.tight_layout()
    fig.savefig(os.path.join(DOC, "fig_vergleich.png"), dpi=130)
    plt.close(fig)
    return d


def main():
    os.makedirs(EXP, exist_ok=True)
    P = ["# PipeNet - Ergebnisse", "",
         "Automatisch aus `cnn/docs/experiments/*.json` erzeugt "
         "(`python cnn/src/report.py`). Entwurfsbegruendung in "
         "[ARCHITEKTUR.md](ARCHITEKTUR.md).", "",
         f"Ausgewerteter Lauf: `{RUN_NAME}` (der von zwei Laeufen mit der "
         "besseren Fitness auf val).", ""]

    rows = fig_training()
    if rows:
        summ = {}
        sp = os.path.join(RUN, "summary.json")
        if os.path.exists(sp):
            summ = json.load(open(sp))
        best = max(rows, key=lambda r: float(r["fitness"]))
        P += ["## Trainingsverlauf", "", "![](fig_training.png)", "",
              md(["Kenngroesse", "Wert"],
                 [("Parameter", f"{summ.get('n_params', 0) / 1e6:.2f} M"),
                  ("Epochen gelaufen", summ.get("epochs_run", len(rows))),
                  ("bestes Modell", f"Epoche {summ.get('best_epoch')}"),
                  ("Laufzeit", f"{summ.get('hours', 0):.2f} h"),
                  ("val mAP@0.5 (bestes)", f"{float(best['mAP50']):.4f}"),
                  ("val mAP@0.5:0.95 (bestes)",
                   f"{float(best['mAP50_95']):.4f}")]), ""]

    sc = fig_scale()
    if sc:
        tab, d = sc
        P += ["## Eingangsskalierung", "",
              "PipeNet ist vollstaendig konvolutional - es gibt keine feste "
              "Netzeingangsgroesse. Skaliert wird nur, um die Objektgroesse in "
              "die Mitte der Trainingsverteilung (Median 22.5 px) zu "
              "verschieben.", "", "![](fig_scale_sweep.png)", "",
              md(["Split", "scale", "Objekt (px)", "mAP50", "mAP75",
                  "mAP50-95", "bestes F1", "R", "Zaehlung"], tab), ""]
        if "results" in d:
            P += ["### Endauswertung (mit und ohne Rotations-TTA)", "",
                  md(["Split", "TTA", "mAP50", "mAP75", "mAP50-95",
                      "bestes F1", "conf*", "P", "R", "mittl. IoU",
                      "Zaehlung", "s"],
                     [(r["split"], r["infer"]["tta"],
                       f"{r['mAP50']:.4f}", f"{r['mAP75']:.4f}",
                       f"{r['mAP50_95']:.4f}", f"{r['best_f1']:.4f}",
                       f"{r['best_f1_conf']:.3f}",
                       f"{r['precision_at_best_f1']:.3f}",
                       f"{r['recall_at_best_f1']:.3f}",
                       f"{r['mean_iou_of_tp']:.3f}",
                       f"{r['count_pred']}/{r['count_gt']}",
                       f"{r['wall_s']:.2f}") for r in d["results"]]), ""]
        if "equivariance" in d:
            P += ["### Aequivarianz-Selbsttest der Rotations-TTA", "",
                  "Bild drehen, durch das Netz schicken, Heatmap "
                  "zurueckdrehen, mit dem untransformierten Durchlauf "
                  "korrelieren. Ein Vorzeichen- oder Achsenfehler im Code "
                  "ergaebe eine Korrelation um 0.", "",
                  md(["Transformation", "Korrelation"],
                     [(k, f"{v:+.4f}") for k, v in
                      d["equivariance"].items()]), ""]

    cmp = fig_compare()
    if cmp:
        P += ["## Vergleich mit den YOLO-Modellen", "",
              "Gleicher synthetischer Datensatz, gleiches Trainingsbudget, "
              "gleiche Metrikimplementierung. Die Betriebseinstellung ist je "
              "Modell **ausschliesslich auf val** bestimmt und dann "
              "unveraendert auf test angewandt.", "",
              "![](fig_vergleich.png)", "",
              md(["Modell", "Familie", "Par. (M)", "Einstellung",
                  "Filterung", "Objekt val (px)", "Objekt test (px)"],
                 [(m["model"], m["family"], m["params_M"], m["setting"],
                   m["nms"], m["obj_px_val"], m["obj_px_test"])
                  for m in cmp]), "",
              md(["Modell", "Split", "mAP50", "mAP75", "mAP50-95",
                  "bestes F1", "P", "R", "mittl. IoU", "Zaehlung"],
                 [(m["model"], sp, f"{m[sp]['mAP50']:.4f}",
                   f"{m[sp]['mAP75']:.4f}", f"{m[sp]['mAP50_95']:.4f}",
                   f"{m[sp]['best_f1']:.4f}",
                   f"{m[sp]['precision_at_best_f1']:.3f}",
                   f"{m[sp]['recall_at_best_f1']:.3f}",
                   f"{m[sp]['mean_iou_of_tp']:.3f}",
                   f"{m[sp]['count_pred']}/{m[sp]['count_gt']}")
                  for m in cmp for sp in ("val", "test")]), ""]

    open(os.path.join(DOC, "ERGEBNISSE.md"), "w").write("\n".join(P) + "\n")
    print("-> cnn/docs/ERGEBNISSE.md")


if __name__ == "__main__":
    main()
