"""
Training der Ablationsarme auf yolo26s-p2.

Uebernimmt die vollstaendige Parametertabelle SPEC aus src/train.py -- also
Optimierer, Lernratenplan, Verlustgewichte und die gesamte
Trainingszeit-Augmentierung samt Begruendungen -- und setzt nur das, was fuer
diese Studie anders ist:

  model        yolo26s-p2.yaml + Gewichte aus yolo26s.pt   (statt yolo11s.pt)
  epochs       30 je Arm                                    (statt 60)
  cache        per Schalter, Vorgabe 'disk'                 (statt 'ram')
  batch        8                                            (statt 16)
  project      runs                                     (statt runs)

Zur Begruendung der drei Abweichungen siehe README_ablationen.md; sie gelten fuer
ALLE Arme gleich und koennen den Vergleich zwischen ihnen deshalb nicht
verzerren.

Aufruf:
    python src/train_neu.py --name A1_baseline \
        --data data/A1_baseline.yaml --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")

import cv2                                              # noqa: E402
cv2.setNumThreads(0)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import train as T                                       # noqa: E402

from ultralytics import YOLO                            # noqa: E402

PROJECT = os.path.join(ROOT, "neu", "runs")

# Abweichungen gegenueber src/train.py -- Wert und Begruendung, damit sie in
# parameter_begruendung.md des jeweiligen Laufs mitgeschrieben werden.
ABWEICHUNG = {
    "model": ("yolo26s-p2.yaml",
              "Alle Arme beider Ablationen laufen auf derselben Architektur: "
              "YOLO26s mit zusaetzlichem P2-Kopf (Strides 4/8/16/32), Gewichte "
              "partiell aus dem COCO-Checkpoint yolo26s.pt uebernommen, der "
              "P2-Kopf zufaellig initialisiert. Begruendung: Der Modellvergleich "
              "bei identischem Budget hatte yolo26s-p2 als beste Variante "
              "ausgewiesen; eine Ablation der Datenerzeugung soll auf dem "
              "Modell laufen, das man auch einsetzen wuerde, damit der gemessene "
              "Effekt nicht bloss eine Schwaeche eines unterlegenen Detektors "
              "kompensiert."),
    "epochs": (30,
               "3000 Bilder / Batch 8 = 375 Iterationen je Epoche. 30 Epochen "
               "sind das Budget, das allen Armen gleichermassen zugeteilt wird. "
               "Der Wert ist aus dem Zeitbudget gewaehlt, nicht aus einem "
               "Konvergenzkriterium; er ist damit eine Einschraenkung der "
               "Aussage (siehe README_ablationen.md), aber keine Verzerrung zwischen "
               "den Armen."),
    "cache": ("disk",
              "ERZWUNGEN, nicht gewaehlt: cache='ram' braucht 3000 x 1024 x "
              "1024 x 3 B = 9.4 GB; auf dieser Maschine sind zur Laufzeit unter "
              "2 GB frei (der Host ist durch Fremdprozesse belegt). cache='disk' "
              "legt die dekodierten Bilder als .npy ab und erspart dem "
              "Hauptprozess - workers=0 ist wegen des DataLoader-Deadlocks "
              "unveraendert erzwungen - die JPEG-Dekodierung, ohne "
              "Arbeitsspeicher zu belegen."),
    "batch": (8,
              "ERZWUNGEN, nicht gewaehlt: yolo26s-p2 haelt bei imgsz=1024 die "
              "P2-Ebene mit 256x256 Zellen vor und belegt bei Batch 16 rund "
              "23.5 GB. Auf dieser Karte sind durch einen Fremdprozess nur "
              "etwa 25 GB frei; ein erster Lauf mit Batch 16 lief zwar, wich "
              "aber wiederholt mit 'CUDA OutOfMemoryError in "
              "TaskAlignedAssigner, using CPU' auf die CPU aus. Das ist nicht "
              "nur langsam, sondern macht Laeufe untereinander unvergleichbar, "
              "weil der Ausweichpfad von der momentanen Fremdlast abhaengt. "
              "Batch 8 belegt rund 13 GB und laeuft durchgaengig auf der GPU. "
              "Der Wert gilt fuer alle fuenf Arme gleich."),
    "project": (PROJECT,
                "Getrenntes Laufverzeichnis, damit die Laeufe der urspruenglichen "
                "Arbeit unter runs/ unberuehrt bleiben und beide Staende "
                "nebeneinander auswertbar sind."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cache", default="disk",
                    help="ram | disk | none")
    ap.add_argument("--model", default="yolo26s-p2.yaml")
    ap.add_argument("--pretrained", default="yolo26s.pt")
    ap.add_argument("--doc-only", action="store_true")
    a = ap.parse_args()

    cache = False if a.cache in ("none", "false", "0") else a.cache
    ov = {"name": a.name, "data": a.data, "epochs": a.epochs,
          "batch": a.batch, "cache": cache, "model": a.model,
          "pretrained": a.pretrained, "project": PROJECT}
    args = T.build_args(ov)

    # Begruendungstexte der abweichenden Parameter ersetzen, damit die je Lauf
    # geschriebene Tabelle den tatsaechlichen Stand beschreibt.
    spec_backup = dict(T.SPEC)
    for k, (val, why) in ABWEICHUNG.items():
        T.SPEC[k] = (val, why)

    out = os.path.join(PROJECT, a.name)
    os.makedirs(out, exist_ok=True)
    T.write_doc(os.path.join(out, "parameter_begruendung.md"), args)
    if a.doc_only:
        T.SPEC.clear()
        T.SPEC.update(spec_backup)
        print("->", os.path.join(out, "parameter_begruendung.md"))
        return

    mpath = args.pop("model")
    pre = args.pop("pretrained", None)
    model = YOLO(mpath)
    if pre:
        model = model.load(pre)
        print(f"Gewichte partiell uebernommen aus: {pre}", flush=True)
    results = model.train(**args)

    with open(os.path.join(out, "train_args_used.json"), "w") as f:
        json.dump({k: str(v) for k, v in args.items()}, f, indent=2)
    print("\nFertig ->", out)
    try:
        print("Val-Metriken:", results.results_dict)
    except Exception:
        pass


if __name__ == "__main__":
    main()
