"""
Duenner Aufsatz auf src/generate.py: taucht die Okkluder-Bibliothek aus.

Der Generator selbst wird NICHT kopiert und NICHT veraendert. Ausgetauscht wird
allein die Funktion, die die Stangen laedt. Damit gilt:

  * Hintergrundwahl, Zoom, Clusterlayout, Instanzgeometrie, Photometrie,
    Tiefenordnung, Sichtbarkeitsrechnung und die globale Bildstufe sind
    Zeile fuer Zeile dieselben;
  * die Zufallsziehungen sind identisch, weil alle drei Bibliotheken genau
    56 Eintraege haben und die Auswahl ueber rng.integers(0, 56) laeuft;
  * die Alphamasken sind byte-identisch (src/procedural_bars.py uebernimmt
    Groesse und capsule_alpha jeder echten Stange).

Folge: die erzeugten Datensaetze unterscheiden sich AUSSCHLIESSLICH in den
RGB-Werten innerhalb der Stangenmasken -- Objektpositionen, Verdeckungsgrade
und damit auch die Labeldateien sind gleich. Genau das macht die Ablation
"Herkunft der Balken" zu einem gepaarten Versuch.

Aufruf (alle Optionen von src/generate.py bleiben gueltig):
    python src/generate_neu.py --out synth/abl_B1_proz_balken \
        --n 3000 --seed 1 --bars-dir assets/bars_proc
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import generate as G                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--bars-dir", default="assets/bars",
                    help="Verzeichnis der Okkluder-PNGs (RGBA)")
    ap.add_argument("--bars-tag", default=None,
                    help="Bezeichner, der in generation_report.json landet")
    known, rest = ap.parse_known_args()

    paths = sorted(glob.glob(os.path.join(known.bars_dir, "*.png")))
    if not paths:
        raise SystemExit(f"keine Stangen in {known.bars_dir}")
    bars = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in paths]
    if any(b is None or b.shape[2] != 4 for b in bars):
        raise SystemExit(f"{known.bars_dir}: alle Stangen muessen RGBA sein")

    orig_load = G.load_assets

    def load_assets():
        profiles, _photo_bars, bgs = orig_load()
        return profiles, bars, bgs

    G.load_assets = load_assets
    print(f"Okkluder-Bibliothek: {known.bars_dir}  ({len(bars)} Stangen)")

    sys.argv = [sys.argv[0]] + rest
    G.main()

    # Herkunft der Balken im Erzeugungsbericht festhalten
    out = None
    for i, a in enumerate(rest):
        if a == "--out":
            out = rest[i + 1]
    if out:
        p = os.path.join(out, "generation_report.json")
        rep = json.load(open(p))
        rep["bars_dir"] = known.bars_dir
        rep["bars_tag"] = known.bars_tag or os.path.basename(known.bars_dir)
        rep["bars_n"] = len(bars)
        json.dump(rep, open(p, "w"), indent=2)


if __name__ == "__main__":
    main()
