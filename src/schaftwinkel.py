#!/usr/bin/env python3
"""Wie oft kreuzen die Rohrschaefte eines Clusters die Rohre des Hintergrunds?

Der Generator zieht die Schaftrichtung je Cluster aus der Saeulenrichtung des
Clusters (cluster_positions, psi_axis), nicht aus der Rohrrichtung des
Hintergrunds. Auf den Clean Plates liegen die Hintergrundrohre waagerecht;
make_background dreht das Bild hoechstens um die Kleinrotation von +-6 Grad,
Spiegelungen lassen waagerechte Rohre waagerecht. Die Eigenneigung der Platten
(unter 1 Grad, src/occluders.py) bleibt unberuecksichtigt.

Nachgespielt wird die Erzeugung mit denselben Seeds wie src/generate.py; je
Bild werden Hintergrundart, Drehwinkel und die Schaftrichtung jedes Clusters
festgehalten. Ein Cluster kreuzt den Hintergrund, wenn seine Schaftrichtung
um mehr als 60 Grad (modulo 180) von der Rohrrichtung abweicht. Fremdszenen
haben keine Rohre und zaehlen nicht mit.

Aufruf (aus dem Projektwurzelverzeichnis):
    python src/schaftwinkel.py [--n 6000] [--seed 1]
                                   -> docs/experiments/schaftwinkel.json
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

import generate as G

GRENZE = 60.0
OUT = "docs/experiments/schaftwinkel.json"
_A = {}


def _init():
    _A["assets"] = G.load_assets()


def _bild(job):
    i, seed = job
    profiles, bars, bgs = _A["assets"]
    info = {"clean": None, "rot": 0.0, "psi": []}
    orig_bg, orig_cl, orig_u = G.make_background, G.cluster_positions, G._u

    def bg(rng, b):
        def u(r, a, c):
            v = orig_u(r, a, c)
            if (a, c) == (-6, 6):          # Kleinrotation in make_background
                info["rot"] = v
            return v
        G._u = u
        try:
            canvas, d, bgd = orig_bg(rng, b)
        finally:
            G._u = orig_u
        info["clean"] = bool(bgd["clean"])
        return canvas, d, bgd

    def cl(rng, d0):
        pts, ctr, psi = orig_cl(rng, d0)
        info["psi"].append(psi)
        return pts, ctr, psi

    G.make_background, G.cluster_positions = bg, cl
    try:
        G.STATS.clear()
        G.generate_one(i, np.random.default_rng(seed * 1_000_003 + i),
                       profiles, bars, bgs)
    finally:
        G.make_background, G.cluster_positions = orig_bg, orig_cl
    # Bildkoordinaten (y nach unten): eine Drehung um +rot gegen den
    # Uhrzeigersinn legt waagerechte Rohre auf die Richtung -rot.
    rohr = -info["rot"]
    abw = [abs((p - rohr + 90.0) % 180.0 - 90.0) for p in info["psi"]]
    return info["clean"], abw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    with Pool(max(1, (os.cpu_count() or 8) - 1), initializer=_init) as p:
        res = p.map(_bild, [(i, a.seed) for i in range(a.n)], chunksize=20)
    cluster = [w for clean, abw in res if clean for w in abw]
    kreuzt = [w > GRENZE for w in cluster]
    bilder = sum(1 for clean, abw in res if clean and any(w > GRENZE for w in abw))
    rep = dict(n_bilder=a.n, seed=a.seed, grenze_grad=GRENZE,
               cluster_auf_clean_plates=len(cluster),
               cluster_kreuzend=int(sum(kreuzt)),
               anteil_cluster=float(np.mean(kreuzt)),
               bilder_mit_kreuzendem_cluster=bilder,
               anteil_bilder=bilder / a.n,
               bilder_auf_clean_plates=sum(1 for c, _ in res if c))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    print("->", OUT)


if __name__ == "__main__":
    main()
