#!/bin/bash
# Quellmaterial, Datensaetze und Laeufe auf dem bereinigten Quellmaterial.
# Umgebung: .venv mit requirements.txt (Ergebnisse haengen an opencv/numpy).
set -e
cd /home/mso/source/yolo-gen
PY=.venv/bin/python

$PY src/clean_plates.py           # bg/img3884_cutted.png, bg/img3886_cutted.png
rm -f assets/bars/*.png
$PY src/occluders.py              # assets/bars (55 Schaefte)
$PY src/ueberlappung.py           # docs/experiments/ueberlappung.json

$PY src/generate.py --out synth/bereinigt_train          --n 6000 --seed 1  --preview 8
$PY src/generate.py --out synth/bereinigt_sval           --n  400 --seed 99
$PY src/generate.py --out synth/bereinigt_C0_kontrolle   --n 3000 --seed 1
$PY src/generate.py --out synth/bereinigt_C1_naive_paste --n 3000 --seed 1  --naive-paste --preview 4
$PY src/generate.py --out synth/bereinigt_sval_C1        --n  400 --seed 99 --naive-paste

$PY src/train_neu.py --name bereinigt_referenz       --data data/bereinigt_referenz.yaml       --epochs 60 > train_bereinigt_referenz.log 2>&1
$PY src/train_neu.py --name bereinigt_C0_kontrolle   --data data/bereinigt_C0_kontrolle.yaml   --epochs 30 > train_bereinigt_C0_kontrolle.log 2>&1
$PY src/train_neu.py --name bereinigt_C1_naive_paste --data data/bereinigt_C1_naive_paste.yaml --epochs 30 > train_bereinigt_C1_naive_paste.log 2>&1
