#!/usr/bin/env bash
# Nach der Ablationskette: Referenzmodell mit demselben Aufbau neu trainieren
# (6000 Bilder, 60 Epochen) und die Analyse darauf wiederholen.
set -u
cd /home/mso/source/yolo-gen
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date +%H:%M:%S)] warte auf das Ende der Ablationskette ..."
until grep -q CHAIN_DONE logs/chain.log 2>/dev/null; do
  if grep -q "FAILED" logs/chain.log 2>/dev/null; then echo ABORT; exit 1; fi
  sleep 60
done
echo "[$(date +%H:%M:%S)] Ablationen fertig, starte Referenzlauf"

cat > data/referenz.yaml <<'YML'
# Referenzmodell: vollstaendiger Datensatz der Pipeline
path: /home/mso/source/yolo-gen
train: synth/train/images
val: data/val/images
test: data/test/images
names:
  0: rohrprofil
YML

$PY -u src/train_neu.py --name referenz --data data/referenz.yaml \
    --epochs 60 --batch 8 --cache disk > logs/train_referenz.log 2>&1 \
    || { echo REF_FAILED; exit 1; }
echo "[$(date +%H:%M:%S)] Referenzlauf fertig"

$PY src/experiment_neu.py main  --weights runs/referenz/weights/best.pt \
    > logs/main_ref.log  2>&1 || echo MAIN_FAILED
$PY src/experiment_neu.py sweep --weights runs/referenz/weights/best.pt \
    > logs/sweep_ref.log 2>&1 || echo SWEEP_FAILED
$PY src/analyse_neu.py  --weights runs/referenz/weights/best.pt \
    > logs/analyse_ref.log 2>&1 || echo ANALYSE_FAILED
echo "[$(date +%H:%M:%S)] Referenzauswertung fertig"
echo REF_DONE
