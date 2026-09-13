#!/usr/bin/env bash
# Alle fuenf Ablationsarme trainieren und anschliessend auswerten.
set -u
cd /home/mso/source/yolo-gen
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PY src/experiment_neu.py train --cache disk --epochs 30 --batch 8 \
    > logs/train_chain.log 2>&1 || { echo TRAIN_FAILED; exit 1; }
echo "[$(date +%H:%M:%S)] alle Trainings fertig"

$PY src/experiment_neu.py eval  > logs/eval.log  2>&1 || echo EVAL_FAILED
$PY src/experiment_neu.py main  > logs/main.log  2>&1 || echo MAIN_FAILED
$PY src/experiment_neu.py sweep > logs/sweep.log 2>&1 || echo SWEEP_FAILED
echo "[$(date +%H:%M:%S)] Auswertung fertig"
echo CHAIN_DONE
