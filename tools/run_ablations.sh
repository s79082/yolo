#!/bin/bash
# Trainiert die verbleibenden Ablationsarme sequenziell, sobald Platz ist.
cd /home/mso/source/yolo-gen
for arm in A1_baseline A2_starke_okklusion; do
  if [ -f "runs/abl_${arm}/weights/best.pt" ]; then continue; fi
  .venv/bin/python -u src/train.py --name "abl_${arm}" \
      --data "data/abl_${arm}.yaml" --epochs 30 > "train_abl_${arm}.log" 2>&1
done
echo "ABLATIONS_DONE"
