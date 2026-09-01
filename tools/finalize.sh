#!/bin/bash
# Abschluss: Endauswertung, Experimente, Diagramme.
set -e
cd /home/mso/source/yolo-gen
PY=.venv/bin/python
mkdir -p docs/experiments

echo "### Checkpoint-Auswahl (best.pt vs last.pt) auf VAL, nativer Massstab"
$PY - <<'PYEOF'
import sys, json, os
sys.path.insert(0, "src")
import cv2; cv2.setNumThreads(0)
from ultralytics import YOLO
from evaluate import evaluate_split
inf = dict(imgsz="native", conf=0.001, iou=0.7, max_det=1000,
           augment=False, quantize=None, device="0")
res = {}
for ck in ("best", "last"):
    w = f"runs/baseline/weights/{ck}.pt"
    r = evaluate_split(YOLO(w), "val", inf)
    res[ck] = dict(mAP50=r["mAP50"], mAP50_95=r["mAP50_95"], f1=r["best_f1"])
    print(f"  {ck}.pt  mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50_95']:.4f}  F1={r['best_f1']:.4f}")
sel = max(res, key=lambda k: res[k]["mAP50_95"])
print("  gewaehlt:", sel)
json.dump(dict(candidates=res, selected=sel, criterion="mAP50-95 auf val, nativer Massstab"),
          open("docs/experiments/checkpoint_selection.json", "w"), indent=2)
open("docs/experiments/.selected_ckpt", "w").write(sel)
PYEOF
CK=$(cat docs/experiments/.selected_ckpt)
W=runs/baseline/weights/${CK}.pt
echo "### Gewaehlter Checkpoint: $W"

echo "### Endauswertung val/test/sval + Ultralytics-Kreuzvergleich + Qualitativbilder"
$PY src/evaluate.py --weights $W --splits val,test,sval \
    --crosscheck --vis docs/qual_SPLIT_NAME.jpg \
    --out docs/experiments/final_eval.json

echo "### Experiment 1 - Inferenzparameter"
$PY src/experiment.py infer --weights $W

echo "### Experiment 2 - Ablation auswerten"
$PY src/experiment.py ablation-eval

echo "### Diagramme und Tabellen"
$PY src/report.py
echo "FINALIZE_DONE"
