#!/bin/bash
# Wartet auf den Abschluss von yolo26s und trainiert danach die P2-Variante
# mit IDENTISCHEM Budget (batch 16), damit der Vergleich nicht durch eine
# abweichende Batchgroesse konfundiert wird.
cd /home/mso/source/yolo-gen
while [ ! -f runs/yolo26s/weights/best.pt ] || \
      ! grep -q "epochs completed" train_yolo26s.log 2>/dev/null; do
  sleep 60
done
.venv/bin/python -u src/train.py --name yolo26s_p2 \
    --model yolo26s-p2.yaml --pretrained yolo26s.pt \
    > train_yolo26s_p2.log 2>&1
echo "P2_DONE"
