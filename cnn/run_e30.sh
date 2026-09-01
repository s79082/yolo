#!/bin/bash
# Wartet auf pipenet und trainiert danach eine Variante, deren Kosinus-Rampe
# innerhalb der tatsaechlichen Laufzeit ausläuft (epochs=30 statt 60).
cd /home/mso/source/yolo-gen
while pgrep -f "cnn/src/train.py --name pipenet$" >/dev/null 2>&1 || \
      ! grep -qE "^Fertig|Early Stopping" cnn/train_pipenet.log 2>/dev/null; do
  sleep 60
done
.venv/bin/python -u cnn/src/train.py --name pipenet_e30 --epochs 30 \
    > cnn/train_pipenet_e30.log 2>&1
echo "E30_DONE"
