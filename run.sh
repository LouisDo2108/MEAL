#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate meal # Change this according to your environment's name.

cd ~/meal

# Train backbone
python train.py \
--img 480 --batch 64 --epochs 1 \
--data vofo/annotations/vofo.yaml \
--weights yolov5s.pt \
--name backbone \
--patience 300

# Validate backbone
python val.py \
--img 480 --batch 1 \
--data vofo/annotations/vofo.yaml \
--weights checkpoints/backbone.pt \
--verbose --save-json \
--name backbone

# Trai MEAL
python train.py

# Validate MEAL
python val.py \
--img 480 --batch 1 \
--data vofo/annotations/vofo.yaml \
--weights checkpoints/meal.pt \
--verbose \
--save-json \
--name test