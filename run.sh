#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate meal

cd meal

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

# # Train MEAL
python meal/train.py

# # Validate MEAL
python meal/val.py \
--img 480 --batch 1 \
--data vofo/annotations/vofo.yaml \
--weights checkpoints/backbone.pt \
--meal-path checkpoints/meal_with_masked_attention.pt \
--meal-type MEAL_with_masked_attention \
--verbose \
--save-json \
--name test