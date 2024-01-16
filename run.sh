#!/bin/bash

module load cuda/11.3
module load cudnn/8.0.5-cuda11
nvidia-smi
nvcc --version

eval "$(conda shell.bash hook)"
conda activate meal
cd /home/thuy0050/code/MEAL

export CUDA_DEVICE_ORDER=PCI_BUS_ID # Change according to GPU availability
# export CUDA_VISIBLE_DEVICES=0 # Change according to GPU availability
export CUDA_LAUNCH_BLOCKING=1

fold=0
data_dir="/home/thuy0050/ft49_scratch/thuy0050/meal/aim_folds/fold_$fold"

# Train backbone
python train.py \
--img 480 --batch 64 --epochs 300 \
--data $data_dir/annotations/aim_fold_$fold.yaml \
--weights yolov5s.pt \
--project /home/thuy0050/ft49_scratch/thuy0050/meal/runs/train \
--name train_backbone_fold_$fold \
--workers 4 \
--patience 100

# Validate backbone
python val.py \
--img 480 --batch 1 \
--data $data_dir/annotations/aim_fold_$fold.yaml \
--weights /home/thuy0050/ft49_scratch/thuy0050/meal/runs/train/train_backbone_fold_$fold/weights/best.pt \
--verbose --save-json \
--project /home/thuy0050/ft49_scratch/thuy0050/meal/runs/val/ \
--name val_backbone_fold_$fold

# # Train MEAL
python meal/train.py --fold $fold

# Validate MEAL
python meal/val.py \
--img 480 --batch 1 \
--data $data_dir/annotations/aim_fold_$fold.yaml \
--weights /home/thuy0050/ft49_scratch/thuy0050/meal/runs/train/train_backbone_fold_$fold/weights/best.pt \
--meal-path /home/thuy0050/ft49_scratch/thuy0050/meal/runs/train/train_meal_fold_$fold/meal_fold_$fold.pt \
--meal-type "MEAL" \
--verbose \
--save-json \
--project /home/thuy0050/ft49_scratch/thuy0050/meal/runs/val/ \
--name val_meal_fold_$fold 
