#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --mode train \
    --model_type DDPM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset mixed_segm \
    --img_dir mixed_segm/data \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --num_epochs 500 \
    --num_workers 10 \
    --seg_dir mixed_segm/mask \
    --segmentation_guided \
    --num_segmentation_classes 4

# training with vessel
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
#     --mode train \
#     --model_type DDPM \
#     --img_size 256 \
#     --num_img_channels 1 \
#     --dataset vessel_segm \
#     --img_dir vessel_segm/data \
#     --train_batch_size 32 \
#     --eval_batch_size 8 \
#     --num_epochs 300 \
#     --num_workers 16 \
#     --seg_dir vessel_segm/mask \
#     --segmentation_guided \
#     --num_segmentation_classes 4 \
#     --resume_epoch 200 \
#     --resume_dir outputs/ddpm-vessel_segm-256-segguided/20240802_092709

# training with clahe data
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
#     --mode train \
#     --model_type DDPM \
#     --img_size 256 \
#     --num_img_channels 1 \
#     --dataset oc_od_segm \
#     --img_dir oc_od_segm/clahe_data \
#     --train_batch_size 32 \
#     --eval_batch_size 8 \
#     --num_epochs 100 \
#     --num_workers 16 \
#     --seg_dir oc_od_segm/mask \
#     --segmentation_guided \
#     --num_segmentation_classes 2

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
#     --mode train \
#     --model_type DDPM \
#     --img_size 256 \
#     --num_img_channels 1 \
#     --dataset oc_od_segm \
#     --img_dir oc_od_segm/gray_data \
#     --train_batch_size 32 \
#     --eval_batch_size 8 \
#     --num_epochs 500 \
#     --num_workers 16 \
#     --seg_dir oc_od_segm/mask \
#     --segmentation_guided \
#     --num_segmentation_classes 4 \
#     --resume_epoch 0 \
#     --resume_dir outputs/ddpm-vessel_segm-256-segguided/20240802_085503
