#!/bin/bash

# EXPERIMENT_NAME=2023-10-27-13-20-44
EXPERIMENT_NAME=bagfiles_casi/automated_sweep_2
# FLOW_CHECKPOINT_PATH=/home/casimir/ETH/SemesterProject/mmflow/checkpoints/pwcnet_ft_4x1_300k_sintel_final_384x768.pth
# FLOW_CONFIG_PATH=/home/casimir/ETH/SemesterProject/mmflow/configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py

FLOW_CHECKPOINT_PATH=/home/casimir/ETH/SemesterProject/IGS/flow/checkpoints_paths/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.pth
FLOW_CONFIG_PATH=/home/casimir/ETH/SemesterProject/IGS/flow/checkpoints_paths/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.py

RGB_PATH=/home/casimir/ETH/SemesterProject/IGS/dataset/dataset_processed/$EXPERIMENT_NAME/rgb
SAVE_PATH=/home/casimir/ETH/SemesterProject/IGS/dataset/dataset_processed/$EXPERIMENT_NAME/estimated_masks
EVAL_PATH=/home/casimir/ETH/SemesterProject/IGS/dataset/dataset_processed/$EXPERIMENT_NAME/gt_masks
FLOW_THRESHOLD=8.0

mkdir -p $SAVE_PATH

python flow_mask.py \
--flow-checkpoint-path $FLOW_CHECKPOINT_PATH \
--flow-config-path $FLOW_CONFIG_PATH \
--rgb-path $RGB_PATH \
--save-path $SAVE_PATH \
--eval-path $EVAL_PATH \
--flow-threshold $FLOW_THRESHOLD