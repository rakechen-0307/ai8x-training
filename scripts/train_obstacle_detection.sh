#!/bin/sh
python train.py --epochs 100 --optimizer Adam --lr 0.001 --compress policies/schedule_obstacle_detection.yaml --model obstacle_detection --use-bias --momentum 0.9 --weight-decay 5e-4 --dataset obstacle_detection --data ./data/obstacle-detection --device MAX78000 --obj-detection --obj-detection-params parameters/obj_detection_params_obstacle_detection.yaml --qat-policy policies/qat_policy_obstacle_detection.yaml --batch-size 32 "$@" 