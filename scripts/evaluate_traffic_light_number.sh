#!/bin/sh
python train.py --deterministic --model obstacle_detection --use-bias --dataset traffic_light_number --data ./data/traffic-light-number --device MAX78000 --obj-detection --obj-detection-params parameters/obj_detection_params_traffic_light_number.yaml --qat-policy policies/qat_policy_traffic_light_number.yaml --evaluate -8 --exp-load-weights-from ../ai8x-synthesis/trained/traffic_light_number-qat8-q.pth.tar --save-sample 5 "$@" 