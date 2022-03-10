#!/bin/bash
#lr=0.0026
#    --data-augmentation retinanet\
python train.py\
    --data-path /data/datasets/coco\
    --batch-size 8\
    --dataset coco\
    --epochs 50\
    --lr-steps 32 40\
    --aspect-ratio-group-factor 3\
    --lr 0.0002\
    --weight-decay 0.0005\
    --model frozen_retinanet_all\
