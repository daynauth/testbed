#!/bin/bash
#lr=0.0026
#    --data-augmentation retinanet\
python train.py\
    --data-path /data/datasets/coco\
    --batch-size 32\
    --dataset coco\
    --epochs 65\
    --lr-steps 43 54\
    --aspect-ratio-group-factor 3\
    --lr 0.001\
    --weight-decay 0.0005\
    --momentum 0.9\
    --data-augmentation ssd\
    --output-dir model_resnetssd_coco_unfreeze\
    --model ssd_resnet_baseline_unfreeze
    
