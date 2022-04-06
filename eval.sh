#!/bin/bash
#lr=0.0026
#    --data-augmentation retinanet\
python eval.py\
    --eval-model model_9.pth\
    --eval-epoch 9\
    --data-path /data/datasets/coco\
    --batch-size 32\
    --dataset coco\
    --data-augmentation ssd\
    --model ssd_vgg_retrained
    
