#!/bin/bash

python cache.py\
    --data-path /data/datasets/coco\
    --dataset coco_cache\
    --cache-dir cached_dataset\
    --model FasterRCNN_resnet50
