#lr=0.0026
python train.py\
    --workers 12\
    --data-path /data/datasets/coco\
    --dataset coco\
    --epochs 50\
    --lr-steps 32 40\
    --aspect-ratio-group-factor 3\
    --lr 0.0002\
    --batch-size 16\
    --weight-decay 0.0005\
    --data-augmentation ssd\
    --model ssd300_resnet50\
