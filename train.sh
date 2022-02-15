#lr=0.0026
#    --data-augmentation retinanet\
python train.py\
    --data-path /data/datasets/coco\
    --batch-size 16\
    --dataset coco\
    --epochs 120\
    --lr-steps 32 40\
    --aspect-ratio-group-factor 3\
    --lr 0.0002\
    --weight-decay 0.0005\
    --data-augmentation ssd\
    --model ssd300_avtn_faster_rcnn_mobilenet_v3_large
