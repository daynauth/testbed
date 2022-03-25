# Testbed
Testbed for multiple object detection models

## Train
The test-bench consist of a training script at 
train.sh.

This consist of the following options
```
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
    --model ssd_frozen\
```
- data-path is the path to the coco dataset
- batch-size is the training batch size
- epochs, the number of epochs to train
- lr-steps, the steps for which to adjust the learning rate. 
- data-augmentation, the image preprocessing done on the image, options available are ssd and ssdlite.
- name of the model to train

To train run

```
bash train.sh
```
