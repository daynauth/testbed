import torch
import torchvision.models as models


#model = models.detection.keypointrcnn_resnet50_fpn()
from models.avtn import fasterrcnn_resnet_ssd_avtn1, fasterrcnn_resnet_ssd_avtn2, fasterrcnn_resnet_ssd_dirty_avtn1, fasterrcnn_resnet_ssd_dirty_avtn2, ssd_resnet_baseline, ssd_resnet_baseline_unfreeze
from models.ssd import ssd_vgg_imagenet_backbone, ssd_vgg_retrained_frozen_backbone
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_paarams = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_paarams += sum(p.numel() for p in parameter)
    return total_params, trainable_paarams

model = ssd_resnet_baseline_unfreeze(False, True, 91, True)

print(count_parameters(model))
