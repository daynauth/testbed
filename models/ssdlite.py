import torch
from typing import Any, Optional, Dict, List, Tuple, Callable
from .ssd import _resnet_extractor
from torch import nn, Tensor

import torchvision.models as models

from functools import partial
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection import _utils as det_utils

__all__ = ['ssdlite_resnet34', 'ssdlite_resnet50']



def ssdlite_resnet(resnet : str = 'resnet34', pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    backbone = _resnet_extractor(resnet, True, 4)

    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    assert len(out_channels) == len(anchor_generator.aspect_ratios)

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, -1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }

    kwargs = {**defaults, **kwargs}

    model = models.detection.SSD(backbone, anchor_generator, size, num_classes,
                head=models.detection.ssdlite.SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer), **kwargs)

    return model

def ssdlite_resnet50(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    return ssdlite_resnet('resnet50', pretrained, progress, num_classes, pretrained_backbone, trainable_backbone_layers, **kwargs)

def ssdlite_resnet34(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    return ssdlite_resnet('resnet34', pretrained, progress, num_classes, pretrained_backbone, trainable_backbone_layers, **kwargs)
