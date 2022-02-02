import torch
from typing import Any, Optional, Dict, List, Tuple, Callable
from .ssd import _resnet_extractor, _xavier_init
from torch import nn, Tensor

from collections import OrderedDict

import torchvision.models
import torchvision.models.detection
import torchvision.models as models

from functools import partial
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection import _utils as det_utils

__all__ = ['ssdlite_resnet34', 'ssdlite_resnet50', 'frozen_ssdlite_resnet50', 'frozen_ssdlite_resnet50_frozen_head']



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

def rand_image(size):
    return torch.rand(1, 3, size[0], size[1])

def get_features(backbone : nn.Module, size, in_shape):
    x = rand_image(size)
    backbone_list = nn.Sequential(*list(backbone.children()))


    #gotta figure this part out
    # conv4_block1 = backbone_list[-1][0]
    # conv4_block1.conv1.stride = (1, 1)
    # conv4_block1.conv2.stride = (1, 1)
    # conv4_block1.downsample[0].stride = (1, 1)    


    count = 0
    for block in backbone_list:
        x = block(x)
        count += 1

        if x.shape[2] < in_shape:
            break
    
    features = nn.ModuleList([
        nn.Sequential(*list(backbone.children())[:count - 1]),
        nn.Sequential(*list(backbone.children())[count - 1])
    ])

    return features

def get_inshapes(features, size):
    x = rand_image(size)
    in_channels = []

    for block in features:
        x = block(x)
        in_channels.append(x.shape[1])

    return in_channels

class LiteResnetAdapter(nn.Module):
    def __init__(self, backbone, extra, size, out_channels):
        super().__init__()
        in_shape = 2**(len(extra))
        self.features = get_features(backbone, size, in_shape)
        
        in_channels = get_inshapes(self.features, size)

        self.reducers = nn.ModuleList()
        for in_channel, out_channel in zip(in_channels, out_channels[0:len(in_channels)]):
            self.reducers.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size = 1, bias = False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True)
                )
            )

        _xavier_init(extra)
        self.extra = extra

    def forward(self, x):
        output = []
        for feature, reducer in zip(self.features, self.reducers):
            x = feature(x)
            output.append(reducer(x))

        x = output[-1]
        for block in self.extra:
            x = block(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])




def frozen_ssdlite_resnet50(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, freeze_head: bool = False, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    size = (300, 300)

    #grab the resnet backbone from retinanet to use as the backbone for ssd
    retinanet_backbone = models.detection.retinanet_resnet50_fpn(pretrained=True).backbone.body

    #freeze all the layers before layer 3
    trainable_backbone_layers = ['layer3', 'layer4']

    for name, parameter in retinanet_backbone.named_parameters():
        if all([not name.startswith(layer) for layer in trainable_backbone_layers]):
            parameter.requires_grad_(False)

    #get the head and extra layers but not frozen
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True, progress=True)

    #grab the ssdlite head and extra layers
    head = model.head
    extra = model.backbone.extra

    #get the input size for the head
    out_channels = []
    for block in head.classification_head.module_list:
        out_channels.append(block[0][0].weight.shape[0])

    #freeze extra layers
    for name, parameter in extra.named_parameters():
        parameter.requires_grad_(False)

    backbone = LiteResnetAdapter(retinanet_backbone, extra, size, out_channels)

    anchor_generator = model.anchor_generator
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



    #freeze head
    if freeze_head is True:
        for name, parameter in head.named_parameters():
            parameter.requires_grad_(False)

    return models.detection.SSD(backbone, anchor_generator, size, num_classes, head=head, **kwargs)



def frozen_ssdlite_resnet50_frozen_head(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    return frozen_ssdlite_resnet50(pretrained, progress, num_classes, pretrained_backbone, True, trainable_backbone_layers, **kwargs)