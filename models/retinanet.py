from torch.nn import parameter
import torchvision.models.detection

from torchvision.models.detection import retinanet_resnet50_fpn

import torch
import torch.nn as nn
from typing import Any, Optional, Dict, List, OrderedDict, Tuple, Callable

__all__ = ['retinanet_resnet50_fpn', 'frozen_retinanet_all']

def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)

class RetinanetAdapter(nn.Module):
    def __init__(self, backbone, fpn):
        super().__init__()

        self.body = backbone
        self.fpn = fpn

        x = torch.rand(1, 3, 300, 300)
        x = self.body(x)

        #in_shapes = [value.shape[1] for key, value in x.items()]
        #print(in_shapes)
        in_shapes = [512, 1024, 2048]

        self.adapter = nn.ModuleDict()
        for (key, value), in_shape in zip(self.body.return_layers.items(), in_shapes):
            self.adapter[value] = nn.Sequential(
                nn.Conv2d(in_shape, in_shape, kernel_size = 1, bias = False),
                nn.BatchNorm2d(in_shape),
                nn.ReLU(inplace=True)
            )
            _xavier_init(self.adapter[value])


        

    def forward(self, x):
        x = self.body(x)

        #output = OrderedDict()
        for key, value in x.items():
            x[key] = self.adapter[key](value)
        x = self.fpn(x)

        return x

    

def frozen_retinanet_all(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, freeze_head: bool = False, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):


    retinanet = retinanet_resnet50_fpn(pretrained=True)

    retinanet_backbone = retinanet.backbone.body
    retinanet_fpn = retinanet.backbone.fpn

    #freeze the layers
    for name, parameter in retinanet_backbone.named_parameters():
        parameter.requires_grad_(False)

    for name, parameter in retinanet_fpn.named_parameters():
        parameter.requires_grad_(False)

    backbone = RetinanetAdapter(retinanet_backbone, retinanet_fpn)
    retinanet.backbone = backbone


    return retinanet




