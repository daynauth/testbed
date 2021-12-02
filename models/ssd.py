#sources 
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/ssd/model.py
# Libtorch
import torch
import warnings
import sys

from collections import OrderedDict
from typing import Any, Optional, Dict

from torch import nn, Tensor

import torchvision.models.resnet as resnet
import torchvision.models as models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import _validate_trainable_layers

sys.path.append('torchvision/models/detection')
from anchor_utils import DefaultBoxGenerator

model_urls = {
    'ssd300_vgg16_coco': 'https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth',
    'ssd512_resnet50_coco': 'https://download.pytorch.org/models/ssd512_resnet50_coco-d6d7edbb.pth',
}


def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)

class SSDFeatureExtractorResNet(nn.Module):
    def __init__(self, backbone: resnet.ResNet):
        super().__init__()

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        )

        # Patch last block's strides to get valid output sizes
        for m in self.features[-1][0].modules():
            if hasattr(m, 'stride'):
                m.stride = 1

        backbone_out_channels = self.features[-1][-1].bn3.num_features

        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_out_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        ])
        _xavier_init(extra)
        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.features(x)
        output = [x]
        
        for block in self.extra:
            x = block(x)
            output.append(x)


        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def _resnet_extractor(backbone_name: str, pretrained: bool, trainable_layers: int):
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained)
    
    #is this part really necessary?
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 4:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return SSDFeatureExtractorResNet(backbone)

def ssd300_resnet50(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    size = (300, 300)

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 4, 4)

    if pretrained:
        pretrained_backbone = False

    backbone = _resnet_extractor("resnet50", pretrained_backbone, trainable_backbone_layers)

    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05])
    model = models.detection.SSD(backbone, anchor_generator, size, num_classes, **kwargs)

    if pretrained:
        weights_name = 'ssd300_resnet50_coco'
        if model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)

    return model




