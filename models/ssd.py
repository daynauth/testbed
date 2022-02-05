#sources 
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/ssd/model.py
# Libtorch
import torch
import torch.nn.functional as F
import warnings
import sys

from functools import partial
from collections import OrderedDict
from typing import Any, Optional, Dict, List, Tuple, Callable

from torch import nn, Tensor


import torchvision.models.resnet as resnet
import torchvision.models as models

from torch.hub import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.ops import boxes as box_ops
from torchvision.models import mobilenet

from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator


__all__ = [
    'ssd300_resnet34', 
    'ssd300_resnet50', 
    'ssd300_resnet101', 
    '_resnet_extractor', 
    'ssd300_mobilenet_v3_large',
    'ssd300_mobilenet_v3_small',
    'ssd300_mobilenet_v2',
    'ssd_frozen',
    'ssd_frozen_mobilenet'
]

model_urls = {
    'ssd300_vgg16_coco': 'https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth',
    'ssd512_resnet50_coco': 'https://download.pytorch.org/models/ssd512_resnet50_coco-d6d7edbb.pth',
}

backbone_urls = {
    # We port the features of a VGG16 backbone trained by amdegroot because unlike the one on TorchVision, it uses the
    # same input standardization method as the paper. Ref: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
    'vgg16_features': 'https://download.pytorch.org/models/vgg16_features-amdegroot.pth'
}


def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)

class SSDFeatureExtractorResNet(nn.Module):
    def __init__(self, backbone: resnet.ResNet, out_channels: List[int] = [1024, 512, 512, 256, 256, 256]):
        super().__init__()

        self.input_size = out_channels[:-1]
        self.output_size = out_channels[1:]
        #self.channels = [256, 256, 128, 128, 128]

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

        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.input_size[0], 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.output_size[0], kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(self.output_size[0]),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(self.input_size[1], 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.output_size[1], kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(self.output_size[1]),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(self.input_size[2], 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, self.output_size[2], kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(self.output_size[2]),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(self.input_size[3], 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, self.output_size[3], kernel_size=3, bias=False),
                nn.BatchNorm2d(self.output_size[3]),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(self.input_size[4], 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, self.output_size[4], kernel_size=3, bias=False),
                nn.BatchNorm2d(self.output_size[4]),
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
    if backbone_name == 'resnet18':
        out_channels = [256, 512, 512, 256, 256, 128]
    elif backbone_name == 'resnet34':
        out_channels = [256, 512, 512, 256, 256, 256]
    elif backbone_name  == 'resnet50':
        out_channels = [1024, 512, 512, 256, 256, 256]
    elif backbone_name == 'resnet101':
        out_channels = [1024, 512, 512, 256, 256, 256]
    else:  # backbone == 'resnet152':
        out_channels = [1024, 512, 512, 256, 256, 256]

    backbone = resnet.__dict__[backbone_name](pretrained=pretrained)
    
    #decide what layers we want to freeze
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 4:
        layers_to_train.append('bn1')

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return SSDFeatureExtractorResNet(backbone, out_channels)


def ssd300_resnet(resnet_name: str = "resnet50", pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")


    size = (300, 300)

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 4, 4)

    if pretrained:
        pretrained_backbone = False

    backbone = _resnet_extractor(resnet_name, pretrained_backbone, trainable_backbone_layers)

    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                            steps=[8, 16, 32, 64, 100, 300])

    model = models.detection.SSD(backbone, anchor_generator, size, num_classes, **kwargs)

    # if pretrained:
    #     weights_name = 'ssd300_resnet50_coco'
    #     if model_urls.get(weights_name, None) is None:
    #         raise ValueError("No checkpoint is available for model {}".format(weights_name))
    #     state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
    #     model.load_state_dict(state_dict)

    return model   

def ssd300_resnet34(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    return ssd300_resnet("resnet34", pretrained, progress, num_classes, pretrained_backbone, trainable_backbone_layers)

def ssd300_resnet50(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    return ssd300_resnet("resnet50", pretrained, progress, num_classes, pretrained_backbone, trainable_backbone_layers)

def ssd300_resnet101(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    return ssd300_resnet("resnet101", pretrained, progress, num_classes, pretrained_backbone, trainable_backbone_layers)

def ssd300_resnet152(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):
    return ssd300_resnet("resnet152", pretrained, progress, num_classes, pretrained_backbone, trainable_backbone_layers)

class SSDFeatureExtractorMobileNet(nn.Module):
    def __init__(self, backbone: nn.Module, **kwargs: Any):
        #print(*backbone[-1])
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(*backbone[0:8]),
            nn.Sequential(*backbone[8:14])
        )

        backbone_out_channels = self.features[-1][-1].conv[-1].num_features

        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_out_channels, 256, kernel_size=1), # conv8_2
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), # conv9_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3), # conv10_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3), # conv11_2
                nn.ReLU(inplace=True),
            )
        ])

        models.detection.ssdlite._normal_init(extra)

        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Get feature maps from backbone and extra. Can't be refactored due to JIT limitations.
        output = []
        for block in self.features:
            x = block(x)
            output.append(x)

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])




def _mobilenet_extractor(backbone_name: str, progress: bool, pretrained: bool, trainable_layers: int,
                         norm_layer: Callable[..., nn.Module], **kwargs: Any):
    backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained, progress=progress,
                                                 norm_layer=norm_layer, **kwargs).features

    return SSDFeatureExtractorMobileNet(backbone)

def _mobilenetv3_extractor(backbone_name: str, progress: bool, pretrained: bool, trainable_layers: int,
                         norm_layer: Callable[..., nn.Module], **kwargs: Any):

    #get the mobilenet backbone
    backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained, progress=progress,
                                                 norm_layer=norm_layer, **kwargs).features

    if not pretrained:
        models.detection.ssdlite._normal_init(backbone)

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = num_stages if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]


    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)                                            
    #print(backbone)
    #print(backbone[-1])

    return models.detection.ssdlite.SSDLiteFeatureExtractorMobileNet(backbone, stage_indices[-2], norm_layer, **kwargs)
    
    # return SSDFeatureExtractorMobileNet(backbone)



def ssd300_mobilenet_v2(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                                  pretrained_backbone: bool = False, trainable_backbone_layers: Optional[int] = None,
                                  norm_layer: Optional[Callable[..., nn.Module]] = None,
                                  **kwargs: Any):
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    # Enable reduced tail if no pretrained backbone is selected
    reduce_tail = not pretrained_backbone

    backbone = _mobilenet_extractor("mobilenet_v2", progress, pretrained_backbone, trainable_backbone_layers,
                                norm_layer)


    size = (300, 300)


    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                        steps=[8, 16, 32, 64, 100, 300])

    model = models.detection.SSD(backbone, anchor_generator, size, num_classes, **kwargs)

    #return models.detection.ssdlite.SSDLiteFeatureExtractorMobileNet(model, progress, pretrained_backbone)
    return model


def ssd300_mobilenet_v3_large(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                                  pretrained_backbone: bool = False, trainable_backbone_layers: Optional[int] = None,
                                  norm_layer: Optional[Callable[..., nn.Module]] = None,
                                  **kwargs: Any):

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 6)

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected
    reduce_tail = not pretrained_backbone

    # Enable reduced tail if no pretrained backbone is selected
    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    #don't freeze anything as yet

    backbone = _mobilenetv3_extractor("mobilenet_v3_large", progress, pretrained_backbone, trainable_backbone_layers,
                                norm_layer)

    size = (300, 300)


    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                        steps=[8, 16, 32, 64, 100, 300])

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

    model = models.detection.SSD(backbone, anchor_generator, size, num_classes, **kwargs)

    return model


def ssd300_mobilenet_v2(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                                  pretrained_backbone: bool = False, trainable_backbone_layers: Optional[int] = None,
                                  norm_layer: Optional[Callable[..., nn.Module]] = None,
                                  **kwargs: Any):
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    # Enable reduced tail if no pretrained backbone is selected
    reduce_tail = not pretrained_backbone

    backbone = _mobilenet_extractor("mobilenet_v2", progress, pretrained_backbone, trainable_backbone_layers,
                                norm_layer)


    size = (300, 300)


    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                        steps=[8, 16, 32, 64, 100, 300])

    model = models.detection.SSD(backbone, anchor_generator, size, num_classes, **kwargs)

    #return models.detection.ssdlite.SSDLiteFeatureExtractorMobileNet(model, progress, pretrained_backbone)
    return model


def ssd300_mobilenet_v3_small(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                                  pretrained_backbone: bool = False, trainable_backbone_layers: Optional[int] = None,
                                  norm_layer: Optional[Callable[..., nn.Module]] = None,
                                  **kwargs: Any):

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 6)

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected
    reduce_tail = not pretrained_backbone

    # Enable reduced tail if no pretrained backbone is selected
    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    #don't freeze anything as yet

    backbone = _mobilenetv3_extractor("mobilenet_v3_small", progress, pretrained_backbone, trainable_backbone_layers,
                                norm_layer)

    size = (300, 300)


    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                        steps=[8, 16, 32, 64, 100, 300])

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

    return models.detection.SSD(backbone, anchor_generator, size, num_classes, **kwargs)



class ResnetAdapter(nn.Module):
    def __init__(self, backbone):
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

        self.reducer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),
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

        _xavier_init(self.reducer)

    def forward(self, x):
        x = self.features(x)
        x = self.reducer(x)
        output = [x]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])

def ssd_resnet50_adapted(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):

    pretrained_model = models.detection.ssd300_vgg16(pretrained=True)
    pretrained_head = pretrained_model.head

    backbone = ResnetAdapter(models.resnet50())
    size = (300, 300)
    anchor_generator = pretrained_model.anchor_generator

    model = models.detection.SSD(backbone, anchor_generator, size, num_classes, head = pretrained_head, **kwargs)
    return model


class ResnetAdapterV2(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.features = nn.Sequential(
            *[u for v, u in list(backbone.items())[:-1]]
        )

        for m in self.features[-1][0].modules():
            if hasattr(m, 'stride'):
                m.stride = 1


        self.reducer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),
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

        _xavier_init(self.reducer)


    def forward(self, x):
        x = self.features(x)
        x = self.reducer(x)

        output = [x]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def ssd_resnet50_adapted_v2(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):  

    #grab pre-trained retinanet model (resnet 50 backbone)
    model = models.detection.retinanet_resnet50_fpn(pretrained=True)
    
    #get the resnet backbone from this model
    pretrained_backbone = model.backbone.body

    #freeze all backbone layers except layer 3 and 4
    layers_to_train = ['layer3', 'layer4']

    for name, parameter in pretrained_backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)



    #attach ssd layers to the backbone
    backbone = ResnetAdapterV2(pretrained_backbone)
    pretrained_model = models.detection.ssd300_vgg16(pretrained=True)

    #don't freeze head. at least not yet.  
    pretrained_head = pretrained_model.head

    #freeze head here but add option to not freeze head
    for name, parameter in pretrained_head.named_parameters():
        parameter.requires_grad_(False)

    size = (300, 300)

    #reuse the same anchor generator from the ssd model
    anchor_generator = pretrained_model.anchor_generator
    
    model = models.detection.SSD(backbone, anchor_generator, size, num_classes, head = pretrained_head, **kwargs)

    return model





class MobilenetFrozenAdapter(nn.Module):
    def __init__(self, backbone, extra, out_size):
        super().__init__()

        self.extra = extra

        #get all the layers up to layer 3
        self.features = nn.Sequential(*list(backbone.body.children())[:7])

        #gotta figure this part out
        conv4_block1 = self.features[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

        in_size = self.features[-1][-1].bn3.weight.shape[0]

        self.reducer = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size = 1, bias = False),
        )

        _xavier_init(self.reducer)

    def forward(self, x):
        x = self.features(x)
        x = self.reducer(x)

        output = [x]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])




def ssd_frozen_mobilenet(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):  

    size = (300, 300)

    ssd_vgg = models.detection.ssd300_vgg16(pretrained=True)
    ssd_vgg_backbone = ssd_vgg.backbone

    #grab the vgg extra to use in ssd
    ssd_vgg_extra = ssd_vgg_backbone.extra

    #get the expected output size for the last layer of the vgg backbone
    conv_layers = [i for i, block in ssd_vgg_backbone.features.named_modules() if type(block) == nn.Conv2d]
    out_size = ssd_vgg_backbone.features[int(conv_layers[-1])].out_channels

    #grab the resnet backbone from retinanet to use as the backbone for ssd
    #retinanet_backbone = models.detection.retinanet_resnet50_fpn(pretrained=True).backbone
    mobilenet_backbone = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).backbone

    #freeze all the layers before layer 3
    trainable_backbone_layers = ['layer3', 'layer4']

    for name, parameter in mobilenet_backbone.body.named_parameters():
        if all([not name.startswith(layer) for layer in trainable_backbone_layers]):
            parameter.requires_grad_(False)

    #freeze extra layers
    for name, parameter in ssd_vgg_extra.named_parameters():
        parameter.requires_grad_(False)


    backbone = MobilenetFrozenAdapter(mobilenet_backbone, ssd_vgg_extra, out_size)

    #reuse the same anchor generator from the ssd model
    anchor_generator = ssd_vgg.anchor_generator


    #don't freeze head. at least not yet.  
    ssd_vgg_head = ssd_vgg.head

    #freeze head here but add option to not freeze head
    for name, parameter in ssd_vgg_head.named_parameters():
        parameter.requires_grad_(False)
    
    model = models.detection.SSD(backbone, anchor_generator, size, num_classes, head = ssd_vgg_head, **kwargs)

    return model

class FrozenAdapter(nn.Module):
    def __init__(self, backbone, extra, out_size):
        super().__init__()

        self.extra = extra

        #get all the layers up to layer 3
        self.features = nn.Sequential(*list(backbone.body.children())[:7])

        #gotta figure this part out
        conv4_block1 = self.features[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

        in_size = self.features[-1][-1].bn3.weight.shape[0]

        self.reducer = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size = 1, bias = False),
        )

        _xavier_init(self.reducer)

    #this is backbone function that does the forward prop.
    def forward(self, x):
        x = self.features(x) #backbone <--- cached

        x = self.reducer(x) #avtn layer

        output = [x]

        #forward prop through the extra layers
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])





def ssd_frozen(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                    pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None, **kwargs: Any):  

    size = (300, 300)

    #we get the backbone from ssd + vgg object detection model
    ssd_vgg = models.detection.ssd300_vgg16(pretrained=True)
    ssd_vgg_backbone = ssd_vgg.backbone

    #grab the vgg extra to use in ssd
    ssd_vgg_extra = ssd_vgg_backbone.extra

    #get the expected output size for the last layer of the vgg backbone
    conv_layers = [i for i, block in ssd_vgg_backbone.features.named_modules() if type(block) == nn.Conv2d]
    out_size = ssd_vgg_backbone.features[int(conv_layers[-1])].out_channels

    #grab the resnet backbone from retinanet to use as the backbone for ssd
    retinanet_backbone = models.detection.retinanet_resnet50_fpn(pretrained=True).backbone

    #freeze all the layers before layer 3
    trainable_backbone_layers = ['layer3', 'layer4']

    for name, parameter in retinanet_backbone.body.named_parameters():
        if all([not name.startswith(layer) for layer in trainable_backbone_layers]):
            parameter.requires_grad_(False)

    #freeze extra layers
    for name, parameter in ssd_vgg_extra.named_parameters():
        parameter.requires_grad_(False)


    backbone = FrozenAdapter(retinanet_backbone, ssd_vgg_extra, out_size)

    #reuse the same anchor generator from the ssd model
    anchor_generator = ssd_vgg.anchor_generator


    #don't freeze head. at least not yet.  
    ssd_vgg_head = ssd_vgg.head

    #freeze head here but add option to not freeze head
    for name, parameter in ssd_vgg_head.named_parameters():
        parameter.requires_grad_(False)
    
    #connect frozen retinanet+resnet50 backbone to frozen ssd+vgg head with adaptive layer
    model = models.detection.SSD(backbone, anchor_generator, size, num_classes, head = ssd_vgg_head, **kwargs)

    return model