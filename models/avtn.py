from collections import OrderedDict

import torch
import torchvision.models
import torch.nn.functional as F
from torch import nn

__all__ = ['ssd300_avtn_retinanet_resnet', 'ssd300_avtn_faster_rcnn_resnet', 'ssd300_avtn_faster_rcnn_mobilenet_v3_large']

def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


class ReducedAVTN2(nn.Module):
    def __init__(self, in_channels, out_channels, out_shape, features, extra):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.extra = extra
        self.features = nn.Sequential(*list(features.children())[:7])
        


        #gotta figure this part out
        conv4_block1 = self.features[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

        print(self.features[-1])


        self.in_channels = self.features[-1][-1].bn3.weight.shape[0]

        self.adjust_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

        _xavier_init(self.adjust_layer)

    def forward(self, x):
        x = self.features(x) #backbone <--- cached
        x = self.adjust_layer(x) #avtn layer

        output = [x]

        #forward prop through the extra layers
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


class ReducedAVTN(nn.Module):
    def __init__(self, in_channels, out_channels, out_shape, features, extra):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.extra = extra
        self.features = nn.Sequential(*list(features.children())[:7])



        #gotta figure this part out
        conv4_block1 = self.features[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)


        self.in_channels = self.features[-1][-1].bn3.weight.shape[0]

        self.adjust_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
            #nn.BatchNorm2d(self.out_channels),
            #nn.ReLU(inplace=True)
        )

        _xavier_init(self.adjust_layer)

    def forward(self, x):
        x = self.features(x) #backbone <--- cached
        x = self.adjust_layer(x) #avtn layer

        output = [x]

        #forward prop through the extra layers
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


class NaiveAVTN(nn.Module):
    def __init__(self, in_channels, out_channels, out_shape, features, extra):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape
        self.features = features

        self.extra = extra

        
        self.adjust_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x):
        x = self.features(x)
        x = list(x.values())[-1]
        x = F.interpolate(x, size = self.out_shape, mode = "nearest")
        x = self.adjust_layer(x)

        output = [x]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])



def calculate_params(shape, original_backbone, new_backbone):
    inputs = torch.rand(1, 3, shape[0], shape[1])
    output = original_backbone(inputs)
    out_channel = output.shape[1]
    output_shape = output.shape[-2:]

    new_backbone_output = new_backbone(inputs)
    in_channel = list(new_backbone_output.values())[-1].shape[1]


    return in_channel, out_channel, output_shape



def _ssd_avtn(features, ssd, pretrained: bool = False, progress: bool = True, num_classes: int = 91, pretrained_backbone: bool = True):
    size = (300, 300)

    extra_layers = ssd.backbone.extra
    anchor_generator = ssd.anchor_generator
    head = ssd.head


    #freeze all the layers before layer 3
    trainable_backbone_layers = ['layer3', 'layer4']

    for name, parameter in features.named_parameters():
        if all([not name.startswith(layer) for layer in trainable_backbone_layers]):
            parameter.requires_grad_(False)


    # for name, parameter in features.named_parameters():
    #     parameter.requires_grad_(False)

    for name, parameter in extra_layers.named_parameters():
        parameter.requires_grad_(False)

    for name, parameter in head.named_parameters():
        parameter.requires_grad_(False)

    in_channel, out_channel, output_shape = calculate_params(size, ssd.backbone.features, features)

    #backbone = NaiveAVTN(in_channel, out_channel, output_shape, features, extra_layers)
    backbone = ReducedAVTN2(in_channel, out_channel, output_shape, features, extra_layers)
    return torchvision.models.detection.SSD(backbone, anchor_generator, size, num_classes, head = head)


def ssd300_avtn_retinanet_resnet(pretrained: bool = False, progress: bool = True, num_classes: int = 91, pretrained_backbone: bool = True):
    backbone = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True).backbone.body
    ssd = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    return _ssd_avtn(backbone, ssd, False, True, 91, True)

def ssd300_avtn_faster_rcnn_resnet(pretrained: bool = False, progress: bool = True, num_classes: int = 91, pretrained_backbone: bool = True):
    backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).backbone.body
    ssd = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    return _ssd_avtn(backbone, ssd, False, True, 91, True)

def ssd300_avtn_faster_rcnn_mobilenet_v3_large(pretrained: bool = False, progress: bool = True, num_classes: int = 91, pretrained_backbone: bool = True):
    backbone = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).backbone.body
    ssd = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    return _ssd_avtn(backbone, ssd, False, True, 91, True)

# def ssdlite300_avtn_retinanet_resnet(pretrained: bool = False, progress: bool = True, num_classes: int = 91, pretrained_backbone: bool = True):
#     backbone = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True).backbone.body
#     ssd = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
#     return _ssd_avtn(backbone, ssd, False, True, 91, True)

def new_avtn():
    backbone = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True).backbone.body
    ssd = torchvision.models.detection.ssd300_vgg16(pretrained=True)

    print(backbone)

    #first lets get the output of the backbone
    inputs = torch.rand(1, 3, 300, 300)

    backbone_output = backbone(inputs)

    for key, values in backbone_output.items():
        print(values.shape)


    ssd.eval()
    ssd_output = ssd.backbone(inputs)

    print('ssd output')
    for key, values in ssd_output.items():
        print(values.shape)

def test(model):
    inputs = torch.rand(1, 3, 300, 300)
    model.eval()
    model(inputs)


model = ssd300_avtn_retinanet_resnet(False, True, 91, True)
#test(model)

# model = ssd300_avtn_faster_rcnn_mobilenet_v3_large(False, True, 91, True)
# test(model)

# model = ssd300_avtn_faster_rcnn_resnet(False, True, 91, True)
# test(model)

#model = ssdlite300_avtn_retinanet_resnet(False, True, 91, True)
#print(model)
#test(model)

#new_avtn()