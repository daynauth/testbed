from torch import nn

import torchvision.models as models


__all__ = [
    'FasterRCNN_resnet50',
]

# for resnet50 FPN
def FasterRCNN_resnet50():
    backbone = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True).backbone.body
    fpn = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True).backbone.fpn
    model = ResNet50(backbone, fpn)
    return model

class ResNet50(nn.Module):
    def __init__(self, backbone, fpn):
        super().__init__()
        self.feature = backbone
        self.fpn = fpn
        
    def forward(self, x):
        out_feature = self.feature(x)
        out_fpn = self.fpn(out_feature)
        return out_feature, out_fpn

