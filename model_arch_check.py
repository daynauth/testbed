import torch
import torchvision.models as models


model = models.detection.keypointrcnn_resnet50_fpn()

print(model)
