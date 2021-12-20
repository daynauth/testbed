import torchvision.models as models
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models.vgg import vgg16
from models.ssd import ssd_frozen

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

resnet34 = models.resnet34()
resnet50 = models.resnet50()
mobilenetv2 = models.mobilenet_v2()
mobilenet_v3_small = models.mobilenet_v3_small()
mobilenet_v3_large = models.mobilenet_v3_large()
vgg = models.vgg16()


ssd_vgg = models.detection.ssd300_vgg16()
ssd_lite = models.detection.ssdlite320_mobilenet_v3_large()
retinanet = models.detection.retinanet_resnet50_fpn()
ssd_frozen = ssd_frozen()
adapter = ssd_frozen.backbone.reducer

head = retinanet.head

#print(count_parameters(adapter))
#print(adapter)

for module in ssd_vgg.modules():
    print(module._get_name())