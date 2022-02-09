from functools import reduce
import torch
from torch.nn.modules.conv import Conv2d
from torchvision.models.detection import retinanet
import torchvision.models
import torchvision.models.detection

from coco_utils import get_kitti, get_coco
import presets

#import torchvision.models as models
import models
from models.ssd import ssd300_resnet34, ssd300_resnet50, ssd300_resnet101, ssd300_mobilenet_v2, ssd_frozen, ssd300_mobilenet_v3_large, ssd300_mobilenet_v3_small
from models.retinanet import frozen_retinanet_all
from models.ssdlite import frozen_ssdlite_resnet50
from torch import nn, Tensor

from torchvision.ops import misc

import models.ssd as ssd

num_classes = 8

inputs = torch.rand(1, 3, 300, 300)

# model = ssd300_resnet101(False, True, num_classes, True)
# model.eval()
# output = model(inputs)

def test_ssd_mobilenet(input):
    mobilenet = models.mobilenet.__dict__["mobilenet_v2"](pretrained = True, progress = True).features
    mobilenet.eval()
    #features = nn.Sequential(*mobilenet[0:14])
    features = nn.Sequential(
        nn.Sequential(*mobilenet[0:8]),
        nn.Sequential(*mobilenet[8:14])
    )
    x = input

    backbone_out_channels = features[-1][-1].conv[-1].num_features

    extra = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(backbone_out_channels, 256, kernel_size=1), # conv8_2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=(2,2)),
            nn.ReLU(inplace=True),
        ),
        nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2), # conv9_2
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
            nn.Conv2d(128, 256, kernel_size=2), # conv11_2
            nn.ReLU(inplace=True),
        )
    ])

    for block in features:
        x = block(x)
        #print(block)
        print(x.shape)

    for block in extra:
        x = block(x)
        print(x.shape)


def test_ssd_mobilenet_v2(input):
    model = ssd300_mobilenet_v2(pretrained = False, pretrained_backbone=True)
    model.eval()
    model(input)


def test_frozen_head():
    model = models.detection.ssd300_vgg16(pretrained=True)
    head = model.head
    
    #print(head)

    regression_head = head.regression_head

    for block in regression_head.module_list:
        print(block)
        #print(block.out_channels)
        




def test_frozen_extra(input):
    ssd_vgg = models.detection.ssd300_vgg16(pretrained=True)
    ssd_vgg_backbone = ssd_vgg.backbone
    
    #grab the vgg extra to use in the ssd backbone
    extra = ssd_vgg_backbone.extra
    
    #grab the resnet backbone from retinanet to use as the backbone for ssd
    retinanet_backbone = models.detection.retinanet_resnet50_fpn(pretrained=True).backbone

    #get all the layers up to layer 3
    feature_extractor = nn.Sequential(*list(retinanet_backbone.body.children())[:7])

    #gotta figure this part out
    conv4_block1 = feature_extractor[-1][0]
    conv4_block1.conv1.stride = (1, 1)
    conv4_block1.conv2.stride = (1, 1)
    conv4_block1.downsample[0].stride = (1, 1)

    in_size = feature_extractor[-1][-1].bn3.weight.shape[0]

    #get the expected output size for the last layer of the vgg backbone
    conv_layers = [i for i, block in ssd_vgg_backbone.features.named_modules() if type(block) == nn.Conv2d]
    out_size = ssd_vgg_backbone.features[int(conv_layers[-1])].out_channels

    reducer = nn.Sequential(
        nn.Conv2d(in_size, out_size, kernel_size = 1, bias = False),
    )

    #do a sample run
    x = input
    x = feature_extractor(x)
    x = reducer(x)
    print(x.shape)

    for block in extra:
        x = block(x)
        print(x.shape)

def test_frozen(input):
    name = "ssd_frozen"
    model = ssd.__dict__[name]()

    for name, parameter in model.named_parameters():
        print(name, parameter.requires_grad)

    model.eval()

    output = model(input)

def test_ssdlite_resnet34(input):
    # from functools import partial
    # from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
    # from torchvision.models.detection import _utils as det_utils

    # backbone = models._resnet_extractor('resnet34', True, 4)

    # #print(backbone)
    # norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    # size = (320, 320)
    # anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    # out_channels = det_utils.retrieve_out_channels(backbone, size)
    # num_anchors = anchor_generator.num_anchors_per_location()
    # assert len(out_channels) == len(anchor_generator.aspect_ratios)
    # defaults = {
    #     "score_thresh": 0.001,
    #     "nms_thresh": 0.55,
    #     "detections_per_img": 300,
    #     "topk_candidates": 300,
    #     # Rescale the input in a way compatible to the backbone:
    #     # The following mean/std rescale the data from [0, 1] to [-1, -1]
    #     "image_mean": [0.5, 0.5, 0.5],
    #     "image_std": [0.5, 0.5, 0.5],
    # }

    # model = models.detection.SSD(backbone, anchor_generator, size, num_classes,
    #             head=models.detection.ssdlite.SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer))
    model = models.ssdlite_resnet34(pretrained=False, progress=True, pretrained_backbone=True)

    model.eval()
    model(input)

def get_backbone_sizes(backbone : nn.Module, size, in_shape: int = 16):
    input = torch.rand(1, 3, size[0], size[1])


    # features = nn.Sequential(
    #     nn.Sequential(*list(backbone.children())[:-4]),
    #     nn.Sequential(*list(backbone.children())[-4])
    # )
        
    #print(features)
    x = input

    backbone_list = nn.Sequential(*list(backbone.children())[:-1])

    

    count = 0
    for block in backbone_list:
        x = block(x)

        if x.shape[2] > in_shape:
            count += 1
        else:
            break

    
    features = nn.ModuleList([
        nn.Sequential(*list(backbone.children())[:count]),
        nn.Sequential(*list(backbone.children())[count])
    ])


    


    return features

def test_ssdlite_resnet50_frozen(input):
    backbone = torchvision.models.resnet50(pretrained=True, progress=True)
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True, progress=True)

    size = (320,320)

    #grab the ssdlite head and extra layers
    head = model.head
    extra = model.backbone.extra

    cls_head = head.classification_head

    #print(cls_head)
    #print(extra[0])

    #get the input size for the head
    out_channels = []
    for block in cls_head.module_list:
        out_channels.append(block[0][0].weight.shape[0])

    #get the length of the extra layers
    #we expect that the size of the image is 1 x 1 at the end of the extra layers
    #print(len(extra))
    
    in_shape = 2**(len(extra))
    #print(in_shape)
    #print(model.backbone.features[-1])
    print(out_channels)

    features = get_backbone_sizes(backbone, size, in_shape)

    #print(features)

    x = input
    in_channels = []
    for block in features:
        x = block(x)
        in_channels.append(x.shape[1])
        #print(x.shape)


    reducers = nn.ModuleList()

    for in_channel, out_channel in zip(in_channels, out_channels[0:len(in_channels)]):
        reducers.append(
            nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size = 1, bias = False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )
        )

    #print(reducer)
    x = input

    output = []
    # for block in features:
    #     x = block(x)
    #     output.append(x)

    assert len(features) == len(reducers)
    for feature, reducer in zip(features, reducers):
        x = feature(x)
        y = reducer(x)
        print(y.shape)
        output.append(y)

def test_ssdlite_resnet50(input):
    my_model = frozen_ssdlite_resnet50()

    my_model.eval()

    output = my_model(input)

    #print(output.shape)
        
    
    # model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True, progress=True)
    # backbone = model.backbone.features
    # extra = model.backbone.extra
    # #print(backbone)

    # x = input
    # for block in backbone:
    #     x = block(x)
    #     print(x.shape)

    
    # extra.eval()

    # for block in extra:
    #     x = block(x)
    #     print(x.shape)
    
    #print(extra)



def test_retinanet_50(input):
    model = frozen_retinanet_all()
    model.eval()
    model(input)

def test_ssd_mobilenetv3_large(input):
    #lsssd300_mobilenet_v2(pretrained=False, progress=True)
    model = ssd300_mobilenet_v3_large(pretrained=False, progress=True)
    model.eval()
    output = model(input)

def test_ssd_mobilenetv3_small(input):
    #lsssd300_mobilenet_v2(pretrained=False, progress=True)
    model = ssd300_mobilenet_v3_small(pretrained=False, progress=True)
    model.eval()
    output = model(input)

def test_print():
    print("Hello World")



def test_evaluate(checkpoint):
    from coco_utils import get_coco, get_coco_kp
    import utils
    from engine import evaluate
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    

    def get_dataset(name, image_set, transform, data_path):
        paths = {
            "coco": (data_path, get_coco, 91),
            "coco_kp": (data_path, get_coco_kp, 2),
            "kitti": (data_path, get_kitti, 8)
        }
        p, ds_fn, num_classes = paths[name]

        ds = ds_fn(p, image_set=image_set, transforms=transform)
        return ds, num_classes

    def get_transform(train, data_augmentation):
        return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()

    model = ssd300_resnet34(pretrained=False, progress=True)
    model_without_ddp = model
    #checkpoint = torch.load('fused_models/ssd_resnet34.pth', map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])


    dataset_test, _ = get_dataset('coco', "val", get_transform(False, 'ssd'), '/data/datasets/coco')

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=4,
        collate_fn=utils.collate_fn)
    
    device = torch.device('cuda')
    model.to(device)

    evaluate(model, data_loader_test, device=device)

def test_file_print(print_fcn, checkpoint):
    import sys

    print('saving results')
    original_stdout = sys.stdout

    with open('output.txt', 'w') as f:
        sys.stdout = f
        print_fcn(checkpoint)
        sys.stdout = original_stdout


#test_ssd_mobilenetv3_small(inputs)

checkpoint = torch.load('fused_models/ssd_resnet34.pth', map_location='cpu')

test_file_print(test_evaluate, checkpoint)