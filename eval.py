from random import seed
import torch
from torch import distributed
from torch.utils import data
import torchvision
import time
import os
import datetime

import presets
import utils

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


#import torchvision.models as models

from models.ssd import ssd300_resnet50, ssd_resnet50_adapted, ssd_resnet50_adapted_v2, ssd300_resnet101,\
    ssd300_resnet152, ssd300_mobilenet_v2, ssd_frozen


import models
from engine import train_one_epoch, evaluate
from coco_utils import get_coco, get_coco_kp, get_kitti

torch.autograd.set_detect_anomaly(False)  
torch.autograd.profiler.profile(False)  
torch.autograd.profiler.emit_nvtx(False)

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
    
def file_print(evaluate, model, data_loader_test, device, epoch):
    import sys

    print('saving results')
    original_stdout = sys.stdout

    output = os.path.join('logs', 'output' + str(epoch) + '.txt')
    with open(output, 'w') as f:
        sys.stdout = f
        evaluate(model, data_loader_test, device=device)
        sys.stdout = original_stdout

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    parser.add_argument('--data-path', default='/data/datasets/coco', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='ssd_frozen', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--eval-model', default='', help='resume from checkpoint')
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')
    parser.add_argument('--eval-epoch', default=1, help='evaluate epoch')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser

def main(args):
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    # Data Loading code
    print('loading data')
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(False, args.data_augmentation), args.data_path)
    
    print('creating data loaders')
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = models.__dict__[args.model](False, True, 91, True)

    model.to(device)

    epoch = args.eval_epoch

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    file_print(evaluate, model, data_loader_test, device, epoch)



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)