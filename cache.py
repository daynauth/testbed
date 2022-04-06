from random import seed
import torch
from torch import distributed
from torch.utils import data
import torchvision
import time
import os
import datetime
from pathlib import Path
import pickle

import presets
import utils

#import torchvision.models as models

from models.cacheNet import FasterRCNN_resnet50

import models
from coco_utils import get_coco, get_coco_kp, get_kitti, get_coco_cache

torch.autograd.set_detect_anomaly(False)  
torch.autograd.profiler.profile(False)  
torch.autograd.profiler.emit_nvtx(False)

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2),
        "kitti": (data_path, get_kitti, 8),
        "coco_cache": (data_path, get_coco_cache, 91),
    }
    p, ds_fn, num_classes = paths[name]
    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()

def cache_image(model, image, image_idx, cache_dir, device):
    model.eval()
    out_feature, out_fpn = model(image)
    '''
    print("=======================================")
    for f in out_feature:
        print(f, out_feature[f].shape)
    
    for f in out_fpn:
        print(f, out_fpn[f].shape)
    
    print("=======================================")
    '''
    out_path = Path(cache_dir)
    '''
    for t in out_feature:
        tensor = out_feature[t]
        torch.save(tensor, out_path/f'{image_idx}_{t}_backbone.pt')
    '''
    with open(out_path/f'{image_idx}_backbone.pickle', 'wb') as handle:
        pickle.dump({
            "feature": out_feature,
            #"fpn": out_fpn
        }, handle)

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    parser.add_argument('--data-path', default='/data/datasets/coco', help='dataset')
    parser.add_argument('--dataset', default='coco_cache', help='dataset')
    parser.add_argument('--model', default='FasterRCNN_resnet50', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--cache-dir', default='.', help='path where to save')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--data-augmentation', default="cache", help='data augmentation policy (default: cache/no aug)')
    return parser

def main(args):
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True


    # Data Loading code
    print('loading data')
    dataset, _ = get_dataset(args.dataset, "train", get_transform(True, args.data_augmentation), args.data_path)

    print('creating data loaders')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=args.workers,
        collate_fn=utils.collate_fn, pin_memory = True)

    # TODO
    print("Creating model")
    model = models.__dict__[args.model]()
    print(model)
    #model.to(device)
    print("Start caching")
    start_time = time.time()
    cache_dir = args.cache_dir
    Path(cache_dir).mkdir(exist_ok=True)
    count = 0
    for image, _, image_idx in data_loader:
        image_idx = image_idx[0]
        if image_idx < 7717:
            continue
        image = torch.stack(image, dim=0)
        #image = image.to(device)
        print(image_idx)
        if(image_idx > 10000):
            # Cache the first 10000 images
            break
        cache_image(model, image, image_idx, cache_dir, device)
    
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)