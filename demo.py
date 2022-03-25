import torch.nn as nn
import torchvision.transforms as T
from torch import Tensor

from numpy import ndarray

import models
import cv2

import os
import json
from typing import Union, Tuple, List, Dict
import random

ColorConstants = [
    (202,255,112),
    (255,140,0),
    (104,34,139),
    (233,150,122),
    (151,255,255),
    (0,206,209),
    (255,20,147),
    (0,191,255),
    (30,144,255),
    (255,48,48),
    (255,125,64),
    (255,193,37)
]

def get_color(id: int) -> Tuple[int, int, int]:
    return ColorConstants[id % len(ColorConstants)]



class TinyCoco:
    def __init__(self, data_dir: str) -> None:
        self.coco_dir = '/data/datasets/coco/'
        val_dir = 'val2017'
        self.data_dir = data_dir
        path = os.path.join(self.data_dir, val_dir)

        self.image_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        self.image_list.sort()

        self._loadcat()

    def __getitem__(self, idx: Union[int, slice]) -> Union[ndarray, List[ndarray]]:
        if isinstance(idx, slice):
            image = [cv2.imread(im) for im in self.image_list[idx]]
            return [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in image]

        image =  cv2.imread(self.image_list[idx])
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        return len(self.image_list)


    def _loadcat(self) -> None:
        ann_file = os.path.join(self.coco_dir, 'annotations', 'instances_val2017.json')
        f = open(ann_file)
        ann = json.load(f)
        cat = ann['categories']

        self.cat : dict[int, str] = dict([(c['id'], c['name']) for c in cat])

def list_infer(model1 : nn.Module, model2 : nn.Module, image_list: List[ndarray]) -> List[Tuple[Tensor, Tensor]]:
    model1.eval()
    model2.eval()

    return [(model1(image)[0], model2(image)[0]) for image in image_list]
    
def to_int(box: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

def draw_bounding_boxes(image: ndarray, outputs, cat ) -> ndarray:
    image = image.copy()
    keep = outputs['scores'] > 0.5
    boxes = outputs['boxes'][keep].tolist()
    labels = outputs['labels'][keep].tolist()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = to_int(box)
        cv2.rectangle(image,(x1,y1),(x2,y2),get_color(labels[i]),2)
        cv2.putText(image, cat[labels[i]], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(labels[i]), 2)

    return image

def get_images(num_images, image_set, rand = True):
    if rand:
        return random.choices(image_set, k=num_images)

    return image_set[0: num_images]


def transforms(image_list):
    t = T.ToTensor()
    return [t(image).unsqueeze(0) for image in image_list]

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='Hitachi Demo', add_help=add_help)
    parser.add_argument('--model1', default='ssdlite320_mobilenet_v3_large', help='model')
    parser.add_argument('--model2', default='retinanet_resnet50_fpn', help='model')
    parser.add_argument('--n', default=5, type=int, help='number of images to test')
    parser.add_argument('--path', default='/data/dataset/coco', help = 'path to your dataset')
    parser.add_argument('--random', action='store_true')


    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()


    model1 = models.__dict__[args.model1](pretrained = True)
    model2 = models.__dict__[args.model2](pretrained = True)
    num_images = args.n


    dataset = TinyCoco(args.path)
    categories = dataset.cat
    image_list = get_images(num_images, dataset, args.random)
    tensor_list = transforms(image_list)
    output = list_infer(model1, model2, tensor_list)

  
    output = [(draw_bounding_boxes(image, o1, categories), draw_bounding_boxes(image, o2, categories)) 
        for (o1, o2), image in zip(output, image_list)]

    for i, (o1, o2) in enumerate(output):
        o1 = cv2.cvtColor(o1, cv2.COLOR_RGB2BGR)
        o2 = cv2.cvtColor(o2, cv2.COLOR_RGB2BGR)
        cv2.imwrite('outputs/output_' + str(i) + '_1.png', o1)
        cv2.imwrite('outputs/output_' + str(i) + '_2.png', o2)
        

