import torch.nn as nn
import torchvision.transforms as T
from torch import Tensor
import torch

import models

from utils import draw_bounding_boxes
from typing import Union, Tuple, List, Dict


import os
import cv2
import json
import progressbar  


def loadcat(dir = None) -> Dict[int, str]:
    coco_dir = '/data/datasets/coco/'
    ann_file = os.path.join(coco_dir, 'annotations', 'instances_val2017.json')
    f = open(ann_file)
    ann = json.load(f)
    cat = ann['categories']

    return dict([(c['id'], c['name']) for c in cat])



def transforms(image_list: List):
    t = T.ToTensor()
    return [t(image).unsqueeze(0) for image in image_list]


def load_model(name, device):
    model = models.__dict__[name](pretrained = True)
    model.eval()
    model.to(device)

    return model

def infer(model, input, image, categories):
    output = model(input)[0]

    o = draw_bounding_boxes(image.copy(), output, categories)
    o = cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
    return o

def video_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading models')
    model1 = load_model(args.model1, device)
    model2 = load_model(args.model2, device)
    print('Models loaded')


    img_path = args.path
    img_list = [os.path.join(img_path, f) for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    img_list.sort()
    img_list = [cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB) for im in img_list]

    tensor_list = transforms(img_list)
    tensor_list = list(t.to(device) for t in tensor_list)

    categories = loadcat()


    height = tensor_list[0][0].shape[1]
    width = tensor_list[0][0].shape[2]
    size = (width, height * 2)

    img_array = []


    print('Running inferences')
    widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    
    bar = progressbar.ProgressBar(max_value=len(img_list), widgets=widgets).start()

    #setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(args.model1, font, 1, 2)[0]

    # get coords based on boundary
    textX = int((width - textsize[0]) / 2)
    textY = int(height - textsize[1])

    
    for i, (img, tensor) in enumerate(zip(img_list, tensor_list)):
        o1 = infer(model1, tensor, img, categories)
        o2 = infer(model2, tensor, img, categories)

        cv2.putText(o1, args.model1, (textX, textY ), font, 1, (0, 255, 0), 3)
        cv2.putText(o2, args.model2, (textX, textY ), font, 1, (0, 255, 0), 3)

        img_array.append(cv2.vconcat([o1, o2]))
        bar.update(i)
        #break

    cv2.imwrite('test.png', img_array[0])
    out = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)


    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='Hitachi Demo', add_help=add_help)
    parser.add_argument('--model1', default='retinanet_resnet50_fpn', help='model')
    parser.add_argument('--model2', default='ssdlite320_mobilenet_v3_large', help='model')
    parser.add_argument('--path', default='/data/datasets/kitti/scenes/scene_02', help = 'path to your dataset')

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    
    video_demo(args)