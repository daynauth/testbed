from typing import Tuple
import torchvision.transforms as T

import models
import cv2

import os


class TinyCoco:
    def __init__(self, coco_dir: str, val: bool = True) -> None:
        val_dir = 'val2017' if val else 'test2017'
        path = os.path.join(coco_dir, val_dir)
        self.image_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        self.image_list.sort()

    def __getitem__(self, idx: int):
        image =  cv2.imread(self.image_list[idx])
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def list_infer(model1, model2, image_list):
    model1.eval()
    model2.eval()

    return [(model1(image)[0], model2(image)[0]) for image in image_list]
    

def draw_bounding_boxes(image, outputs):
    image = image.copy()
    keep = outputs['scores'] > 0.5
    boxes = outputs['boxes'][keep].tolist()

    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)

    return image

def get_images(num_images, image_set):
    image_list = []
    for i in range(num_images):
        image_list.append(image_set[i])

    return image_list

def transforms(image_list):
    t = T.ToTensor()
    return [t(image).unsqueeze(0) for image in image_list]

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='Hitachi Demo', add_help=add_help)
    parser.add_argument('--model1', default='ssdlite320_mobilenet_v3_large', help='model')
    parser.add_argument('--model2', default='retinanet_resnet50_fpn', help='model')
    parser.add_argument('--n', default=5, type=int, help='number of images to test')

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    model1 = models.__dict__[args.model1](pretrained = True)
    model2 = models.__dict__[args.model2](pretrained = True)
    num_images = args.n


    dataset = TinyCoco('/data/datasets/coco/', True)
    image_list = get_images(num_images, dataset)
    tensor_list = transforms(image_list)
    output = list_infer(model1, model2, tensor_list)

  
    output = [(draw_bounding_boxes(image, o1), draw_bounding_boxes(image, o2)) for (o1, o2), image in zip(output, image_list)]

    for i, (o1, o2) in enumerate(output):
        o1 = cv2.cvtColor(o1, cv2.COLOR_RGB2BGR)
        o2 = cv2.cvtColor(o2, cv2.COLOR_RGB2BGR)
        cv2.imwrite('outputs/output_' + str(i) + '_1.png', o1)
        cv2.imwrite('outputs/output_' + str(i) + '_2.png', o2)
        

