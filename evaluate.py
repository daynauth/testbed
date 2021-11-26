import fiftyone as fo
import fiftyone.zoo as foz

import torchvision.models as models

from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

def load_coco_dataset(dataset_name : str) -> fo.Dataset:
    if not fo.dataset_exists(dataset_name):
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            dataset_name=dataset_name
        )

        dataset.persistent = True

    else:
        dataset = fo.load_dataset(dataset_name)
    
    return dataset

#dataset = load_coco_dataset('efficientdet-coco')

model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
#model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)




#print(model.head)

#print(model)