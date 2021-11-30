from random import seed
from mongoengine.queryset import transform
import torch
import torchvision

import utils
from  dataset import load_coco_dataset
from models.ssd import ssd300_resnet50
from engine import train_one_epoch
from coco_utils import get_coco, get_coco_kp


import fiftyone.utils.torch as fout

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


#all the configs for training
batch_size = 2
lr = 0.02
momentum = 0.9
weight_decay = 1e-4
lr_scheduler = "multisteplr"
lr_steps = [16,22]
lr_gamma = 0.1
epochs = 4
start_epoch = 0
print_freq = 20

device = torch.device('cuda')

# print('loading data')
# dataset = load_coco_dataset('ssd300_coco_train', 'train')
# num_classes = len(dataset.default_classes)

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor()
    ]
)

dataset = torchvision.datasets.CocoDetection(
    '/data/datasets/train/data', 
    '/data/datasets/train/labels.json',
    transforms=transforms
)


#load a small sample of the dataset for now
#view = dataset.take(500, seed = 51)

#print(view.first())

#use the whole dataset for training in the future
#image_paths, sample_ids = zip(*[(s.filepath, s.id) for s in view])

#load the dataset as a pytorch dataset


# dataset = fout.TorchImageDataset(
#     image_paths, sample_ids=sample_ids, transform = transforms
# )

print('creating data loaders')
# train_sampler = torch.utils.data.RandomSampler(dataset)

# train_batch_sampler = torch.utils.data.BatchSampler(
#     train_sampler, batch_size, drop_last = True
# )

# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_sampler = train_batch_sampler, collate_fn = utils.collate_fn
# )



# print('creating model')
# model = ssd300_resnet50(
#     pretrained=False, 
#     progress=True, 
#     num_classes=num_classes, 
#     pretrained_backbone=True
# )

# model.to(device)
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(
#         params, lr=lr, momentum=momentum, weight_decay=weight_decay)

# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)

# print("start training")
# for epoch in range(start_epoch, epochs):
#     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)
#     lr_scheduler.step()
