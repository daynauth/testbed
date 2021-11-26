from random import seed
import torch
import torchvision

import utils
from  dataset import load_coco_dataset
from models.ssd import ssd300_resnet50

import fiftyone.utils.torch as fout

model = ssd300_resnet50(pretrained=False, progress=True, pretrained_backbone=True)

device = torch.device('cuda')

print('loading data')
dataset = load_coco_dataset('ssd300_coco_train', 'train')
num_classes = len(dataset.default_classes)


#load a small sample of the dataset for now
view = dataset.take(500, seed = 51)

#use the whole dataset for training in the future
image_paths, sample_ids = zip(*[(s.filepath, s.id) for s in view])

#load the dataset as a pytorch dataset
transforms = {
    torchvision.transforms.ToTensor()
}

dataset = fout.TorchImageDataset(
    image_paths, sample_ids=sample_ids, transform = transforms
)

print('creating data loaders')
train_sampler = torch.utils.data.RandomSampler(dataset)
batch_size = 2
train_batch_sampler = torch.utils.data.BatchSampler(
    train_sampler, batch_size, drop_last = True
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_sampler = train_batch_sampler, collate_fn = utils.collate_fn
)

#num_classes = 91
#print(dataset.default_classes)
#print(len(dataset.default_classes))
#print(len(dataset))